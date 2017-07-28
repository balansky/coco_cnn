from cores import model
import tensorflow as tf
import os
import json
import argparse
from tensorflow.python.saved_model import signature_constants as sig_constants
import threading


class EvaluationRunHook(tf.train.SessionRunHook):
    """EvaluationRunHook performs continuous evaluation of the model.
    Args:
      checkpoint_dir (string): Dir to store model checkpoints
      metric_dir (string): Dir to store metrics like accuracy and auroc
      graph (tf.Graph): Evaluation graph
      eval_frequency (int): Frequency of evaluation every n train steps
      eval_steps (int): Evaluation steps to be performed
    """
    def __init__(self, trainer, checkpoint_dir,eval_batch_size,
                 eval_frequency, eval_steps=None, **kwargs):

        self._eval_steps = eval_steps
        self._checkpoint_dir = checkpoint_dir
        self._kwargs = kwargs
        self._eval_every = eval_frequency
        self._latest_checkpoint = None
        self._checkpoints_since_eval = 0

        # With the graph object as default graph
        # See https://www.tensorflow.org/api_docs/python/tf/Graph#as_default
        # Adds ops to the graph object
        # evaluation_graph = tf.Graph()
        with tf.Graph().as_default() as evaluation_graph:

            eval_inputs, eval_labels = trainer.input_fn('val', eval_batch_size, None)
            cross_entropy, correct_prediction = trainer.eval_fn(eval_inputs, eval_labels)

            # Op that creates a Summary protocol buffer by merging summaries
            self._summary_op = tf.summary.merge_all()

            global_step = tf.contrib.framework.get_or_create_global_step()

            # Saver class add ops to save and restore
            # variables to and from checkpoint
            self._saver = tf.train.Saver()

            # Creates a global step to contain a counter for
            # the global training step
            self._gs = global_step

            self._loss_ops = cross_entropy
            self._eval_ops = correct_prediction

        # MonitoredTrainingSession runs hooks in background threads
        # and it doesn't wait for the thread from the last session.run()
        # call to terminate to invoke the next hook, hence locks.
        self._graph = evaluation_graph
        self._eval_lock = threading.Lock()
        self._checkpoint_lock = threading.Lock()
        self._file_writer = tf.summary.FileWriter(
            os.path.join(checkpoint_dir, 'eval'))

    def after_run(self, run_context, run_values):
        # Always check for new checkpoints in case a single evaluation
        # takes longer than checkpoint frequency and _eval_every is >1
        self._update_latest_checkpoint()

        if self._eval_lock.acquire(False):
            try:
                if self._checkpoints_since_eval > self._eval_every:
                    self._checkpoints_since_eval = 0
                    self._run_eval()
            finally:
                self._eval_lock.release()

    def _update_latest_checkpoint(self):
        """Update the latest checkpoint file created in the output dir."""
        if self._checkpoint_lock.acquire(False):
            try:
                latest = tf.train.latest_checkpoint(self._checkpoint_dir)
                if not latest == self._latest_checkpoint:
                    self._checkpoints_since_eval += 1
                    self._latest_checkpoint = latest
            finally:
                self._checkpoint_lock.release()

    def end(self, session):
        """Called at then end of session to make sure we always evaluate."""
        self._update_latest_checkpoint()

        with self._eval_lock:
            self._run_eval()

    def _run_eval(self):
        """Run model evaluation and generate summaries."""
        coord = tf.train.Coordinator(clean_stop_exception_types=(
            tf.errors.CancelledError, tf.errors.OutOfRangeError))

        with tf.Session(graph=self._graph) as session:
            # Restores previously saved variables from latest checkpoint
            self._saver.restore(session, self._latest_checkpoint)

            session.run([
                tf.tables_initializer(),
                tf.local_variables_initializer(),
                # tf.global_variables_initializer()
            ])
            tf.train.start_queue_runners(coord=coord, sess=session)
            train_step = session.run(self._gs)

            tf.logging.info('Starting Evaluation For Step: {}'.format(train_step))
            total_loss = 0
            total_acc = 0
            with coord.stop_on_exception():
                eval_step = 0
                while not coord.should_stop() and (self._eval_steps is None or
                                                           eval_step < self._eval_steps):
                    summaries, eval_loss, eval_accuracy = session.run(
                        [self._summary_op, self._loss_ops, self._eval_ops])
                    if eval_step % 100 == 0:
                        tf.logging.info("On Evaluation Step: {}".format(eval_step))
                    eval_step += 1
                    total_loss += eval_loss
                    total_acc += eval_accuracy
            tf.logging.info("Average Loss : %s" % str(total_loss/eval_step))
            tf.logging.info("Average Accuracy : %s" % str(total_acc/eval_step))
            # Write the summaries
            self._file_writer.add_summary(summaries, global_step=train_step)
            self._file_writer.flush()
            tf.logging.info(eval_accuracy)



def run(target, cluster_spec, is_chief, job_dir, data_dir, config_dir, sup_cats,
        train_steps, train_batch_size, eval_steps,eval_batch_size, eval_frequency,
        learning_rate, decay_frequency, decay_rate, num_epochs, num_threads):

    trainer = model.MultiLabelTrainer(data_dir, config_dir, sup_cats)
    if is_chief:
        hooks = [EvaluationRunHook(trainer, job_dir, eval_batch_size,
                                   eval_frequency, eval_steps=eval_steps)]
    else:
        hooks = []
    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
            batch_inputs, batch_labels = trainer.input_fn('train', train_batch_size, num_epochs, num_threads)

            train_op, global_step = trainer.train_fn(batch_inputs, batch_labels, learning_rate, decay_frequency, decay_rate)
        with tf.train.MonitoredTrainingSession(master=target, is_chief=is_chief, checkpoint_dir=job_dir,
                                               hooks=hooks, save_checkpoint_secs=60,
                                               save_summaries_steps=50) as session:
            step = global_step.eval(session=session)
            while (train_steps is None or
                           step < train_steps) and not session.should_stop():
                step, _ = session.run([global_step, train_op])
            latest_checkpoint = tf.train.latest_checkpoint(job_dir)
            if is_chief:
                build_and_run_exports(trainer, latest_checkpoint, job_dir)


def build_and_run_exports(trainer, latest, job_dir):
    prediction_graph = tf.Graph()
    export_dir = os.path.join(job_dir, 'export')
    exporter = tf.saved_model.builder.SavedModelBuilder(export_dir)
    with prediction_graph.as_default():
        incoming_data = tf.placeholder(tf.string, name='incoming_data')
        top_classes, class_values, _ = trainer.inference(incoming_data)
        saver = tf.train.Saver()
        predict_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(incoming_data)
        classes_output_tensor_info = tf.saved_model.utils.build_tensor_info(top_classes)
        scores_output_tensor_info = tf.saved_model.utils.build_tensor_info(class_values)
        signature_def = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                'images': predict_inputs_tensor_info
            },
            outputs={
                'classes': classes_output_tensor_info,
                'scores': scores_output_tensor_info
            },
            method_name=sig_constants.PREDICT_METHOD_NAME
        )

    with tf.Session(graph=prediction_graph) as session:
        session.run([tf.local_variables_initializer(), tf.tables_initializer()])
        saver.restore(session, latest)
        exporter.add_meta_graph_and_variables(
            session,
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                sig_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
            },
            legacy_init_op=tf.saved_model.main_op.main_op()
        )

        exporter.save()
    print('Successfully exported model to %s' % export_dir)


def dispatch(*args, **kwargs):
    tf_config = os.environ.get('TF_CONFIG')

    if not tf_config:
        return run('', cluster_spec=None, is_chief=True, *args, **kwargs)

    tf_config_json = json.loads(tf_config)

    cluster = tf_config_json.get('cluster')
    job_name = tf_config_json.get('task', {}).get('type')
    task_index = tf_config_json.get('task', {}).get('index')

    # If cluster information is empty run local
    if job_name is None or task_index is None:
        return run('', cluster_spec=None, is_chief=True, *args, **kwargs)

    cluster_spec = tf.train.ClusterSpec(cluster)
    server = tf.train.Server(cluster_spec,
                             job_name=job_name,
                             task_index=task_index)
    if job_name == 'ps':
        server.join()
        return
    elif job_name in ['master', 'worker']:
        return run(server.target, cluster_spec, job_name == 'master', *args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',
                        default='gs://lace-data/tfrecords',
                        type=str,
                        help='Local Tfrecord Data')
    parser.add_argument('--sup-cats',
                        default=None,
                        type=str,
                        help='Evaluation files local or GCS', nargs='+')
    parser.add_argument('--config-dir',
                        required=True,
                        type=str,
                        help="""\
                        GCS or local dir for checkpoints, exports, and
                        summaries. Use an existing directory to load a
                        trained model, or a new directory to retrain""")
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help="""\
                        GCS or local dir for checkpoints, exports, and
                        summaries. Use an existing directory to load a
                        trained model, or a new directory to retrain""")
    parser.add_argument('--train-steps',
                        type=int,
                        help='Maximum number of training steps to perform.')
    parser.add_argument('--eval-steps',
                        help="""\
                        Number of steps to run evalution for at each checkpoint.
                        If unspecified, will run for 1 full epoch over training
                        data""",
                        default=None,
                        type=int)
    parser.add_argument('--train-batch-size',
                        type=int,
                        default=40,
                        help='Batch size for training steps')
    parser.add_argument('--eval-batch-size',
                        type=int,
                        default=40,
                        help='Batch size for evaluation steps')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.003,
                        help='Learning rate for SGD')
    parser.add_argument('--eval-frequency',
                        default=100,
                        help='Perform one evaluation per n steps',
                        type=int
                        )
    parser.add_argument('--decay-rate',
                        type=float,
                        default=0.96,
                        help="""\
                        Rate of decay size of layer for Deep Neural Net. \
                        """)
    parser.add_argument('--decay-frequency',
                        default=100,
                        type=int,
                        help='Perform decay per n steps')
    parser.add_argument('--num-epochs',
                        type=int,
                        default=None,
                        help='Maximum number of epochs on which to train')
    parser.add_argument('--num-threads',
                        type=int,
                        default=8,
                        help='Maximum number of threads to read tfrecord')
    parser.add_argument('--verbosity',
                        choices=[
                            'DEBUG',
                            'ERROR',
                            'FATAL',
                            'INFO',
                            'WARN'
                        ],
                        default='INFO',
                        help='Set logging verbosity')
    parse_args, unknown = parser.parse_known_args()
    # Set python level verbosity
    tf.logging.set_verbosity(parse_args.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
        tf.logging.__dict__[parse_args.verbosity] / 10)
    del parse_args.verbosity

    if unknown:
        tf.logging.warn('Unknown arguments: {}'.format(unknown))

    dispatch(**parse_args.__dict__)