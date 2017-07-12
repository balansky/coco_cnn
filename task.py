from cores import dataset, model
from inception import inception_model
import tensorflow as tf
import os
import json
import argparse



def run(target, is_chief, num_classes, train_steps, eval_steps, job_dir, train_files, eval_files,
        train_batch_size, eval_batch_size, learning_rate, eval_frequency, scale_factor, num_epochs):
    coco_date = dataset.CoCoTfRecord(tf)
    hooks = [tf.train.StopAtStepHook(last_step=1000000)]
    # if is_chief:
    #     with tf.Graph().as_default() as evaluation_graph:
    #         images, labels = coco_date.input_fn(eval_files, eval_batch_size, eval_steps)
    #         logits, end_points = inception_model.inference(images, num_classes)

    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter()):
            images, labels = coco_date.input_fn(train_files, train_batch_size, num_epochs)
            logits, end_points = inception_model.inference(images, num_classes, for_training=True)
            global_step = tf.contrib.framework.get_or_create_global_step()
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
            lr = tf.train.exponential_decay(learning_rate, global_step, 3000, scale_factor, staircase=True)
            train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy, global_step=global_step)
        with tf.train.MonitoredTrainingSession(master=target, is_chief=is_chief, checkpoint_dir=job_dir,
                                               hooks=hooks, save_checkpoint_secs=20,
                                               save_summaries_steps=50) as session:
            step = global_step.eval(session=session)
            while (train_steps is None or
                           step < train_steps) and not session.should_stop():
                step, _ = session.run([global_step, train_op])
            latest_checkpoint = tf.train.latest_checkpoint(job_dir)
            # if is_chief:
            #     build_and_run_exports(latest_checkpoint,
            #                           job_dir,
            #                           model.SERVING_INPUT_FUNCTIONS[export_format],
            #                           hidden_units)

def dispatch(*args, **kwargs):
    tf_config = os.environ.get('TF_CONFIG')

    if not tf_config:
        return run('', True, *args, **kwargs)

    tf_config_json = json.loads(tf_config)

    cluster = tf_config_json.get('cluster')
    job_name = tf_config_json.get('task', {}).get('type')
    task_index = tf_config_json.get('task', {}).get('index')

    # If cluster information is empty run local
    if job_name is None or task_index is None:
        return run('', True, *args, **kwargs)

    cluster_spec = tf.train.ClusterSpec(cluster)
    server = tf.train.Server(cluster_spec,
                             job_name=job_name,
                             task_index=task_index)
    if job_name == 'ps':
        server.join()
        return
    elif job_name in ['master', 'worker']:
        return run(server.target, job_name == 'master', *args, **kwargs)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-files',
                        required=True,
                        type=str,
                        help='Training files local or GCS', nargs='+')
    parser.add_argument('--eval-files',
                        required=True,
                        type=str,
                        help='Evaluation files local or GCS', nargs='+')
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
                        default=50,
                        help='Perform one evaluation per n steps')
    parser.add_argument('--scale-factor',
                        type=float,
                        default=0.25,
                        help="""\
                        Rate of decay size of layer for Deep Neural Net. \
                        """)
    parser.add_argument('--num-epochs',
                        type=int,
                        help='Maximum number of epochs on which to train')
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