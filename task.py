from cores import dataset, model
from inception import inception_model
import tensorflow as tf
import os
import json
import argparse
from tensorflow.python.saved_model import signature_constants as sig_constants
from utils import tfrecord
from tensorflow.python.ops import control_flow_ops


def preprocess_image(raw_image):
    image = tf.image.decode_jpeg(raw_image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    max_side = tf.reduce_max(tf.shape(image))
    target_size = tf.constant(299)
    target_smaller_than_max = tf.less_equal(target_size, max_side)
    resize_size = control_flow_ops.cond(target_smaller_than_max, lambda: max_side, lambda: target_size)
    image = tf.image.resize_image_with_crop_or_pad(image, resize_size, resize_size)
    image = tf.image.per_image_standardization(image)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bicubic(
        image, [target_size, target_size], align_corners=False)

    image = tf.squeeze(image, squeeze_dims=[0])
    return image

def batch_inputs(tf_example, batch_size, classes, num_threads=8, min_after_dequeue=1000):

    idx_table = tf.contrib.lookup.index_table_from_tensor(mapping=classes, num_oov_buckets=1, default_value=-1)

    tf_image = preprocess_image(tf_example['image_raw'])
    tf_labels = tf.cast(tf_example['image_labels'], tf.string)
    batch_images, batch_labels = tf.train.shuffle_batch([tf_image, tf_labels], num_threads=num_threads,
                                                        batch_size=batch_size,
                                                        capacity=min_after_dequeue + 3 * batch_size,
                                                        min_after_dequeue=min_after_dequeue)
    labels_idx = idx_table.lookup(batch_labels)
    labels = tf.cast(tf.sparse_to_indicator(labels_idx, len(classes)), tf.float32)
    return batch_images, labels


def run(target, cluster_spec, is_chief, train_steps, eval_steps, job_dir, data_dir, sup_cats, num_threads,
        train_batch_size, eval_batch_size, learning_rate, eval_frequency, scale_factor, num_epochs):
    tf_record = tfrecord.TfRecord(data_dir)
    cats = tf_record.get_tf_categories(sup_cats)
    hooks = [tf.train.StopAtStepHook(last_step=1000000)]
    # if is_chief:
    #     with tf.Graph().as_default() as evaluation_graph:
    #         images, labels = coco_date.input_fn(eval_files, eval_batch_size, eval_steps)
    #         logits, end_points = inception_model.inference(images, num_classes)
    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
            tf_example = tf_record.decode_tfrecords('train', sup_cats, num_epochs)
            batch_images, batch_labels = batch_inputs(tf_example, train_batch_size, cats, num_threads)
            logits, end_points = inception_model.inference(batch_images, len(cats), for_training=True)
            global_step = tf.contrib.framework.get_or_create_global_step()
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=batch_labels)
            lr = tf.train.exponential_decay(learning_rate, global_step, 3000, scale_factor, staircase=True)
            train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy, global_step=global_step)
        with tf.train.MonitoredTrainingSession(master=target, is_chief=is_chief, checkpoint_dir=job_dir,
                                               hooks=hooks, save_checkpoint_secs=20,
                                               save_summaries_steps=50) as session:
            step = global_step.eval(session=session)
            while (train_steps is None or
                           step < train_steps) and not session.should_stop():
                step, _ = session.run([global_step, train_op])
            # latest_checkpoint = tf.train.latest_checkpoint(job_dir)
            # if is_chief:
            #     build_and_run_exports(latest_checkpoint,
            #                           job_dir,
            #                           model.SERVING_INPUT_FUNCTIONS[export_format],
            #                           hidden_units)


def build_and_run_exports(latest, job_dir, serving_input_fn, hidden_units):
    """Given the latest checkpoint file export the saved model.
    Args:
    latest (string): Latest checkpoint file
    job_dir (string): Location of checkpoints and model files
    name (string): Name of the checkpoint to be exported. Used in building the
      export path.
    hidden_units (list): Number of hidden units
    learning_rate (float): Learning rate for the SGD
    """

    prediction_graph = tf.Graph()
    exporter = tf.saved_model.builder.SavedModelBuilder(os.path.join(job_dir, 'export'))
    with prediction_graph.as_default():
        features, inputs_dict = serving_input_fn()
        prediction_dict = model.model_fn(
            model.PREDICT,
            features,
            None,  # labels
            hidden_units=hidden_units,
            learning_rate=None  # learning_rate unused in prediction mode
        )
        saver = tf.train.Saver()

        inputs_info = {
            name: tf.saved_model.utils.build_tensor_info(tensor)
            for name, tensor in inputs_dict.iteritems()
        }
        output_info = {
            name: tf.saved_model.utils.build_tensor_info(tensor)
            for name, tensor in prediction_dict.iteritems()
        }
        signature_def = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs_info,
            outputs=output_info,
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
                        required=True,
                        type=str,
                        help='Local Tfrecord Data')
    parser.add_argument('--sup-cats',
                        default=None,
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