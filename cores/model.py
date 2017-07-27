from inception import inception_model
import tensorflow as tf
from utils import tfrecord
from tensorflow.python.ops import control_flow_ops


TRAIN, EVAL, PREDICT = 'TRAIN', 'EVAL', 'PREDICT'
CNN_IMAGE_SIZE = 299


class Inception(object):

    def __init__(self, clf_classes):
        self.clf_classes = clf_classes
        self.cls_num = len(clf_classes)

    def preprocess_tf_image(self, raw_image, image_height, image_width):
        image_rows = tf.cast(image_height, tf.int32)
        image_cols = tf.cast(image_width, tf.int32)
        max_side = tf.reduce_max((image_rows, image_cols))
        target_smaller_than_max = tf.less_equal(CNN_IMAGE_SIZE, max_side)
        resize_size = control_flow_ops.cond(target_smaller_than_max, lambda: max_side, lambda: CNN_IMAGE_SIZE)
        image_shape = tf.stack([image_rows, image_cols, 3])
        image_data = tf.decode_raw(raw_image, tf.uint8)
        image = tf.reshape(image_data, image_shape)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_image_with_crop_or_pad(image, resize_size, resize_size)
        image = tf.image.per_image_standardization(image)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bicubic(
            image, [CNN_IMAGE_SIZE, CNN_IMAGE_SIZE], align_corners=False)
        image = tf.squeeze(image, squeeze_dims=[0])
        return image

    def preprocess_jpeg_data(self, jpegs):
        image = tf.image.decode_jpeg(jpegs, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        max_side = tf.reduce_max(tf.shape(image))
        target_size = tf.constant(CNN_IMAGE_SIZE)
        target_smaller_than_max = tf.less_equal(target_size, max_side)
        resize_size = control_flow_ops.cond(target_smaller_than_max, lambda: max_side, lambda: target_size)
        image = tf.image.resize_image_with_crop_or_pad(image, resize_size, resize_size)
        image = tf.image.per_image_standardization(image)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bicubic(
            image, [target_size, target_size], align_corners=False)

        image = tf.squeeze(image, squeeze_dims=[0])
        return image

    def _parse_incoming_data(self, incoming_data):
        feature_configs = {
            'image': tf.FixedLenFeature(shape=[], dtype=tf.string)
        }
        tf_examples = tf.parse_example(incoming_data, feature_configs)
        jpegs = tf_examples['image']
        inputs = tf.map_fn(self.preprocess_jpeg_data, jpegs, dtype=tf.float32)
        return inputs


class MultiLabelTrainer(Inception):

    def __init__(self, tf_dir, sup_cats=None):
        self.tf_record = tfrecord.TfRecord(tf_dir, sup_cats)
        tf_classes = self.tf_record.get_tf_categories()
        super(MultiLabelTrainer, self).__init__(tf_classes)


    def _cross_entropy(self, batch_images, batch_labels, for_training=False):
        logits, end_points = inception_model.inference(batch_images, self.cls_num, for_training=for_training)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=batch_labels)
        total_loss = tf.reduce_mean(cross_entropy)
        return total_loss, logits


    def input_fn(self, input_type, batch_size, num_epochs, num_threads=8, min_after_dequeue=1000):
        tf_example = self.tf_record.decode_tfrecords(input_type, num_epochs)
        tf_image = self.preprocess_tf_image(tf_example['image_raw'], tf_example['image_height'], tf_example['image_width'])
        tf_labels = tf.cast(tf_example['image_labels'], tf.string)
        batch_images, batch_labels = tf.train.shuffle_batch([tf_image, tf_labels], num_threads=num_threads,
                                                            batch_size=batch_size,
                                                            capacity=min_after_dequeue + 3 * batch_size,
                                                            min_after_dequeue=min_after_dequeue)
        idx_table = tf.contrib.lookup.index_table_from_tensor(mapping=self.clf_classes, num_oov_buckets=1,
                                                              default_value=-1)
        labels_idx = idx_table.lookup(batch_labels)
        labels = tf.cast(tf.sparse_to_indicator(labels_idx, self.cls_num), tf.float32)
        return batch_images, labels

    def train_fn(self, batch_images, batch_labels, learning_rate, decay_frequency=3000, decay_rate=0.96):
        global_step = tf.contrib.framework.get_or_create_global_step()
        total_loss, _ = self._cross_entropy(batch_images, batch_labels, True)
        lr = tf.train.exponential_decay(learning_rate, global_step, decay_frequency, decay_rate, staircase=True)
        train_op = tf.train.AdamOptimizer(lr).minimize(total_loss, global_step=global_step)
        return train_op, global_step

    def eval_fn(self, batch_images, batch_labels):
        total_loss, logits = self._cross_entropy(batch_images, batch_labels, False)
        tf.summary.scalar('total_loss', total_loss)
        sigmoid_tensor = tf.nn.sigmoid(logits, name='sigmoid_tensor')
        tf.summary.histogram('activations', sigmoid_tensor)
        correct_prediction = tf.equal(tf.round(sigmoid_tensor), batch_labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', evaluation_step)
        return total_loss, evaluation_step

    def inference(self, incoming_data, tops=10):
        inputs = self._parse_incoming_data(incoming_data)
        logits, end_points = inception_model.inference(inputs, self.cls_num, for_training=False)
        sigmoid_tensor = tf.nn.sigmoid(logits, name='sigmoid_tensor')
        class_tensor = tf.constant(self.clf_classes)
        table = tf.contrib.lookup.index_to_string_table_from_tensor(class_tensor)
        values, indices = tf.nn.top_k(sigmoid_tensor, tops)
        classes = table.lookup(tf.to_int64(indices))
        return classes, values, logits




# class InceptionMultiLabels(object):
#
#     def __init__(self, images, cat_num, mode):
#         self.cat_num = cat_num
#         if mode in (TRAIN, EVAL):
#             self.logits, self.end_points = inception_model.inference(images, self.cat_num, for_training=True)
#         else:
#             self.logits, self.end_points = inception_model.inference(images, self.cat_num, for_training=False)
#
#
#     # def _build_graph(self, images, is_training=True):
#     #     logits, end_points = inception_model.inference(images, self.cat_num + 1, for_training=is_training)
#     #     inception_embeddings = end_points['prelogits']
#     #     sigmoids = tf.nn.sigmoid(logits, name='sigmoid_result')
#     #     return sigmoids, logits, inception_embeddings
#
#     #
#     # def _restore_graph(self, sess, checkpoint_dir):
#     #     global_step = None
#     #     variable_averages = tf.train.ExponentialMovingAverage(0.9999)
#     #     variables_to_restore = variable_averages.variables_to_restore()
#     #     saver = tf.train.Saver(variables_to_restore)
#     #     ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
#     #     if ckpt and ckpt.model_checkpoint_path:
#     #         saver.restore(sess, ckpt.model_checkpoint_path)
#     #         global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#     #         print('Successfully loaded model from %s at step=%s.' % (ckpt.model_checkpoint_path, global_step))
#     #     else:
#     #         print('No checkpoint file found at %s') % checkpoint_dir
#     #     return global_step
#
#     def train_fn(self, images, labels, learning_rate, fully_train=True):
#         idx_table = tf.contrib.lookup.index_table_from_tensor(mapping=labels, num_oov_buckets=1, default_value=-1)
#
#         logits, self.end_points = inception_model.inference(images, self.cat_num, for_training=fully_train)
#         global_step = tf.Variable(0, trainable=False)
#         cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
#         loss = tf.reduce_mean(cross_entropy)
#         lr = tf.train.exponential_decay(learning_rate, global_step, 3000, 0.96, staircase=True)
#         train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy, global_step=global_step)
#         return train_step, loss
#
#
#     def predict_fn(self):
#         pass
#
#     def model_fn(self, mode, learning_rate=0.01):
#         if mode in (PREDICT, EVAL):
#             pass
#
#     def inference(self, sigmoids):
#         return tf.round(sigmoids)
#
#     def eval(self, sigmoids, labels):
#         correct_prediction = tf.reduce_mean(tf.equal(tf.round(sigmoids), labels))
#         return correct_prediction
