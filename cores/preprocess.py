import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


def decode_image(raw_image):
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





def batch_decode_tfrecords(tfrecord, tfrecord_files, batch_size, num_epochs=1, capacity=60,
                           num_threads=8, min_after_dequeue=1000):
    filename_queue = tf.train.string_input_producer(tfrecord_files, num_epochs=num_epochs,
                                                    shuffle=True, capacity=capacity)
    image_feature = tfrecord.read_image_tfrecord(filename_queue)
    image = decode_image(image_feature['image_raw'])
    labels = tf.cast(image_feature['image_labels'],  tf.string)
    batch_images, batch_labels = tf.train.shuffle_batch([image, labels], num_threads=num_threads, batch_size=batch_size,
                                                        capacity=min_after_dequeue + 3 * batch_size,
                                                        min_after_dequeue=min_after_dequeue)
    return batch_images, batch_labels


def convert_coco_to_tfrecord():
    pass