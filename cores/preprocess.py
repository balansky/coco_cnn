import tensorflow as tf

def decode_image():
    pass


def decode_tfrecords(tfrecord, tfrecord_files):
    filename_queue = tf.train.string_input_producer(tfrecord_files, num_epochs=num_epochs,
                                                         shuffle=True, capacity=capacity)
    image_feature = tfrecord.read_image_tfrecord(tfrecord_files)