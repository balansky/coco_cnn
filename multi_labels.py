from tensorflow.contrib.slim import nets
import tensorflow as tf
from cores.dataset import DataSet
from tensorflow.python.platform import gfile

RootDataDir = "/home/andy/Data/coco"
IMAGE_SIZE = 299


def process_image(img):
    raw_image = gfile.FastGFile(img['file_name'], 'rb').read()
    decoded_image = tf.image.decode_image(raw_image, channels=3)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    max_side = max(img['width'], img['height'])
    crop_image = tf.image.resize_image_with_crop_or_pad(decoded_image_as_float, max_side, max_side)
    stand_image = tf.image.per_image_standardization(crop_image)
    stand_image_4d = tf.expand_dims(stand_image, 0)
    resize_image = tf.image.resize_bicubic(stand_image_4d, (IMAGE_SIZE, IMAGE_SIZE))
    resize_image_3d = tf.squeeze(resize_image, squeeze_dims=[0])
    return resize_image_3d

def train():
    coco_set = DataSet(RootDataDir, 'food')
    total_cats = len(coco_set.cats)
    batch_size = 2
    with tf.Session() as sess:
        inputs = tf.placeholder(tf.float32, shape=[batch_size, 299, 299, 3])
        targets = tf.placeholder(tf.float32, shape=[None, total_cats])
        global_step = tf.Variable(0, trainable=False)
        logits, end_points = nets.inception.inception_v3(inputs, num_classes=total_cats)
        sigmoids = tf.nn.sigmoid(logits, name='sigmoid_result')
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets))
        correct_prediction = tf.reduce_mean(tf.equal(tf.round(sigmoids), targets))
        lr = tf.train.exponential_decay(0.01, global_step, 3000, 0.96, staircase=True)
        train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy, global_step=global_step)
        sess.run(tf.global_variables_initializer())
        for step in range(10000):
            images, labels = coco_set.next_batch(batch_size)
            resized_images = [process_image(img) for img in images]
            resized_images = tf.stack(resized_images, 0)
            resized_images = sess.run(resized_images)
            loss = sess.run([train_step], {inputs: resized_images, targets: labels})
            print("step %s : %s loss" % (step, loss))
            if step % 1000 == 0:
                val_images, val_labels = coco_set.next_batch(batch_size, 'val')
                loss, acc = sess.run([cross_entropy, correct_prediction], {inputs: val_images, targets: val_labels})
                print("step %s : %s loss, %s accuracy" % (step, loss, acc))



if __name__== "__main__":
    # coco_set = DataSet(RootDataDir, 'food')
    train()