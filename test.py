import tensorflow as tf
from pycocotools.coco import COCO
import os
from cores import dataset
from tensorflow.python.platform import gfile
from PIL import Image
from io import BytesIO
import numpy as np

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    if isinstance(value, list):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_test_data():
    annotation_file = os.path.join('/home/andy/Data/coco', 'annotations', 'instances_%s2014.json' % 'train')
    coco = COCO(annotation_file)
    cat_ids = coco.getCatIds(supNms='food')
    cats = coco.loadCats(cat_ids)
    imgs = coco.loadImgs(coco.getImgIds(catIds=cat_ids))
    id_to_cat = {cat['id']: cat['name'] for cat in cats}

    img_to_cats = {img['id']: list(set([id_to_cat[ann['category_id']] for ann in coco.imgToAnns[img['id']]
                                        if ann['category_id'] in cat_ids])) for img in imgs}
    return imgs[:5], cats, img_to_cats


def decode_and_resize(image_str_tensor):
    """Decodes jpeg string, resizes it and returns a uint8 tensor."""
    features = tf.parse_single_example(image_str_tensor, features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'image_height': tf.FixedLenFeature([], tf.int64),
        'image_width': tf.FixedLenFeature([], tf.int64)
    })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image_height = tf.cast(features['image_height'], tf.int32)
    image_width = tf.cast(features['image_width'], tf.int32)
    max_side = tf.reduce_max((image_height, image_width))
    image_shape = tf.stack([image_height, image_width, 3])
    image = tf.reshape(image, image_shape)
    image = tf.cast(image, tf.float32)

    image = tf.image.resize_image_with_crop_or_pad(image, max_side, max_side)
    image = tf.image.per_image_standardization(image)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bicubic(
        image, [299, 299], align_corners=False)

    image = tf.squeeze(image, squeeze_dims=[0])

    return image


def serialize_images(images):
    examples = []
    for image in images:
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_height': int64_feature(image.size[1]),
            'image_width': int64_feature(image.size[0]),
            'image_raw': bytes_feature(image.tobytes())
        }))
        examples.append(example.SerializeToString())
    return examples


def test_batch():
    batch_size = 8
    coco_tfrecord = dataset.CoCoTfRecord(tf, '/home/andy/Data/coco')
    cats = coco_tfrecord.get_coco_cats('food')
    with tf.Session() as sess:
        batch_images, batch_labels = coco_tfrecord.batch_data('train', batch_size)
        idx_table = tf.contrib.lookup.index_table_from_tensor(
                    mapping=cats, num_oov_buckets=1, default_value=-1)
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        labels_idx = idx_table.lookup(batch_labels)
        labels = tf.cast(tf.sparse_to_indicator(labels_idx, 10), tf.float32)
        # label_tensor[labels_idx] = 1.0
        # rel_training_batch, rel_training_label = sess.run([batch_images, batch_labels])
        # str_training_label = tf.constant([label.decode('utf-8') for label in rel_training_label.values])
        print(sess.run(labels))
        coord.request_stop()
        coord.join(threads)

def main():
    test_batch()
    # coco_tfrecord = dataset.CoCoTfRecord(tf, '/home/andy/Data/coco')
    # coco_tfrecord.write_to_tfrecords('food', 'train')
    # image_dir = os.path.join('/home/andy/Data/coco', 'train')
    # imgs = [os.path.join(image_dir, img) for img in os.listdir(image_dir)][300:303]
    # pil_imgs = [Image.open(img) for img in imgs]
    # serialized_images = serialize_images(pil_imgs)
    # with tf.Session() as sess:
    #     serialized_tf_example = tf.placeholder(tf.string, shape=[None], name='tf_example')
    #     imgs_tf = tf.map_fn(decode_and_resize, serialized_tf_example, back_prop=False, dtype=tf.float32)
    #     x = tf.identity(imgs_tf, 'input')
    #
    #     ff = sess.run(x, {serialized_tf_example: serialized_images})
    #     print(ff)



if __name__== "__main__":
    main()