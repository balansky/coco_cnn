import os

import tensorflow as tf

from cores import model
from pycocotools.coco import COCO

# from tensorflow.contrib.slim.python.slim.nets import inception_v3 as inception
# from tensorflow.contrib import slim
from inception import inception_model

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
    image = tf.image.decode_jpeg(features['image_raw'], channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

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
    batch_size = 1
    trainer = model.MultiLabelTrainer('/home/andy/Data/coco/tfrecords', 'food')
    with tf.Session() as sess:
        batch_images, batch_labels = trainer.input_fn('train', batch_size, 10)
        # idx_table = tf.contrib.lookup.index_table_from_tensor(
        #             mapping=trainer.clf_classes, num_oov_buckets=1, default_value=-1)
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # labels_idx = idx_table.lookup(batch_labels)
        # labels = tf.cast(tf.sparse_to_indicator(labels_idx, 10), tf.float32)
        # label_tensor[labels_idx] = 1.0
        # rel_training_batch, rel_training_label = sess.run([batch_images, batch_labels])
        # str_training_label = tf.constant([label.decode('utf-8') for label in rel_training_label.values])

        _, lb = sess.run([batch_images, batch_labels])
        print(lb)
        # print(lbidx)
        coord.request_stop()
        coord.join(threads)

def process_image(serialized_tf_example):

    image = tf.decode_raw(serialized_tf_example, tf.uint8)
    image = tf.reshape(image, [299, 299, 3])
    image = tf.cast(image, tf.float32)
    # feature_configs = {
    #     'image_raw': tf.FixedLenFeature(shape=[], dtype=tf.string),
    # }
    # tf_exmple = tf.parse_example(serialized_tf_example, feature_configs)
    # raw_image = tf_exmple['image_raw']
    # image = tf.decode_raw(raw_image, tf.uint8)
    #
    # # image_shape = tf.stack([image_height, image_width, 3])
    # image = tf.reshape(image, [299, 299, 3])
    # image = tf.cast(image, tf.float32)
    return image



def export_incept3():
    NUM_CLASSES = 1000
    NUM_TOP_CLASSES = 5
    checkpoint_dir = "/home/andy/Data/models/inception-v3"
    # WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
    SYNSET_FILE = os.path.join(checkpoint_dir, 'imagenet_lsvrc_2015_synsets.txt')
    METADATA_FILE = os.path.join(checkpoint_dir, 'imagenet_metadata.txt')

    output_dir = "/home/andy/Data/models/servable"
    model_version = "1"


    # synsets = []
    with open(SYNSET_FILE) as f:
        synsets = f.read().splitlines()
    # Create synset->metadata mapping
    texts = {}
    with open(METADATA_FILE) as f:
        for line in f.read().splitlines():
            parts = line.split('\t')
            assert len(parts) == 2
            texts[parts[0]] = parts[1]


    with tf.Graph().as_default():
        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        feature_configs = {
            'image': tf.FixedLenFeature(shape=[], dtype=tf.string)
        }
        tf_examples = tf.parse_example(serialized_tf_example, feature_configs)
        jpegs = tf_examples['image']
        images = tf.map_fn(process_image, jpegs,  dtype=tf.float32)
        # with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception_model.inference(images, NUM_CLASSES + 1)
        inception_embeddings = end_points['prelogits']
        values, indices = tf.nn.top_k(logits, NUM_TOP_CLASSES)
        class_descriptions = ['unused background']
        for s in synsets:
            class_descriptions.append(texts[s])
        class_tensor = tf.constant(class_descriptions)

        table = tf.contrib.lookup.index_to_string_table_from_tensor(class_tensor)
        classes = table.lookup(tf.to_int64(indices))

        # Restore variables from training checkpoint.
        variable_averages = tf.train.ExponentialMovingAverage(0.9999)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:
            # Restore variables from training checkpoints.
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/imagenet_train/model.ckpt-0,
                # extract global_step from it.
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('Successfully loaded model from %s at step=%s.' % (ckpt.model_checkpoint_path, global_step))
            else:
                print('No checkpoint file found at %s') % checkpoint_dir
                return

            # Export inference model.
            output_path = os.path.join(
                tf.compat.as_bytes(output_dir),
                tf.compat.as_bytes(str(model_version)))
            print('Exporting trained model to', output_path)
            builder = tf.saved_model.builder.SavedModelBuilder(output_path)
            classify_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(
                serialized_tf_example)
            classes_output_tensor_info = tf.saved_model.utils.build_tensor_info(
                classes)
            scores_output_tensor_info = tf.saved_model.utils.build_tensor_info(values)

            classification_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                            classify_inputs_tensor_info
                    },
                    outputs={
                        tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                            classes_output_tensor_info,
                        tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                            scores_output_tensor_info
                    },
                    method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME
                ))
            predict_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(jpegs)
            embedding_tensor_info = tf.saved_model.utils.build_tensor_info(inception_embeddings)
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'images': predict_inputs_tensor_info},
                    outputs={
                        'classes': classes_output_tensor_info,
                        'embedding': embedding_tensor_info
                    },
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                ))
            legacy_init_op = tf.group(
                tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict_images':
                        prediction_signature,
                    tf.saved_model.signature_constants.
                        DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        classification_signature,
                },
                legacy_init_op=legacy_init_op)

            builder.save()
            print('Successfully exported model to %s' % output_dir)


def test_gc_tfrecord():
    from utils import tfrecord
    gc_tfrecord = tfrecord.GcloudTfrecord("/home/andy/Data/coco/tfrecords", 'ss', sup_cats='food')
    gc_tfrecord.get_tfrecords('train')



def main():
    # export_incept3()
    test_gc_tfrecord()
    # test_batch()
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
