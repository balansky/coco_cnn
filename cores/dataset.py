from pycocotools.coco import COCO
import numpy as np
from datetime import datetime
import cv2
import os

class CoCoSet(object):

    def __init__(self, coco_root, image_cats=[]):
        self.data_root = coco_root
        self.train_imgs, self.cats = self._load_imgs(image_cats, image_type='train')
        self.val_imgs, _ = self._load_imgs(image_cats, image_type='val')


    def _load_imgs(self, image_cats, image_type='train'):
        annotation_file = os.path.join(self.data_root, 'annotations', 'instances_%s2014.json' % image_type)
        coco = COCO(annotation_file)
        cat_ids = coco.getCatIds(supNms=image_cats)
        cats = coco.loadCats(cat_ids)
        if cat_ids:
            imgs = coco.loadImgs(coco.getImgIds(catIds=cat_ids))
        else:
            imgs = []
        for img in imgs:
            img_anns = coco.imgToAnns[img['id']]
            img['categories'] = list(set([ann['category'] for ann in img_anns if ann['category_id'] in cat_ids]))
        return imgs, cats

    def get_image_set(self, data_type):
        return self.train_imgs if data_type == 'train' else self.val_imgs

    #
    # def next_batch(self, batch_size, batch_type='train'):
    #     image_set = self.train_imgs if batch_type == 'train' else self.val_imgs
    #     label_set = self.train_to_cats if batch_type == 'train' else self.val_to_cats
    #     # image_set = self.train_imgs
    #     # label_set = self.train_to_cats
    #     random_idx = np.random.choice(len(image_set), batch_size)
    #     picked_imgs = [image_set[idx] for idx in random_idx]
    #     picked_labels = np.zeros((batch_size, len(self.cats)), np.float32)
    #     for i, img in enumerate(picked_imgs):
    #         img['file_name'] = os.path.join(self.data_root, batch_type, img['file_name'])
    #         labels = [self.cat_to_label[cat] for cat in label_set[img['id']]]
    #         picked_labels[i][labels] = 1.0
    #     return picked_imgs, picked_labels


class CoCoTfRecord(object):

    def __init__(self, tf, data_dir):
        self.data_dir = data_dir
        self.tf = tf

    def int64_feature(self, value):
        return self.tf.train.Feature(int64_list=self.tf.train.Int64List(value=[value]))

    def bytes_feature(self, value):
        if isinstance(value, list):
            return self.tf.train.Feature(bytes_list=self.tf.train.BytesList(value=value))
        else:
            return self.tf.train.Feature(bytes_list=self.tf.train.BytesList(value=[value]))

    def get_coco_cats(self, supcats):
        coco = self._load_coco_set('train')
        cat_ids = coco.getCatIds(supNms=supcats)
        cats = coco.loadCats(cat_ids)
        cat_nms = [cat['name'] for cat in cats]
        return cat_nms

    def _load_coco_set(self, image_type):
        annotation_file = os.path.join(self.data_dir, 'annotations', 'instances_%s2014.json' % image_type)
        coco = COCO(annotation_file)
        return coco

    def _dump_to_tfrecord(self, coco, sup_cat, image_type):
        cat_ids = coco.getCatIds(supNms=sup_cat)
        cats = coco.loadCats(cat_ids)
        catid_to_label = {cat['id']: cat['name'] for cat in cats}
        imgs = coco.loadImgs(coco.getImgIds(catIds=cat_ids))
        writer = self.tf.python_io.TFRecordWriter(
            os.path.join(self.data_dir, 'tfrecords', image_type + "_" + sup_cat + ".tfrecords"))
        for img in imgs[:20]:
            print("[%s]Convertint Image(%s) to Tfrecord..." %
                  (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), img['file_name']))
            try:
                raw_img = cv2.cvtColor(cv2.imread(os.path.join(self.data_dir, image_type, img['file_name'])),
                                       cv2.COLOR_BGR2RGB)
                img_anns = coco.imgToAnns[img['id']]
                categories = list(set([catid_to_label[ann['category_id']] for ann in img_anns
                                       if ann['category_id'] in cat_ids]))
                example = self.tf.train.Example(features=self.tf.train.Features(feature={
                    'image_raw': self.bytes_feature(raw_img.tostring()),
                    'image_labels': self.bytes_feature([cat.encode('utf-8') for cat in categories]),
                    'image_height': self.int64_feature(img['height']),
                    'image_width': self.int64_feature(img['width'])
                }))
                writer.write(example.SerializeToString())
            except Exception as err:
                print("Error Occur : " + str(err))
        writer.close()

    def decode_image(self, raw_image, image_height, image_width):
        image = self.tf.decode_raw(raw_image, self.tf.uint8)
        max_side = self.tf.reduce_max((image_height, image_width))
        image_shape = self.tf.stack([image_height, image_width, 3])
        image = self.tf.reshape(image, image_shape)
        image = self.tf.cast(image, self.tf.float32)

        image = self.tf.image.resize_image_with_crop_or_pad(image, max_side, max_side)
        image = self.tf.image.per_image_standardization(image)
        image = self.tf.expand_dims(image, 0)
        image = self.tf.image.resize_bicubic(
            image, [299, 299], align_corners=False)

        image = self.tf.squeeze(image, squeeze_dims=[0])
        return image

    def write_to_tfrecords(self, image_cats, image_type):
        coco = self._load_coco_set(image_type)
        if isinstance(image_cats, list):
            for cat in image_cats:
                self._dump_to_tfrecord(coco, cat, image_type)
        else:
            self._dump_to_tfrecord(coco, image_cats, image_type)


    def _read_and_decode(self, filename_queue):
        reader = self.tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = self.tf.parse_single_example(
            serialized_example,
            features={
            "image_raw" : self.tf.FixedLenFeature([], self.tf.string),
            "image_labels": self.tf.VarLenFeature(self.tf.string),
            "image_height" : self.tf.FixedLenFeature([], self.tf.int64),
            "image_width" : self.tf.FixedLenFeature([], self.tf.int64),
            })
        image = self.decode_image(features['image_raw'],
                                  self.tf.cast(features['image_height'], self.tf.int32),
                                  self.tf.cast(features['image_width'], self.tf.int32))
        labels = self.tf.cast(features['image_labels'], self.tf.string)
        return image, labels

    def _get_tfrecords(self, require_type):
        records = []
        tfrecord_dir = os.path.join(self.data_dir, 'tfrecords')
        for record in os.listdir(tfrecord_dir):
            record_name, record_extension = record.split('.')
            if record_extension != 'tfrecords': continue
            record_type, record_cat = record_name.split('_')
            if record_type == require_type:
                records.append(os.path.join(tfrecord_dir, record))
        return records


    def batch_data(self, batch_type, batch_size, capacity=60, num_threads=8, min_after_dequeue=1000):
        tfrecords = self._get_tfrecords(batch_type)
        filename_queue = self.tf.train.string_input_producer(tfrecords, shuffle=True, capacity=capacity)
        record_image, record_labels = self._read_and_decode(filename_queue)
        batch_images, batch_labels = self.tf.train.shuffle_batch([record_image, record_labels],
                                                                 num_threads=num_threads, batch_size=batch_size,
                                                                 capacity=min_after_dequeue + 3*batch_size,
                                                                 min_after_dequeue=min_after_dequeue,)
        return batch_images, batch_labels