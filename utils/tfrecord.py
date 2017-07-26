import tensorflow as tf
import cv2
import os
from datetime import datetime
# from cores import dataset
import json

class TfRecord(object):

    def __init__(self, tfrecord_dir):
        self.tfrecord_dir = tfrecord_dir
        self.tf_categories = self._load_tf_categories()

    def _int64_feature(self, value):
        if isinstance(value, list):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        else:
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        if isinstance(value, list):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
        else:
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def save_tf_categories(self, json_cats):
        cat_path = os.path.join(self.tfrecord_dir, 'categories.json')
        with open(cat_path, 'w') as f:
            json.dump(json_cats, f)

    def _load_tf_categories(self):
        categories = {}
        cat_path = os.path.join(self.tfrecord_dir, 'categories.json')
        if os.path.exists(cat_path):
            with open(cat_path, 'r') as f:
                categories = json.load(f)
        return categories

    def get_tf_categories(self, sup_cats=None):
        cats = []
        if sup_cats and isinstance(sup_cats, list):
            for cat in sup_cats:
                if cat in self.tf_categories:
                    cats.extend(self.tf_categories[cat])
        elif sup_cats and isinstance(sup_cats, str):
            cats.extend(self.tf_categories[sup_cats])
        return cats

    def write_image_tfrecord(self, image_files, tf_type, tf_cat):
        tfrecord_path = os.path.join(self.tfrecord_dir, tf_type + "_" + tf_cat + ".tfrecords")
        writer = tf.python_io.TFRecordWriter(tfrecord_path)
        for image_file in image_files:
            print("[%s]Convertint Image(%s) to Tfrecord..." %
                  (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), image_file['path']))
            try:
                img = cv2.imread(image_file['path'])
                raw_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                categories = image_file['category']
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image_raw': self._bytes_feature(raw_img.tostring()),
                    'image_height': self._int64_feature(raw_img.shape[0]),
                    'image_width': self._int64_feature(raw_img.shape[1]),
                    'image_labels': self._bytes_feature([cat.encode('utf-8') for cat in categories]),
                }))
                writer.write(example.SerializeToString())
            except Exception as err:
                print("Error Occur : " + str(err))
        writer.close()

    def read_image_tfrecord(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
            "image_raw" : tf.FixedLenFeature([], tf.string),
            "image_height" : tf.FixedLenFeature([], tf.int64),
            "image_width" : tf.FixedLenFeature([], tf.int64),
            "image_labels": tf.VarLenFeature(tf.string),
            })
        return features

    def get_tfrecords(self, tf_type, sup_cats):
        records = []
        for record in os.listdir(self.tfrecord_dir):
            record_name, record_extension = record.split('.')
            if record_extension != 'tfrecords':
                continue
            record_type, record_cat = record_name.split('_')
            if record_type == tf_type and record_cat in sup_cats:
                records.append(os.path.join(self.tfrecord_dir, record))
        return records

    def decode_tfrecords(self, tf_type, sup_cats, num_epochs=1, capacity=60):
        if not sup_cats:
            sup_cats = [c for c in self.tf_categories.keys()]
        tfrecord_files = self.get_tfrecords(tf_type, sup_cats)
        filename_queue = tf.train.string_input_producer(tfrecord_files, num_epochs=num_epochs,
                                                        shuffle=True, capacity=capacity)
        tf_example = self.read_image_tfrecord(filename_queue)
        return tf_example



# class CocoImageTfRecord(dataset.CoCoSet, TfRecord):
#
#     def __init__(self, coco_root, coco_type):
#         super(CocoImageTfRecord, self).__init__(coco_root, coco_type)
#
#     def convert_to_tfrecord(self, tf_dir, sup_cats=None):
#         if sup_cats and isinstance(sup_cats, list):
#             for cat in sup_cats:
#                 image_files = self.coco_images(cat)
#                 self.write_image_tfrecord(image_files, os.path.join(tf_dir, self.coco_type, cat))
#         elif sup_cats and isinstance(sup_cats, str):
#             image_files = self.coco_images(sup_cats)
#             self.write_image_tfrecord(image_files, os.path.join(tf_dir, self.coco_type, sup_cats))
#         else:
#             sup_cats = self.coco_sup_cats()
#             for cat in sup_cats:
#                 image_files = self.coco_images(cat)
#                 self.write_image_tfrecord(image_files, os.path.join(tf_dir, self.coco_type, cat))
#
#     def get_tfrecords(self, tfrecord_dir):
#         records = []
#         for record in os.listdir(tfrecord_dir):
#             record_name, record_extension = record.split('.')
#             if record_extension != 'tfrecords':
#                 continue
#             record_type, record_cat = record_name.split('_')
#             if record_type == self.coco_type:
#                 records.append(os.path.join(tfrecord_dir, record))
#         return records