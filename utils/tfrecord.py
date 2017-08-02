import tensorflow as tf
import cv2
import os
from datetime import datetime
# from cores import dataset
import numpy as np
import json
from google.cloud import storage


class TfRecord(object):

    def __init__(self, tfrecord_dir, config_dir, sup_cats=None):
        self.tfrecord_dir = tfrecord_dir
        self.config_dir = config_dir
        self.sup_cats = sup_cats
        self.tf_categories = self._load_tf_categories(sup_cats)

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

    def _save_json(self, json_data, save_path):
        with open(save_path, 'w') as f:
            json.dump(json_data, f)

    def _load_json(self, json_path):
        json_dict = {}
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                json_dict = json.load(f)
        return json_dict

    def save_tf_categories(self, json_cats):
        cat_path = os.path.join(self.config_dir, 'categories.json')
        self._save_json(json_cats, cat_path)


    def save_tf_info(self, tf_info):
        tf_info_path = os.path.join(self.config_dir, "tfrecord.json")
        self._save_json(tf_info, tf_info_path)


    def _load_tf_categories(self, sup_cats):
        cat_path = os.path.join(self.config_dir, 'categories.json')
        full_categories = self._load_json(cat_path)
        tf_cats = []
        if sup_cats and isinstance(sup_cats, list):
            for cat in sup_cats:
                if cat in full_categories:
                    tf_cats.extend(full_categories[cat])
        elif sup_cats and isinstance(sup_cats, str):
            if sup_cats in full_categories:
                tf_cats.extend(full_categories[sup_cats])
        elif full_categories and not sup_cats:
            for sub_cats in full_categories.values():
                tf_cats.extend([cat for cat in sub_cats])
        return tf_cats

    def get_tf_categories(self):
        return self.tf_categories

    # def get_tfrecords(self, tf_type):
    #     records = []
    #     for record in os.listdir(self.tfrecord_dir):
    #         record_name, record_extension = record.split('.')
    #         if record_extension != 'tfrecords':
    #             continue
    #         record_type, record_cat, _ = record_name.split('_')
    #         if record_type == tf_type and record_cat in self.sup_cats:
    #             records.append(os.path.join(self.tfrecord_dir, record))
    #     return records

    def get_tfrecords(self, tf_type):
        tf_files = []
        info_path = os.path.join(self.config_dir, 'tfrecord.json')
        tf_info = self._load_json(info_path)
        tf_data_set = tf_info[tf_type]
        for tf_cat, tf_data in tf_data_set.items():
            if self.sup_cats and tf_cat in self.sup_cats:
                tf_files.extend([self.tfrecord_dir + "/" + tf_file for tf_file in tf_data])
            elif not self.sup_cats:
                tf_files.extend([self.tfrecord_dir + "/" + tf_file for tf_file in tf_data])
        return tf_files


    def write_image_tfrecord(self, image_files, tf_type, tf_cat, tf_size=2000):
        tfrecords = []
        image_file_chunks = [image_files[i:i+tf_size] for i in range(0, len(image_files), tf_size)]
        for i, image_chk in enumerate(image_file_chunks):
            tfrecord_filename = tf_type + "_" + tf_cat + "_%05d" % i + ".tfrecords"
            tfrecord_path = os.path.join(self.tfrecord_dir, tfrecord_filename)
            writer = tf.python_io.TFRecordWriter(tfrecord_path)
            for image_file in image_chk:
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
                    tfrecords.append(tfrecord_filename)
                except Exception as err:
                    print("Error Occur : " + str(err))
            writer.close()
        return tfrecords

    def read_image_tfrecord(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
            "image_raw" : tf.FixedLenFeature([], tf.string),
            "image_height" : tf.FixedLenFeature([], tf.int64),
            "image_width" : tf.FixedLenFeature([], tf.int64),
            "image_labels": tf.VarLenFeature(tf.string)
            })
        return features



    def decode_tfrecords(self, tf_type, num_epochs=1, capacity=60):
        # if self.tfrecord_dir.split(':')[0] == "gs":
        #     tfrecord_files = self.get_gs_tfrecords(tf_type)
        # else:
        tfrecord_files = self.get_tfrecords(tf_type)
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