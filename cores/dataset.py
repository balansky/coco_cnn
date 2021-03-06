from pycocotools.coco import COCO
import os

class CoCoSet(object):

    def __init__(self, coco_root, coco_type):
        self.data_root = coco_root
        self.coco_type = coco_type
        self.coco = self._load_coco(coco_root, coco_type)

    def _load_coco(self, coco_dir, image_type):
        annotation_file = os.path.join(coco_dir, 'annotations', 'instances_%s2014.json' % image_type)
        coco = COCO(annotation_file)
        return coco

    def coco_images(self, sup_cat):
        images = []
        cat_ids = self.coco.getCatIds(supNms=sup_cat)
        cats = self.coco.loadCats(cat_ids)
        catid_to_label = {cat['id']: cat['name'] for cat in cats}
        coco_images = self.coco.loadImgs(self.coco.getImgIds(catIds=cat_ids))
        for image in coco_images:
            img_anns = self.coco.imgToAnns[image['id']]
            categories = list(set([catid_to_label[ann['category_id']] for ann in img_anns
                                   if ann['category_id'] in cat_ids]))
            images.append({'path': os.path.join(self.data_root, self.coco_type, image['file_name']),
                           'category': categories})
        return images

    def coco_sup_cats(self):
        cats = self.coco.loadCats(self.coco.getCatIds())
        sup_cats = set([cat['supercategory'] for cat in cats])
        return sup_cats

    def coco_cat_nms(self, supcats):
        cat_ids = self.coco.getCatIds(supNms=supcats)
        cats = self.coco.loadCats(cat_ids)
        cat_nms = [cat['name'] for cat in cats]
        return cat_nms



# class CocoTfRecord(TfRecord):
#
#     def _coco_images(self, coco, sup_cat):
#         cat_ids = coco.getCatIds(supNms=sup_cat)
#         cats = coco.loadCats(cat_ids)
#         catid_to_label = {cat['id']: cat['name'] for cat in cats}
#         coco_images = coco.loadImgs(coco.getImgIds(catIds=cat_ids))
#         for image in coco_images:
#             img_anns = coco.imgToAnns[image['id']]
#             categories = list(set([catid_to_label[ann['category_id']] for ann in img_anns
#                                    if ann['category_id'] in cat_ids]))
#             yield image['file_name'], categories
#
#     def _image_file(self, image_dir, image_name, categories):
#         return {'path': os.path.join(image_dir, image_name), 'category': categories}
#
#     def load_sup_cats(self, coco):
#         cats = coco.loadCats(coco.getCatIds())
#         sup_cats = set([cat['supercategory'] for cat in cats])
#         return sup_cats
#
#     def load_coco(self, coco_dir, image_type):
#         annotation_file = os.path.join(coco_dir, 'annotations', 'instances_%s2014.json' % image_type)
#         coco = COCO(annotation_file)
#         return coco
#
#     def convert_to_tfrecord(self, coco_dir, tf_dir, image_type, sup_cats=None):
#         coco = self.load_coco(coco_dir, image_type)
#         image_dir = os.path.join(coco_dir, image_type)
#         if sup_cats and isinstance(sup_cats, list):
#             for cat in sup_cats:
#                 image_files = [self._image_file(image_dir, file_name, categories) for file_name, categories
#                                in self._coco_images(coco, sup_cats)]
#                 self.write_image_tfrecord(image_files, os.path.join(tf_dir, image_type, cat))
#         elif sup_cats and isinstance(sup_cats, str):
#             image_files = [self._image_file(image_dir, file_name, categories) for file_name, categories
#                            in self._coco_images(coco, sup_cats)]
#             self.write_image_tfrecord(image_files, os.path.join(tf_dir, image_type, sup_cats))
#         else:
#             sup_cats = self.load_sup_cats(coco)
#             for cat in sup_cats:
#                 image_files = [self._image_file(image_dir, file_name, categories) for file_name, categories
#                                in self._coco_images(coco, cat)]
#                 self.write_image_tfrecord(image_files, os.path.join(tf_dir, image_type, cat))
#
#     def get_tfrecords(self, tfrecord_dir, require_type):
#         records = []
#         for record in os.listdir(tfrecord_dir):
#             record_name, record_extension = record.split('.')
#             if record_extension != 'tfrecords': continue
#             record_type, record_cat = record_name.split('_')
#             if record_type == require_type:
#                 records.append(os.path.join(tfrecord_dir, record))
#         return records



# class CoCoTfRecord(object):
#
#     def __init__(self, tf, data_dir=None):
#         self.data_dir = data_dir
#         self.tf = tf
#
#     def int64_feature(self, value):
#         if isinstance(value, list):
#             return self.tf.train.Feature(int64_list=self.tf.train.Int64List(value=value))
#         else:
#             return self.tf.train.Feature(int64_list=self.tf.train.Int64List(value=[value]))
#
#     def bytes_feature(self, value):
#         if isinstance(value, list):
#             return self.tf.train.Feature(bytes_list=self.tf.train.BytesList(value=value))
#         else:
#             return self.tf.train.Feature(bytes_list=self.tf.train.BytesList(value=[value]))
#
#     def get_coco_cats(self, supcats):
#         coco = self._load_coco_set('train')
#         cat_ids = coco.getCatIds(supNms=supcats)
#         cats = coco.loadCats(cat_ids)
#         cat_nms = [cat['name'] for cat in cats]
#         return cat_nms
#
#     def _load_coco_set(self, image_type):
#         annotation_file = os.path.join(self.data_dir, 'annotations', 'instances_%s2014.json' % image_type)
#         coco = COCO(annotation_file)
#         return coco
#
#     def _dump_to_tfrecord(self, coco, sup_cat, image_type):
#         cat_ids = coco.getCatIds(supNms=sup_cat)
#         cats = coco.loadCats(cat_ids)
#         catid_to_label = {cat['id']: cat['name'] for cat in cats}
#         imgs = coco.loadImgs(coco.getImgIds(catIds=cat_ids))
#         writer = self.tf.python_io.TFRecordWriter(
#             os.path.join(self.data_dir, 'tfrecords', image_type + "_" + sup_cat + ".tfrecords"))
#         for img in imgs:
#             print("[%s]Convertint Image(%s) to Tfrecord..." %
#                   (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), img['file_name']))
#             try:
#                 raw_img = cv2.cvtColor(cv2.imread(os.path.join(self.data_dir, image_type, img['file_name'])),
#                                        cv2.COLOR_BGR2RGB)
#                 img_anns = coco.imgToAnns[img['id']]
#                 categories = list(set([catid_to_label[ann['category_id']] for ann in img_anns
#                                        if ann['category_id'] in cat_ids]))
#                 example = self.tf.train.Example(features=self.tf.train.Features(feature={
#                     'image_raw': self.bytes_feature(raw_img.tostring()),
#                     'image_labels': self.bytes_feature([cat.encode('utf-8') for cat in categories]),
#                     'image_height': self.int64_feature(img['height']),
#                     'image_width': self.int64_feature(img['width'])
#                 }))
#                 writer.write(example.SerializeToString())
#             except Exception as err:
#                 print("Error Occur : " + str(err))
#         writer.close()
#
#     def decode_image(self, raw_image, image_height, image_width):
#         image = self.tf.decode_raw(raw_image, self.tf.uint8)
#         max_side = self.tf.reduce_max((image_height, image_width))
#         image_shape = self.tf.stack([image_height, image_width, 3])
#         image = self.tf.reshape(image, image_shape)
#         image = self.tf.cast(image, self.tf.float32)
#
#         image = self.tf.image.resize_image_with_crop_or_pad(image, max_side, max_side, 299)
#         image = self.tf.image.per_image_standardization(image)
#         image = self.tf.expand_dims(image, 0)
#         image = self.tf.image.resize_bicubic(
#             image, [299, 299], align_corners=False)
#
#         image = self.tf.squeeze(image, squeeze_dims=[0])
#         return image
#
#     def write_to_tfrecords(self, image_cats, image_type):
#         coco = self._load_coco_set(image_type)
#         if isinstance(image_cats, list):
#             for cat in image_cats:
#                 self._dump_to_tfrecord(coco, cat, image_type)
#         else:
#             self._dump_to_tfrecord(coco, image_cats, image_type)
#
#
#     def _read_and_decode(self, filename_queue):
#         reader = self.tf.TFRecordReader()
#         _, serialized_example = reader.read(filename_queue)
#         features = self.tf.parse_single_example(
#             serialized_example,
#             features={
#             "image_raw" : self.tf.FixedLenFeature([], self.tf.string),
#             "image_labels": self.tf.VarLenFeature(self.tf.string),
#             "image_height" : self.tf.FixedLenFeature([], self.tf.int64),
#             "image_width" : self.tf.FixedLenFeature([], self.tf.int64),
#             })
#         image = self.decode_image(features['image_raw'],
#                                   self.tf.cast(features['image_height'], self.tf.int32),
#                                   self.tf.cast(features['image_width'], self.tf.int32))
#         labels = self.tf.cast(features['image_labels'], self.tf.string)
#         return image, labels
#
#     def _get_tfrecords(self, require_type):
#         records = []
#         tfrecord_dir = os.path.join(self.data_dir, 'tfrecords')
#         for record in os.listdir(tfrecord_dir):
#             record_name, record_extension = record.split('.')
#             if record_extension != 'tfrecords': continue
#             record_type, record_cat = record_name.split('_')
#             if record_type == require_type:
#                 records.append(os.path.join(tfrecord_dir, record))
#         return records
#
#     def input_fn(self, tfrecords, batch_size, num_epochs=1, capacity=60, num_threads=8, min_after_dequeue=1000):
#         filename_queue = self.tf.train.string_input_producer(tfrecords, num_epochs=num_epochs,
#                                                              shuffle=True, capacity=capacity)
#         record_image, record_labels = self._read_and_decode(filename_queue)
#         batch_images, batch_labels = self.tf.train.shuffle_batch([record_image, record_labels],
#                                                                  num_threads=num_threads, batch_size=batch_size,
#                                                                  capacity=min_after_dequeue + 3 * batch_size,
#                                                                  min_after_dequeue=min_after_dequeue, )
#         return batch_images, batch_labels
#
#
#     def batch_data(self, batch_type, batch_size, num_epochs=1, capacity=60, num_threads=8, min_after_dequeue=1000):
#         tfrecords = self._get_tfrecords(batch_type)
#         return self.input_fn(tfrecords, batch_size, num_epochs, capacity, num_threads, min_after_dequeue)