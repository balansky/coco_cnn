from pycocotools.coco import COCO
import numpy as np
import os

class DataSet(object):

    def __init__(self, coco_root, image_cats=[]):
        self.data_root = coco_root
        self.train_imgs, self.cats, self.train_to_cats = self._load_imgs(image_cats, image_type='train')
        self.cat_to_label = {cat['id']: i for i, cat in enumerate(self.cats)}
        self.val_imgs, _, self.val_to_cats = self._load_imgs(image_cats, image_type='val')


    def _load_imgs(self, image_cats, image_type='train'):
        annotation_file = os.path.join(self.data_root, 'annotations', 'instances_%s2014.json' % image_type)
        coco = COCO(annotation_file)
        cat_ids = coco.getCatIds(supNms=image_cats)
        cats = coco.loadCats(cat_ids)
        if cat_ids:
            imgs = coco.loadImgs(coco.getImgIds(catIds=cat_ids))
        else:
            imgs = []
        img_to_cats = {img['id']: list(set([ann['category_id'] for ann in coco.imgToAnns[img['id']]
                                            if ann['category_id'] in cat_ids])) for img in imgs}
        return imgs, cats, img_to_cats


    def next_batch(self, batch_size, batch_type='train'):
        image_set = self.train_imgs if batch_type == 'train' else self.val_imgs
        label_set = self.train_to_cats if batch_type == 'train' else self.val_to_cats
        # image_set = self.train_imgs
        # label_set = self.train_to_cats
        random_idx = np.random.choice(len(self.train_imgs), batch_size)
        picked_imgs = [image_set[idx] for idx in random_idx]
        picked_labels = np.zeros((batch_size, len(self.cats)), np.float32)
        for i, img in enumerate(picked_imgs):
            img['file_name'] = os.path.join(self.data_root, batch_type, img['file_name'])
            labels = [self.cat_to_label[cat] for cat in label_set[img['id']]]
            picked_labels[i][labels] = 1.0
        return picked_imgs, picked_labels



