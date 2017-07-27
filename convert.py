from cores import dataset
from utils import tfrecord
import argparse
import os
import json

def convert_to_tfrecord(tf_record, coco_root, coco_type, sup_cats):
    train_coco_set = dataset.CoCoSet(coco_root, coco_type)
    if not sup_cats:
        sup_cats = train_coco_set.coco_sup_cats()
    for sup_cat in sup_cats:
        image_files = train_coco_set.coco_images(sup_cat)
        tf_record.write_image_tfrecord(image_files, coco_type, sup_cat)

def save_image_categories(coco_root, tf_record, sup_cats):
    coco_set = dataset.CoCoSet(coco_root, 'train')
    if not sup_cats:
        sup_cats = coco_set.coco_sup_cats()
    cat_dict = {}
    for sup_cat in sup_cats:
        image_cats = coco_set.coco_cat_nms(sup_cat)
        cat_dict[sup_cat] = image_cats
    tf_record.save_tf_categories(cat_dict)


def main(coco_root, tf_dir, sup_cats):
    tf_record = tfrecord.TfRecord(tf_dir)
    convert_to_tfrecord(tf_record, coco_root, 'train', sup_cats)
    # convert_to_tfrecord(coco_root, tf_dir, 'val', sup_cats)
    # save_image_categories(coco_root, tf_dir, sup_cats)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_root',
                        type=str,
                        default="/home/andy/Data/coco",
                        help='Coco Data Directory')
    parser.add_argument('--tf_dir',
                        type=str,
                        default="/home/andy/Data/coco/tfrecords",
                        help='Tfrecord Directory')
    parser.add_argument('--sup-cats',
                        default=None,
                        type=str,
                        help='Super Categorie of Coco Dataset', nargs='+')
    parse_args, unknown = parser.parse_known_args()
    main(**parse_args.__dict__)