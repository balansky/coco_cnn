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

def get_image_categories(coco_root, sup_cats):
    coco_set = dataset.CoCoSet(coco_root, 'train')
    if not sup_cats:
        sup_cats = coco_set.coco_sup_cats()
    cat_dict = {}
    for sup_cat in sup_cats:
        image_cats = coco_set.coco_cat_nms(sup_cat)
        cat_dict[sup_cat] = image_cats
    return cat_dict
    # tf_record.save_tf_categories(cat_dict)

def get_tfrecord_files(tf_dir, tf_type):
    tfrecord_files = {}
    for file in os.listdir(tf_dir):
        file_info, file_ext = file.split('.')
        if file_ext == 'tfrecords':
            ftype, fcat, fidx = file_info.split('_')
            if ftype == tf_type:
                if fcat not in tfrecord_files:
                    tfrecord_files[fcat] = []
                tfrecord_files[fcat].append(file)
    return tfrecord_files


def main(coco_root, tf_dir, config_dir, sup_cats):
    tf_record = tfrecord.TfRecord(tf_dir, config_dir)
    train_files = convert_to_tfrecord(tf_record, coco_root, 'train', sup_cats)
    val_files = convert_to_tfrecord(tf_record, coco_root, 'val', sup_cats)
    cats = get_image_categories(coco_root, sup_cats)
    # train_files = get_tfrecord_files(tf_dir, 'train')
    # val_files = get_tfrecord_files(tf_dir, 'val')
    tf_record.save_tf_categories(cats)
    tf_info = {'train': train_files, 'val': val_files}
    tf_record.save_tf_info(tf_info)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-root',
                        type=str,
                        default="/home/andy/Data/coco",
                        help='Coco Data Directory')
    parser.add_argument('--tf-dir',
                        type=str,
                        default="/home/andy/Data/coco/tfrecords",
                        help='Tfrecord Directory')
    parser.add_argument('--config-dir',
                        type=str,
                        default="configs",
                        help='Tfrecord Directory')
    parser.add_argument('--sup-cats',
                        default=None,
                        type=str,
                        help='Super Categorie of Coco Dataset', nargs='+')
    parse_args, unknown = parser.parse_known_args()
    main(**parse_args.__dict__)