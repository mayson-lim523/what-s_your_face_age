from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import hashlib
import re
import glob
from shutil import copyfile
import logging
from pathlib import Path

logger = logging.getLogger('train_testsplit')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

FLAGS = None

MAX_NUM_IMAGES_PER_CLASS = 2**27 - 1

def create_image_lists(image_dir, testing_percentage, validation_percentage):

    image_dir_path = Path(image_dir)
    if not image_dir_path.exists:
        logger.error(
            "Image directory'" + image_dir_path.name +"'not found.")
        return None
    result = {}
    sub_dirs = [x for x in image_dir_path.iterdir() if x.is_dir()]
    for sub_dir in sub_dirs:
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = sub_dir.name
        logger.info("Looking for images in '{}'".format(sub_dir))
        for extension in extensions:
            file_list.extend(sub_dir.glob('*.'+extension))
        if not file_list:
            logger.warning('No files found')
            continue
        if len(file_list) < 20:
            logger.warning('WARNING : Folder ha less than 20 images, which may cause issues. Skipping.')
            continue
        elif len(file_list) > FLAGS.max_per_file:
            logger.warning(
                'WARNING: Folder {} has more than {} images. Pruning.'.format(
                    dir_name, FLAGS.max_per_file))
            file_list = file_list[:FLAGS.max_per_file]
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = Path(file_name).name
            hash_name = re.sub(r'_nohash_.*$', '', str(file_name))
            hash_name_hashed = hashlib.sha1(
                hash_name.encode(errors='replace')).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (
                    testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result


def divide_images():
    img_dir = FLAGS.image_dir
    testing_pct = FLAGS.pct_test
    validation_pct = FLAGS.pct_validation
    out_dir = Path(FLAGS.output_dir)
    image_lists = create_image_lists(img_dir, testing_pct, validation_pct)
    class_count = len(image_lists.keys())
    if class_count == 0:
        logger.error('No valid folders of images found at ' + FLAGS.image_dir)
        return -1
    if class_count == 1:
        logger.error(
            'Only one valid folder of images found at ' + FLAGS.image_dir +
            ' - multiple classes are needed for classification.')
        return -1
    out_dir.mkdir()
    train_dir = out_dir / 'training'
    test_dir = out_dir / 'testing'
    val_dir = out_dir / 'validation'
    train_dir.mkdir()
    test_dir.mkdir()
    val_dir.mkdir()
    for cl in image_lists.keys():
        td_cl = train_dir / cl
        te_cl = test_dir / cl
        v_cl = val_dir / cl
        td_cl.mkdir()
        te_cl.mkdir()
        v_cl.mkdir()
        indir = Path(img_dir) / image_lists[cl]['dir']
        for img in image_lists[cl]['training']:
            copyfile(indir / img, td_cl / img)
        for img in image_lists[cl]['testing']:
            copyfile(indir / img, te_cl / img)
        for img in image_lists[cl]['validation']:
            copyfile(indir / img, v_cl / img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help='Path to folders of labeled images.')
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Where to save the divided images.')
    parser.add_argument(
        '--pct_test',
        type=int,
        default=10,
        help='Percentage of input set to use for training.')
    parser.add_argument(
        '--pct_validation',
        type=int,
        default=20,
        help='Percentage of images to use in validation.')
    parser.add_argument(
        '--seed', type=float, default=1337, help='Random seed.')
    parser.add_argument(
        '--max_per_file',
        type=int,
        default=MAX_NUM_IMAGES_PER_CLASS,
        help='Limit the maximum number of images in a given class')
    FLAGS, _ = parser.parse_known_args()
    divide_images()
        
