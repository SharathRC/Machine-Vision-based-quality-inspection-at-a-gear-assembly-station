from multiprocessing import cpu_count
from typing import List, Dict, Tuple
from functools import partial
import tensorflow as tf
from tensorflow import keras as tk
import glob
import os

from api.types import DatasetType
from config.consts import ClassesNames, ClassLabelMapping, IMAGE_HEIGHT, IMAGE_WIDTH, NumClasses


def find_img_paths(path: str):
    filepaths = list()
    file_extensions = ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG", "avif", "AVIF"]
    # file_extensions = ["jpg"]
    img_wildcards = ["**/*" + ext for ext in file_extensions]

    for w in img_wildcards:
        filepaths.extend(glob.glob(os.path.join(path, w)))
    print(f"found {len(filepaths)} in '{path}'")

    return filepaths


def get_train_val_test_ds(dataset_path: str) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    :param dataset_path: dataset root
    :return: returns all 3 tf datasets
    """
    return get_train_ds(dataset_path), get_val_ds(dataset_path), get_test_ds(dataset_path)


def get_train_ds(dataset_path: str) -> tf.data.Dataset:
    """
    :param dataset_path: dataset root
    :return: returns train dataset
    """
    return _get_dataset(DatasetType.TRAIN, dataset_path)


def get_val_ds(dataset_path: str) -> tf.data.Dataset:
    """
    :param dataset_path: dataset root
    :return: returns validation dataset
    """
    return _get_dataset(DatasetType.VALIDATION, dataset_path)


def get_test_ds(dataset_path: str) -> tf.data.Dataset:
    """
    :param dataset_path: dataset root
    :return: returns test dataset
    """
    return _get_dataset(DatasetType.TEST, dataset_path)




# def _get_dataset(ds_type: DatasetType, dataset_path: str) -> tf.data.Dataset:
#     # path to dataset type (train, val, ...)
#     subpath = os.path.join(dataset_path, ds_type.value)

#     img_paths = list()
#     labels = list()
#     for class_name in ClassesNames:
#         # class directory name (no-mask, ...)
#         class_dirname = class_name.value
#         # complete path for current class
#         class_path = os.path.join(subpath, class_dirname)
#         # all images for that class and dataset type
#         buffer_img_paths = find_img_paths(class_path)
#         # save img paths
#         img_paths.extend(buffer_img_paths)
#         # labels
#         labels.extend([ClassLabelMapping[class_name] for imp in buffer_img_paths])

#     final_ds = _create_dataset(img_paths, labels)

#     return final_ds

def _get_dataset(text_file):
    img_paths, labels = _get_files_and_labels(text_file)
    final_ds = _create_dataset(img_paths, labels)
    return final_ds

def _get_files_and_labels(text_file):
    with open(text_file, 'r') as f:
        img_paths = []
        labels = []
        for path in f.readlines():
            img_paths.append(path[:-1])
            label = int(path.split('/')[-1].split('_')[0])
            labels.append(label)
    f.close()
    return img_paths, labels

def _get_all_datasets(train_file=None, val_file=None, test_file=None):
    train_ds, val_ds, test_ds = None, None, None
    if not train_file is None:
        train_ds = _get_dataset(train_file)
    if not val_file is None:
        val_ds = _get_dataset(val_file)
    if not test_file is None:
        test_ds = _get_dataset(test_file)
    
    return train_ds, val_ds, test_ds




def _create_dataset(img_paths: List[str], labels: List[int]) -> tf.data.Dataset:
    ds_img = tf.data.Dataset.from_tensor_slices(img_paths)
    ds_img = ds_img.map(_load_img, num_parallel_calls=cpu_count())

    ds_lbl = tf.data.Dataset.from_tensor_slices(labels)
    ds_lbl = ds_lbl.map(_load_lbl, num_parallel_calls=cpu_count())

    ds = tf.data.Dataset.zip((ds_img, ds_lbl))

    return ds


def _load_img(im_path):
    im = tf.io.read_file(im_path)
    im = tf.io.decode_jpeg(im, channels=3)
    im = tf.cast(im, tf.float32)
    im = tf.image.resize(im, size=(IMAGE_HEIGHT, IMAGE_WIDTH), method=tf.image.ResizeMethod.BILINEAR)
    im /= 255.0

    return im


def _load_lbl(label):
    label = tf.cast(label, tf.uint8)
    label = tf.one_hot(indices=label, depth=NumClasses, dtype=tf.float32)

    return label
