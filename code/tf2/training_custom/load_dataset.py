from multiprocessing import cpu_count
from typing import List
import tensorflow as tf

class DataLoader:
    def __init__(self, img_h, img_w, num_classes):
        self.IMAGE_HEIGHT = img_h
        self.IMAGE_WIDTH = img_w
        self.NumClasses = num_classes

    def _get_dataset(self, text_file):
        img_paths, labels = self._get_files_and_labels(text_file)
        final_ds = self._create_dataset(img_paths, labels)
        return final_ds

    def _get_files_and_labels(self, text_file):
        with open(text_file, 'r') as f:
            img_paths = []
            labels = []
            for path in f.readlines():
                img_paths.append(path[:-1])
                label = int(path.split('/')[-1].split('_')[0])
                labels.append(label)
        f.close()
        return img_paths, labels

    def _create_dataset(self, img_paths: List[str], labels: List[int]) -> tf.data.Dataset:
        ds_img = tf.data.Dataset.from_tensor_slices(img_paths)
        ds_img = ds_img.map(self._load_img, num_parallel_calls=cpu_count())

        ds_lbl = tf.data.Dataset.from_tensor_slices(labels)
        ds_lbl = ds_lbl.map(self._load_lbl, num_parallel_calls=cpu_count())

        ds = tf.data.Dataset.zip((ds_img, ds_lbl))

        return ds

    def _load_img(self, im_path):
        im = tf.io.read_file(im_path)
        im = tf.io.decode_jpeg(im, channels=3)
        im = tf.cast(im, tf.float32)
        im = tf.image.resize(im, size=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH), method=tf.image.ResizeMethod.BILINEAR)
        im /= 255.0

        return im

    def _load_lbl(self, label):
        label = tf.cast(label, tf.uint8)
        label = tf.one_hot(indices=label, depth=self.NumClasses, dtype=tf.float32)

        return label
    
    def get_all_datasets(self, train_file=None, val_file=None, test_file=None):
        train_ds, val_ds, test_ds = None, None, None
        if not train_file is None:
            train_ds = self._get_dataset(train_file)
        if not val_file is None:
            val_ds = self._get_dataset(val_file)
        if not test_file is None:
            test_ds = self._get_dataset(test_file)
        
        return train_ds, val_ds, test_ds
