import os
import sys
import cv2
import numpy as np
import argparse
import tensorflow as tf
from tensorflow import keras as tk
from functools import partial

from load_dataset import DataLoader

from models import get_efficientnet_model

IMAGE_HEIGHT = 400
IMAGE_WIDTH = 400

import time


def get_date_time_str() -> str:
    return time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())



def augment(image, label):
    image = tf.image.resize_with_crop_or_pad(
        image,
        target_height=IMAGE_HEIGHT + 16,
        target_width=IMAGE_WIDTH + 16
    )
    image = tf.image.random_crop(image, size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    image = tf.image.random_brightness(image, max_delta=0.25)
    image = tf.image.random_hue(image, max_delta=0.25)
    image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
    image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
    image = tf.image.random_flip_left_right(image)

    return image, label


def eval_failures(
        fw: tf.summary.SummaryWriter,
        ds: tf.data.Dataset,
        epoch,
        logs):
    failures = list()
    num_fail_no_mask = 0
    num_fail_oth_mask = 0
    num_fail_med_mask = 0
    for x, y in ds:
        res = model.predict(x)

        for i in range(len(x)):
            lbl = np.argmax(y[i])
            if np.argmax(res[i]) != lbl:
                failures.append((x[i], res[i], y[i]))
                if lbl == 0:
                    num_fail_no_mask += 1
                elif lbl == 1:
                    num_fail_oth_mask += 1
                elif lbl == 2:
                    num_fail_med_mask += 1

    with fw.as_default():
        tf.summary.scalar(f"number of val failures total", len(failures), step=epoch)
        tf.summary.scalar(f"number of val failures no mask", num_fail_no_mask, step=epoch)
        tf.summary.scalar(f"number of val failures other mask", num_fail_oth_mask, step=epoch)
        tf.summary.scalar(f"number of val failures medical mask", num_fail_med_mask, step=epoch)
        for i, (im, x, y) in enumerate(failures):
            im = im.numpy()
            im = cv2.rectangle(im, (0, 0), (IMAGE_WIDTH, 15), color=(0, 0, 0), thickness=-1)
            im = cv2.putText(
                img=im,
                text=f"{np.round(x, decimals=2)} | {y}",
                org=(3, 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.25,
                color=(255, 255, 255),
                lineType=cv2.LINE_AA
            )

            tf.summary.image(f"failures epoch {epoch}", im[np.newaxis, :], step=i)


if __name__ == "__main__":

    data_dir = '/workspace/training_custom/images'
    save_dir = '/workspace/training_custom/weights'
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    
    IMG_H = 225
    IMG_W = 225
    NUM_CLASSES = 2
    EPOCHS = 10

    dl = DataLoader(img_h=IMG_H, img_w=IMG_W, num_classes=NUM_CLASSES)

    train_file_path = '/workspace/training_custom/train.txt'
    val_file_path = '/workspace/training_custom/val.txt'
    test_file_path = '/workspace/training_custom/test.txt'

    ds_train, ds_val, ds_test = dl.get_all_datasets(train_file=train_file_path, \
                                                    val_file=val_file_path)
    ds_train = ds_train.shuffle(buffer_size=100)
    # ds_train = ds_train.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.batch(16)
    ds_train = ds_train.prefetch(64)

    ds_eval_failures = ds_val
    ds_val = ds_val.batch(16)

    ds_eval_failures = ds_eval_failures.batch(8)

    # print(ds_train)
    # sys.exit()
    model, base_model = get_efficientnet_model(IMAGE_HEIGHT=IMG_H, IMAGE_WIDTH=IMG_W, NumClasses=NUM_CLASSES)
    model.summary()

    logpath = os.path.join(save_dir, "efficientnet", "b2", model.name, get_date_time_str())
    if not os.path.exists(logpath):
        os.makedirs(logpath)

    callback_tb = tk.callbacks.TensorBoard(
        log_dir=os.path.join(logpath, "tb")
    )

    callback_save = tk.callbacks.ModelCheckpoint(
        filepath=os.path.join(logpath, "save", "epoch_{epoch:03d}"),
        save_weights_only=False,
        save_freq="epoch",
        verbose=1
    )

    fw = tf.summary.create_file_writer(os.path.join(logpath, "tb", "val-failure"))
    callback_failures = tk.callbacks.LambdaCallback(
        on_epoch_end=partial(eval_failures, fw, ds_eval_failures)
    )

    """
        regular training
    """
    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=[
            tk.callbacks.ReduceLROnPlateau(patience=2, factor=0.8, min_delta=2e-3),
            callback_tb,
            callback_save,
            callback_failures
        ]
    )

    """
        training/fine-tuning
    """
    # base_model.trainable = False
    # model.compile(
    #     loss=tk.losses.CategoricalCrossentropy(from_logits=True),
    #     optimizer=tk.optimizers.Adam(learning_rate=8e-4),
    #     metrics=["acc"]
    # )
    #
    # history = model.fit(
    #     ds_train,
    #     validation_data=ds_val,
    #     epochs=15,
    #     callbacks=[
    #         tk.callbacks.ReduceLROnPlateau(patience=2, factor=0.8, min_delta=2e-3),
    #         callback_tb,
    #         callback_save,
    #         callback_failures
    #     ]
    # )
    #
    # base_model.trainable = True
    # model.compile(
    #     loss=tk.losses.CategoricalCrossentropy(from_logits=True),
    #     optimizer=tk.optimizers.Adam(learning_rate=1e-4),
    #     metrics=["acc"]
    # )
    #
    # model.fit(
    #     ds_train,
    #     validation_data=ds_val,
    #     epochs=EPOCHS,
    #     initial_epoch=history.epoch[-1],
    #     callbacks=[
    #         tk.callbacks.ReduceLROnPlateau(patience=2, factor=0.8, min_delta=1e-3),
    #         callback_tb,
    #         callback_save,
    #         callback_failures
    #     ]
    # )
