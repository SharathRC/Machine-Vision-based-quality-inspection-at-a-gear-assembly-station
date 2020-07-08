import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as tk

import tensorflow_datasets as tfds

from efficientnet.tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2



IMAGE_HEIGHT = 255
IMAGE_WIDTH = 255
NumClasses = 2


def get_efficient_model():
    base_model = EfficientNetB2(
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        include_top=False,
        weights="imagenet"
    )

    input_layer = tk.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    base_model = base_model(input_layer)

    maxpool = tk.layers.GlobalMaxPooling2D()(base_model)
    avgpool = tk.layers.GlobalAveragePooling2D()(base_model)

    features_maxpool = tk.layers.Dense(20)(maxpool)
    features_maxpool = tk.layers.LeakyReLU()(features_maxpool)
    features_maxpool = tk.layers.Dropout(0.25)(features_maxpool)

    features_avgpool = tk.layers.Dense(20)(avgpool)
    features_avgpool = tk.layers.LeakyReLU()(features_avgpool)
    features_avgpool = tk.layers.Dropout(0.25)(features_avgpool)

    x = tk.layers.Concatenate(axis=-1)([features_maxpool, features_avgpool])
    x = tk.layers.Dense(NumClasses)(x)

    model = tk.Model(inputs=input_layer, outputs=x, name="max_avg_pool")

    # model = tk.Sequential()
    # model.add(base_model)
    # model.add(tk.layers.Flatten())
    # model.add(tk.layers.BatchNormalization())
    # model.add(tk.layers.Dense(20))
    # model.add(tk.layers.BatchNormalization())
    # model.add(tk.layers.LeakyReLU())
    # model.add(tk.layers.Dropout(0.5))
    # model.add(tk.layers.Dense(NumClasses))
    # model.add(tk.layers.Softmax())

    model.compile(
        loss=tk.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=tk.optimizers.Adam(learning_rate=4e-5),
        metrics=["acc"]
    )

    return model, base_model


