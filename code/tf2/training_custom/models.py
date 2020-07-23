
import tensorflow as tf
from tensorflow import keras as tk

# from efficientnet.tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2


def get_efficientnet_model(IMAGE_HEIGHT, IMAGE_WIDTH, NumClasses, LEARNING_RATE=4e-5):
    base_model = tf.keras.applications.EfficientNetB2(
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
        optimizer=tk.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=["acc"]
    )

    return model, base_model

def get_resnet18_model(IMAGE_HEIGHT, IMAGE_WIDTH, NumClasses):
    pass

def get_alexnet(IMAGE_HEIGHT, IMAGE_WIDTH, NumClasses, LEARNING_RATE):
    model = tf.keras.Sequential([
        # layer 1
        tf.keras.layers.Conv2D(filters=96,
                               kernel_size=(11, 11),
                               strides=4,
                               padding="valid",
                               activation=tf.keras.activations.relu,
                               input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="valid"),
        tf.keras.layers.BatchNormalization(),
        # layer 2
        tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(5, 5),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="same"),
        tf.keras.layers.BatchNormalization(),
        # layer 3
        tf.keras.layers.Conv2D(filters=384,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
        # layer 4
        tf.keras.layers.Conv2D(filters=384,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
        # layer 5
        tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="same"),
        tf.keras.layers.BatchNormalization(),
        # layer 6
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=4096,
                              activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(rate=0.2),
        # layer 7
        tf.keras.layers.Dense(units=4096,
                              activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(rate=0.2),
        # layer 8
        tf.keras.layers.Dense(units=NumClasses,
                              activation=tf.keras.activations.softmax)
    ])

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  metrics=['accuracy'])

    return model