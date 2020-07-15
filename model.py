import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import Sequential

from config import *

import configparser
cfg_parser = configparser.ConfigParser()
cfg_parser.read('hyper_param.ini')

dropout_rate = cfg_parser['default'].getfloat('dropout_rate', fallback=0.25)
learning_rate = cfg_parser['default'].getfloat('learning_rate', fallback=0.0001)
batch_size = cfg_parser['default'].getint('batch_size', fallback=32)


def print_hyper_params():
    print(f'Hyper-params: dropout_rate={dropout_rate}, learning_rate={learning_rate}, batch_size={batch_size}')


def char_accuracy(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, len(char_set)))
    y_true = K.reshape(y_true, (-1, len(char_set)))
    y_p = K.argmax(y_pred, axis=1)
    y_t = K.argmax(y_true, axis=1)
    r = K.mean(K.cast(K.equal(y_p, y_t), 'float32'))
    return r


def build_model() -> Sequential:
    model = Sequential(
        [
            # Conv Layer 1
            layers.Conv2D(32,
                          (3, 3),
                          input_shape=(image_height, image_width, 1),
                          padding='same',
                          activation='relu'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Dropout(dropout_rate),

            # Conv Layer 2
            layers.Conv2D(64,
                          (3, 3),
                          padding='same',
                          activation='relu'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Dropout(dropout_rate),

            # Conv Layer 3
            layers.Conv2D(128,
                          (3, 3),
                          padding='same',
                          activation='relu'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Dropout(dropout_rate),

            # Conv Layer 4
            layers.Conv2D(256,
                          (3, 3),
                          padding='same',
                          activation='relu'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Dropout(dropout_rate),
            #
            # # Conv Layer 5
            # layers.Conv2D(32,
            #               (3, 3),
            #               padding='same',
            #               activation='relu'),
            # layers.MaxPooling2D((2, 2), padding='same'),
            # layers.Dropout(dropout_rate),

            # Dense 1
            layers.Flatten(),
            layers.Dense(1024,
                         activation='relu'),
            layers.Dropout(dropout_rate),

            # Dense 2
            layers.Dense(char_count * len(char_set))
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[char_accuracy])

    return model


if __name__ == '__main__':
    m = build_model()
    print(m.summary())
