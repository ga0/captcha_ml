from model import build_model
import tensorflow as tf
from tensorflow.keras import Model
from dataset import CaptchaDataset
import sys
from config import *


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def train(model: Model, init_epoch):
    checkpoint_path = "checkpoint/cp-{epoch:04d}.ckpt"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq='epoch')

    model.fit(
        CaptchaDataset("samples/train", batch_size=batch_size),
        validation_data=CaptchaDataset("samples/test", batch_size=batch_size),
        epochs=init_epoch+100,
        callbacks=[cp_callback],
        initial_epoch=init_epoch,
        verbose=2)
    return model


if __name__ == '__main__':
    print('Training start')
    print_hyper_params()
    model = build_model()
    if len(sys.argv) > 1:
        init_epoch = int(sys.argv[1])
        model.load_weights(f'checkpoint/cp-{init_epoch :04d}.ckpt')
    else:
        init_epoch = 0
    train(model, init_epoch)
    model.save('captcha_model')
