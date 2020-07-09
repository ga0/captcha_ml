from model import build_model
import tensorflow as tf
from tensorflow.keras import Model
from dataset import CaptchaDataset
from config import *
import sys


def train(model: Model, init_epoch):
    checkpoint_path = "checkpoint/cp-{epoch:04d}.ckpt"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq='epoch')

    batch_size = 32
    model.fit(
        CaptchaDataset("samples/py_captcha", batch_size=batch_size, max_samples=1000),
        validation_data=CaptchaDataset("samples/test", batch_size=batch_size),
        epochs=init_epoch+100,
        callbacks=[cp_callback],
        initial_epoch=init_epoch)
    return model


def evaluate(model: Model):
    loss, acc = model.evaluate(
        CaptchaDataset('samples/test', batch_size=1000),
        steps=1,
        verbose=1)
    print('loss: {}, accuracy: {}'.format(loss, acc))


if __name__ == '__main__':
    model = build_model()
    if len(sys.argv) > 1:
        init_epoch = int(sys.argv[1])
        model.load_weights(f'checkpoint/cp-{init_epoch :04d}.ckpt')
    else:
        init_epoch = 0
    train(model, init_epoch)
    model.save('captcha_model')
