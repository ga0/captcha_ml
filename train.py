from model import build_model
import tensorflow as tf
from tensorflow.keras import Model
from dataset import samples
from config import *


def train(model: Model):
    checkpoint_path = "checkpoint/cp-{epoch:04d}.ckpt"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=5)

    validation_data = []
    for v in samples('samples/test', batch_size=1000, max_samples=1000):
        validation_data = v
        break

    batch_size = 32
    model.fit(
        samples("samples/train", batch_size=batch_size, max_samples=train_set_size),
        validation_data=validation_data,
        steps_per_epoch=train_set_size//batch_size,
        epochs=100,
        callbacks=[cp_callback],
        initial_epoch=0)
    return model


def evaluate(model: Model):
    loss, acc = model.evaluate(
        samples("samples/test", batch_size=1000, max_samples=1000),
        steps=1,
        verbose=1)
    print('loss: {}, accuracy: {}'.format(loss, acc))


if __name__ == '__main__':
    model = build_model()
    train(model)
    model.save('captcha_model')
