from model import build_model
from tensorflow.keras import Model
from sample import samples
import numpy as np


def train(model: Model):
    g = samples('samples/train')
    g = list(g)
    x = np.expand_dims(np.asarray([i[0] for i in g]), 3)
    y = np.asarray([i[1] for i in g])
    model.fit(x, y,
              epochs=2,
              use_multiprocessing=True)

    # model.fit_generator(g,
    #                     steps_per_epoch=10,
    #                     epochs=10,
    #                     use_multiprocessing=True)
    return model

def evaluate(model: Model):
    g = samples('samples/test')
    g = list(g)
    x = np.expand_dims(np.asarray([i[0] for i in g]), 3)
    y = np.asarray([i[1] for i in g])
    test_loss, test_acc = model.evaluate(x, y, verbose=2)
    print('\nTest accuracy:', test_acc)


if __name__ == '__main__':
    model = build_model()
    train(model)
    evaluate(model)
