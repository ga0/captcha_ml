import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
import sys
import os
from dataset import preprocess_image, label2vec
from PIL import Image
import numpy as np
from config import *

from train import evaluate

if __name__ == '__main__':
    image_name = sys.argv[1]

    model: Model = keras.models.load_model("captcha_model")
    # model = keras.Sequential([model, layers.Softmax()])
    # evaluate(model)

    img = preprocess_image(Image.open(image_name))
    x = np.expand_dims(img, 0)
    x = np.expand_dims(x, 3)
    preds = model.predict(x)

    print(preds)

    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0

    print(preds)

    y = preds.reshape((-1, char_count, len(char_set)))
    y0 = y[0]

    for a in y0:
        am = np.argmax(a)
        print(am, char_set[am])


