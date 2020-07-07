import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
import sys
import os
from dataset import preprocess_image, label2vec, preprocess_label
from model import char_accuracy
from PIL import Image
import numpy as np
from config import *
from model import build_model
from train import evaluate


if __name__ == '__main__':
    image_dir = sys.argv[2]

    model: Model = build_model()
    model.load_weights(sys.argv[1])

    correct, miss = 0, 0
    for image_name in os.listdir(image_dir):
        fullpath = image_dir + '/' + image_name
        img = preprocess_image(Image.open(fullpath))
        x = np.expand_dims(img, 0)
        x = np.expand_dims(x, 3)
        preds = model.predict(x)

        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0

        y = preds.reshape((-1, char_count, len(char_set)))
        y0 = y[0]

        label = ''
        for a in y0:
            am = np.argmax(a)
            label += char_set[am]

        label_true = preprocess_label(image_name.split('_')[0])

        if label_true == label:
            print('+ [%s] [%s] [%s]' % (image_name, label_true, label))
            correct += 1
        else:
            print('- [%s] [%s] [%s]' % (image_name, label_true, label))
            miss += 1

    print('correct=%d, miss=%d' % (correct, miss))



