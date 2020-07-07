import random
from tensorflow.keras import Model
import sys
import os
from dataset import preprocess_image, label2vec, preprocess_label
from PIL import Image
import numpy as np
from config import *
from model import build_model


if __name__ == '__main__':
    image_dir = sys.argv[2]
    image_files = os.listdir(image_dir)
    random.shuffle(image_files)
    image_files = image_files[:1000]

    model: Model = build_model()
    model.load_weights(sys.argv[1])

    correct, miss = 0, 0
    X = []

    for image_name in image_files:
        fullpath = image_dir + '/' + image_name
        img = preprocess_image(Image.open(fullpath))
        X.append(np.expand_dims(img, 2))

    X = np.asarray(X)
    preds = model.predict(X)

    Y = preds.reshape((-1, char_count, len(char_set)))
    Y = np.argmax(Y, axis=2)

    for i, y in enumerate(Y):
        label = ''
        for ch_index in y:
            label += char_set[ch_index]

        image_name = image_files[i]
        label_true = preprocess_label(image_name.split('_')[0])

        if label_true == label:
            print('+ [%s] [%s] [%s]' % (image_name, label_true, label))
            correct += 1
        else:
            print('- [%s] [%s] [%s]' % (image_name, label_true, label))
            miss += 1

    print('correct=%d, miss=%d' % (correct, miss))



