import random
from tensorflow.keras import Model
import sys
import os
from dataset import preprocess_image
from PIL import Image
import numpy as np
from config import *
from model import build_model


if __name__ == '__main__':
    image_dir = sys.argv[2]
    if os.path.isdir(image_dir):
        image_files = os.listdir(image_dir)
    else:
        image_files = [image_dir.split('/')[-1]]
        image_dir = '/'.join(image_dir.split('/')[:-1])
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

    Y_pred = preds.reshape((-1, char_count, len(char_set)))
    Y_pred = np.argmax(Y_pred, axis=2)
    Y = []

    for i, y in enumerate(Y_pred):
        label = ''
        for ch_index in y:
            label += char_set[ch_index]

        image_name = image_files[i]
        print(image_name, label)



