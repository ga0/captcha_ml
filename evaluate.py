import random
from tensorflow.keras import Model
import sys
import os
from dataset import preprocess_image, label2vec, preprocess_label
from PIL import Image
import numpy as np
from config import *
from model import build_model
from dataset import CaptchaDataset
import matplotlib.pyplot as plt
import seaborn as sns


def draw_heatmap(Y_map):
    row_sums = Y_map.sum(axis=1)
    Y_map = Y_map / row_sums[:, np.newaxis]

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    f, ax = plt.subplots()
    sns.heatmap(Y_map, linewidths=0.1, linecolor='grey', xticklabels=char_set, yticklabels=char_set, ax=ax, cmap='YlGnBu')
    plt.setp(ax.get_yticklabels(), horizontalalignment='right')
    plt.setp(ax.get_xticklabels(), rotation=0, horizontalalignment='right')
    #plt.show()
    plt.savefig('heatmap.png')


def in_charset(label):
    for ch in label:
        if char_set.find(ch) == -1:
            return False
    return True

if __name__ == '__main__':
    image_dir = sys.argv[2]
    if os.path.isdir(image_dir):
        image_files = os.listdir(image_dir)
    else:
        image_files = [image_dir.split('/')[-1]]
        image_dir = '/'.join(image_dir.split('/')[:-1])
    image_files = [i for i in image_files if in_charset(i.split('_')[0])]
    random.shuffle(image_files)
    image_files = image_files[:10000]

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

    Y_map = np.zeros(shape=(len(char_set), len(char_set)), dtype='float32')

    for i, y in enumerate(Y_pred):
        label = ''
        for ch_index in y:
            label += char_set[ch_index]

        image_name = image_files[i]
        label_true = preprocess_label(image_name.split('_')[0])

        # Ignore case
        # label_true = label_true.lower()
        # label = label.lower()

        for ch_true, ch_pred in zip(label_true, label):
            id_true = char_set.index(ch_true)
            id_pred = char_set.index(ch_pred)
            Y_map[id_true][id_pred] += 1

        Y.append(label2vec(label_true))

        if label_true == label:
            print('+ [%s] [%s] [%s]' % (image_name, label_true, label))
            correct += 1
        else:
            print('- [%s] [%s] [%s]' % (image_name, label_true, label))
            miss += 1

    print('correct: %d, miss: %d' % (correct, miss))

    draw_heatmap(Y_map)

    loss, acc = model.evaluate(
        X,
        np.asarray(Y),
        steps=1,
        verbose=1)
    print('loss: {}, accuracy: {}'.format(loss, acc))



