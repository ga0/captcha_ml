import os
import random

from PIL import Image
import numpy as np
from config import *


def label2vec(label):
    vector = np.zeros(len(label) * len(char_set))

    for i, ch in enumerate(label):
        idx = i * len(char_set) + char_set.index(ch)
        vector[idx] = 1
    return vector


def preprocess_label(label):
    """
        Replace C->c, K->k, P->p, S->s, W->w, V->v, U->u, X->x, Y->y, Z->z
    """
    return label.replace('C', 'c').replace('K', 'k').replace('P', 'p').replace('S', 's').replace('W', 'w') \
        .replace('V', 'v').replace('U', 'u').replace('X', 'x').replace('Y', 'y').replace('Z', 'z')


def preprocess_image(img: Image):
    # resize image
    img = img.resize((image_width, image_height))

    # convert to array
    img_array = np.array(img)

    # make it gray scale
    if len(img_array.shape) > 2:
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        gray = (0.2989 * r + 0.5870 * g + 0.1140 * b) / 255
        # gray_image = Image.fromarray(gray * 255)
        # gray_image.show()
        return gray
    else:
        return img_array / 255


def samples(dir: str):
    images = [f for f in os.listdir(dir) if f.endswith('.png')]
    random.shuffle(images)
    images = images[:1000]

    for img in images:
        label = img.split('_')[0]
        captcha_image = Image.open(os.path.join(dir, img))

        yield preprocess_image(captcha_image), label2vec(preprocess_label(label))


if __name__ == '__main__':
    for x, y in samples('/Users/gaoxueyao/projects/captcha/samples'):
        print(x, y)
    print('Done')
