import cv2
import numpy as np
import time
import os
import random

from util import utils

from util.tf_utils import PIC_SIZE, NUM_LABELS, get_charsets, cwd

__FONT_FILE_LIST = ["training_set"]


def __random_evaluate_set(set_size, max_size, seed):
    index_set = set()
    random.seed(seed)
    while len(index_set) != set_size:
        nxt = random.randint(0, max_size - 1)
        index_set.add(nxt)
    return index_set


def generate(batch_size, seed=time.time()):
    images = np.zeros([batch_size, PIC_SIZE, PIC_SIZE, 1], dtype=np.float32)
    labels = np.zeros([batch_size], dtype=np.int32)
    print("Generating Evaluate Set ...")
    temp_dir = os.path.join(cwd, __FONT_FILE_LIST[0])
    for i, j in enumerate(__random_evaluate_set(batch_size, NUM_LABELS, seed)):
        filename = os.path.join(temp_dir, str(j) + ".jpg")
        img = cv2.imread(filename)
        img = utils.binary(img)
        img = utils.resize(img, height=PIC_SIZE)
        images[i, :, :, 0] = np.array(img, np.float32)
        labels[i] = j
    print("Generated " + str(batch_size) + " Test Units")
    images = np.array(images)
    return images, labels
