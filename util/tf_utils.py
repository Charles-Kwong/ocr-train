# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np
import cv2
from PIL import Image

# FONT_LIST = ["chinese-msyh", "chinese-simfang", "chinese-simhei", "chinese-simkai", "chinese-simsun",
#              "chinese-AdobeSongStd-Light"]
FONT_LIST = ["training_set", "training_set_1"]
FONT_NUM = len(FONT_LIST)
INNER_SIZE = 24
# 总样本数为 INNER_SIZE * 字体数 * 汉字数
# 类别数
NUM_LABELS = 500
# 图片大小 pixel default 28
PIC_SIZE = 28

cwd = "training_data/" + str(NUM_LABELS) + "/"
file_list = []
label_list = []


def get_charsets():
    """
    :rtype: list
    """
    # 读取类别列表
    f = open("training_data/" + str(NUM_LABELS) + ".txt", "r", encoding="utf-8")
    tmp_str = f.readline()
    f.close()
    # 定义类别
    classes = []
    for s in tmp_str:
        classes.append(s)
    return classes


def get_char_code(input_str):
    charsets = get_charsets()
    labels = list()
    for s in input_str:
        for i, char in enumerate(charsets):
            if char == s:
                labels.append(i)
                break
    return labels


def load_images(path):
    img_list = os.listdir(path)
    batch_size = len(img_list)
    images = np.zeros([batch_size, PIC_SIZE, PIC_SIZE, 1], dtype=np.float32)
    labels = np.zeros([batch_size], dtype=np.int32)
    for i, img in enumerate(img_list):
        image = Image.open(os.path.join(path, img))
        # image = image.resize((PIC_SIZE, PIC_SIZE), Image.ANTIALIAS)
        image = resize(image)
        images[i, :, :, 0] = np.array(image, np.float32)
        # 获取标记(label)
        if img.__contains__(".jpg"):
            labels[i] = int(img.replace(".jpg", ""))
        elif img.__contains__(".png"):
            labels[i] = int(img.replace(".png", ""))
    return images, labels


def resize(img):
    temp_img = np.ones([PIC_SIZE, PIC_SIZE]) * 255
    image = np.array(img, np.uint8)
    h, w = image.shape[:2]
    if h > PIC_SIZE:
        temp = img.resize((int(PIC_SIZE * (w / h)), PIC_SIZE), Image.ANTIALIAS)
        image = np.array(temp, np.uint8)
    h, w = image.shape[:2]
    j = int(PIC_SIZE - w) / 2
    j = int(j)
    for k in range(j, w + j):
        for l in range(PIC_SIZE):
            temp_img[l, k] = image[l, k - j]
    return np.array(temp_img, np.uint8)


def get_batch(file_dir, batch_size, shuffle=False):
    # 读取TFRecord文件，创建文件列表，并通过文件列表创建输入文件队列。在调用输入数据处理流程前，需要统一所有原始数据的格式并将它们存储到TFRecord文件中。
    files = tf.train.match_filenames_once(file_dir)
    filename_queue = tf.train.string_input_producer(files, shuffle=shuffle)

    # 解析TFRecord文件里的数据
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           # 'pixels': tf.FixedLenFeature([], tf.int64),
                                       })

    # 得到图像原始数据、尺寸、标签。
    # image, label, pixels = features['image_raw'], features['label'], features['pixels']
    image, label = features['image_raw'], features['label']

    # 从原始图像数据解析出像素矩阵，并根据图像尺寸还原图像
    decode_image = tf.decode_raw(image, tf.uint8)
    decode_image = tf.reshape(decode_image, [28, 28, 1])

    # 将图像和标签数据通过tf.train.shuffle_batch整理成神经网络训练时需要的batch
    min_after_dequeue = 10000
    # min_after_dequeue = 3000
    num_threads = 8
    capacity = min_after_dequeue + (num_threads + 3 * batch_size)
    image_batch, label_batch = tf.train.shuffle_batch([decode_image, label], batch_size=batch_size,
                                                      num_threads=num_threads,
                                                      capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue,
                                                      )

    image_batch = tf.cast(image_batch, tf.float32)
    # 返回batch数据
    return image_batch, label_batch


def next_batch(batch_size):
    if not file_list:
        for i in range(FONT_NUM):
            font_dir = FONT_LIST[i]
            for single_img in os.listdir(cwd + font_dir):
                file_list.append(cwd + font_dir + "/" + single_img)
            for j in range(NUM_LABELS):
                label_list.append(j)
