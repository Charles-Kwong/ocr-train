# import math
import copy
import sys

import cv2
import numpy as np
# import os
import time
# import matplotlib.pylab as plt
# from scipy import signal
from skimage.measure import label, regionprops

# from preprocess import otsu


# def binary(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # OTSU 大津法
#     ret, img_binary = cv2.threshold(gray, otsu.thresh(gray), 255, cv2.THRESH_BINARY)
#     return img_binary


def resize(img, width=0, height=0):
    global w
    global h
    oh, ow = img.shape[:2]
    if width == 0 and height == 0:
        print("err")
        return None
    if width != 0 and height != 0:
        img = cv2.resize(img, (width, height), cv2.INTER_CUBIC)
        return img
    elif width != 0:
        w = width
        h = int(w * oh / ow)
    elif height != 0:
        h = height
        w = int(h * ow / oh)
    img = cv2.resize(img, (w, h), cv2.INTER_CUBIC)
    return img


# def do_split_binary(img, times):
#     global split_times
#     split_times = times
#
#     arr = np.array(img)
#     img_arrays_y = np.split(arr, split_times, axis=1)
#
#     global res_y, res_x, res
#     res_y = np.empty([0, 0])
#     res_x = np.empty([0, 0])
#
#     for i in range(split_times):
#         img_arrays_x = np.split(img_arrays_y[i], split_times, axis=0)
#         for j in range(split_times):
#             img_tmp_x = binary(img_arrays_x[j])
#             if j > 0:
#                 res_x = np.concatenate((res_x, img_tmp_x), axis=0)
#             else:
#                 res_x = img_tmp_x
#         res_y = copy.deepcopy(res_x)
#         res_x = np.empty([0, 0])
#         if i > 0:
#             res = np.concatenate((res, res_y), axis=1)
#         else:
#             res = res_y
#     return res


def normalize(img, size=28):
    temp_img = np.ones([size, size], np.uint8) * 255
    height, width = img.shape[:2]
    left = -1
    right = -1
    for j in range(width):
        for i in range(height):
            if left == -1 and img[i, j] == 0:
                left = j
                break
            if right == -1 and img[i, width - j - 1] == 0:
                right = width - j - 1
                break

    image = np.ones([height, right - left + 1], np.uint8) * 255

    try:
        for i in range(height):
            for j in range(image.shape[1]):
                image[i][j] = img[i][j + left]
    except IndexError:
        cv2.imshow(None, image)
        cv2.waitKey(0)

    if width / height < 0.5:
        return None
    image = resize(image, height=size)
    height, width = image.shape[:2]
    xc = int((size - width) / 2)
    if size - width < 0:
        return None
    try:
        for i in range(height):
            for j in range(xc, w + xc):
                temp_img[i, j] = image[i, j - xc]
    except IndexError:
        print(sys.exc_info())
        cv2.imshow(None, image)
        cv2.waitKey(0)
    return temp_img


def get_connected_domain(img):
    img = cv2.bitwise_not(img, img)

    # 四连通域
    label_img = label(img, connectivity=1)
    props = regionprops(label_img)
    # dst = color.label2rgb(label_img)

    return props


def draw_sliced_rect(img, props, min_area=200):
    for prop in props:
        if prop.bbox_area < min_area:
            continue
        y, x, y1, x1 = prop.bbox
        x = int(x)
        y = int(y)
        x1 = int(x1)
        y1 = int(y1)
        cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)
    return
