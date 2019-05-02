import multiprocessing
import os
import pickle
import time
import tensorflow as tf
from PIL import Image
from typing import List

from util.tf_utils import FONT_LIST
from util.tf_utils import FONT_NUM
from util.tf_utils import INNER_SIZE
from util.tf_utils import PIC_SIZE
from util.tf_utils import NUM_LABELS
from util.tf_utils import cwd

output = "train_" + str(NUM_LABELS) + "_" + str(INNER_SIZE * FONT_NUM)


# 生成字符型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def gen(cls, start_index, end_index, file_index):
    cls = cls[start_index: end_index]
    _arr = []
    # writer = tf.python_io.TFRecordWriter(cwd + output + ".tfrecords")
    for index, name in enumerate(cls):
        ls = FONT_LIST
        for _i in range(FONT_NUM):
            for j in range(INNER_SIZE):
                class_path = cwd + ls[_i] + '/'
                # 每个图片的路径
                img_path = class_path + str(index + start_index) + ".jpg"

                img = Image.open(img_path)
                img = img.resize((PIC_SIZE, PIC_SIZE))
                # 将图片转化为二进制格式
                img_raw = img.tobytes()
                # 获取图像尺寸
                # (img_W, img_H) = img.size
                # 图像通道数
                # channels = 1
                # 将一个样例转化成Example Protocol Buffer，并将所有的信息写入这个数据结构
                example = tf.train.Example(features=tf.train.Features(feature={
                    # 'pixels': _int64_feature(img.size[1]),
                    # 'img_W': _int64_feature(img_W),
                    # 'img_H': _int64_feature(img_H),
                    # 'channels': _int64_feature(channels),
                    'label': _int64_feature(index + start_index),
                    'image_raw': _bytes_feature(img_raw)}))

                # 序列化为字符串
                _arr.append(example.SerializeToString())
                # writer.write(example.SerializeToString())

    # 保存临时文件
    file = open(cwd + "tmp_" + str(file_index), "wb")
    pickle.dump(_arr, file)
    file.close()


class GenProcess(multiprocessing.Process):
    def __init__(self, cls, start_index, end_index, process_id):
        multiprocessing.Process.__init__(self)
        self.cls = cls
        self.start_index = start_index
        self.end_index = end_index
        self.process_id = process_id

    def run(self):
        gen(self.cls, self.start_index, self.end_index, self.process_id)


if __name__ == "__main__":

    # 读取类别列表
    f = open("training_data/" + str(NUM_LABELS) + ".txt", "r", encoding="utf-8")
    tmp_str = f.readline()
    f.close()

    # 定义类别
    classes: List[str] = []
    for s in tmp_str:
        classes.append(s)

    s1 = time.time()

    indexes = []
    length = len(classes)
    for i in range(4):
        t = int(length / 4)
        indexes.append(int(i * t))

    process_1 = GenProcess(classes, indexes[0], indexes[1], 1)
    process_2 = GenProcess(classes, indexes[1], indexes[2], 2)
    process_3 = GenProcess(classes, indexes[2], indexes[3], 3)
    process_4 = GenProcess(classes, indexes[3], length, 4)

    print("Processes started ...")
    print("Current working directory is " + cwd)

    process_1.start()
    process_2.start()
    process_3.start()
    process_4.start()

    process_1.join()
    process_2.join()
    process_3.join()
    process_4.join()

    print("Processes exited ...")
    print("Start merging files ...")

    # 输出tfrecord文件
    writer = tf.python_io.TFRecordWriter(cwd + output + ".tfrecords")
    for i in range(4):
        path = cwd + "tmp_" + str(i + 1)
        f = open(path, "rb")
        arr_s = pickle.load(f)
        for s in arr_s:
            writer.write(s)
        f.close()
        os.remove(path)

    # gen(classes, indexes[2], indexes[3], 0)

    print("Progress took %.2f second(s)" % (time.time() - s1))

    writer.close()
