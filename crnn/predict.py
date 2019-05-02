# -*- utf-8 -*-
"""
    @describe: text recognition with images path or images ndarray list
    @author: xushen
    @date: 2018-12-25
"""
import time

import tensorflow as tf
import numpy as np
from crnn.modules import CRNN
from crnn import config as config


def predict(batch_images, batch_labels, models_path):
    """
    predict images
    :param batch_images: images numpy array
    :param batch_labels: labels array
    :param models_path:
    :return:
    """

    global result_set
    crnn_graph = tf.Graph()
    with crnn_graph.as_default():
        crnn = CRNN(image_shape=config.image_shape,
                    lstm_hidden=config.lstm_hidden,
                    pool_size=config.pool_size,
                    learning_decay_rate=config.learning_decay_rate,
                    learning_rate=config.learning_rate,
                    learning_decay_steps=config.learning_decay_steps,
                    is_training=False,
                    )

    sess = tf.Session(graph=crnn_graph)
    with sess.as_default():
        with crnn_graph.as_default():
            tf.global_variables_initializer().run()
            crnn_saver = tf.train.Saver(tf.global_variables())
            crnn_ckpt = tf.train.get_checkpoint_state(models_path)
            crnn_saver.restore(sess, crnn_ckpt.model_checkpoint_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        batch_len = len(batch_images)
        # batch_image_widths = np.pad([], int(batch_len / 2), "constant",
        #                             constant_values=config.image_shape[0])
        predict_label_list = list()
        images = np.split(batch_images, batch_len)
        for i in range(batch_len):
            # if i + batch_len >= batch_len:
            #     batch_size = batch_len - i
            rs = sess.run(crnn.dense_predicts,
                          feed_dict={
                              crnn.images: images[i],
                              crnn.image_widths: [config.image_shape[0]]}
                          )
            predict_label_list.append(rs)

        counter = 0
        result_set = []
        for j, predict_label in enumerate(predict_label_list):
            predict_index = predict_label[0]
            correct_index = batch_labels[j]
            if predict_index == correct_index:
                counter += 1
            predict_text = "" if len(predict_index) == 0 else crnn.charsets[predict_index[0]]
            result_set.append(predict_text)
            print("truth: [" + crnn.charsets[correct_index] + "] predict: [" + predict_text + "]")
        print("Accuracy is " + str(counter / float(batch_len)))
    except tf.errors.OutOfRangeError:
        print("Insufficient batches ")
    finally:
        coord.request_stop()
        print("End at " + time.asctime(time.localtime(time.time())))
    coord.join(threads)
    return result_set
