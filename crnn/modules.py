import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.rnn import BasicLSTMCell
from util.tf_utils import get_charsets

from util import tf_utils

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.intra_op_parallelism_threads = 1
config.inter_op_parallelism_threads = 1


class CRNN(object):
    def __init__(self,
                 image_shape,
                 lstm_hidden,
                 pool_size,
                 learning_decay_rate,
                 learning_rate,
                 learning_decay_steps,
                 is_training):
        self.lstm_hidden = lstm_hidden
        self.pool_size = pool_size
        self.learning_decay_rate = learning_decay_rate
        self.learning_rate = learning_rate
        self.learning_decay_steps = learning_decay_steps
        self.is_training = is_training
        self.charsets = get_charsets()
        self.image_shape = image_shape
        self.images = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.image_shape[0], self.image_shape[1], self.image_shape[2]])
        self.image_widths = tf.placeholder(dtype=tf.int32, shape=[None])
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.seq_len_inputs = tf.divide(self.image_widths, self.pool_size, name='seq_len_input_op') - 1
        self.logprob = self.forward(self.is_training)
        self.global_step = tf.Variable(0, trainable=False)
        self.train_op, self.loss_ctc = self.create_train_op(self.logprob)
        self.dense_predicts = self.decode_predict(self.logprob)

    def vgg_net(self, inputs, is_training, scope='vgg'):
        batch_norm_params = {
            'is_training': is_training
        }
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
                with slim.arg_scope([slim.max_pool2d], padding='SAME'):
                    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                        net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
                        net = slim.max_pool2d(net, [2, 2], scope='pool1')
                        net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
                        net = slim.max_pool2d(net, [2, 2], scope='pool2')
                        net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
                        net = slim.max_pool2d(net, [2, 2], stride=[2, 1], scope='pool3')
                        net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
                        net = slim.max_pool2d(net, [2, 2], stride=[2, 1], scope='pool4')
                        net = slim.repeat(net, 1, slim.conv2d, 512, [3, 3], scope='conv5')
                        return net

    def forward(self, is_training):
        dropout_keep_prob = 0.7 if is_training else 1.0
        cnn_net = self.vgg_net(self.images, is_training)

        with tf.variable_scope('Reshaping_cnn'):
            shape = cnn_net.get_shape().as_list()  # [batch, height, width, features]
            transposed = tf.transpose(cnn_net, perm=[0, 2, 1, 3],
                                      name='transposed')  # [batch, width, height, features]
            conv_reshaped = tf.reshape(transposed, [-1, shape[2], shape[1] * shape[3]],
                                       name='reshaped')  # [batch, width, height x features]

        list_n_hidden = [self.lstm_hidden, self.lstm_hidden]

        with tf.name_scope('deep_bidirectional_lstm'):
            # Forward direction cells
            fw_cell_list = [BasicLSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]
            # Backward direction cells
            bw_cell_list = [BasicLSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]

            lstm_net, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_cell_list,
                                                                            bw_cell_list,
                                                                            conv_reshaped,
                                                                            dtype=tf.float32
                                                                            )
            # Dropout layer
            lstm_net = tf.nn.dropout(lstm_net, keep_prob=dropout_keep_prob)

        with tf.variable_scope('fully_connected'):
            shape = lstm_net.get_shape().as_list()  # [batch, width, 2*n_hidden]
            fc_out = slim.layers.linear(lstm_net, len(self.charsets) + 1)  # [batch x width, n_class]

            lstm_out = tf.reshape(fc_out, [-1, shape[1], len(self.charsets) + 1],
                                  name='lstm_out')  # [batch, width, n_classes]

            # Swap batch and time axis
            logprob = tf.transpose(lstm_out, [1, 0, 2], name='transpose_time_major')  # [width(time), batch, n_classes]

        return logprob

    def create_loss(self, logprob):
        sparse_code_target = self.dense_to_sparse(self.labels, blank_symbol=len(self.charsets) + 1)
        with tf.control_dependencies(
                [tf.less_equal(sparse_code_target.dense_shape[1],
                               tf.reduce_max(tf.cast(self.seq_len_inputs, tf.int64)))]):
            loss_ctc = tf.nn.ctc_loss(labels=sparse_code_target,
                                      inputs=logprob,
                                      sequence_length=tf.cast(self.seq_len_inputs, tf.int32),
                                      preprocess_collapse_repeated=False,
                                      ctc_merge_repeated=True,
                                      ignore_longer_outputs_than_inputs=True,
                                      # returns zero gradient in case it happens -> ema loss = NaN
                                      time_major=True)
            loss_ctc = tf.reduce_mean(loss_ctc)
        return loss_ctc

    def create_train_op(self, logprob):
        loss_ctc = self.create_loss(logprob)
        tf.losses.add_loss(loss_ctc)

        self.global_step = tf.train.get_or_create_global_step()

        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                   self.learning_decay_steps, self.learning_decay_rate,
                                                   staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        train_op = slim.learning.create_train_op(total_loss=tf.losses.get_total_loss(), optimizer=optimizer,
                                                 update_ops=update_ops)
        return train_op, loss_ctc

    def decode_predict(self, logprob):
        with tf.name_scope('decode_conversion'):
            sparse_code_pred, log_probability = tf.nn.ctc_greedy_decoder(logprob,
                                                                         sequence_length=tf.cast(
                                                                             self.seq_len_inputs,
                                                                             tf.int32
                                                                         ))
            sparse_code_pred = sparse_code_pred[0]
            dense_predicts = tf.sparse_to_dense(sparse_code_pred.indices,
                                                sparse_code_pred.dense_shape,
                                                sparse_code_pred.values, default_value=-1)

        return dense_predicts

    def dense_to_sparse(self, dense_tensor, blank_symbol):
        """
        将标签转化为稀疏表示
        :param dense_tensor: 原始的密集标签
        :param blank_symbol: padding的符号
        :return:
        """
        indices = tf.where(tf.not_equal(dense_tensor, blank_symbol))
        values = tf.gather_nd(dense_tensor, indices)
        sparse_target = tf.SparseTensor(indices, values, [-1, self.image_shape[1]])
        return sparse_target

    def train(self,
              epoch=100,
              batch_size=32,
              restore=False,
              model_path=None,
              model_name=None,
              log_path=None,
              batch_path=None
              ):

        # 创建目录
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        model_save_path = os.path.join(model_path, model_name)

        # summary for tensorboard
        tf.summary.scalar('loss_ctc', self.loss_ctc)
        merged = tf.summary.merge_all()

        xs, ys = tf_utils.get_batch(batch_path, batch_size)

        with tf.Session(config=config) as sess:
            writer = tf.summary.FileWriter(log_path, sess.graph)
            saver = tf.train.Saver(max_to_keep=2)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # 从checkpoint恢复
            last_epoch = 0
            init_steps = 0
            if restore:
                ckpt = tf.train.latest_checkpoint(model_path)
                if ckpt:
                    # last_epoch = int(ckpt.split('-')[1]) + 1
                    init_steps = int(ckpt.split('-')[1])
                    saver.restore(sess, ckpt)

            # 计算batch的数量
            # batch_nums = 1000
            # if self.mode == 1:
            #     batch_nums = 1000
            # # else:
            # #     train_img_list, train_label_list = get_img_label(train_label_path, train_images_path)
            # #     batch_nums = int(np.ceil(len(train_img_list) / batch_size))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            # batch_num = int(np.ceil(len(self.charsets) / batch_size))
            batch_num = 1
            try:
                print("Start at " + time.asctime(time.localtime(time.time())))
                s1 = time.time()
                global_step = 0
                samples = [0] * len(self.charsets)
                for i in range(last_epoch, epoch):
                    if coord.should_stop():
                        break
                    batch_images, batch_labels = sess.run([xs, ys])
                    batch_labels = batch_labels.reshape(batch_size, 1)
                    batch_image_widths = np.pad([], int(batch_size / 2), "constant",
                                                constant_values=self.image_shape[0])
                    for lbs in batch_labels:
                        samples[lbs[0]] += 1
                    # images = np.split(batch_images, batch_size)
                    # labels = np.split(batch_labels, batch_size)
                    # width = [self.image_shape[0]]
                    counter = 0
                    for sms in samples:
                        if sms != 0:
                            counter += 1
                    # print(samples)
                    for j in range(batch_num):
                        _, loss, predict_label, global_step = sess.run(
                            [self.train_op, self.loss_ctc, self.dense_predicts, self.global_step],
                            feed_dict={self.images: batch_images,
                                       self.image_widths: batch_image_widths,
                                       self.labels: batch_labels}
                        )
                        if global_step % 10 == 0:
                            remaining_steps = init_steps + epoch * batch_num - global_step - 1
                            s2 = int(time.time() - s1)
                            # 预计剩余时间(s)
                            est_rem_sec = int(remaining_steps / 10) * s2
                            est_sec = est_rem_sec % 60
                            est_min = int(est_rem_sec / 60)
                            est_hour = int(est_rem_sec / 3600)
                            if est_min >= 60:
                                est_min = est_min - est_hour * 60
                            s1 = time.time()
                            est = "Estimated remaining time is %d:%02d:%02d" % (est_hour, est_min, est_sec)
                            print('epoch:%d/%d, global:%d/%d loss:%.5f. %s ' % (
                                i, epoch,
                                global_step, batch_num * epoch + init_steps,
                                loss,
                                # ''.join(
                                #     [self.charsets[k] for k in batch_labels[j] if k != (len(self.charsets) + 1)]),
                                # ''.join([self.charsets[v] for v in predict_label[0] if v != -1]),
                                est
                            ))
                        if global_step % 10 == 0:
                            summary = sess.run(merged,
                                               feed_dict={
                                                   self.images: batch_images,
                                                   self.image_widths: batch_image_widths,
                                                   self.labels: batch_labels
                                               })
                            writer.add_summary(summary, global_step=global_step)
                        if global_step % 1000 == 0:
                            saver.save(sess, save_path=model_save_path, global_step=global_step)
                saver.save(sess, save_path=model_save_path, global_step=global_step)
            except tf.errors.OutOfRangeError:
                print("Insufficient batches ")
            finally:
                coord.request_stop()
                print("End at " + time.asctime(time.localtime(time.time())))
                coord.join(threads)
                writer.close()
            coord.join(threads)
