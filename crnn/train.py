import tensorflow as tf
from crnn.modules import CRNN
from crnn import config as crnn_config

from util.tf_utils import NUM_LABELS, INNER_SIZE, FONT_NUM
from tfrecords_gen import output, cwd

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 模型保存的路径和文件名。
MODEL_PATH = cwd + "model/" + str(NUM_LABELS) + str(INNER_SIZE * FONT_NUM)
MODEL_NAME = "model_" + str(NUM_LABELS) + str(INNER_SIZE * FONT_NUM) + ".ckpt"
LOGS_PATH = cwd + "logs"


def main(_):
    crnn = CRNN(image_shape=crnn_config.image_shape,
                lstm_hidden=crnn_config.lstm_hidden,
                pool_size=crnn_config.pool_size,
                learning_decay_rate=crnn_config.learning_decay_rate,
                learning_rate=crnn_config.learning_rate,
                learning_decay_steps=crnn_config.learning_decay_steps,
                is_training=True,
                )
    crnn.train(epoch=27000,
               batch_size=128,
               restore=True,
               batch_path=cwd + output + ".tfrecords",
               log_path=LOGS_PATH,
               model_path=MODEL_PATH,
               model_name=MODEL_NAME
               )

    if __name__ == '__main__':
        tf.app.run()
