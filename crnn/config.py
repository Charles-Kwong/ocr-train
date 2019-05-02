from util.tf_utils import PIC_SIZE

# 图像通道
CHANNELS = 1

# data
image_shape = [PIC_SIZE, PIC_SIZE, CHANNELS]  # 图像尺寸

# data icpr
train_test_ratio = 0.9  # 训练测试集的比例
is_transform = True  # 是否进行仿射变换
angle_range = [-15.0, 15.0]  # 不进行仿射变换的倾斜角度范围
epsilon = 1e-4  # 原始图像的顺时针变换参数
filter_ratio = 1.3  # 图片过滤的高宽比例，高于该比例的图片将被过滤
filter_height = 16  # 高度过滤，切图后的图像高度低于该值的将被过滤掉，[int]

# model
lstm_hidden = 256

# train
pool_size = 2 * 2  # pool层总共对图像宽度的缩小倍数
learning_rate = 1e-3  # 学习率
learning_decay_steps = 3000  # 学习率每多少次递减一次
learning_decay_rate = 0.95  # 学习率每次递减时，变为原来的多少
