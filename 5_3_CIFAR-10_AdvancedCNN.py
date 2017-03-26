# CIAR-10 数据集，包含60000张32x32的彩色图像，训练集50000张，测试集10000张。标注为10类
# 分别是airplane、automobile、bird、cat、deer、dog、frog、horse、ship和trunk
#
# #
# 载入一些常用库，比如Numpy和time，然后是自动读取和下载的Model
from cifar10 import cifar10_input
from cifar10 import cifar10
import tensorflow as tf
import numpy as np
import time

# 定义batch_size，训练轮数max_steps，以及下载默认路径
max_steps = 3000
batch_size = 128
data_dir = 'cifar10_data'

# 1.依然用tf.truncated_normal截断正态分布来初始化权重。这里给weight加一个L2的loss
# 2.使用wl控制L2 loss的大小，使用tf.nn.l2_loss函数计算weight的L2 weight，再使用tf.multiply
# 让L2 loss乘以wl，得到最后的weight loss。
# 3.接着使用tf.add_to_collection把weight loss同意存到一个collection，这个collection名为
# ‘loss’，在计算总体loss 的时候用上
def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name="weight_loss")
        tf.add_to_collection('losses', weight_loss)
    return var

# 下载数据集并解压
cifar10.maybe_download_and_extract()

# cifar10_input.distorted_inputs产生训练需要使用的数据，包括对应的label，返回的是封装好的tensor
images_train, labels_train = cifar10_input.distorted_inputs(
                        data_dir=data_dir, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                data_dir=data_dir,
                                                batch_size=batch_size)
#
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

#
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2,
                                    wl=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1],
                       padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

#
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2,
                                    wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='SAME')

#
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

#
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3,weight4) + bias4)

#
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[0]))
logits5 = tf.add(tf.matmul(local4, weight5), bias5)
# 完成了整个网络inference的


