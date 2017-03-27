# CIAR-10 数据集，包含60000张32x32的彩色图像，训练集50000张，测试集10000张。标注为10类
# 分别是airplane、automobile、bird、cat、deer、dog、frog、horse、ship和trunk
#
# #
# 载入一些常用库，比如Numpy和time，然后是自动读取和下载的Model
import cifar10_input
import cifar10
import tensorflow as tf
import numpy as np
import time

# 定义batch_size，训练轮数max_steps，以及下载默认路径
max_steps = 3000
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'


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
# 输入特征和label,由于定义网络结构需要使用，因此不能None，图片裁剪后24x24，颜色通道3
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

# 第一个卷积层conv1
# 1.variable_with_weight_loss函数创建卷积核参数并初始化
# 2.使用5x5的卷积核大小，3通道，64个卷积核,设置weight标准差为0.05
# 3.第一个卷积层不做L2正则因此wl=0，步长为1=[1,1,1,1]。这层的bias=0，再把卷积的结果加上bias
# 4.最后使用一个ReLU激活函数进行非线性化
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2,
                                    wl=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
# 5.使用一个3x3步长2x2的最大池化层，这里的尺寸和步长不一致，可以增加数据的丰富性
# 6.使用tf.nn.lrn函数面对结果进行处理。（侧抑制）
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1],
                       padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# 第二个卷积层
# 和第一层的区别是上一层输出64，变成本层输入64；这的bias初始化为0.1，最后调换了最大池化层和LRN层顺序
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2,
                                    wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='SAME')

# 接下来是全连接层，
# 1.先把前两个卷积层的输出结果flatten，使用reshape变成一维向量，再使用get.shape函数获取扁平化后的长度
# 2.接着使用variable_with_weight_loss对全连接层的weight初始化，隐含节点384，正态分布的标准差0.04
# 3.bias的值初始化为0.1
# 4.我们希望的是全连接层不要过拟合，因此设置weight loss为0.04，让这一层的所有参数都被L2正则约束
# 5.最后依然用ReLU激活函数进行非线性化
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# 和前一层很像，不过隐含节点下降了一半，只有192个，其他的超参数保持不变。
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# 最后一层
# 1.创建这一层的weight，正态分布标准设为上一个隐含层的节点数的倒数，不计入L2正则
# 2.不使用softmax计算结果，是因为把它放在了计算loss的部分。
# 3.不需要对Inference的输出进行softmax就可以获得最终分类结果，计算softmax主要是计算loss值，因此放在后面
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)
# 完成了整个网络inference的部分


# 计算CNN的loss，用的cross_entropy,tf.nn.sparse_softmax_cross_entropy_with_logits函数把softmax和
# cross entropy loss的计算合在了一起。
# cross_entropy_mean对cross entropy计算均值。
# 再用tf.add_to_collection添加loss到整体losses的collection中。
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                   labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                        name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')
# 将loss，和label_holder传入loss函数得到最终的loss
loss = loss(logits, label_holder)
# 优化器Adam，学习率0.003
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
# 使用 tf.nn.in_top_k函数求输出结果中top k的准确率，默认使用top 1，也就是输出分数最高的那一类的准确率
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)
# 使用tf.InteractiveSession创建默认的session，接着初始化全部模型参数
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 启动图片数据增强的线程队列，这里用了16个线程来进行加速。若不启动线程，后面的inference及训练无法开始。
tf.train.start_queue_runners()
# 开始训练，现在多了个time函数，用来展示当前loss、每秒钟能训练的样本数量，以及训练一个batch数据所花费的时间。
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss],
                             feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time
    if step % 10 == 0:
        example_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        format_str = ('step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, example_per_sec, sec_per_batch))

# 评测模型 ，测试集10000个样本，执行top_k_op计算模型，预测在top 1上预测正确的样本数
# 最后汇总所有结果计算预测正确的数量
num_examples = 10000
import math
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch,
                                                  label_holder: label_batch})
    true_count += np.sum(predictions)
    step += 1

prediction = true_count / total_sample_count
print('precision @ 1 = %.3f' % prediction)


