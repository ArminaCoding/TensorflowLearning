# 首先载入MNIST数据集，并创建默认的Interactive Session。
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST-data", one_hot=True)
sess = tf.InteractiveSession()


# 定义初始化函数。给权重制造一些随机的噪声来打破完全对称，比如截断的正态分布噪声
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 同时因为使用ReLU，也给偏置增加一些小的正值（0.1）用来避免死亡节点（dead neurons）
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# tf.nn.conv2d是Tensorflow中的2维卷积函数，x是输入，W是卷积的参数
# 函数参数[5,5,1,32],输入5x5的卷积核尺寸，channal=1灰度（3彩色），卷积核数量32（特征数）
# Strides：卷积模板移动步长，都是1，代表不会遗漏图片每一点；Padding：边界的处理方式。
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# tf.nn.max_pool（2x2）最大池化函数。为了缩小尺寸，strides设横竖两个方向以2为步长
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')

# x是特征，y_是真实label。将1D转2D，1x784->28x28。tf.reshape变形函数
# 故[-1,28,28,1],-1代表样本数量不固定，1代表颜色通道数量
x = tf.placeholder(tf.float32, [None, 784], name="input")  # 输入节点
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 参数weights和bias初始化。[5,5,1,32]卷积核5x5,1个颜色通道，32个不同卷积核
# conv2D进行卷积操作，加上偏置，接着用ReLU进行非线性处理，用最大池化函数对结果池化
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二个卷积层，卷积核的数量变成了64
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 因为经历两个max_pool,边长变成1/4，图片28x28->7x7。尺寸为7x7x64
# 用tf.reshape函数对第2层输出tensor进行变形，转成1D的向量
# 然后连接一个全连接层，隐含节点为1024，并使用ReLU激活函数
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 为了减轻过拟合，使用一个Dropout层，随机丢弃一部分节点数据
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 将Dropout层的输出连接一个Softmax层，得到最后的概率输出
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name="output")  # 输出节点

# 定义损失函数为cross entropy，优化器用Adam，较小的学习率1e-4
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
                                              reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 定义评测准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 训练过程
tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 尝试保存模型，output_node_names=指定输出节点名称
output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['output'])
with tf.gfile.FastGFile('model/mnist.pb', mode='wb') as f:
    f.write(output_graph_def.SerializeToString())

# 由于一次性载入测试集，导致内存溢出OOM [10000,32,28,28], 所以分批测试
for i in range(10):
    batch_ = mnist.test.next_batch(1000)
    print("step %d, test accuracy %g" % (i, accuracy.eval(feed_dict={
        x: batch_[0], y_: batch_[1], keep_prob: 1.0})))

