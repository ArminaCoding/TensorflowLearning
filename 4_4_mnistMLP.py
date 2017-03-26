# 多层感知器神经网络（Multi-layer perceptron neural networks，MLP neural netwoks）
# 创建一个Tensorflow默认的InteractiveSession，这样后面执行无须指定Session
#
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST-data", one_hot=True)
sess = tf.InteractiveSession()

# ### step1：定义算法公式 ####
in_units = 784  # 输入节点数
h1_units = 300  # 隐含层的输出节点数
# W1和b1是隐含层的权重和偏置，将偏置全部赋值为0，设置为截断的正态分布，标准差stddev为0.1
# 可以通过tf.truncated_normal实现
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
# 因为模型使用的是ReLU几乎函数，所以需要用正态分布加一点噪声打破完全对称，和避免0梯度
# 其他模型可能还需要给偏置赋上一点小的零值来避免dead neuron,输出层Softmax，W2和b2初始化为0
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

# x的输入Dropout的比率keep_prob是不一样的，通常在训练时小于1，在预测时等于1，
x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)

# 首先一个ReLU的隐含层，调用Dropout。keep_prob为保留数据的比例
# 预测时应该等于1，用全部特征来预测样本的类别
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

# ### step2：定义loss，选定优化器 ####
# 交叉信息熵，AdagradOptimizer 和 学习率0.3
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                              reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# ### step3：训练 #####
# 输入数据集，设置keep_prob为0.75
tf.global_variables_initializer().run()
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(1000)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})
    if i % 100 == 0:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Epoch:", '%02d' % i,  "accuracy=", "{:.5f}".format(accuracy.eval(
                {x: batch_xs, y_: batch_ys, keep_prob: 1.0})))

# ### step4：在测试集或验证集上对准确率进行评测 ####
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels,
                     keep_prob: 1.0}))

