import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# one_hot是10维的向量
mnist = input_data.read_data_sets("MNIST-data/", one_hot=True)
# 查看一下数据集的情况
# print(mnist.train.image.shape, mnist.train.labels.shape)
# print(mnist.test.image.shape, mnist.test.labels.shape)
# print(mnist.validation.image.shape, mnist.validation.labels.shape)
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
# 定义cross-entropy
y_ = tf.placeholder(tf.float32, [None,10])
# 用一个placeholder 输入真实的label，y_ * tf.log(y)就是计算yi*log(yi)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# 然后定义一个随机梯度下降算法SGD(Stochastic Dradient Desent)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 全局参数初始化
tf.global_variables_initializer().run()

for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_:batch_ys})
    if i % 1000 == 0:
        print(i)
# 对模型的准确率进行验证tf.argmax(y, 1)预测的最大值，和样本的真实数字比较
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 用tf.cast将correct_prediction输出的bool值转换为float32
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))



