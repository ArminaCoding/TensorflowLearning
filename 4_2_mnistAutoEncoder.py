import numpy as np
import sklearn.preprocessing as prep
# 数据预处理的模块，还有使用数据标准化的功能
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 使用的是一种参数初始化方法xavier initialization。Xavier初始化器
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input                   # 输入变量数
        self.n_hidden = n_hidden                 # 隐含层节点数
        self.transfer = transfer_function        # 隐含层的激活函数
        self.scale = tf.placeholder(tf.float32)  # 优化器，默认为Adam
        self.training_scale = scale              # 高斯噪声技术
        network_weights = self._initialize_weights()
        self.weights = network_weights
        # 定义网络结构，建立n_input维度的placeholder然后建立隐含层，将输入的x加上噪声
        # 之后x*w1+b1，用transfer对结果进行激活函数处理
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(
            self.x + scale * tf.random_normal((n_input,)),
            self.weights['w1']),self.weights['b1']))
        # 在输出层进行数据复原、重建操作（即reconstruction），只要输出self.hidden*w2+b2
        self.reconstruction = tf.add(tf.matmul(self.hidden,
                                               self.weights['w2']),self.weights['b2'])
        # 定义自编码的损失函数
        # 计算平方误差和优化损失cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(
            self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
        # 初始化模型
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    # 创建初始化函数
    def _initialize_weights(self):
        # 把w1,b1,w2,b2存入all_weights,w1要用到xavier初始化，后三个变量使用tf.zeros置0
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,
                                                    self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],
                                                 dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,
                                                  self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],
                                                 dtype=tf.float32))
        return all_weights

    # 计算损失cost以及执行一步训练的函数partial_fit
    # feed_dict输入了数据x和噪声系数sacle
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict={self.x: X, self.scale:self.training_scale})
        return cost

    # 只求损失cost的函数，评测性能会用到
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X,
                                                   self.scale: self.training_scale})

    # 提供一个接口获取抽象后的特征
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X,
                                                     self.scale: self.training_scale
                                                     })

    # 将高阶特征复原为原始数据
    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction,
                             feed_dict={self.hidden: hidden})

    # 整体运行一遍复原过程,包括提取高阶特征和通过高阶特征复原数据
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X,
                                                             self.scale: self.training_scale
                                                             })

    # 获取隐含层权重w1
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    # 获取隐含层偏置系数b1
    def getBiases(self):
        return self.sess.run(self.weights['b1'])

# 使用定义好的AGN自编码
mnist = input_data.read_data_sets('MNIST-data', one_hot=True)


# 标准化处理函数，把数据变为均值0，标准差1的分布
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


# 定义一个随机block数据的函数：取一个0~（len(data) - batch_size）之间的随机整数
def get_random_block_from_data(data, batch_size_1):
    start_index = np.random.randint(0, len(data) - batch_size_1)
    return data[start_index:(start_index + batch_size_1)]

X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
n_samples = int(mnist.train.num_examples)
training_epochs = 40
batch_size = 128
display_step = 1

# 创建一个AGN实例
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784,
                                               n_hidden=200,
                                               transfer_function=tf.nn.softplus,
                                               optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                               scale=0.01)

# 开始训练过程
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples/batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost/n_samples*batch_size
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))
