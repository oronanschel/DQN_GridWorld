import tensorflow as tf
import numpy as np
from ops import linear



class Tabular(object):
    def __init__(self, bins):
        self.bins = bins
        self.V = np.random.randn(bins,bins)

    def learn(self, s, target):
        for i in range(len(s)):
            idx = np.unravel_index(s[i].argmax(),s[0].shape)
            ###### alpha = 0.5 ######
            self.V[idx] = 0.5*self.V[idx] + 0.5*target[i]

    def predict(self, s):
        pred = np.zeros(len(s))
        for i in range(len(s)):
            idx = np.unravel_index(s[i].argmax(),s[0].shape)
            pred[i] = self.V[idx]

        return pred


class NN_1_hot(object):
    def __init__(self, sess,bins,sgd_batch_size,hid_size):
        self.sess = sess
        self.mini_batch_size = sgd_batch_size
        self.bins = bins
        self.hid_size=hid_size


        self.build_net()

    def learn(self, s, target, iternum):

        target = target.reshape(target.shape[0],1)

        for itr in range(iternum):
            idx = np.random.randint(low=0,high=target.shape[0],size=self.mini_batch_size)
            mini_batch = target[idx,...]
            s_t = s[idx,...]
            _, loss = self.sess.run([self.optim, self.loss], {self.s_t: s_t, self.target: mini_batch})
            if itr % 50 == 0:
                    print 'itr:' + str(itr) + ' loss:' + str(loss)
        print 'itr:' + str(itr) + ' loss:' + str(loss)


    def predict(self, s):
        pred = self.sess.run([self.pred], {self.s_t: s})[0]
        return pred


    def build_net(self):
        self.w = {}

        hidden_size = self.hid_size
        self.s_t = tf.placeholder('float32', [None, self.bins, self.bins], name='s_t')
        self.target = tf.placeholder('float32', [None, 1], name='target')
        shape = self.s_t.get_shape().as_list()

        with tf.variable_scope('prediction'):
            self.s_t_flat = tf.reshape(self.s_t, [-1, reduce(lambda x, y: x * y, shape[1:])])
            hid, self.w['l0_w'], self.w['l0_b'] = linear(self.s_t_flat, hidden_size, activation_fn=tf.nn.relu, name='l0')
            self.pred ,self.w['l1_w'], self.w['l1_b'] = linear(hid, 1, activation_fn=None, name='l1')


        with tf.variable_scope('optimizer'):

            self.delta = tf.reshape(self.pred, shape=[self.mini_batch_size]) - tf.reshape(self.target,shape=[self.mini_batch_size])
            self.loss = tf.reduce_mean(tf.square(self.delta))
            self.optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
