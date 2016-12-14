import tensorflow as tf
import numpy as np
from ops import linear



class Tabular(object):
    def __init__(self, bins):

        self.bins = bins
        self.V = np.random.randn(bins,bins)


    def learn(self, s, target, N):
        for i in range(len(s)):
            idx = np.unravel_index(s[i].argmax(),s[0].shape)
            self.V[idx] = 0.5*self.V[idx] + 0.5*target[i]

        return 1

    def predict(self, s,testing,alpha):
        pred = np.zeros(len(s))
        for i in range(len(s)):
            idx = np.unravel_index(s[i].argmax(),s[0].shape)
            pred[i] = self.V[idx]

        return pred




class NN_1_hot(object):
    def __init__(self, sess, batch_size,num_of_prime,bins,in_train,in_test,sgd_batch_size,hid_size,lr,random_theta):
        self.sess = sess
        self.batch_size = batch_size
        self.mini_batch_size =sgd_batch_size
        self.num_of_prime = num_of_prime
        self.bins = bins
        self.in_train = in_train
        self.in_test = in_test
        self.hid_size=hid_size
        self.lr=lr
        self.t =0
        self.random_theta = random_theta
        self.g_noise = None

        self.build_net()

    def learn(self, s, target, N):

        target = target.reshape(target.shape[0],1)
        print_rnd = np.random.uniform()
        # print_rnd = 0
        if self.random_theta:
        # Assaing random values for learning network
            for name in self.w.keys():
             w = self.sess.run([tf.random_normal(shape=self.w[name].get_shape().as_list(),stddev=0.02)])
             self.sess.run([self.w_assign_op[name]], {self.t_w_input[name]: w[0]})
        if print_rnd<0.05:
          print 'loss'
        for itr in range(N):
            idx = np.random.randint(low=0,high=target.shape[0],size=self.mini_batch_size)
            mini_batch = target[idx,...]
            s_t = s[idx,...]
            comp_lr = self.lr*(N)/(N+3*itr)
            _, loss = self.sess.run([self.optim, self.loss], {self.s_t: s_t, self.target: mini_batch,self.learning_rate:comp_lr})

            if itr % 400 == 0:
                if print_rnd<0.05:
                    print 'itr:' + str(itr) + ' loss:' + str(loss)
        if print_rnd<0.05:
            print 'itr:' + str(itr) + ' loss:' + str(loss)

        self.update_target_q_network()
        return

    def predict(self, s,testing,alpha):
        # pred = self.sess.run([self.pred], {self.s_t: s})[0]
        # sum = 1
        # for k in range(1,self.num_of_prime):
        #     pred += self.sess.run([self.target_list[k-1]], {self.s_t: s})[0] *(1-alpha)**(k)
        #     sum  += (1-alpha)**(k)
        # pred = pred / (sum)
        # #

        pred = self.sess.run([self.pred], {self.s_t: s})[0] * alpha
        sum = alpha
        for k in range(1, self.num_of_prime):
            pred += self.sess.run([self.target_list[k-1]], {self.s_t: s})[0] * (1 - alpha)
            sum+=(1-alpha)
        pred = pred/sum



        return pred


    def build_net(self):
        self.w = {}
        self.t_w = []
        self.target_list =[[] for k in range(self.num_of_prime)]

        hidden_size = self.hid_size
        self.s_t = tf.placeholder('float32', [None, self.bins, self.bins], name='s_t')
        self.target = tf.placeholder('float32', [None, 1], name='target')
        shape = self.s_t.get_shape().as_list()

        with tf.variable_scope('prediction'):
            self.s_t_flat = tf.reshape(self.s_t, [-1, reduce(lambda x, y: x * y, shape[1:])])
            hid, self.w['l0_w'], self.w['l0_b'] = linear(self.s_t_flat, hidden_size, activation_fn=tf.nn.relu, name='l0')
            self.pred ,self.w['l1_w'], self.w['l1_b'] = linear(hid, 1, activation_fn=None, name='l1')

        for k in range(self.num_of_prime):
            self.t_w.append({})
            with tf.variable_scope('target'+str(k)):
                hid, self.t_w[k]['l0_w'], self.t_w[k]['l0_b'] = linear(self.s_t_flat, hidden_size, activation_fn=tf.nn.relu,
                                                             name='l0')
                self.target_list[k], self.t_w[k]['l1_w'], self.t_w[k]['l1_b'] = linear(hid, 1, activation_fn=None, name='l1')

        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}


            self.w_assign_op = {}

            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder('float32', self.t_w[0][name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[0][name].assign(self.t_w_input[name])
                self.w_assign_op[name] = self.w[name].assign(self.t_w_input[name])

            self.t_w_input_primes = []
            self.t_w_assign_op_primes = []
            for k in range(0,self.num_of_prime-1):
                self.t_w_assign_op_primes.append({})
                for name in self.w.keys():
                    self.t_w_assign_op_primes[k][name] = self.t_w[k+1][name].assign(self.t_w_input[name])



        with tf.variable_scope('optimizer'):

            self.delta = tf.reshape(self.pred, shape=[self.mini_batch_size]) - tf.reshape(self.target,shape=[self.mini_batch_size])
            self.loss = tf.reduce_mean(tf.square(self.delta))

            self.learning_rate = tf.placeholder('float32', name='learning_rate')

            # self.optim = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate ).minimize(self.loss)
            # self.optim = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10).minimize(self.loss)
            self.optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def update_target_q_network(self):

        for k in range(self.num_of_prime-2,-1,-1):
            for name in self.w.keys():
                w = self.sess.run(self.t_w[k][name])
                self.sess.run([self.t_w_assign_op_primes[k][name]],{self.t_w_input[name]: w})

        for name in self.w.keys():
            w = self.sess.run(self.w[name])
            self.sess.run([self.t_w_assign_op[name]], {self.t_w_input[name]: w})

    def random_init_w(self):
        # Assaing random values for learning network
            for name in self.w.keys():
             w = self.sess.run([tf.random_normal(shape=self.w[name].get_shape().as_list(),stddev=0.02)])
             self.sess.run([self.w_assign_op[name]], {self.t_w_input[name]: w[0]})


