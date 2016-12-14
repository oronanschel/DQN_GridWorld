import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import model as approximator

import tensorflow as tf
import time
from matplotlib.backends.backend_pdf import PdfPages
import os
class Trainer(object):
    def __init__(self,in_test=True,in_train=True,samp_size=256,filename=None,num_of_prime=10,plot=False,
                 plot_interval=3,alpha=1,sgd_batch_size=32,hid_size=80,lr=0.01,random_theta=False):
      self.v_str_time_limit = 150
      self.bins = 40
      self.batch_size = samp_size
      self.num_of_prime = num_of_prime
      self.plot_interval = plot_interval
      self.gamma = 0.8
      self.screen = np.zeros(shape=[self.bins,self.bins])
      self.dist_l = []

      self.dist_l_p = [[] for _ in range(3)]
      self.itr_l = []
      self.filename= filename
      self.plot = plot
      self.alpha = alpha
      self.sgd_batch_size = sgd_batch_size
      self.save_itr =0


      self.sess = tf.Session()
      model = approximator.NN(self.sess,batch_size,num_of_prime)
      self.model = approximator.NN_1_hot(self.sess,self.batch_size,self.num_of_prime,self.bins,in_train=in_train,
                                        in_test=in_test,sgd_batch_size=self.sgd_batch_size,hid_size=hid_size,lr=lr,random_theta=random_theta)

      # self.model = approximator.Tabular(bins=self.bins)


      self.Vstr =self.computeVstr()


      init_op = tf.initialize_all_variables()
      self.sess.run(init_op)


      if filename:
          self.createResultFile()

      if plot:
          # self.fig = plt.figure()
          # self.ax0 = self.fig.add_subplot(331, projection='3d')
          # self.ax1 = self.fig.add_subplot(332)
          # self.ax2 = self.fig.add_subplot(333)
          # self.ax3 = self.fig.add_subplot(334)
          # self.ax4 = self.fig.add_subplot(335)

          self.f_tmp = plt.figure()
          self.ax_tmp = self.f_tmp.add_subplot(111, projection='3d')
          # plt.ion()

    def computeVstr(self):
        V_str = np.zeros(shape=[self.bins,self.bins])
        V_str_new = np.ones(shape=[self.bins,self.bins])

        start_time = time.time()
        iternum = 10000
        for itr in range(iternum):
            V_str = np.array(V_str_new)
            for idx in range(self.bins):
                for idy in range(self.bins):
                    V_list = [ V_str[min(idx + 1, self.bins - 1), idy]
                             , V_str[max(idx - 1, 0)       , idy]
                             , V_str[idx                   ,max(idy - 1, 0)]
                             , V_str[idx                   ,min(idy + 1, self.bins - 1)]]
                    V_max = np.max(V_list)

                    V_str_new[idx,idy] = self.gamma*V_max + self.computeR_idx(idx,idy)

            time_now = time.time()
            if  time_now - start_time > self.v_str_time_limit:
                print 'time limit reached'
                break
            if np.mean(np.square(V_str.reshape(-1)-V_str_new.reshape(-1)))<10**-12:
                break

        print 'Vstr eps='+str(np.mean(np.square(V_str.reshape(-1)-V_str_new.reshape(-1))))
        return V_str_new

    def computeR_idx(self,idx,idy):
        r = 0
        if idx==39 and idy==39:
            r = 1
        # if idx == 5 and idy == 10:
        #     r = 1
        return r

    def plotVstr(self):

        x = np.repeat(np.arange(self.bins),self.bins)
        y = np.arange(self.bins)
        for _ in range(1,self.bins):
            y = np.append(y,np.arange(self.bins))

        V_str= np.zeros(len(x))
        for i in range(len(x)):
            V_str[i] = self.Vstr[x[i],y[i]]
        # plot V_Str
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(x, y, V_str, cmap=cm.jet, linewidth=0.2)

        plt.title('$V^*$')
        plt.xlabel('x')
        plt.ylabel('y')

        pp = PdfPages("Vstr.pdf")
        pp.savefig(fig)
        pp.close()

        # plot problem example path
        fig = plt.figure()
        ax = fig.gca()

        x = [1,1,1]
        y = [2,3,4]
        ax.plot(x, y,color='b',marker=6)

        x = [1,2]
        y = [4,4]
        ax.plot(x, y,color='b')

        x = [2,3,4,5,6,7,8]
        y = [4,4,4,4,4,4,4]
        ax.plot(x, y,color='b', marker='>')

        x = [8,8]
        y = [4,5]
        ax.plot(x, y,color='b')

        x = [8,8,8,8,8,8]
        y = [5,6,7,8,9,10]
        ax.plot(x, y, color='b', marker=6)

        x = [8,7]
        y = [10,10]
        ax.plot(x, y,color='b')


        x = [7,6,5]
        y = [10,10,10]
        ax.plot(x, y,color='b', marker='<')

        ax.plot([5],[5], '*r',markersize=18)
        ax.text(5.2, 5.2, '$r=1$', fontsize=22)

        ax.set_xlim(0,10.5)
        ax.set_ylim(0,10.5)

        plt.xticks([0,2.5,5,7.5,10],[0,10,20,30,40])
        plt.yticks([0,2.5,5,7.5,10],[0,10,20,30,40])

        plt.title('Gridworld')
        plt.xlabel('x')
        plt.ylabel('y')

        pp = PdfPages("Gridworld.pdf")
        pp.savefig(fig)
        pp.close()


        return

    def sample_and_target(self,batch_size):
        s = np.zeros(shape=[batch_size,self.bins,self.bins])
        V_t = np.zeros(batch_size)

        # TODO: "confuse" with histoy of state
        for i in range(batch_size):
            idx = np.random.randint(self.bins)
            idy = np.random.randint(self.bins)
            s[i,idx,idy] = 1
            r = self.computeR_idx(idx,idy)
            s1 = np.zeros(shape=[4,self.bins,self.bins])
            s1[0,min(idx + 1, self.bins - 1), idy] = 1
            s1[1,max(idx - 1, 0), idy] = 1
            s1[2,idx,max(idy - 1, 0)] =1
            s1[3,idx,min(idy + 1, self.bins - 1)]=1
            alpha = self.computeAlpha()
            V_t[i] = r+self.gamma*np.max(self.model.predict(s1,testing=False,alpha=alpha))

        return s,V_t

    def full_ss_and_target(self):
        s = np.zeros(shape=[self.bins**2,self.bins,self.bins])
        V_t = np.zeros(self.bins**2)

        # TODO: "confuse" with histoy of state
        i=0
        for idx in range(self.bins):
            for idy in range(self.bins):
                s[i,idx,idy] = 1
                r = self.computeR_idx(idx,idy)
                s1 = np.zeros(shape=[4,self.bins,self.bins])
                s1[0,min(idx + 1, self.bins - 1), idy] = 1
                s1[1,max(idx - 1, 0), idy] = 1
                s1[2,idx,max(idy - 1, 0)] =1
                s1[3,idx,min(idy + 1, self.bins - 1)]=1
                alpha = self.computeAlpha()
                V_t[i] = r+self.gamma*np.max(self.model.predict(s1,testing=False,alpha=alpha))
                i+=1

        return s,V_t

    def createResultFile(self):

        self.res_file = os.path.join(os.getcwd(), self.filename+'.csv')
        res_file = open(self.res_file, 'wb')
        res_file.write('itr,dist\n')
        res_file.close()

    def computeDist(self):
        s  = np.zeros(shape=[self.bins*self.bins,self.bins,self.bins])
        itr = 0
        V_str_flat = np.zeros(shape=[self.bins*self.bins])
        for idx in range(self.bins):
            for idy in range(self.bins):
                s[itr,idx,idy]=1
                V_str_flat[itr]=self.Vstr[idx,idy]
                itr+=1

        V_pred = self.model.predict(s,testing=True,alpha=self.alpha)
        V_diff = V_pred.reshape(itr)-V_str_flat.reshape(itr)
        dist = np.mean(np.square(V_pred.reshape(itr)-V_str_flat.reshape(itr)))

        # self.dist_l_p[0].append(np.square(V_pred[5]-V_str_flat[5]))
        # self.dist_l_p[1].append(np.square(V_pred[44]-V_str_flat[44]))
        # self.dist_l_p[2].append(np.square(V_pred[102]-V_str_flat[102]))


        self.dist_l_p[0].append((V_pred[5]))
        self.dist_l_p[1].append((V_pred[44]))
        self.dist_l_p[2].append((V_pred[102]))

        self.dist_l.append(dist)
        self.itr_l.append(self.itr)
        if self.filename:
            res_file = open(self.res_file, 'a')
            res_file.write("%d,%.11f\n" % (self.itr,dist))
            res_file.close()

        x = np.repeat(np.arange(self.bins),self.bins)
        y = np.arange(self.bins)
        for _ in range(1,self.bins):
            y = np.append(y,np.arange(self.bins))

        if self.plot and self.itr % 5 == 0:
            self.ax_tmp.clear()
            self.ax_tmp.plot_trisurf(x, y, V_pred.reshape(len(x)), cmap=cm.jet, linewidth=0.2,vmin=0,vmax=5)
            # self.ax_tmp.axis(0,40,0,40,0,5)
            # self.f_tmp.set_size_inches(5, 5)
            self.f_tmp.savefig('q_learning'+str(self.save_itr), dpi=50)
            self.save_itr+=1

            # self.ax0.clear()
            # self.ax1.clear()
            #
            # self.ax2.clear()
            # self.ax3.clear()
            # self.ax4.clear()

            # self.ax0.plot_trisurf(x, y, V_pred.reshape(len(x)), cmap=cm.jet, linewidth=0.2)
            # self.ax0.plot_trisurf(x, y, V_diff.reshape(len(x)), cmap=cm.jet, linewidth=0.2)
            # self.ax1.semilogy(self.itr_l,self.dist_l)

            # self.ax2.semilogy(self.itr_l,self.dist_l_p[0])
            # self.

            # plt.axis((self.itr-300,self.itr, 10**-12, 10**-1))
            # self.ax3.semilogy(self.itr_l, self.dist_l_p[1])
            # plt.axis((self.itr - 300, self.itr, 10 ** -12, 10 ** -1))
            # self.ax4.semilogy(self.itr_l, self.dist_l_p[2])
            # plt.axis((self.itr - 300, self.itr, 10 ** -12, 10 ** -1))

            # plt.pause(0.05)

        return dist

    def Train(self,iternum,sgd_itrenum,full_state_update=True):

        for self.itr in range(iternum):
            if self.itr % self.plot_interval == 0:
                    self.computeDist()
            if self.itr-1%5==0:
                #plt.pause(0.01)
                print '------ itr:'+str(self.itr)+ '------'

            if full_state_update:
                s,V_t = self.full_ss_and_target()
            else:
                s, V_t = self.sample_and_target(self.batch_size)

            self.model.learn(s,V_t,sgd_itrenum)



    def computeAlpha(self):
        return self.alpha
        # return self.alpha*(10)/(10+self.itr)


iternum = 700
sgd_itrenum = 100
samp_size = 300
primes = 1
full_state_update = False
alpha = 0.001
# alpha = 1 # TODO: (decline in time)
batch_size = 32
hid_size = 80
random_theta = False

lr = 0.8

# name = 'results/Bfull_s_sgd_'+str(sgd_itrenum)+'_prime_'+str(primes)+'_alpha_'+str(alpha)+'_batch_size_'+str(batch_size)+'_hidsize_'+str(hid_size)


name = 'results/full_s_sgd_'+str(sgd_itrenum)+'_prime_'+str(primes)+'_alpha_'+str(alpha)+'_batch_size_'+str(batch_size)\
       +'_hidsize_'+str(hid_size)+'_lr_'+str(lr)+'_random_theta_'+str(random_theta)


# name = 'results/samp_size_'+str(samp_size)+'_sgd_'+str(sgd_itrenum)+'_prime_'+str(primes)+'_alpha_'+str(alpha)+'_batch_size_'+str(batch_size)+'_hidsize_'+str(hid_size)

# name = None
# plot_interval  =1
# plot=True

print name
# train = Trainer(samp_size,filename=name,num_of_prime=primes,plot_interval=plot_interval,plot=plot)
train = Trainer(filename=name,num_of_prime=primes,in_test=True,in_train=True,alpha=alpha,
                sgd_batch_size=batch_size,hid_size=hid_size,lr=lr,random_theta=random_theta,plot=True)


train.Train(iternum,sgd_itrenum,full_state_update=full_state_update)


train.plotVstr()
