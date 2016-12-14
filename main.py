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
    def __init__(self,hid_size=80,sgd_batch_size=32,filename=None,plot_on_screen=False):

      ###### MDP Params ######
      self.bins = 40
      self.gamma = 0.95

      ###### Compute optimal V (baseline) ######
      self.Vstr =self.computeVstr()

      ###### Initalzie a tf session with Value model ######
      self.sess = tf.Session()
      self.model = approximator.NN_1_hot(self.sess,self.bins,
                                         sgd_batch_size=sgd_batch_size, hid_size=hid_size)

      # self.model = approximator.Tabular(bins=self.bins)
      self.sess.run(tf.initialize_all_variables())

      ###### Results/Plot ######
      self.plot_interval = 1
      self.filename = filename
      if filename:
          self.createResultFile()
      self.plot_on_screen = plot_on_screen
      if plot_on_screen:
          plt.ion()
          self.fig = plt.figure()
          self.ax0 = self.fig.add_subplot(121, projection='3d')
          self.ax1 = self.fig.add_subplot(122)

      ###### Miscellaneous ######
      self.epoch = 0
      self.dist_l = []
      self.epoch_l = []



    def computeVstr(self):
        V_str = np.zeros(shape=[self.bins,self.bins])
        V_str_new = np.ones(shape=[self.bins,self.bins])

        iternum = 1000
        for itr in range(iternum):
            V_str = np.array(V_str_new)
            for idx in range(self.bins):
                for idy in range(self.bins):
                    V_list = [ V_str[min(idx + 1, self.bins - 1), idy]
                             , V_str[max(idx - 1, 0)       , idy]
                             , V_str[idx                   ,max(idy - 1, 0)]
                             , V_str[idx                   ,min(idy + 1, self.bins - 1)]]
                    V_max = np.max(V_list)

                    V_str_new[idx,idy] = self.gamma*V_max + self.reward(idx,idy)
            if np.mean(np.square(V_str.reshape(-1)-V_str_new.reshape(-1)))<10**-13:
                break

        print 'Optimal V with eps='+str(np.mean(np.square(V_str.reshape(-1)-V_str_new.reshape(-1)))) \
              + ' in '+str(itr)+' iterations'
        return V_str_new

    def reward(self,idx,idy):
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

        # # plot problem example path
        # fig = plt.figure()
        # ax = fig.gca()
        #
        # x = [1,1,1]
        # y = [2,3,4]
        # ax.plot(x, y,color='b',marker=6)
        #
        # x = [1,2]
        # y = [4,4]
        # ax.plot(x, y,color='b')
        #
        # x = [2,3,4,5,6,7,8]
        # y = [4,4,4,4,4,4,4]
        # ax.plot(x, y,color='b', marker='>')
        #
        # x = [8,8]
        # y = [4,5]
        # ax.plot(x, y,color='b')
        #
        # x = [8,8,8,8,8,8]
        # y = [5,6,7,8,9,10]
        # ax.plot(x, y, color='b', marker=6)
        #
        # x = [8,7]
        # y = [10,10]
        # ax.plot(x, y,color='b')
        #
        #
        # x = [7,6,5]
        # y = [10,10,10]
        # ax.plot(x, y,color='b', marker='<')
        #
        # ax.plot([5],[5], '*r',markersize=18)
        # ax.text(5.2, 5.2, '$r=1$', fontsize=22)
        #
        # ax.set_xlim(0,10.5)
        # ax.set_ylim(0,10.5)
        #
        # plt.xticks([0,2.5,5,7.5,10],[0,10,20,30,40])
        # plt.yticks([0,2.5,5,7.5,10],[0,10,20,30,40])
        #
        # plt.title('Gridworld')
        # plt.xlabel('x')
        # plt.ylabel('y')
        #
        # pp = PdfPages("Gridworld.pdf")
        # pp.savefig(fig)
        # pp.close()
        return

    # def sample_and_target(self,batch_size):
    #     s = np.zeros(shape=[batch_size,self.bins,self.bins])
    #     V_t = np.zeros(batch_size)
    #
    #     for i in range(batch_size):
    #         idx = np.random.randint(self.bins)
    #         idy = np.random.randint(self.bins)
    #         s[i,idx,idy] = 1
    #         r = self.reward(idx,idy)
    #         s1 = np.zeros(shape=[4,self.bins,self.bins])
    #         s1[0,min(idx + 1, self.bins - 1), idy] = 1
    #         s1[1,max(idx - 1, 0), idy] = 1
    #         s1[2,idx,max(idy - 1, 0)] =1
    #         s1[3,idx,min(idy + 1, self.bins - 1)]=1
    #         alpha = self.computeAlpha()
    #         V_t[i] = r+self.gamma*np.max(self.model.predict(s1))
    #
    #     return s,V_t

    def full_ss_and_target(self):
        s = np.zeros(shape=[self.bins**2,self.bins,self.bins])
        V_t = np.zeros(self.bins**2)

        i=0
        for idx in range(self.bins):
            for idy in range(self.bins):
                s[i,idx,idy] = 1
                r = self.reward(idx,idy)
                s1 = np.zeros(shape=[4,self.bins,self.bins])
                s1[0,min(idx + 1, self.bins - 1), idy] = 1
                s1[1,max(idx - 1, 0), idy] = 1
                s1[2,idx,max(idy - 1, 0)] =1
                s1[3,idx,min(idy + 1, self.bins - 1)]=1
                V_t[i] = r+self.gamma*np.max(self.model.predict(s1))
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

        V_pred = self.model.predict(s)
        dist = np.mean(np.square(V_pred.reshape(itr)-V_str_flat.reshape(itr)))


        self.dist_l.append(dist)
        self.epoch_l.append(self.epoch)
        if self.filename:
            res_file = open(self.res_file, 'a')
            res_file.write("%d,%.11f\n" % (self.epoch,dist))
            res_file.close()

        x = np.repeat(np.arange(self.bins),self.bins)
        y = np.arange(self.bins)
        for _ in range(1,self.bins):
            y = np.append(y,np.arange(self.bins))

        if self.plot_on_screen and self.epoch % 1 == 0:
            self.ax0.clear()
            self.ax1.clear()
            self.ax0.plot_trisurf(x, y, V_pred.reshape(len(x)), cmap=cm.jet, linewidth=0.2,vmin=0,vmax=5)
            self.ax1.plot(self.epoch_l,self.dist_l)
            self.ax1.set_yscale('log')
            plt.pause(0.05)






        return dist

    def Train(self,epoch_num,sgd_itrenum,full_state_update=True):

        for self.epoch in range(epoch_num):
            if self.epoch % self.plot_interval == 0:
                    self.computeDist()
            if self.epoch-1%5==0:
                print '------ epoch:'+str(self.epoch)+ '------'

            if full_state_update:
                s,V_t = self.full_ss_and_target()
            else:
                s, V_t = self.sample_and_target(self.batch_size)

            self.model.learn(s,V_t,sgd_itrenum)



    def computeAlpha(self):
        return self.alpha
        # return self.alpha*(10)/(10+self.itr)



epoch_num = 700
sgd_itrenum = 100
samp_size = 300
full_state_update = True
sgd_batch_size = 32
hid_size = 80
plot_on_screen = True

filename = 'results/full_s_sgd_'+str(sgd_itrenum)+'_sgdbatchsize_'+str(sgd_batch_size)\
       +'_hidsize_'+str(hid_size)

print filename

train = Trainer(hid_size=hid_size,sgd_batch_size=sgd_batch_size,filename=filename,plot_on_screen=plot_on_screen)
# train.plotVstr()
train.Train(epoch_num,sgd_itrenum,full_state_update=full_state_update)


