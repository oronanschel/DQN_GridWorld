import matplotlib.pyplot as plt
import csv, pdb, sys
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import csv, pdb, sys
from matplotlib.backends.backend_pdf import PdfPages
import os

_xlabel = 'iterations$\\times 10^3$'
_ylabel = '$||V^*-\hat{V}||$'

def plot_subplot(p,ax,title,name,tau):
    for idx in range(len(p)):
        f1 = os.path.join('results/', name + str(p[idx]) + '.csv')
        f1 = os.path.join(os.getcwd(), f1)
        with open(f1) as csvfile:
            reader = csv.DictReader(csvfile)
            itr = []
            dist = []
            for row in reader:
                itr.append(int(row['itr']))
                dist.append(float(row['dist']))
        itr=(np.array((itr))+.0)*tau/1000

        ax.semilogy(itr, dist, label='k=' + str(p[idx]))
        plt.xlabel(_xlabel)
        plt.ylabel(_ylabel)
        plt.title(title)
        plt.legend()

def plot_subplot2(p,ax,title,name1,name2,tau):
    for idx in range(len(p)):
        f1 = os.path.join('results/', name1 + str(p[idx]) + name2+ '.csv')
        f1 = os.path.join(os.getcwd(), f1)
        with open(f1) as csvfile:
            reader = csv.DictReader(csvfile)
            itr = []
            dist = []
            for row in reader:
                itr.append(int(row['itr']))
                dist.append(float(row['dist']))
        itr=(np.array((itr))+.0)*tau/1000

        ax.semilogy(itr, dist, label='k=' + str(p[idx]))
        plt.xlabel(_xlabel)
        plt.ylabel(_ylabel)
        plt.title(title)
        plt.legend()


def plot_ax(ax,max_itr, name,label):
    f1 = os.path.join('results/', name + '.csv')
    f1 = os.path.join(os.getcwd(), f1)
    with open(f1) as csvfile:
        reader = csv.DictReader(csvfile)
        itr_t = []
        dist_t = []
        for row in reader:
            itr_t.append(int(row['itr']))
            dist_t.append(float(row['dist']))

    # dilute the plot
    dist = []
    itr  = []
    for i in range(len(dist_t)):
        if i % 1==0:
            dist.append(dist_t[i])
            itr.append(itr_t[i])

    ax.semilogy(itr[:max_itr], dist[:max_itr], label=label)




# 3 plot in_train in_test sgd=[5,10,50,300] prime=[1,10,40,80]
def plot_compare_tau():
    fig = plt.figure()
    n = 2
    m = 1
    max_itr = -1

    # ax = fig.add_subplot(n,m,1)
    # p = ['1','10','40']
    # name = 'full_state_space_sgd_5_prime_'
    # title  = '$\\tau=5$'
    # plot_subplot(p,ax,title,max_itr,name)

    # ax = fig.add_subplot(n,m,1)
    # p = ['1','40','80']
    # name = 'full_state_space_sgd_10_prime_'
    # title  = '$\\tau=10$'
    # plot_subplot(p,ax,title,max_itr,name)
    # plt.grid()

    ax = fig.add_subplot(n,m,1)
    p = ['1','40','80']
    title  = '$\\tau=50$'
    tau = 50
    name = 'full_state_space_sgd_50_prime_'
    plot_subplot(p,ax,title,name,tau)
    plt.grid()

    x1, x2, y1, y2 = plt.axis()
    plt.axis((0,100, 10**-9, 10**-1))

    ax = fig.add_subplot(n,m,2)
    p = ['1','40','80']
    title  = '$\\tau=200$'
    tau = 200
    name = 'full_s_sgd_200_prime_'
    plot_subplot(p,ax,title,name,tau)
    plt.grid()

    x1, x2, y1, y2 = plt.axis()
    plt.axis((0,400, 10**-9, 10**-1))




    plt.tight_layout()


    pp = PdfPages('compare_tau.pdf')
    pp.savefig(fig)
    pp.close()

def plot_compare_theta_dependent():
    fig = plt.figure()
    n = 2
    m = 1
    max_itr = -1

    # ax = fig.add_subplot(n,m,1)
    # # p = ['1','40','80']
    # p = ['1',]
    # title  = '$\\theta_i$ was learned from random initialization every $\\tau=5000$'
    # tau = 400
    # name1= 'full_s_sgd_400_prime_'
    # name2 ='_alpha_0.01_batch_size_64_hidsize_80_lr_0.1_random_theta_True'
    # plot_subplot2(p,ax,title,name1,name2,tau)
    # plt.grid()


    # ax = fig.add_subplot(n,m,2)
    # # p = ['1','40','80']
    # p = ['1','40']
    # title  = '$\\theta_i$ was learned from random initialization every $\\tau=5000$'
    # tau = 5000
    # name1= 'full_s_sgd_5000_prime_'
    # name2 ='_alpha_0.01_batch_size_64_hidsize_80_lr_0.1_random_theta'
    # plot_subplot2(p,ax,title,name1,name2,tau)
    # plt.grid()


    ax = fig.add_subplot(n,m,1)
    p = ['1','40','80']
    # p = ['1','40']
    tau = 1000
    title  = '$\\theta_i$ was learned from random initialization every $\\tau=$' +str(tau)
    name1= 'full_s_sgd_'+str(tau)+'_prime_'
    name2 ='_alpha_0.01_batch_size_64_hidsize_80_lr_0.8_random_theta_True'
    plot_subplot2(p,ax,title,name1,name2,tau)
    plt.grid()

    p = ['80']
    tau = 1000
    # title  = '$\\theta_i$ was learned from random initialization every $\\tau=$' +str(tau)
    name1= 'full_s_sgd_'+str(tau)+'_prime_'
    name2 ='_alpha_1_batch_size_64_hidsize_80_lr_0.8_random_theta_True'
    plot_subplot2(p,ax,title,name1,name2,tau)
    plt.grid()


    x1, x2, y1, y2 = plt.axis()
    plt.axis((0,2500, 10**-12, 10**-1))

    ax = fig.add_subplot(n,m,2)
    p = ['1','40']
    title  = '$\\theta_i$ was learned using warm start with $\\theta_{i-1}$ every $\\tau=5000$'
    tau = 5000
    name1= 'full_s_sgd_5000_prime_'
    name2 ='_alpha_0.01_batch_size_64_hidsize_80_lr_0.1_random_theta_false'
    plot_subplot2(p,ax,title,name1,name2,tau)
    plt.grid()

    # name = 'full_s_sgd_200_prime_'
    # plot_subplot(p,ax,title,name,tau)
    # plt.grid()

    x1, x2, y1, y2 = plt.axis()
    plt.axis((0,2500, 10**-12, 10**-1))

    plt.tight_layout()


    pp = PdfPages('compare_theta_dependent.pdf')
    pp.savefig(fig)
    pp.close()

# 2 plot sgd=[10,50] in=[train,test,train+test]
def plot_in_train_in_test_compare():
    fig = plt.figure()
    n = 1
    m = 1
    max_itr = 400
    title  = '$\\tau=50,K=40$'

    # ax = fig.add_subplot(n,m,1)
    # name = 'full_state_space_sgd_50_prime_40'
    # plot_ax(ax,max_itr, name,label='PL')
    #
    # name = 'full_s_sgd_in_test_50_prime_40'
    # plot_ax(ax,max_itr, name,label='P')
    #
    # name = 'full_s_sgd_in_train_50_prime_40'
    # plot_ax(ax,max_itr, name,label='L')
    #
    # plt.xlabel(_xlabel)
    # plt.ylabel(_ylabel)
    # plt.title(title)
    # plt.legend()


    title  = '$\\tau=200,K=40$'
    ax = fig.add_subplot(n,m,1)
    name = 'full_s_sgd_200_prime_40'
    plot_ax(ax,max_itr, name,label='PL')

    name = 'full_s_sgd_in_test_200_prime_40'
    plot_ax(ax,max_itr, name,label='P')

    name = 'full_s_sgd_in_train_200_prime_40'
    plot_ax(ax,max_itr, name,label='L')

    plt.xlabel(_xlabel)
    plt.ylabel(_ylabel)
    plt.title(title)
    plt.legend()

    plt.tight_layout()

    pp = PdfPages('compare_test_train.pdf')
    pp.savefig(fig)
    pp.close()
#
# fig = plt.figure()
# n = 3
# m = 1
# ax  = fig.add_subplot(n,m,1)
# p = ['1','10','40']
# name = 'sgd_5_samp_size_100_prime_'
# for idx in range(len(p)):
#     f1 = os.path.join(os.getcwd(), name +str(p[idx])+'.csv')
#     with open(f1) as csvfile:
#         reader = csv.DictReader(csvfile)
#         itr = []
#         dist = []
#         for row in reader:
#             itr.append(int(row['itr']))
#             dist.append(float(row['dist']))
#     ax.semilogy(itr,dist,label=str(p[idx])+ (' target net' if idx==0 else ' targets nets'))
#
# plt.xlabel(_xlabel)
# plt.title(name)
# plt.legend()
# #
# ax2 = fig.add_subplot(n,m,2)
# p = ['1']
# name = 'sgd_10_samp_size_40_prime_'
# for idx in range(len(p)):
#     f1 = os.path.join(os.getcwd(), name +str(p[idx])+'.csv')
#     with open(f1) as csvfile:
#         reader = csv.DictReader(csvfile)
#         itr = []
#         dist = []
#         for row in reader:
#             itr.append(int(row['itr']))
#             dist.append(float(row['dist']))
#     ax2.semilogy(itr,dist,label=str(p[idx])+ (' full target net' if idx==0 else 'full targets nets'))
#
# plt.xlabel(_xlabel)
# plt.legend()
# plt.title(name)
def plot_compare_ads():
    fig = plt.figure()
    n = 3
    m = 1
    max_itr = 500


    ax = fig.add_subplot(n,m,1)
    p = ['1','40']
    name = 'sgd_300_samp_size_40_prime_'
    title  = '$\\tau=300$'
    plot_subplot(p,ax,title,max_itr,name)
    plt.grid()

    ax = fig.add_subplot(n,m,2)
    p = ['1','40','80']
    title  = '$\\tau=50$'
    name = 'full_state_space_sgd_50_prime_'
    plot_subplot(p,ax,title,max_itr,name)
    plt.grid()

    ax = fig.add_subplot(n,m,3)
    p = ['4','80']
    title  = '$\\tau=200$'
    name = 'full_s_sgd_200_prime_'
    plot_subplot(p,ax,title,max_itr,name)
    plt.grid()


    plt.tight_layout()


    pp = PdfPages('compare_tmp.pdf')
    pp.savefig(fig)
    pp.close()

# compare alpha - tau=200, prime=40
def plot_compare_alpha():
    fig = plt.figure()
    n = 1
    m = 1
    max_itr = 400

    title  = '$\\tau=200,K=40$'
    ax = fig.add_subplot(n,m,1)
    name = 'full_s_sgd_in_test_200_prime_40_alpha_0.01'
    plot_ax(ax,max_itr, name,label='$\\alpha=0.01$')

    name = 'full_s_sgd_in_test_200_prime_40_alpha_0.1'
    plot_ax(ax,max_itr, name,label='$\\alpha=0.1$')

    name = 'full_s_sgd_in_test_200_prime_40_alpha_0.5'
    plot_ax(ax,max_itr, name,label='$\\alpha=0.5$')

    name = 'full_s_sgd_in_test_200_prime_40_alpha_0.9'
    plot_ax(ax,max_itr, name,label='$\\alpha=0.9$')

    plt.xlabel(_xlabel)
    plt.ylabel(_ylabel)
    plt.title(title)
    plt.legend()

    x1, x2, y1, y2 = plt.axis()

    plt.axis((0,500, y1, y2))

    plt.tight_layout()

    pp = PdfPages('compare_alpha.pdf')
    pp.savefig(fig)
    pp.close()

def plot_compare_tau_noise():
    fig = plt.figure()
    n = 1
    m = 1
    max_itr = -1

    title  = '$K=1$'
    ax = fig.add_subplot(n,m,1)


    name = 'full_s_sgd_10_prime_1_alpha_1_batch_size_32'
    plot_ax(ax,max_itr, name,label='$\\tau=10$')

    name = 'full_s_sgd_100_prime_1_alpha_1_batch_size_32'
    plot_ax(ax, max_itr, name, label='$\\tau=100$')


    name = 'full_s_sgd_300_prime_1_alpha_1_batch_size_32'
    plot_ax(ax,max_itr, name,label='$\\tau=300$')


    plt.xlabel(_xlabel)
    plt.ylabel(_ylabel)
    plt.title(title)
    plt.legend()

    x1, x2, y1, y2 = plt.axis()
    plt.axis((0,5000, y1, y2))

    plt.tight_layout()

    pp = PdfPages('compare_tau_noise.pdf')
    pp.savefig(fig)
    pp.close()

def plot_compare_batch_size_noise():
    fig = plt.figure()
    n = 1
    m = 1
    max_itr = -1

    title  = '$K=1$'
    ax = fig.add_subplot(n,m,1)


    name = 'full_s_sgd_100_prime_1_alpha_1_batch_size_16'
    plot_ax(ax,max_itr, name,label='mini batch size=16')

    # name = 'full_s_sgd_100_prime_1_alpha_1_batch_size_32'
    # plot_ax(ax, max_itr, name, label='mini batch size=32')

    name = 'full_s_sgd_100_prime_1_alpha_1_batch_size_64'
    plot_ax(ax, max_itr, name, label='mini batch size=64')

    # name = 'full_s_sgd_100_prime_1_alpha_1_batch_size_128'
    # plot_ax(ax, max_itr, name, label='mini batch size=128')

    name = 'full_s_sgd_100_prime_1_alpha_1_batch_size_256'
    plot_ax(ax, max_itr, name, label='mini batch size=256')

    plt.xlabel(_xlabel)
    plt.ylabel(_ylabel)
    plt.title(title)
    plt.legend()

    x1, x2, y1, y2 = plt.axis()
    plt.axis((0,5000, y1, y2))

    plt.tight_layout()

    pp = PdfPages('compare_batch_size_noise.pdf')
    pp.savefig(fig)
    pp.close()

def plot_compare_hid_size_noise():
    fig = plt.figure()
    n = 1
    m = 1
    max_itr = -1

    title  = '$K=1$'
    ax = fig.add_subplot(n,m,1)


    name = 'full_s_sgd_200_prime_1_alpha_1_batch_size_32_hidsize_5'
    plot_ax(ax,max_itr, name,label='hidden layer size=5')


    # name = 'full_s_sgd_200_prime_1_alpha_1_batch_size_32_hidsize_20'
    # plot_ax(ax,max_itr, name,label='hidden layer size=20')


    name = 'full_s_sgd_200_prime_1_alpha_1_batch_size_32_hidsize_40'
    plot_ax(ax,max_itr, name,label='hidden layer size=40')


    # name = 'full_s_sgd_200_prime_1_alpha_1_batch_size_32_hidsize_80'
    # plot_ax(ax,max_itr, name,label='hidden layer size=80')


    name = 'full_s_sgd_200_prime_1_alpha_1_batch_size_32_hidsize_140'
    plot_ax(ax,max_itr, name,label='hidden layer size=140')
    plt.xlabel(_xlabel)
    plt.ylabel(_ylabel)
    plt.title(title)
    plt.legend()

    x1, x2, y1, y2 = plt.axis()
    plt.axis((0,5000, y1, y2))

    plt.tight_layout()

    pp = PdfPages('compare_hid_size_noise.pdf')
    pp.savefig(fig)
    pp.close()

def plot_compare_samp_size_noise():
    fig = plt.figure()
    n = 1
    m = 1
    max_itr = -1

    title  = '$K=1$'
    ax = fig.add_subplot(n,m,1)


    name = 'samp_size_10_sgd_200_prime_1_alpha_1_batch_size_32_hidsize_80'
    plot_ax(ax,max_itr, name,label='sample size=10')


    name = 'samp_size_50_sgd_200_prime_1_alpha_1_batch_size_32_hidsize_80'
    plot_ax(ax,max_itr, name,label='sample size=50')

    name = 'samp_size_200_sgd_200_prime_1_alpha_1_batch_size_32_hidsize_80'
    plot_ax(ax, max_itr, name, label='sample size=200')

    name = 'samp_size_400_sgd_200_prime_1_alpha_1_batch_size_32_hidsize_80'
    plot_ax(ax, max_itr, name, label='full state sample')

    plt.xlabel(_xlabel)
    plt.ylabel(_ylabel)
    plt.title(title)
    plt.legend()

    x1, x2, y1, y2 = plt.axis()
    plt.axis((0,5000, y1, y2))

    plt.tight_layout()

    pp = PdfPages('compare_samp_size_noise.pdf')
    pp.savefig(fig)
    pp.close()

def plot_compare_added_noise():
    fig = plt.figure()
    n = 1
    m = 1
    max_itr = -1

    title  = '$K=40$'
    ax = fig.add_subplot(n,m,1)


    name = 'no_added_noise_full_s_sgd_200_prime_40_alpha_0.01_batch_size_32_hidsize_80'
    plot_ax(ax,max_itr, name,label='no noise')


    name = 'with_added_noise_full_s_sgd_200_prime_40_alpha_0.01_batch_size_32_hidsize_80'
    plot_ax(ax,max_itr, name,label='noise std=10^-3')


    name = 'with_added_noise_10-2_full_s_sgd_200_prime_40_alpha_0.01_batch_size_32_hidsize_80'
    plot_ax(ax,max_itr, name,label='noise std=10^-2')



    plt.xlabel(_xlabel)
    plt.ylabel(_ylabel)
    plt.title(title)
    plt.legend()

    x1, x2, y1, y2 = plt.axis()
    plt.axis((0,5000, y1, y2))

    plt.tight_layout()

    pp = PdfPages('compare_added_noise.pdf')
    pp.savefig(fig)
    pp.close()

def plot_compare_learning_rate():
    fig = plt.figure()
    n = 1
    m = 1
    max_itr = -1

    title  = '$K=40,\\tau=200$'
    ax = fig.add_subplot(n,m,1)


    name = 'full_s_sgd_200_prime_1_alpha_0.01_batch_size_32_hidsize_80_lr_0.01'
    plot_ax(ax,max_itr, name,label='lr=0.01, k=1')


    name = 'full_s_sgd_200_prime_1_alpha_0.01_batch_size_32_hidsize_80_lr_0.01_to_0.0001'
    plot_ax(ax, max_itr, name, label='lr=0.01 to 0.0001')


    name = 'full_s_sgd_200_prime_1_alpha_0.01_batch_size_32_hidsize_80_lr_0.001'
    plot_ax(ax,max_itr, name,label='lr=0.001')


    name = 'full_s_sgd_200_prime_40_alpha_0.01_batch_size_32_hidsize_80_lr_0.01'
    plot_ax(ax,max_itr, name,label='lr=0.01, k=40')



    plt.xlabel(_xlabel)
    plt.ylabel(_ylabel)
    plt.title(title)
    plt.legend()

    x1, x2, y1, y2 = plt.axis()
    plt.axis((0,5000, y1, y2))

    plt.tight_layout()

    pp = PdfPages('compare_learning_rate.pdf')
    pp.savefig(fig)
    pp.close()


def plot_compare_ADAM_learning_rate():
    fig = plt.figure()
    n = 1
    m = 1
    max_itr = -1

    title  = '$K=40,\\tau=200$'
    ax = fig.add_subplot(n,m,1)

    name = 'Bfull_s_sgd_200_prime_1_alpha_0.01_batch_size_32_hidsize_80'
    plot_ax(ax,max_itr, name,label='lr=0.01')

    name = 'full_s_sgd_200_prime_1_alpha_1_batch_size_32_hidsize_80'
    plot_ax(ax,max_itr, name,label='lr=0.001')


    name = 'Afull_s_sgd_200_prime_1_alpha_0.01_batch_size_32_hidsize_80'
    plot_ax(ax,max_itr, name,label='lr=0.0001')



    plt.xlabel(_xlabel)
    plt.ylabel(_ylabel)
    plt.title(title)
    plt.legend()

    x1, x2, y1, y2 = plt.axis()
    plt.axis((0,5000, y1, y2))

    plt.tight_layout()

    pp = PdfPages('compare_ADAM_learning_rate.pdf')
    pp.savefig(fig)
    pp.close()




# plot_ax(ax,max_itr, name,label='hidden layer size=80')

# plot_compare_tau()
#plot_compare_alpha( )
# plot_compare_ads()
# plot_in_train_in_test_compare()
# plt.show()
# plot_compare_tau_noise()
# plot_compare_batch_size_noise()
# plot_compare_hid_size_noise()
# plot_compare_samp_size_noise()
# plot_compare_added_noise()
# plot_compare_learning_rate()
# plot_compare_ADAM_learning_rate()
plot_compare_theta_dependent()