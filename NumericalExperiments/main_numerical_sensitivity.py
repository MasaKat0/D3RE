import numpy as np
import torch
import torch.optim as optim

from dataset import *
from train import train, loss_func, test 
from model import NN, CNN


def main():
    dataset = 'cifar10'
    ite = 10
    num_nu_data = 1000
    num_de_data = 1000
    Net = CNN

    learning_rate = 1e-4

    epoch = 500
    batchsize = 256

    seed = 2020

    train_loss_nnpu_15 = np.zeros((ite, epoch))
    test_loss_nnpu_15 = np.zeros((ite, epoch))
    test_auc_nnpu_15 = np.zeros((ite, epoch))
    mean_dr_nnpu_15 = np.zeros((ite, epoch))
    train_loss_nnpu_20 = np.zeros((ite, epoch))
    test_loss_nnpu_20 = np.zeros((ite, epoch))
    test_auc_nnpu_20 = np.zeros((ite, epoch))
    mean_dr_nnpu_20 = np.zeros((ite, epoch))
    train_loss_nnpu_50 = np.zeros((ite, epoch))
    test_loss_nnpu_50 = np.zeros((ite, epoch))
    test_auc_nnpu_50 = np.zeros((ite, epoch))
    mean_dr_nnpu_50 = np.zeros((ite, epoch))

    train_loss_nnulsif_15 = np.zeros((ite, epoch))
    test_loss_nnulsif_15 = np.zeros((ite, epoch))
    test_auc_nnulsif_15 = np.zeros((ite, epoch))
    mean_dr_nnulsif_15 = np.zeros((ite, epoch))
    train_loss_nnulsif_20 = np.zeros((ite, epoch))
    test_loss_nnulsif_20 = np.zeros((ite, epoch))
    test_auc_nnulsif_20 = np.zeros((ite, epoch))
    mean_dr_nnulsif_20 = np.zeros((ite, epoch))
    train_loss_nnulsif_50 = np.zeros((ite, epoch))
    test_loss_nnulsif_50 = np.zeros((ite, epoch))
    test_auc_nnulsif_50 = np.zeros((ite, epoch))
    mean_dr_nnulsif_50 = np.zeros((ite, epoch))

    for i in range(ite):
        np.random.seed(seed)

        x_train, t_train, x_test, t_test = load_dataset(dataset)
        dim = x_train.shape[1]

        perm = np.random.permutation(len(x_train))
        x_train_de = x_train[perm[:num_de_data]]

        x_train_nu = x_train[t_train==1]
        perm = np.random.permutation(len(x_train_nu))
        x_train_nu = x_train_nu[perm[:num_nu_data]]

        x_data = np.concatenate([x_train_nu, x_train_de], axis=0)
        
        t_train_nu = np.ones(len(x_train_nu))
        t_train_de = np.zeros(len(x_train_de))
        t_data = np.concatenate([t_train_nu, t_train_de], axis=0)

        temp_t_test = t_test.copy()
        temp_t_test[t_test==1] = 1
        temp_t_test[t_test!=1] = 0
        t_test = temp_t_test

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_nnpu_15 = Net(dim).to(device)
        optimizer_nnpu_15 = optim.Adam(params=model_nnpu_15.parameters(), lr = learning_rate)

        model_nnpu_20 = Net(dim).to(device)
        optimizer_nnpu_20 = optim.Adam(params=model_nnpu_20.parameters(), lr = learning_rate)

        model_nnpu_50 = Net(dim).to(device)
        optimizer_nnpu_50 = optim.Adam(params=model_nnpu_50.parameters(), lr = learning_rate)

        model_nnulsif_15 = Net(dim).to(device)
        optimizer_nnulsif_15 = optim.Adam(params=model_nnulsif_15.parameters(), lr = learning_rate)

        model_nnulsif_20 = Net(dim).to(device)
        optimizer_nnulsif_20 = optim.Adam(params=model_nnulsif_20.parameters(), lr = learning_rate)

        model_nnulsif_50 = Net(dim).to(device)
        optimizer_nnulsif_50 = optim.Adam(params=model_nnulsif_50.parameters(), lr = learning_rate)

        train_nnpu_15, test_nnpu_15, auc_nnpu_15, mean_nnpu_15 = train(x_data, t_data, x_test, t_test, epoch, model_nnpu_15, optimizer_nnpu_15, device, batchsize=batchsize, method='nnPU', upper_bound=1.5)
        train_nnpu_20, test_nnpu_20, auc_nnpu_20, mean_nnpu_20 = train(x_data, t_data, x_test, t_test, epoch, model_nnpu_20, optimizer_nnpu_20, device, batchsize=batchsize, method='nnPU', upper_bound=2)
        train_nnpu_50, test_nnpu_50, auc_nnpu_50, mean_nnpu_50 = train(x_data, t_data, x_test, t_test, epoch, model_nnpu_50, optimizer_nnpu_50, device, batchsize=batchsize, method='nnPU', upper_bound=5)

        train_nnulsif_15, test_nnulsif_15, auc_nnulsif_15, mean_nnulsif_15 = train(x_data, t_data, x_test, t_test, epoch, model_nnulsif_15, optimizer_nnulsif_15, device, batchsize=batchsize, method='nnuLSIF', upper_bound=1.2)
        train_nnulsif_20, test_nnulsif_20, auc_nnulsif_20, mean_nnulsif_20 = train(x_data, t_data, x_test, t_test, epoch, model_nnulsif_20, optimizer_nnulsif_20, device, batchsize=batchsize, method='nnuLSIF', upper_bound=3)
        train_nnulsif_50, test_nnulsif_50, auc_nnulsif_50, mean_nnulsif_50 = train(x_data, t_data, x_test, t_test, epoch, model_nnulsif_50, optimizer_nnulsif_50, device, batchsize=batchsize, method='nnuLSIF', upper_bound=5)

        train_loss_nnpu_15[i] = train_nnpu_15
        test_loss_nnpu_15[i] = test_nnpu_15
        test_auc_nnpu_15[i] = auc_nnpu_15
        mean_dr_nnpu_15[i] = mean_nnpu_15
        train_loss_nnpu_20[i] = train_nnpu_20
        test_loss_nnpu_20[i] = test_nnpu_20
        test_auc_nnpu_20[i] = auc_nnpu_20
        mean_dr_nnpu_20[i] = mean_nnpu_20
        train_loss_nnpu_50[i] = train_nnpu_50
        test_loss_nnpu_50[i] = test_nnpu_50
        test_auc_nnpu_50[i] = auc_nnpu_50
        mean_dr_nnpu_50[i] = mean_nnpu_50

        train_loss_nnulsif_15[i] = train_nnulsif_15
        test_loss_nnulsif_15[i] = test_nnulsif_15
        test_auc_nnulsif_15[i] = auc_nnulsif_15
        mean_dr_nnulsif_15[i] = mean_nnulsif_15
        train_loss_nnulsif_20[i] = train_nnulsif_20
        test_loss_nnulsif_20[i] = test_nnulsif_20
        test_auc_nnulsif_20[i] = auc_nnulsif_20
        mean_dr_nnulsif_20[i] = mean_nnulsif_20
        train_loss_nnulsif_50[i] = train_nnulsif_50
        test_loss_nnulsif_50[i] = test_nnulsif_50
        test_auc_nnulsif_50[i] = auc_nnulsif_50
        mean_dr_nnulsif_50[i] = mean_nnulsif_50

        print(test_nnpu_20[-1])
        print(test_nnulsif_20[-1])

        seed += 1

        np.savetxt('results/train_loss_nnpu_15_%s_%f.csv'%(dataset, learning_rate), train_loss_nnpu_15, delimiter=',')
        np.savetxt('results/test_loss_nnpu_15_%s_%f.csv'%(dataset, learning_rate), test_loss_nnpu_15, delimiter=',')
        np.savetxt('results/test_auc_nnpu_15_%s_%f.csv'%(dataset, learning_rate), test_auc_nnpu_15, delimiter=',')
        np.savetxt('results/mean_dr_nnpu_15_%s_%f.csv'%(dataset, learning_rate), mean_dr_nnpu_15, delimiter=',')
        np.savetxt('results/train_loss_nnpu_30_%s_%f.csv'%(dataset, learning_rate), train_loss_nnpu_20, delimiter=',')
        np.savetxt('results/test_loss_nnpu_30_%s_%f.csv'%(dataset, learning_rate), test_loss_nnpu_20, delimiter=',')
        np.savetxt('results/test_auc_nnpu_30_%s_%f.csv'%(dataset, learning_rate), test_auc_nnpu_20, delimiter=',')
        np.savetxt('results/mean_dr_nnpu_30_%s_%f.csv'%(dataset, learning_rate), mean_dr_nnpu_20, delimiter=',')
        np.savetxt('results/train_loss_nnpu_50_%s_%f.csv'%(dataset, learning_rate), train_loss_nnpu_50, delimiter=',')
        np.savetxt('results/test_loss_nnpu_50_%s_%f.csv'%(dataset, learning_rate), test_loss_nnpu_50, delimiter=',')
        np.savetxt('results/test_auc_nnpu_50_%s_%f.csv'%(dataset, learning_rate), test_auc_nnpu_50, delimiter=',')
        np.savetxt('results/mean_dr_nnpu_50_%s_%f.csv'%(dataset, learning_rate), mean_dr_nnpu_50, delimiter=',')

        np.savetxt('results/train_loss_nnulsif_15_%s_%f.csv'%(dataset, learning_rate), train_loss_nnulsif_15, delimiter=',')
        np.savetxt('results/test_loss_nnulsif_15_%s_%f.csv'%(dataset, learning_rate), test_loss_nnulsif_15, delimiter=',')
        np.savetxt('results/test_auc_nnulsif_15_%s_%f.csv'%(dataset, learning_rate), test_auc_nnulsif_15, delimiter=',')
        np.savetxt('results/mean_dr_nnulsif_15_%s_%f.csv'%(dataset, learning_rate), mean_dr_nnulsif_15, delimiter=',')
        np.savetxt('results/train_loss_nnulsif_30_%s_%f.csv'%(dataset, learning_rate), train_loss_nnulsif_20, delimiter=',')
        np.savetxt('results/test_loss_nnulsif_30_%s_%f.csv'%(dataset, learning_rate), test_loss_nnulsif_20, delimiter=',')
        np.savetxt('results/test_auc_nnulsif_30_%s_%f.csv'%(dataset, learning_rate), test_auc_nnulsif_20, delimiter=',')
        np.savetxt('results/mean_dr_nnulsif_30_%s_%f.csv'%(dataset, learning_rate), mean_dr_nnulsif_20, delimiter=',')
        np.savetxt('results/train_loss_nnulsif_50_%s_%f.csv'%(dataset, learning_rate), train_loss_nnulsif_50, delimiter=',')
        np.savetxt('results/test_loss_nnulsif_50_%s_%f.csv'%(dataset, learning_rate), test_loss_nnulsif_50, delimiter=',')
        np.savetxt('results/test_auc_nnulsif_50_%s_%f.csv'%(dataset, learning_rate), test_auc_nnulsif_50, delimiter=',')
        np.savetxt('results/mean_dr_nnulsif_50_%s_%f.csv'%(dataset, learning_rate), mean_dr_nnulsif_50, delimiter=',')


if __name__ == "__main__":
    main()

