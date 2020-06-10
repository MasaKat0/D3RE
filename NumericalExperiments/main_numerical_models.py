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

    learning_rate = 1e-3

    epoch = 500
    batchsize = 256

    seed = 2020

    train_loss_bkl = np.zeros((ite, epoch))
    test_loss_bkl = np.zeros((ite, epoch))
    test_auc_bkl = np.zeros((ite, epoch))
    mean_dr_bkl = np.zeros((ite, epoch))
    train_loss_ukl = np.zeros((ite, epoch))
    test_loss_ukl = np.zeros((ite, epoch))
    test_auc_ukl = np.zeros((ite, epoch))
    mean_dr_ukl = np.zeros((ite, epoch))

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

        model_bkl = Net(dim).to(device)
        optimizer_bkl = optim.Adam(params=model_bkl.parameters(), lr = learning_rate)

        model_ukl = Net(dim).to(device)
        optimizer_ukl = optim.Adam(params=model_ukl.parameters(), lr = learning_rate)

        train_bkl, test_bkl, auc_bkl, mean_bkl = train(x_data, t_data, x_test, t_test, epoch, model_pu, optimizer_pu, device, batchsize=batchsize, method='BKL')
        train_ukl, test_ukl, auc_ukl, mean_ukl = train(x_data, t_data, x_test, t_test, epoch, model_ulsif, optimizer_ulsif, device, batchsize=batchsize, method='UKL')

        train_loss_bkl[i] = train_bkl
        test_loss_bkl[i] = test_bkl
        test_auc_bkl[i] = auc_bkl
        mean_dr_bkl[i] = mean_bkl
        train_loss_ukl[i] = train_ukl
        test_loss_ukl[i] = test_ukl
        test_auc_ukl[i] = auc_ukl
        mean_dr_ukl[i] = mean_ukl

        seed += 1

        np.savetxt('results/train_loss_bkl_%s_%f.csv'%(dataset, learning_rate), train_loss_bkl, delimiter=',')
        np.savetxt('results/test_loss_bkl_%s_%f.csv'%(dataset, learning_rate), test_loss_bkl, delimiter=',')
        np.savetxt('results/test_auc_bkl_%s_%f.csv'%(dataset, learning_rate), test_auc_bkl, delimiter=',')
        np.savetxt('results/mean_dr_bkl_%s_%f.csv'%(dataset, learning_rate), mean_dr_bkl, delimiter=',')
        np.savetxt('results/train_loss_ukl_%s_%f.csv'%(dataset, learning_rate), train_loss_ukl, delimiter=',')
        np.savetxt('results/test_loss_ukl_%s_%f.csv'%(dataset, learning_rate), test_loss_ukl, delimiter=',')
        np.savetxt('results/test_auc_ukl_%s_%f.csv'%(dataset, learning_rate), test_auc_ukl, delimiter=',')
        np.savetxt('results/mean_dr_ukl_%s_%f.csv'%(dataset, learning_rate), mean_dr_ukl, delimiter=',')
    
if __name__ == "__main__":
    main()

