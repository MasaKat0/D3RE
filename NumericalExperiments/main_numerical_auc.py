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

    train_loss_pu = np.zeros((ite, epoch))
    test_loss_pu = np.zeros((ite, epoch))
    test_auc_pu = np.zeros((ite, epoch))
    mean_dr_pu = np.zeros((ite, epoch))
    train_loss_ulsif = np.zeros((ite, epoch))
    test_loss_ulsif = np.zeros((ite, epoch))
    test_auc_ulsif = np.zeros((ite, epoch))
    mean_dr_ulsif = np.zeros((ite, epoch))
    train_loss_kliep = np.zeros((ite, epoch))
    test_loss_kliep = np.zeros((ite, epoch))
    test_auc_kliep = np.zeros((ite, epoch))
    mean_dr_kliep = np.zeros((ite, epoch))

    train_loss_nnpu = np.zeros((ite, epoch))
    test_loss_nnpu = np.zeros((ite, epoch))
    test_auc_nnpu = np.zeros((ite, epoch))
    mean_dr_nnpu = np.zeros((ite, epoch))
    train_loss_nnulsif = np.zeros((ite, epoch))
    test_loss_nnulsif = np.zeros((ite, epoch))
    test_auc_nnulsif = np.zeros((ite, epoch))
    mean_dr_nnulsif = np.zeros((ite, epoch))

    train_loss_boundedulsif = np.zeros((ite, epoch))
    test_loss_boundedulsif = np.zeros((ite, epoch))
    test_auc_boundedulsif = np.zeros((ite, epoch))
    mean_dr_boundedulsif = np.zeros((ite, epoch))

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

        model_pu = Net(dim).to(device)
        optimizer_pu = optim.Adam(params=model_pu.parameters(), lr = learning_rate)

        model_ulsif = Net(dim).to(device)
        optimizer_ulsif = optim.Adam(params=model_ulsif.parameters(), lr = learning_rate)

        model_kliep = Net(dim).to(device)
        optimizer_kliep = optim.Adam(params=model_kliep.parameters(), lr = learning_rate)

        model_nnpu = Net(dim).to(device)
        optimizer_nnpu = optim.Adam(params=model_nnpu.parameters(), lr = learning_rate)

        model_nnulsif = Net(dim).to(device)
        optimizer_nnulsif = optim.Adam(params=model_nnulsif.parameters(), lr = learning_rate)

        model_boundedulsif = Net(dim).to(device)
        optimizer_boundedulsif = optim.Adam(params=model_boundedulsif.parameters(), lr = learning_rate)

        train_pu, test_pu, auc_pu, mean_pu = train(x_data, t_data, x_test, t_test, epoch, model_pu, optimizer_pu, device, batchsize=batchsize, method='PU')
        train_ulisf, test_ulsif, auc_ulsif, mean_ulsif = train(x_data, t_data, x_test, t_test, epoch, model_ulsif, optimizer_ulsif, device, batchsize=batchsize, method='uLSIF')
        train_kliep, test_kliep, auc_kliep, mean_kliep = train(x_data, t_data, x_test, t_test, epoch, model_kliep, optimizer_kliep, device, batchsize=batchsize, method='KLIEP')

        train_nnpu, test_nnpu, auc_nnpu, mean_nnpu = train(x_data, t_data, x_test, t_test, epoch, model_nnpu, optimizer_nnpu, device, batchsize=batchsize, method='nnPU')
        train_nnulisf, test_nnulsif, auc_nnulsif, mean_nnulsif = train(x_data, t_data, x_test, t_test, epoch, model_nnulsif, optimizer_nnulsif, device, batchsize=batchsize, method='nnuLSIF')

        train_boundedulisf, test_boundedulsif, auc_boundedulsif, mean_boundedulsif = train(x_data, t_data, x_test, t_test, epoch, model_boundedulsif, optimizer_boundedulsif, device, batchsize=batchsize, method='boundeduLSIF')

        train_loss_pu[i] = train_pu
        test_loss_pu[i] = test_pu
        test_auc_pu[i] = auc_pu
        mean_dr_pu[i] = mean_pu
        train_loss_ulsif[i] = train_ulisf
        test_loss_ulsif[i] = test_ulsif
        test_auc_ulsif[i] = auc_ulsif
        mean_dr_ulsif[i] = mean_ulsif
        train_loss_kliep[i] = train_kliep
        test_loss_kliep[i] = test_kliep
        test_auc_kliep[i] = auc_kliep
        mean_dr_kliep[i] = mean_kliep

        train_loss_nnpu[i] = train_nnpu
        test_loss_nnpu[i] = test_nnpu
        test_auc_nnpu[i] = auc_nnpu
        mean_dr_nnpu[i] = mean_nnpu
        train_loss_nnulsif[i] = train_nnulisf
        test_loss_nnulsif[i] = test_nnulsif
        test_auc_nnulsif[i] = auc_nnulsif
        mean_dr_nnulsif[i] = mean_nnulsif

        train_loss_boundedulsif[i] = train_boundedulisf
        test_loss_boundedulsif[i] = test_boundedulsif
        test_auc_boundedulsif[i] = auc_boundedulsif
        mean_dr_boundedulsif[i] = mean_boundedulsif

        print(test_pu[-1])
        print(test_ulsif[-1])
        print(test_kliep[-1])

        print(test_nnpu[-1])
        print(test_nnulsif[-1])

        print(test_boundedulsif[-1])

        seed += 1

        np.savetxt('results/train_loss_pu_%s_%f.csv'%(dataset, learning_rate), train_loss_pu, delimiter=',')
        np.savetxt('results/test_loss_pu_%s_%f.csv'%(dataset, learning_rate), test_loss_pu, delimiter=',')
        np.savetxt('results/test_auc_pu_%s_%f.csv'%(dataset, learning_rate), test_auc_pu, delimiter=',')
        np.savetxt('results/mean_dr_pu_%s_%f.csv'%(dataset, learning_rate), mean_dr_pu, delimiter=',')
        np.savetxt('results/train_loss_ulsif_%s_%f.csv'%(dataset, learning_rate), train_loss_ulsif, delimiter=',')
        np.savetxt('results/test_loss_ulsif_%s_%f.csv'%(dataset, learning_rate), test_loss_ulsif, delimiter=',')
        np.savetxt('results/test_auc_ulisf_%s_%f.csv'%(dataset, learning_rate), test_auc_ulsif, delimiter=',')
        np.savetxt('results/mean_dr_ulsif_%s_%f.csv'%(dataset, learning_rate), mean_dr_ulsif, delimiter=',')
        np.savetxt('results/train_loss_kliep_%s_%f.csv'%(dataset, learning_rate), train_loss_kliep, delimiter=',')
        np.savetxt('results/test_loss_kliep_%s_%f.csv'%(dataset, learning_rate), test_loss_kliep, delimiter=',')
        np.savetxt('results/test_auc_kliep_%s_%f.csv'%(dataset, learning_rate), test_auc_kliep, delimiter=',')
        np.savetxt('results/mean_dr_klipe_%s_%f.csv'%(dataset, learning_rate), mean_dr_kliep, delimiter=',')

        np.savetxt('results/train_loss_nnpu_%s_%f.csv'%(dataset, learning_rate), train_loss_nnpu, delimiter=',')
        np.savetxt('results/test_loss_nnpu_%s_%f.csv'%(dataset, learning_rate), test_loss_nnpu, delimiter=',')
        np.savetxt('results/test_auc_nnpu_%s_%f.csv'%(dataset, learning_rate), test_auc_nnpu, delimiter=',')
        np.savetxt('results/mean_dr_nnpu_%s_%f.csv'%(dataset, learning_rate), mean_dr_nnpu, delimiter=',')
        np.savetxt('results/train_loss_nnulsif_%s_%f.csv'%(dataset, learning_rate), train_loss_nnulsif, delimiter=',')
        np.savetxt('results/test_loss_nnulsif_%s_%f.csv'%(dataset, learning_rate), test_loss_nnulsif, delimiter=',')
        np.savetxt('results/test_auc_nnulsif_%s_%f.csv'%(dataset, learning_rate), test_auc_nnulsif, delimiter=',')
        np.savetxt('results/mean_dr_nnulsif_%s_%f.csv'%(dataset, learning_rate), mean_dr_nnulsif, delimiter=',')

        np.savetxt('results/train_loss_boundedulsif_%s_%f.csv'%(dataset, learning_rate), train_loss_boundedulsif, delimiter=',')
        np.savetxt('results/test_loss_boundedulsif_%s_%f.csv'%(dataset, learning_rate), test_loss_boundedulsif, delimiter=',')
        np.savetxt('results/test_auc_boundedulsif_%s_%f.csv'%(dataset, learning_rate), test_auc_boundedulsif, delimiter=',')
        np.savetxt('results/mean_dr_boundedulsif_%s_%f.csv'%(dataset, learning_rate), mean_dr_boundedulsif, delimiter=',')


if __name__ == "__main__":
    main()

