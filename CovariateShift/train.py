import numpy as np
import six
from scipy import optimize
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


UPPER_BOUND = 3
CLASS_PIOR = 1/UPPER_BOUND

def train(X0, X1, epoch, model, optimizer, device, batchsize=5000, method='nnPU'):  
    x_train = np.concatenate([X0, X1])
    
    print(x_train.shape)
    
    N = len(x_train)
    
    t_train = np.zeros(N)
    #X0 is denominator
    t_train[len(X0):] = 1

    for ep in range(1, epoch+1):
        train_loss_step = 0
        count = 0

        perm = np.random.permutation(N)

        for i in six.moves.range(0, N, batchsize):
            model.train()
            optimizer.zero_grad()

            x = x_train[perm[i:i + batchsize]]
            x = torch.tensor(x, dtype=torch.float32)
            x = x.to(device)

            t = np.array([t_train[perm[i:i + batchsize]]]).T
            t_nu = torch.tensor(t, dtype=torch.float32)
            t_nu = t_nu.to(device)
            t_de = torch.tensor(1-t, dtype=torch.float32)
            t_de = t_de.to(device)
            
            output = model(x)

            output_temp = output.clone()

            loss = loss_func(output, t_nu, t_de, method)

            loss.backward()
            optimizer.step()
            
            count += 1

            if (method == 'PU') or (method == 'nnPU'):
                output_temp = sigmoid_func(output_temp)*UPPER_BOUND
            elif method == 'KLIEP':
                output_temp = F.relu(output_temp)

            train_loss = loss_func(output_temp, t_nu, t_de, 'uLSIF').cpu().detach().numpy()
            train_loss_step += train_loss

    return model

def test(xt, model, device, batchsize=100, method='nnPU'):
    f = np.array([])

    for i in six.moves.range(0, len(xt), batchsize):
        xs = xt[i:i + batchsize]
        x = torch.tensor(xs, dtype=torch.float32)
        x = x.to(device)

        output = model(x)

        if (method == 'PU') or (method == 'nnPU'):
            output = sigmoid_func(output)*UPPER_BOUND
        elif method == 'KLIEP':
            output = F.relu(output)

        output = output.cpu().detach().numpy().T[0]

        f = np.append(f, output, axis=0)

    return f

def sigmoid_func(output):
    return 1/(1+torch.exp(-output))

def loss_func(output, t_nu, t_de, method='nnPU'):
    if method == 'nnPU':
        n_positive, n_unlabeled = max([1., torch.sum(t_nu)]), max([1., torch.sum(t_de)])

        g_positive = torch.log(1/(1+torch.exp(-output)))
        g_unlabeled = torch.log(1-1/(1+torch.exp(-output)))
    
        loss_positive = -CLASS_PIOR*torch.sum(g_positive*t_nu)/n_positive
        loss_negative = -torch.sum(g_unlabeled*t_de)/n_unlabeled + CLASS_PIOR*torch.sum(g_unlabeled*t_nu)/n_positive
        
        if loss_negative < 0:
            loss = - loss_negative
        else:
            loss = loss_positive + loss_negative

    if method == 'PU':
        n_positive, n_unlabeled = max([1., torch.sum(t_nu)]), max([1., torch.sum(t_de)])

        g_positive = torch.log(torch.sigmoid(output))
        g_unlabeled = torch.log(1-torch.sigmoid(output))

        loss_positive = -CLASS_PIOR*torch.sum(g_positive*t_nu)/n_positive
        loss_negative = -torch.sum(g_unlabeled*t_de)/n_unlabeled + CLASS_PIOR*torch.sum(g_unlabeled*t_nu)/n_positive

        loss = loss_positive + loss_negative

    elif method == 'uLSIF':
        loss_nu = -(2*output*t_nu).sum()/t_nu.sum()
        loss_de = (output*t_de**2).sum()/t_de.sum()
        
        loss = loss_nu + loss_de

    elif method == 'boundeduLSIF':
        output[output > UPPER_BOUND] = UPPER_BOUND
        loss_nu = -(2*output*t_nu).sum()/t_nu.sum()
        loss_de = (output*t_de**2).sum()/t_de.sum()

        loss = loss_nu + loss_de
    
    elif method == 'nnuLSIF':
        loss_nu = ((-2*output+output**2/UPPER_BOUND)*t_nu).sum()/t_nu.sum()
        loss_nu_middle = (-output**2*t_nu/UPPER_BOUND).sum()/t_nu.sum()
        loss_de = (output**2*t_de).sum()/t_de.sum()

        if loss_de + loss_nu_middle < 0:
            loss = - (loss_de + loss_nu_middle)
        else:
            loss = loss_nu + loss_nu_middle + loss_de

    elif method == 'KLIEP':
        output = F.relu(output)

        loss_nu = torch.log(output*t_nu).sum()/t_nu.sum()
        loss_de = (output*t_de).sum()/t_de.sum()

        loss = -loss_nu + (1-loss_de)**2

    return loss
