from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

import scipy.stats
from scipy.special import psi, polygamma

import abc
import itertools

from keras.preprocessing.image import apply_affine_transform



class GTdata(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        
        self.datanum = len(data)
        
    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        return out_data, out_label

class AffineTransformation(object):
    def __init__(self, flip, tx, ty, k_90_rotate):
        self.flip = flip
        self.tx = tx
        self.ty = ty
        self.k_90_rotate = k_90_rotate

    def __call__(self, x):
        res_x = x
        if self.flip:
            res_x = np.fliplr(res_x)
        if self.tx != 0 or self.ty != 0:
            #res_x = apply_affine_transform(res_x, tx=self.tx, ty=self.ty, channel_axis=-1, fill_mode='reflect')
            res_x = apply_affine_transform(res_x, tx=self.tx, ty=self.ty, channel_axis=2, fill_mode='reflect')
        if self.k_90_rotate != 0:
            res_x = np.rot90(res_x, self.k_90_rotate)

        return res_x


class AbstractTransformer(abc.ABC):
    def __init__(self):
        self._transformation_list = None
        self._create_transformation_list()

    @property
    def n_transforms(self):
        return len(self._transformation_list)

    @abc.abstractmethod
    def _create_transformation_list(self):
        return

    def transform_batch(self, x_batch, t_inds):
        assert len(x_batch) == len(t_inds)

        transformed_batch = x_batch.copy()
        for i, t_ind in enumerate(t_inds):
            transformed_batch[i] = self._transformation_list[t_ind](transformed_batch[i])
        return transformed_batch


class Transformer(AbstractTransformer):
    def __init__(self, translation_x=8, translation_y=8):
        self.max_tx = translation_x
        self.max_ty = translation_y
        super().__init__()

    def _create_transformation_list(self):
        transformation_list = []
        for is_flip, tx, ty, k_rotate in itertools.product((False, True),
                                                           (0, -self.max_tx, self.max_tx),
                                                           (0, -self.max_ty, self.max_ty),
                                                           range(4)):
            transformation = AffineTransformation(is_flip, tx, ty, k_rotate)
            transformation_list.append(transformation)

        self._transformation_list = transformation_list

class SimpleTransformer(AbstractTransformer):
    def _create_transformation_list(self):
        transformation_list = []
        for is_flip, k_rotate in itertools.product((False, True),
                                                    range(4)):
            transformation = AffineTransformation(is_flip, 0, 0, k_rotate)
            transformation_list.append(transformation)

        self._transformation_list = transformation_list

def calc_approx_alpha_sum(observations):
    N = len(observations)
    f = np.mean(observations, axis=0)

    return (N * (len(f) - 1) * (-psi(1))) / (
            N * np.sum(f * np.log(f)) - np.sum(f * np.sum(np.log(observations), axis=0)))

def inv_psi(y, iters=5):
    # initial estimate
    cond = y >= -2.22
    x = cond * (np.exp(y) + 0.5) + (1 - cond) * -1 / (y - psi(1))

    for _ in range(iters):
        x = x - (psi(x) - y) / polygamma(1, x)
    return x

def fixed_point_dirichlet_mle(alpha_init, log_p_hat, max_iter=1000):
    alpha_new = alpha_old = alpha_init
    for _ in range(max_iter):
        alpha_new = inv_psi(psi(np.sum(alpha_old)) + log_p_hat)
        if np.sqrt(np.sum((alpha_old - alpha_new) ** 2)) < 1e-9:
            break
        alpha_old = alpha_new
    return alpha_new

def dirichlet_normality_score(alpha, p):
    return np.sum((alpha - 1) * np.log(p), axis=-1)


class GTTrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)
        
        # Optimization parameters
        self.eps = 1e-6

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()
        
        # Set device for network
        net = net.to(self.device)

        # Geometric transformation
        self.transformer = Transformer(8, 8)

        x_train = dataset.train_set.data
        y_train = dataset.train_set.targets

        print(y_train.shape)
        print(x_train.shape)
        x_train_task = x_train[y_train == 0]

        transformations_inds = np.tile(np.arange(self.transformer.n_transforms), len(x_train_task))
        x_train_task_transformed = self.transformer.transform_batch(np.repeat(x_train_task, self.transformer.n_transforms, axis=0), transformations_inds)

        self.softmax = nn.Softmax(dim=1)

        dataset = GTdata(x_train_task_transformed, transformations_inds)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle = True, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0)

        # Training
        loss_func = nn.CrossEntropyLoss()
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            
            for data in train_loader:
                inputs, targets = data

                inputs = np.float64(inputs)
                inputs = np.transpose(inputs, (0, 3, 1, 2))/255

                inputs = torch.FloatTensor(inputs)
                targets = torch.LongTensor(targets)

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)

                loss = loss_func(outputs, targets)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Get test data loader

        x_train = dataset.train_set.data
        y_train = dataset.train_set.targets

        x_train_task = x_train[y_train == 0]
        y_train_task = y_train[y_train == 0]

        x_test = dataset.test_set.data
        labels = dataset.test_set.targets

        train_dataset = GTdata(x_train_task, y_train_task)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs_dataloader)

        test_dataset = GTdata(x_test, labels)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        
        scores = np.zeros(len(x_test))

        with torch.no_grad():

            for t_ind in range(self.transformer.n_transforms):

                init = True

                for data in train_loader:
                    inputs, targets = data

                    inputs = np.float64(inputs)
                    inputs = np.transpose(self.transformer.transform_batch(inputs, [t_ind]*len(inputs)), (0, 3, 1, 2))/255
                    inputs = torch.FloatTensor(inputs)

                    inputs = inputs.to(self.device)
                    outputs = net(inputs)
                    outputs = self.softmax(outputs)
                    #outputs = torch.exp(outputs)
                    #outputs = outputs/torch.sum(outputs, axis=0)

                    observed_dirichlet_temp = outputs.cpu().detach().numpy()
                    
                    if init:
                        observed_dirichlet = observed_dirichlet_temp
                        init = False
                    else:
                        observed_dirichlet = np.append(observed_dirichlet, observed_dirichlet_temp, axis=0)

                log_p_hat_train = np.log(observed_dirichlet).mean(axis=0)

                alpha_sum_approx = calc_approx_alpha_sum(observed_dirichlet)
                alpha_0 = observed_dirichlet.mean(axis=0) * alpha_sum_approx
                
                #bar_s = np.mean(observed_dirichlet, axis=0)
                #bar_l = np.mean(log_p_hat_train, axis=0)

                #alpha_0 = bar_s*((self.transformer.n_transforms-1)*(-psi(1)))/(np.dot(bar_s, np.log(bar_s))-np.dot(bar_s, bar_l))

                mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, log_p_hat_train)

                init = True

                for data in test_loader:
                    inputs, targets = data

                    inputs = np.float64(inputs)
                    inputs = np.transpose(self.transformer.transform_batch(inputs, [t_ind]*len(inputs)), (0, 3, 1, 2))/255
                    inputs = torch.FloatTensor(inputs)
                    
                    inputs = inputs.to(self.device)

                    outputs = net(inputs)
                    outputs = self.softmax(outputs)
                    #outputs = torch.exp(outputs)
                    #outputs = outputs/torch.sum(outputs, axis=0)

                    x_test_p_temp = outputs.cpu().detach().numpy()
                    
                    if init:
                        x_test_p = x_test_p_temp
                        init = False
                    else:
                        x_test_p = np.append(x_test_p, x_test_p_temp, axis=0)

                scores += dirichlet_normality_score(mle_alpha_t, x_test_p)

        scores /= self.transformer.n_transforms
        scores = - scores

        self.test_time = time.time() - start_time

        # Compute AUC
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)

        # Log results
        #logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing.')
