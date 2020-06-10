from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset

import torch
import torchvision.transforms as transforms
import random
import numpy as np

class MNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int = 0):
        super().__init__(root)

        # MNIST preprocessing: feature scaling to [0, 1]
        transform = transforms.ToTensor()

        # Get train set
        self.train_set = MyMNIST(root=self.root, train=True, transform=transform, download=True)

        self.test_set = MyMNIST(root=self.root, train=False, transform=transform, download=True)
        
        normal = self.train_set.data[self.train_set.targets==normal_class]
        anomaly = self.test_set.data

        self.train_set.data = torch.cat((normal, anomaly), 0)
        semi_targets = torch.ones(len(self.train_set))
        semi_targets[:len(normal)] = 0
        self.train_set.semi_targets = semi_targets
        self.train_set.targets = semi_targets

        targets = np.array(self.test_set.targets)
        targets_temp = targets.copy()
        targets[targets_temp == normal_class] = 0
        targets[targets_temp != normal_class] = 1
        self.test_set.targets = targets


class MyMNIST(MNIST):
    """
    Torchvision MNIST class with additional targets for the semi-supervised setting and patch of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    def __init__(self, *args, **kwargs):
        super(MyMNIST, self).__init__(*args, **kwargs)
        
        self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
        img, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, semi_target, index
