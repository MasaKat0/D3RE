from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .fmnist import FashionMNIST_Dataset


def load_dataset(dataset_name, data_path, normal_class, random_state=None):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'fmnist', 'cifar10')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'fmnist':
        dataset = FashionMNIST_Dataset(root=data_path, normal_class=normal_class)

    return dataset
