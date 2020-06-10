import click
import torch
import logging
import random
import numpy as np

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from D3RE_uLSIF import D3REuLSIF
from datasets.main import load_dataset


################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'cifar10', 'fmnist']))
@click.argument('net_name', type=click.Choice(['mnist_LeNet', 'cifar10_LeNet', 'fmnist_LeNet']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--upper_bound', type=float, default=10, help='Deep DR hyperparameter pi (must be 0 < pi < 1).')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for Deep DR network training.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for Deep DR network training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep DR objective.')
@click.option('--num_threads', type=int, default=0,
              help='Number of threads used for parallelizing CPU operations. 0 means that all resources are used.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
def main(dataset_name, net_name, xp_path, data_path, load_config, load_model, upper_bound,
         device, seed, optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay,
         num_threads, n_jobs_dataloader, normal_class):
    """
    Deep DR, a method for deep semi-supervised anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print paths
    logger.info('Log file is %s' % log_file)
    logger.info('Data path is %s' % data_path)
    logger.info('Export path is %s' % xp_path)

    # Print experimental setup
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print model configuration
    logger.info('Pi-parameter: %.2f' % cfg.settings['upper_bound'])

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        torch.cuda.manual_seed(cfg.settings['seed'])
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    # Set the number of threads used for parallelizing CPU operations
    if num_threads > 0:
        torch.set_num_threads(num_threads)

    logger.info('Computation device: %s' % device)
    logger.info('Number of threads: %d' % num_threads)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class, random_state=np.random.RandomState(cfg.settings['seed']))

    # Initialize DeepSAD model and set neural network phi
    D3RE = D3REuLSIF(cfg.settings['upper_bound'])
    D3RE.set_network(net_name, rep_dim=1)

    # If specified, load Deep DR model (center c, network weights, and possibly autoencoder weights)
    if load_model:
        D3RE.load_model(model_path=load_model, load_ae=True, map_location=device)
        logger.info('Loading model from %s.' % load_model)

    # Log training details
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    # Train model on dataset
    D3RE.train(dataset,
                  optimizer_name=cfg.settings['optimizer_name'],
                  lr=cfg.settings['lr'],
                  n_epochs=cfg.settings['n_epochs'],
                  lr_milestones=cfg.settings['lr_milestone'],
                  batch_size=cfg.settings['batch_size'],
                  weight_decay=cfg.settings['weight_decay'],
                  device=device,
                  n_jobs_dataloader=n_jobs_dataloader)

    # Test model
    D3RE.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # Save results, model, and configuration
    D3RE.save_results(export_json=xp_path + '/results.json')
    D3RE.save_model(export_model=xp_path + '/model.tar')
    cfg.save_config(export_json=xp_path + '/config.json')


if __name__ == '__main__':
    main()
