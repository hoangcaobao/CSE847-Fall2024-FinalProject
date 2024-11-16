import os
from configs.default_configs import get_default_configs

def init_cfg(cfg_proj):
    n_solver = cfg_proj.solver
    config = get_default_configs()

    if cfg_proj.dataset_name in ["MNIST", "CIFAR10", "STL10"]:
        config.data.dim_out = 10
    elif cfg_proj.dataset_name in ["Cat_and_Dog"]:
        config.data.dim_out = 2
    
    if cfg_proj.dataset_name == "MNITS":
        config.data.in_channels = 1
    else:
        config.data.in_channels = 3

    config.training.epochs = 100
    config.training.batch_size = 128
    config.training.lr_init = 1.0e-3
    config.training.tol = 1e-4

    if n_solver in ['FixMatch_solver', 'MixMatch_solver']:
        config.training.batch_size = 64
        config.training.epochs = 200
    
    if cfg_proj.numberOfClients > 1:
        config.training.epochs = 5
        config.training.globalEpochs = 40

    return config
