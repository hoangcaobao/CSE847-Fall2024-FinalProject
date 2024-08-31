import os
from configs.default_configs import get_default_configs

def init_cfg(cfg_proj):
    n_solver = cfg_proj.solver
    config = get_default_configs()

    if cfg_proj.dataset_name in ["MNIST", "CIFAR10"]:
        config.data.dim_out = 10
    
    if cfg_proj.dataset_name == "MNITS":
        config.data.in_channels = 1
    else:
        config.data.in_channels = 3

    if n_solver in ["Standard_solver"]:
        config.training.epochs = 20
        config.training.batch_size = 512
        config.training.lr_init = 1.0e-3
        config.training.tol = 1e-4
   
    return config
