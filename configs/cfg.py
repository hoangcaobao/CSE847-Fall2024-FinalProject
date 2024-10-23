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

    if n_solver in ["Standard_solver", "SelfTraining_solver", "MeanTeachers_solver"]:
        config.training.epochs = 200
        if n_solver == "MeanTeachers_solver" and cfg_proj.dataset_name == "Cat_and_Dog":
            config.training.epochs = 50
        config.training.batch_size = 32
        config.training.lr_init = 1.0e-3
        config.training.tol = 1e-4
    
    if cfg_proj.numberOfClients > 1:
        config.training.epochs = 5

    return config
