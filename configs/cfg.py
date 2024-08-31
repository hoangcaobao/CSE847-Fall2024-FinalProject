import os
from configs.default_configs import get_default_configs

def init_cfg(cfg_proj):
    n_solver = cfg_proj.solver
    config = get_default_configs()

    if cfg_proj.dataset_name == "MNIST":
        config.data.dim_out = 10

    if n_solver in ["Standard_solver"]:
        config.training.epochs = 10
        config.training.batch_size = 512
        config.training.lr_init = 1.0e-3
        config.training.tol = 1e-4
   
    return config
