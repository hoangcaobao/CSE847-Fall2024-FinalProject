import numpy as np
from Solvers.Solver_Base import Solver_Base
import sys

from Solvers.Standard_solver import Standard_solver
from Solvers.SelfTraining_solver import SelfTraining_solver
from Solvers.FixMatch_solver import FixMatch_solver
from Solvers.MeanTeachers_solver import MeanTeachers_solver
from Solvers.MixMatch_solver import MixMatch_solver

import torch
from Models.model import Conv2DModel
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import copy 

def local_solver_loader(cfg_proj, cfg_m):
    s = None
    
    if cfg_proj.solver == "Standard_solver":
        s = Standard_solver(cfg_proj, cfg_m)
    if cfg_proj.solver == "SelfTraining_solver":
        s = SelfTraining_solver(cfg_proj, cfg_m)
    if cfg_proj.solver == "FixMatch_solver":
        s = FixMatch_solver(cfg_proj, cfg_m)
    if cfg_proj.solver == "MeanTeachers_solver":
        s = MeanTeachers_solver(cfg_proj, cfg_m)
    if cfg_proj.solver == "MixMatch_solver":
        s = MixMatch_solver(cfg_proj, cfg_m)
  
    return s

class FedAvg_solver(Solver_Base):
    
    def __init__(self, cfg_proj, cfg_m, name = "Std"):
        Solver_Base.__init__(self, cfg_proj, cfg_m, name)
    
    def fedavg(self, local_models):
        global_model = local_models[0]

        global_state_dict = global_model.state_dict()

        for key in global_state_dict.keys():
            global_state_dict[key] = torch.stack([model.state_dict()[key] for model in local_models]).mean(dim=0)

        global_model.load_state_dict(global_state_dict)

        return global_model

    def run(self, train_labeled_loader, train_unlabeled_loader, test_loader):
        self.set_random_seed(self.cfg_proj.seed)
        
        local_solver = local_solver_loader(self.cfg_proj, self.cfg_m)

        # Initialize the model, loss function, and optimizer
        global_model = Conv2DModel(dim_out=self.cfg_m.data.dim_out, in_channels=self.cfg_m.data.in_channels, dataset_name=self.cfg_proj.dataset_name)
        
        for iter in range(40):
            local_models = []
            for i in range(len(train_labeled_loader)):
                local_models.append(local_solver.run(train_labeled_loader[i], train_unlabeled_loader[i], test_loader, model = copy.deepcopy(global_model)))
            global_model = self.fedavg(local_models)

        acc = self.eval_func(global_model, test_loader)

        return acc