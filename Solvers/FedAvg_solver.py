import numpy as np
from Solvers.Solver_Base import Solver_Base
import sys
from Models.model_loader import model_loader
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

import torch.distributed as dist
import torch.multiprocessing as mp 
from torch.nn.parallel import DistributedDataParallel as DDP

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
            # Stack the tensors for the specific parameter across all local models
            stacked_tensors = torch.stack([model.state_dict()[key] for model in local_models])

            # Ensure the tensor is in floating-point type
            if not stacked_tensors.is_floating_point():
                stacked_tensors = stacked_tensors.float()

            # Compute the mean
            global_state_dict[key] = stacked_tensors.mean(dim=0)

        # Load the averaged state_dict back into the global model
        global_model.load_state_dict(global_state_dict)

        return global_model

    def run(self, train_labeled_loader, train_unlabeled_loader, test_loader):
        self.set_random_seed(self.cfg_proj.seed)

        if self.cfg_proj.solver == "SelfTraining_solver":
            return self.run_selfTraining(train_labeled_loader, train_unlabeled_loader, test_loader)    
            
        local_solver = local_solver_loader(self.cfg_proj, self.cfg_m)

        # Initialize the model, loss function, and optimizer
        if self.cfg_proj.solver == "MeanTeachers_solver":
            global_model = [model_loader(self.cfg_proj, self.cfg_m), model_loader(self.cfg_proj, self.cfg_m)]
        else:
            global_model = model_loader(self.cfg_proj, self.cfg_m)
            
        acc_max = 0

        for iter in range(self.cfg_m.training.globalEpochs):
            local_models = []
            for i in range(len(train_labeled_loader)):
                local_models.append(local_solver.run(train_labeled_loader[i], train_unlabeled_loader[i], test_loader, model = copy.deepcopy(global_model)))
            if self.cfg_proj.solver == "MeanTeachers_solver":
                global_model[0] = self.fedavg([local_model[0] for local_model in local_models])
                global_model[1] = self.fedavg([local_model[1] for local_model in local_models])
                acc_max = max(self.eval_func(global_model[0], test_loader), acc_max)
            else:
                global_model = self.fedavg(local_models)
                acc_max = max(self.eval_func(global_model, test_loader), acc_max)
        return acc_max
    
    def run_selfTraining(self, train_labeled_loader, train_unlabeled_loader, test_loader):
        local_solvers = [local_solver_loader(self.cfg_proj, self.cfg_m) for i in range(len(train_unlabeled_loader))]
        global_model = model_loader(self.cfg_proj, self.cfg_m)
        for i in range(len(local_solvers)):
            local_solvers[i].run(train_labeled_loader[i], train_unlabeled_loader[i], test_loader, model = copy.deepcopy(global_model))
        acc_max = 0
        
        for _ in range(10):
            global_model = model_loader(self.cfg_proj, self.cfg_m)
            for iter in range(self.cfg_m.training.globalEpochs):
                local_models = []
                for i in range(len(local_solvers)):
                    local_models.append(local_solvers[i].basic_train(model = copy.deepcopy(global_model), train_labeled_loader = local_solvers[i].train_labeled_loader, train_pseudo_labeled_loader = local_solvers[i].train_pseudo_labeled_loader))
                
                global_model = self.fedavg(local_models)
                acc_max = max(self.eval_func(global_model, test_loader), acc_max)
                
            for i in range(len(local_solvers)):
                local_solvers[i].add_pseudo(copy.deepcopy(global_model))
        return acc_max