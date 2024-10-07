import numpy as np
from Solvers.Solver_Base import Solver_Base
import torch
from Models.model import Conv2DModel
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class Standard_solver(Solver_Base):
    
    def __init__(self, cfg_proj, cfg_m, name = "Std"):
        Solver_Base.__init__(self, cfg_proj, cfg_m, name)
        
    def run(self, train_labeled_loader, train_unlabeled_loader, test_loader):
        self.set_random_seed(self.cfg_proj.seed)
        
        # Initialize the model, loss function, and optimizer
        model = Conv2DModel(dim_out=self.cfg_m.data.dim_out, in_channels=self.cfg_m.data.in_channels, dataset_name=self.cfg_proj.dataset_name)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.cfg_m.training.lr_init)

        model = self.basic_train(model, train_labeled_loader, criterion, optimizer, test_loader)

        acc = self.eval_func(model, test_loader)

        return acc