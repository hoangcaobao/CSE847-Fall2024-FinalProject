import numpy as np
from Solvers.Solver_Base import Solver_Base
import torch
# from Models.model import Conv2DModel, ResNet50, VGG16
from Models.model import Conv2DModel, ResNet50
# from Models.ViT import ViT
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class Standard_solver(Solver_Base):
    
    def __init__(self, cfg_proj, cfg_m, name = "Std"):
        Solver_Base.__init__(self, cfg_proj, cfg_m, name)
        
    def run(self, train_labeled_loader, train_unlabeled_loader, test_loader, model = None):
        self.set_random_seed(self.cfg_proj.seed)
        
        local_train = True if model else False

        # Initialize the model, loss function, and optimizer
        if not model:
            
            if self.cfg_proj.model == "simpleCNN":
                model = Conv2DModel(dim_out=self.cfg_m.data.dim_out, in_channels=self.cfg_m.data.in_channels, dataset_name=self.cfg_proj.dataset_name)
            if self.cfg_proj.model == "resnet50":
                model = ResNet50(dim_out=self.cfg_m.data.dim_out, in_channels=self.cfg_m.data.in_channels, dataset_name=self.cfg_proj.dataset_name)
            # model = ViT(image_size = 32, patch_size = 4, num_classes = 10, dim = 512, depth = 6, heads = 8, mlp_dim = 512, dropout = 0.1, emb_dropout = 0.1)
            # model = VGG16(dim_out=self.cfg_m.data.dim_out, in_channels=self.cfg_m.data.in_channels, dataset_name=self.cfg_proj.dataset_name) 
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.cfg_m.training.lr_init)
        
        # import pdb; pdb.set_trace()

        model = self.basic_train(model, train_labeled_loader, criterion, optimizer, test_loader)

        if local_train:
            return model
        
        acc = self.eval_func(model, test_loader)

        return acc