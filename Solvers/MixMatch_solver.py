from Solvers.Solver_Base import Solver_Base 
from Models.model import Conv2DModel
import torch.nn as nn 
import torch.optim as optim 
import torch 
import numpy as np 
from Data.DataPreProcessing import get_cifar10
from torch.utils.data import DataLoader
from tqdm import tqdm 
import torch.nn.functional as F 


class MixMatch_solver(Solver_Base):
    def __init__(self, cfg_proj, cfg_m, name="MixMatch"):
        Solver_Base.__init__(self, cfg_proj, cfg_m, name)
        
    def run(self, train_labeled_loader, train_unlabeled_loader, test_loader):
        model = Conv2DModel(dim_out=self.cfg_m.data.dim_out, in_channels=self.cfg_m.data.in_channels)
        optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        
        num_epochs = 200
        best_acc = 0.0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        for epoch in range(num_epochs):
            model.train()
            
            labeled_iter = iter(train_labeled_loader)
            unlabeled_iter = iter(train_unlabeled_loader)
            
            for batch_idx in tqdm(range(len(train_labeled_loader))):
                try:
                    inputs_x, targets_x = next(labeled_iter)
                except:
                    labeled_iter = iter(train_labeled_loader)
                    inputs_x, targets_x = next(labeled_iter)
                    
                try:
                    inputs_u, _ = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(train_unlabeled_loader)
                    inputs_u, _ = next(unlabeled_iter)
                    
                inputs_x, targets_x = inputs_x.to(device), targets_x.to(device)
                inputs_u = inputs_u.to(device)
                
                batch_size = inputs_x.shape[0]
                
                #Generate guessed labels for unlabeled data
                with torch.no_grad():
                    logits_u = model(inputs_u)
                    pseudo_label_u = torch.softmax(logits_u, dim=1)
                    
                #MixUp: combine labeled and unlabeled data using MixMatch's augmentation
                mixed_input, mixed_target = self.mixup(inputs_x, targets_x, inputs_u, pseudo_label_u)
                
                #Forward pass
                logits = model(mixed_input)
                logits_x = logits[:batch_size]
                logits_u = logits[batch_size:]
                
                Lx = F.cross_entropy(logits_x, targets_x)
                Lu = F.mse_loss(torch.softmax(logits_u, dim=1), mixed_target[batch_size:])
                
                loss = Lx + Lu 
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
            acc = self.eval_func(model, test_loader)
            print(acc)
            
        return model
    
    def mixup(self, inputs_x, targets_x, inputs_u, targets_u, alpha=0.75):
        all_inputs = torch.cat([inputs_x, inputs_u], dim=0)
        all_targets = torch.cat([targets_x, targets_u], dim=0)
        
        lam = np.random.beta(alpha, alpha)
        batch_size = inputs_x.shape[0]
        index = torch.randperm(all_inputs.size(0)).to(inputs_x.device)
        
        input_a, input_b = all_inputs, all_inputs[index]
        target_a, target_b = all_targets, all_targets[index]
        
        mixed_input = lam * input_a + (1 - lam) * input_b 
        mixed_target = lam * target_a + (1 - lam) * target_b 
        
        return mixed_input, mixed_target
                    
                    