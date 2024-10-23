from Solvers.Solver_Base import Solver_Base
from Models.model import Conv2DModel 
import torch.nn as nn 
import torch.optim as optim 
import torch 
import numpy as np 
from Data.DataPreProcessing import get_cifar10
from torch.utils.data import TensorDataset, DataLoader, random_split 
from tqdm import tqdm
 

class FixMatch_solver(Solver_Base):
    def __init__(self, cfg_proj, cfg_m, name = "Std"):
        Solver_Base.__init__(self, cfg_proj, cfg_m, name)
            
    def run(self, train_labeled_loader, train_unlabeled_loader, test_loader, model = None):
        self.set_random_seed(self.cfg_proj.seed)
        
        local_train = True if model else False

        model = self.train(train_labeled_loader, train_unlabeled_loader, test_loader, model = model)
        
        if local_train:
            return model
        
        acc = self.eval_func(model, test_loader)
        
        return acc 
    
    def train(self, train_labeled_loader, train_unlabeled_loader, test_loader, model = None):
        if not model:
            model = Conv2DModel(dim_out=self.cfg_m.data.dim_out, in_channels=self.cfg_m.data.in_channels, dataset_name=self.cfg_proj.dataset_name)
    
        optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        ema_model = None
        
        
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
                    (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(train_unlabeled_loader)
                    (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
                    
                inputs_x, targets_x = inputs_x.to(device), targets_x.to(device)
                inputs_u_w, inputs_u_s = inputs_u_w.to(device), inputs_u_s.to(device)
                
                batch_size = inputs_x.shape[0]
                # import pdb; pdb.set_trace()
                inputs = torch.cat([inputs_x, inputs_u_w, inputs_u_s], dim=0)
                logits = model(inputs)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                
                Lx = nn.CrossEntropyLoss()(logits_x, targets_x)
                pseudo_label = torch.softmax(logits_u_w.detach() / 1.0, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(0.95).float()
                
                Lu = (nn.CrossEntropyLoss(reduction='none')(logits_u_s, targets_u)*mask).mean()
                
                loss = Lx + Lu
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            acc = self.eval_func(model, test_loader)
            print(acc)
        
        return model