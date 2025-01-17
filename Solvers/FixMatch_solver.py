from Solvers.Solver_Base import Solver_Base
from Models.model import Conv2DModel 
import torch.nn as nn 
import torch.optim as optim 
import torch 
import numpy as np 
from Data.DataPreProcessing import get_cifar10
from torch.utils.data import TensorDataset, DataLoader, random_split 
from tqdm import tqdm
from Models.model_loader import model_loader

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
            model = model_loader(self.cfg_proj, self.cfg_m)
        
        optimizer = optim.Adam(model.parameters(), lr=self.cfg_m.training.lr_init, weight_decay=5e-4)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        # scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_epochs)
        ema_model = None
        
        # import pdb; pdb.set_trace()
        best_acc = 0.0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        for epoch in range(5):
            model.train()
            
            labeled_iter = iter(train_labeled_loader)
            unlabeled_iter = iter(train_unlabeled_loader)
            
            for batch_idx in tqdm(range(len(train_labeled_loader))):
                try:
                    inputs_x, targets_x = next(labeled_iter)
                except:
                    print(batch_idx)
                    labeled_iter = iter(train_labeled_loader)
                    inputs_x, targets_x = next(labeled_iter)
                # inputs_x, targets_x = next(labeled_iter)
                    
                try:
                    (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(train_unlabeled_loader)
                    (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
                    
                inputs_x, targets_x = inputs_x.to(device), targets_x.to(device)
                inputs_u_w, inputs_u_s = inputs_u_w.to(device), inputs_u_s.to(device)
                
                batch_size = inputs_x.shape[0]
                inputs = torch.cat([inputs_x, inputs_u_w, inputs_u_s], dim=0)
                logits = model(inputs)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                # logits_x = model(inputs_x)
                
                Lx = nn.CrossEntropyLoss()(logits_x, targets_x)
                pseudo_label = torch.softmax(logits_u_w.detach() / 1.0, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(0.95).float()
                
                Lu = (nn.CrossEntropyLoss(reduction='none')(logits_u_s, targets_u)*mask).mean()
                
                loss = Lx + Lu
                # loss = Lx
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()
            
            acc = self.eval_func(model, test_loader)
            best_acc = max(best_acc, acc)
            # print(best_acc)
        
        return model