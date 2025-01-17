from Solvers.Solver_Base import Solver_Base 
from Models.model import Conv2DModel
import torch.nn as nn 
import torch.optim as optim 
import torch 
from Models.model_loader import model_loader
import numpy as np 
from Data.DataPreProcessing import get_cifar10
from torch.utils.data import DataLoader
from tqdm import tqdm 
import torch.nn.functional as F 


class MixMatch_solver(Solver_Base):
    def __init__(self, cfg_proj, cfg_m, name="MixMatch"):
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
        
        # num_epochs = 100
        best_acc = 0.0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        criterion = SemiLoss()
        
        for epoch in range(self.cfg_m.training.epochs):
            model.train()
            
            labeled_iter = iter(train_labeled_loader)
            unlabeled_iter = iter(train_unlabeled_loader)
            
            train_iteration = len(train_unlabeled_loader) + 50
            
            for batch_idx in tqdm(range(train_iteration)):
                
                try:
                    inputs_x, targets_x = next(labeled_iter)
                except:
                    labeled_iter = iter(train_labeled_loader)
                    inputs_x, targets_x = next(labeled_iter)
                    
                try:
                    (inputs_u, inputs_u2), _ = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(train_unlabeled_loader)
                    (inputs_u, inputs_u2), _ = next(unlabeled_iter)
                    
                batch_size = inputs_x.shape[0]
                targets_x = torch.zeros(batch_size, self.cfg_m.data.dim_out, device=targets_x.device).scatter_(1, targets_x.view(-1,1).long(), 1)
                
                inputs_x, targets_x = inputs_x.to(device), targets_x.to(device)
                inputs_u, inputs_u2 = inputs_u.to(device), inputs_u2.to(device)
                
                #Generate guessed labels for unlabeled data
                with torch.no_grad(): 
                    outputs_u = model(inputs_u)
                    outputs_u2 = model(inputs_u2)
                    p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
                    pt = p**(1/0.5)
                    targets_u = pt / pt.sum(dim=1, keepdim=True)
                    targets_u = targets_u.detach()
                
                all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
                all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)
                
                alpha = 0.75
                l = np.random.beta(alpha, alpha)
                l = max(l, 1 - l)
                
                idx = torch.randperm(all_inputs.size(0)).to(device)
                input_a, input_b = all_inputs, all_inputs[idx]
                targets_a, targets_b = all_targets, all_targets[idx]
                mixed_input = l * input_a + (1 - l) * input_b
                mixed_target = l * targets_a + (1 - l) * targets_b
                
                mixed_input = list(torch.split(mixed_input, batch_size))
                mixed_input = self.interleave(mixed_input, batch_size)
                
                
                #Forward pass
                logits = [model(mixed_input[0])]
                for inp in mixed_input[1:]:
                    logits.append(model(inp))
                
                logits = self.interleave(logits, batch_size)
                logits_x = logits[0]
                logits_u = torch.cat(logits[1:], dim=0)
                
                Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch + batch_idx/train_iteration)
                # print(w, " ", batch_idx)
                loss = Lx + w*Lu
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()
                
            acc = self.eval_func(model, test_loader)
            best_acc = max(best_acc, acc)
            # print(best_acc)
            
        return model

    @staticmethod
    def interleave_offsets(batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch 
        return offsets
        
    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p] : offsets[p+1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
            
        return [torch.cat(v, dim=0) for v in xy]

def linear_rampup(current, rampup_length=200):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current) 
   
class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)
        
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lx, Lu, 75 * linear_rampup(epoch)
        
                    
                    