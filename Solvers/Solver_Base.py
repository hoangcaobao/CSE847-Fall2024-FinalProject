import random
import torch
import numpy as np
import torch.nn as nn

class Solver_Base:
    
    def __init__(self, cfg_proj, cfg_m, name):
        self.name = name
        self.cfg_proj = cfg_proj
        self.cfg_m = cfg_m
        self.acc = 0
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def eval_func(self, model, test_loader):
        # Testing loop
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total
    
    def to_parallel_model(self, model):
        if torch.cuda.device_count() > 1:
            # print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
            model = model.to(self.device)
            return model, model.module
        else:
            model = model.to(self.device)
            return model, model
        
    def set_random_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available(): 
            torch.cuda.manual_seed(seed)
            if torch.cuda.device_count() > 1: torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.seed_current = seed

    def predict(self, model, X):
        pass 

    def predict_proba(self, model, X):

        model.eval()
        with torch.no_grad():
            X = X.float().to(self.device)
            pred = model(X)
            pred = torch.nn.functional.softmax(pred, dim = 1)
        
        return pred.detach().cpu().numpy()   
      
    def basic_train(self, model, dataloader_train, criterion, optimizer, test_loader):
        
        model = model.to(self.device)
        # Training loop with validation
        for epoch in range(self.cfg_m.training.epochs):
            model.train()

            epoch_loss = []

            for images, labels in dataloader_train:
                # Forward pass
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                epoch_loss.append(loss.item())

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc = self.eval_func(model, test_loader)
            self.acc = max(acc, self.acc)
            if epoch % 5 == 0:
                print(f'Epoch [{epoch+1}/{self.cfg_m.training.epochs}], Loss: {np.mean(epoch_loss):.4f}, Accuracy: {acc}')

        return model
    
    def freeze_grad(self, model, except_full_names = [None], except_str = [None]):
        for n, para in model.named_parameters():
            para.requires_grad = False
        for n, para in model.named_parameters():
            for f_n in except_full_names: 
                if f_n == n: para.requires_grad = True
            for s in except_str:
                if s is not None:
                    if s in n: para.requires_grad = True
        return model    