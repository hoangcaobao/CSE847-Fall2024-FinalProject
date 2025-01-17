from Solvers.Solver_Base import Solver_Base
from Models.model import Conv2DModel
from Models.model_loader import model_loader
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from tqdm import tqdm
import copy 

class MeanTeachers_solver(Solver_Base):
    
    def __init__(self, cfg_proj, cfg_m, name = "Std"):
        Solver_Base.__init__(self, cfg_proj, cfg_m, name)
        
    def run(self, train_labeled_loader, train_unlabeled_loader, test_loader, model = None):
        self.set_random_seed(self.cfg_proj.seed)
        
        local_train = True if model else False

        # Initialize the model, loss function, and optimizer
        model = self.train(train_labeled_loader, train_unlabeled_loader, test_loader, model = model)

        if local_train:
            return model
        
        return self.acc

    def train(self, train_labeled_loader, train_unlabeled_loader, test_loader, alpha = 0.99, model = None):
        
        if not model:
            teacher_model = model_loader(self.cfg_proj, self.cfg_m)
            student_model = model_loader(self.cfg_proj, self.cfg_m)

        else:
            teacher_model = model[0]
            student_model = model[1]

        teacher_model = teacher_model.to(self.device)
        student_model = student_model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(student_model.parameters(), lr=self.cfg_m.training.lr_init)

        labeled_iter = iter(train_labeled_loader)
        unlabeled_iter = iter(train_unlabeled_loader)
        
        # Training loop with validation
        for epoch in range(self.cfg_m.training.epochs):
            epoch_loss = []
            student_model.train()
            teacher_model.train()

            for batch_idx in tqdm(range(len(train_labeled_loader))):
                loss = 0

                try:
                    labeled_images, labels = next(labeled_iter)
                except:
                    labeled_iter = iter(train_labeled_loader)
                    labeled_images, labels = next(labeled_iter)
                    
                try:
                    unlabeled_images, _ = next(unlabeled_iter)
                except:
                    unlabeled_iter = iter(train_unlabeled_loader)
                    unlabeled_images, _ = next(unlabeled_iter)

                labeled_images, unlabeled_images, labels = labeled_images.to(self.device), unlabeled_images.to(self.device), labels.to(self.device)
                loss = 0

                loss += criterion(student_model(labeled_images), labels)
                with torch.no_grad():
                    epoch_loss.append(criterion(teacher_model(labeled_images), labels).item())
                
                with torch.no_grad():
                    teacher_output = teacher_model(unlabeled_images)
                student_output = student_model(unlabeled_images)

                loss += torch.mean((teacher_output - student_output) ** 2)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update teacher model weights with EMA
                for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
                    teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data
            
            acc = self.eval_func(teacher_model, test_loader)
            self.acc = max(self.acc, acc)

            print(f'Epoch [{epoch+1}/{self.cfg_m.training.epochs}], Loss: {np.mean(epoch_loss):.4f}, Accuracy: {acc}')

        return [teacher_model, student_model]
