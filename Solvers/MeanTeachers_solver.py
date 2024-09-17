from Solvers.Solver_Base import Solver_Base
from Models.model import Conv2DModel
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from tqdm import tqdm

class MeanTeachers_solver(Solver_Base):
    
    def __init__(self, cfg_proj, cfg_m, name = "Std"):
        Solver_Base.__init__(self, cfg_proj, cfg_m, name)
        
    def run(self, train_labeled_loader, train_unlabeled_loader, test_loader):
        self.set_random_seed(self.cfg_proj.seed)
        
        # Initialize the model, loss function, and optimizer
        model = self.train(train_labeled_loader, train_unlabeled_loader, test_loader)

        acc = self.eval_func(model, test_loader)

        return acc

    def train(self, train_labeled_loader, train_unlabeled_loader, test_loader, alpha = 0.95):
        teacher_model = Conv2DModel(dim_out=self.cfg_m.data.dim_out, in_channels=self.cfg_m.data.in_channels)
        student_model = Conv2DModel(dim_out=self.cfg_m.data.dim_out, in_channels=self.cfg_m.data.in_channels)

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
            
            print(f'Epoch [{epoch+1}/{self.cfg_m.training.epochs}], Loss: {np.mean(epoch_loss):.4f}, Accuracy: {self.eval_func(student_model, test_loader)}')

        return teacher_model
