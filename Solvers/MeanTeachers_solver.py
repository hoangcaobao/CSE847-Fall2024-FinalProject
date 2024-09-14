from Solvers.Solver_Base import Solver_Base
from Models.model import Conv2DModel
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split

class MeanTeachers_solver(Solver_Base):
    
    def __init__(self, cfg_proj, cfg_m, name = "Std"):
        Solver_Base.__init__(self, cfg_proj, cfg_m, name)
        
    def run(self, train_labeled_loader, train_unlabeled_loader, test_loader):
        self.set_random_seed(self.cfg_proj.seed)
        
        # Initialize the model, loss function, and optimizer
        model = self.train(train_labeled_loader, train_unlabeled_loader)

        acc = self.eval_func(model, test_loader)

        return acc

    def train(self, train_labeled_loader, train_unlabeled_loader, alpha = 0.99):
        teacher_model = Conv2DModel(dim_out=self.cfg_m.data.dim_out, in_channels=self.cfg_m.data.in_channels)
        student_model = Conv2DModel(dim_out=self.cfg_m.data.dim_out, in_channels=self.cfg_m.data.in_channels)

        teacher_model = teacher_model.to(self.device)
        student_model = student_model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(student_model.parameters(), lr=self.cfg_m.training.lr_init)

        images_dataset = []
        labels_dataset = []
        original_labeled_dataset = []

        for images, labels in train_labeled_loader:
            images_dataset.append(images)
            labels_dataset.append(labels)
            original_labeled_dataset.append(torch.tensor([1 for _ in range(images.shape[0])]))

        for images, labels in train_unlabeled_loader:
            images_dataset.append(images)
            labels_dataset.append(labels)
            original_labeled_dataset.append(torch.tensor([0 for _ in range(images.shape[0])]))

        images_dataset = torch.cat(images_dataset, dim = 0)
        labels_dataset = torch.cat(labels_dataset, dim = 0)    
        original_labeled_dataset = torch.cat(original_labeled_dataset, dim = 0) 
        
        dataloader = DataLoader(dataset=TensorDataset(images_dataset, labels_dataset, original_labeled_dataset), batch_size=self.cfg_m.training.batch_size, shuffle=True)

        # Training loop with validation
        for epoch in range(self.cfg_m.training.epochs):
            epoch_loss = []
            student_model.train()
            teacher_model.train()

            for images, labels, original in dataloader:
                images, labels, original = images.to(self.device), labels.to(self.device), original.to(self.device)
                
                loss = 0

                if torch.any(original == 1):
                    loss += criterion(student_model(images[original == 1]), labels[original == 1])
                    with torch.no_grad():
                        epoch_loss.append(criterion(teacher_model(images[original == 1]), labels[original == 1]).item())

                if torch.any(original == 0):
                    with torch.no_grad():
                        teacher_output = teacher_model(images[original == 0])
                    student_output = student_model(images[original == 0])

                    loss += torch.mean((teacher_output - student_output) ** 2)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update teacher model weights with EMA
                for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
                    teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data
               
            if epoch % 25 == 0:
                print(f'Epoch [{epoch+1}/{self.cfg_m.training.epochs}], Loss: {np.mean(epoch_loss):.4f}')

        return teacher_model
