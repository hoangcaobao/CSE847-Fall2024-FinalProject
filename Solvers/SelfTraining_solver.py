from Solvers.Solver_Base import Solver_Base
from Models.model import Conv2DModel
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split

class SelfTraining_solver(Solver_Base):
    
    def __init__(self, cfg_proj, cfg_m, name = "Std"):
        Solver_Base.__init__(self, cfg_proj, cfg_m, name)
        
    def run(self, train_labeled_loader, train_unlabeled_loader, test_loader):
        self.set_random_seed(self.cfg_proj.seed)
        
        # Initialize the model, loss function, and optimizer
        model = self.train(train_labeled_loader, train_unlabeled_loader, test_loader)

        acc = self.eval_func(model, test_loader)

        return acc

    def train(self, train_labeled_loader, train_unlabeled_loader, test_loader):
        images_labeled_dataset = []
        labels_labeled_dataset = []
        original_labeled_dataset = []
        
        for images, labels in train_labeled_loader:
            images_labeled_dataset.append(images)
            labels_labeled_dataset.append(labels)
            original_labeled_dataset.append(torch.tensor([1 for _ in range(images.shape[0])]))

        images_labeled_dataset = torch.cat(images_labeled_dataset, dim = 0)
        labels_labeled_dataset = torch.cat(labels_labeled_dataset, dim = 0)  
        original_labeled_dataset = torch.cat(original_labeled_dataset, dim = 0) 

        train_labeled_loader = DataLoader(dataset=TensorDataset(images_labeled_dataset, labels_labeled_dataset, original_labeled_dataset), batch_size=self.cfg_m.training.batch_size, shuffle=True)

        for _ in range(10):
            model = Conv2DModel(dim_out=self.cfg_m.data.dim_out, in_channels=self.cfg_m.data.in_channels)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.cfg_m.training.lr_init)

            model.train()

            model = self.basic_train(model, train_labeled_loader, criterion, optimizer)

            model.eval()

            images_labeled_dataset = []
            labels_labeled_dataset = []
            original_labeled_dataset = []

            images_unlabeled_dataset = []
            labels_unlabeled_dataset = []

            for images, labels, original in train_labeled_loader:
                images_labeled_dataset.append(images)
                labels_labeled_dataset.append(labels)
                original_labeled_dataset.append(original)

            with torch.no_grad():
                for images, l in train_unlabeled_loader:
                    outputs = model(images.to(self.device))
                    outputs = torch.nn.functional.softmax(outputs, dim = 1).detach().cpu()
                    outputs, labels = torch.max(outputs.data, 1)
                    
                    # choose most confidence one
                    index = outputs >= 0.999

                    images_labeled_dataset.append(images[index])
                    labels_labeled_dataset.append(labels[index])
                    original_labeled_dataset.append(torch.tensor([0 for _ in range(index.sum())]))

                    images_unlabeled_dataset.append(images[~index])
                    labels_unlabeled_dataset.append(labels[~index])
                
            images_labeled_dataset = torch.cat(images_labeled_dataset, dim = 0)
            labels_labeled_dataset = torch.cat(labels_labeled_dataset, dim = 0)    
            original_labeled_dataset = torch.cat(original_labeled_dataset, dim = 0) 

            print(f"Number of labeled data: {images_labeled_dataset.shape[0]}")

            images_unlabeled_dataset = torch.cat(images_unlabeled_dataset, dim = 0)
            labels_unlabeled_dataset = torch.cat(labels_unlabeled_dataset, dim = 0)   
            
            train_labeled_loader = DataLoader(dataset=TensorDataset(images_labeled_dataset, labels_labeled_dataset, original_labeled_dataset), batch_size=self.cfg_m.training.batch_size, shuffle=True)
            train_unlabeled_loader = DataLoader(dataset=TensorDataset(images_unlabeled_dataset, labels_unlabeled_dataset), batch_size=self.cfg_m.training.batch_size, shuffle=False)

            print(f"Test performance: {self.eval_func(model, test_loader)}")
            
        return model

    def basic_train(self, model, dataloader_train, criterion, optimizer):
        
        model = model.to(self.device)
        # Training loop with validation
        for epoch in range(self.cfg_m.training.epochs):
            model.train()

            epoch_loss = []

            for images, labels, original in dataloader_train:
                # Forward pass
                images, labels, original = images.to(self.device), labels.to(self.device), original.to(self.device)

                outputs = model(images)
                loss = 0

                # Compute the loss for original labeled dataset
                if torch.any(original == 1):
                    loss += criterion(outputs[original == 1], labels[original == 1])

                # Compute the loss for pseudo labeled dataset
                if torch.any(original == 0):
                    loss += 0.2 * criterion(outputs[original == 0], labels[original == 0])
                
                epoch_loss.append(loss.item())

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 25 == 0:
                print(f'Epoch [{epoch+1}/{self.cfg_m.training.epochs}], Loss: {np.mean(epoch_loss):.4f}')

        return model
    