from Solvers.Solver_Base import Solver_Base
from Models.model import Conv2DModel
from Models.model_loader import model_loader
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from Data.DataPreProcessing import CustomDataset, CustomDatasetSelfTraining
from tqdm import tqdm
from torchvision import datasets, transforms


class SelfTraining_solver(Solver_Base):
    
    def __init__(self, cfg_proj, cfg_m, name = "Std"):
        Solver_Base.__init__(self, cfg_proj, cfg_m, name)
        
    def run(self, train_labeled_loader, train_unlabeled_loader, test_loader):
        self.set_random_seed(self.cfg_proj.seed)
        
        if self.cfg_proj.dataset_name == "CIFAR10":
            cifar10_mean = (0.4914, 0.4822, 0.4465)
            cifar10_std = (0.2471, 0.2435, 0.2616)
            self.transform_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
            ])
            self.transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32,
                                    padding=int(32*0.125),
                                    padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
            ])
        elif self.cfg_proj.dataset_name == "STL10":
            stl10_mean = (0.4467, 0.4398, 0.4066)
            stl10_std = (0.2243, 0.2214, 0.2236)

            self.transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=96, padding=int(96 * 0.125), padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean=stl10_mean, std=stl10_std)
            ])

            self.transform_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=stl10_mean, std=stl10_std)
            ])
        elif self.cfg_proj.dataset_name == "Cat_and_Dog":
            self.transform_val = transforms.Compose([transforms.ToTensor(),transforms.Resize((64,64)), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            self.transform_train = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.RandomCrop(size=64,
                                    padding=int(64*0.125),
                                    padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        # Initialize the model, loss function, and optimizer
        model = self.train(train_labeled_loader, train_unlabeled_loader, test_loader)

        acc = self.eval_func(model, test_loader)

        return acc

    def train(self, train_labeled_loader, train_unlabeled_loader, test_loader):
        unlabeled_imgs = [item[0] for item in train_unlabeled_loader.dataset.dataset[:]]
        unlabeled_imgs = torch.stack(unlabeled_imgs)

        pseudo_imgs = []
        pseudo_labels = []

        train_pseudo_labeled_loader = None
        
        for _ in range(10):
            model = model_loader(self.cfg_proj, self.cfg_m)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.cfg_m.training.lr_init)

            model.train()

            model = self.basic_train(model, train_labeled_loader, train_pseudo_labeled_loader, criterion, optimizer)

            model.eval()

            print(f"Test performance: {self.eval_func(model, test_loader)}")

            train_unlabeled_loader = DataLoader(dataset=CustomDatasetSelfTraining(unlabeled_imgs, transform=self.transform_val), batch_size=self.cfg_m.training.batch_size, shuffle=True, drop_last=False)
            unlabeled_imgs = []

            if len(pseudo_imgs):
                pseudo_imgs = [pseudo_imgs]
                pseudo_labels = [pseudo_labels]

            with torch.no_grad():
                for original_images, images in train_unlabeled_loader:
                    outputs = model(images.to(self.device))
                    outputs = torch.nn.functional.softmax(outputs, dim = 1).detach().cpu()
                    outputs, labels = torch.max(outputs.data, 1)
                    
                    # choose most confidence one
                    index = outputs >= 0.8
                    pseudo_imgs.append(original_images[index])
                    pseudo_labels.append(labels[index])

                    unlabeled_imgs.append(original_images[~index])
                
            pseudo_imgs = torch.cat(pseudo_imgs, dim = 0)
            pseudo_labels = torch.cat(pseudo_labels, dim = 0)    
            print(f"Number of pseudo labeled data: {pseudo_imgs.shape[0]}")

            unlabeled_imgs = torch.cat(unlabeled_imgs, dim = 0)
            
            train_pseudo_labeled_loader =  DataLoader(dataset=CustomDataset(TensorDataset(pseudo_imgs, pseudo_labels), transform=self.transform_train), batch_size=self.cfg_m.training.batch_size, shuffle=True, drop_last=False)

        return model

    def basic_train(self, model, train_labeled_loader, train_pseudo_labeled_loader, criterion, optimizer):
        
        model = model.to(self.device)
        # Training loop with validation

        labeled_iter = iter(train_labeled_loader)
        if train_pseudo_labeled_loader:
            unlabeled_iter = iter(train_pseudo_labeled_loader)

        for epoch in range(self.cfg_m.training.epochs):
            model.train()

            epoch_loss = []

            for batch_idx in range(len(train_labeled_loader)):

                try:
                    labeled_images, labels = next(labeled_iter)
                except:
                    labeled_iter = iter(train_labeled_loader)
                    labeled_images, labels = next(labeled_iter)
                
                if train_pseudo_labeled_loader:
                    try:
                        unlabeled_images, pseudo_labels = next(unlabeled_iter)
                    except:
                        unlabeled_iter = iter(train_pseudo_labeled_loader)
                        unlabeled_images, pseudo_labels = next(unlabeled_iter)

                # Forward pass
                labeled_images, labels = labeled_images.to(self.device), labels.to(self.device)

                if train_pseudo_labeled_loader:
                    unlabeled_images, pseudo_labels = unlabeled_images.to(self.device), pseudo_labels.to(self.device)
                loss = 0
                loss += criterion(model(labeled_images), labels)
                if train_pseudo_labeled_loader:
                    loss += 0.2 * criterion(model(unlabeled_images), pseudo_labels)
                
                epoch_loss.append(loss.item())

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 25 == 0:
                print(f'Epoch [{epoch+1}/{self.cfg_m.training.epochs}], Loss: {np.mean(epoch_loss):.4f}')

        return model
    