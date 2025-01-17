import logging 
import math
import os
import random

import torch
import numpy as np
from PIL import Image
from torchvision import datasets 
from torchvision import transforms 
from torch.utils.data import Dataset

from Data.randaugment import RandAugmentMC


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx

class TransformFixMatch(object):
    def __init__(self, mean, std, image_size):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=image_size, padding=int(image_size * 0.125), padding_mode='reflect')
        ])
        
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=image_size, padding=int(image_size * 0.125), padding_mode='reflect'), 
            RandAugmentMC(n=2, m=10)
        ])
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target
    
class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform 
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        if isinstance(image, torch.Tensor):
            image = image.numpy()
            
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
            
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        if isinstance(image, np.ndarray):  
            image = Image.fromarray(image)
            
        if self.transform:
            image = self.transform(image)
        return image, target

class CustomDatasetSelfTraining(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform 
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image = self.dataset[idx]
        if isinstance(image, torch.Tensor):
            image = image.numpy()
            
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        if isinstance(image, np.ndarray):    
            image_transform = Image.fromarray(image)
            
        if self.transform:
            image_transform = self.transform(image_transform)

        return image, image_transform
    
class TransformTwice:
    def __init__(self, transform):
        self.transform = transform 
        
    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2
    
def LoadCatAndDogDataset(root_dir, transform=None):
    all_data = []
    for class_dir in ["cats", "dogs"]:
        class_label = 0 if class_dir == "cats" else 1
        class_path = os.path.join(root_dir, class_dir)
        for img_file in os.listdir(class_path):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                all_data.append((transform(Image.open(os.path.join(class_path, img_file)).convert('RGB')), class_label))   
    return all_data