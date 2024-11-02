from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from Data.DataPreProcessing import get_cifar10, TransformFixMatch, CustomDataset, TransformTwice, LoadCatAndDogDataset
import argparse
import numpy as np 
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split

def random_split_dataset(dataset, num_clients):
    # Length of dataset and calculate sizes for each client
    dataset_size = len(dataset)
    split_sizes = [dataset_size // num_clients for _ in range(num_clients)]
    
    client_datasets = []

    # Account for any leftover data
    leftover = dataset_size % num_clients
    for i in range(leftover):
        split_sizes[i] += 1
    
    # Random Split
    for client in range(num_clients-1):
        dataset, client_dataset = train_test_split(dataset, test_size=split_sizes[client]/len(dataset), random_state=42)
        client_datasets.append(client_dataset)

    client_datasets.append(dataset)

    return client_datasets

def data_init(cfg_proj, cfg_m):

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    stl10_mean = (0.4467, 0.4398, 0.4066)
    stl10_std = (0.2243, 0.2214, 0.2236)
    
    if cfg_proj.dataset_name == "MNIST":
        # MNIST dataset
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        
    elif cfg_proj.dataset_name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
        
        if cfg_proj.model == "simpleCNN":
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32,
                                    padding=int(32*0.125),
                                    padding_mode='reflect'),
                # transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
            ])
            
            transform_val = transforms.Compose([
                # transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
            ])
            
        if cfg_proj.model == "resnet50":
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32,
                                    padding=int(32*0.125),
                                    padding_mode='reflect'),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
            ])
            
            transform_val = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
            ])
        
        train_labeled_dataset, train_unlabeled_dataset = train_test_split(train_dataset, test_size=0.9, stratify=train_dataset.targets, random_state=42)

    elif cfg_proj.dataset_name == "STL10":
        transform = transforms.Compose([transforms.ToTensor()])
        train_labeled_dataset = datasets.STL10(root='./data', split='train', transform=transform, download=True)
        train_unlabeled_dataset = datasets.STL10(root='./data', split='unlabeled', transform=transform, download=True)
        test_dataset = datasets.STL10(root='./data', split='test', transform=transform, download=True)
        
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=96, padding=int(96 * 0.125), padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=stl10_mean, std=stl10_std)
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=stl10_mean, std=stl10_std)
        ])

        train_unlabeled_dataset, _ = train_test_split(train_unlabeled_dataset, test_size=0.5, random_state=42)
        train_labeled_dataset = [(image, label) for image, label in train_labeled_dataset]
        train_unlabeled_dataset = [(image, label) for image, label in train_unlabeled_dataset]
    
    elif cfg_proj.dataset_name == "Cat_and_Dog":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64))])
        transform_val = transforms.Compose([
            transforms.ToTensor(),transforms.Resize((64,64)), 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        transform_train = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(size=64,
                                padding=int(64*0.125),
                                padding_mode='reflect'),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        if cfg_proj.solver == "MixMatch_solver":
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=64, padding=int(64*0.125), padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        train_dataset = LoadCatAndDogDataset(root_dir="./data/training_set/training_set/",  transform=transform)    
        test_dataset = LoadCatAndDogDataset(root_dir="./data/test_set/test_set/", transform=transform)
        train_labeled_dataset, train_unlabeled_dataset = train_test_split(train_dataset, test_size=0.9, stratify=[i[1] for i in train_dataset], random_state=42)
        
    train_labeled_datasets = random_split_dataset(train_labeled_dataset, cfg_proj.numberOfClients)
    train_unlabeled_datasets = random_split_dataset(train_unlabeled_dataset, cfg_proj.numberOfClients)

    train_labeled_loaders = []
    train_unlabeled_loaders = []

    for train_labeled_dataset in train_labeled_datasets:
        # GOLDEN BASELINE
        if cfg_proj.golden_baseline: 
            train_labeled_dataset = CustomDataset(train_dataset, transform=transform_train)
            train_labeled_loader = DataLoader(dataset=train_labeled_dataset, batch_size=cfg_m.training.batch_size, shuffle=True, drop_last=False)
        elif cfg_proj.solver == "FixMatch_solver":
            train_labeled_dataset = CustomDataset(train_labeled_dataset, transform=transform_train)
            train_labeled_loader = DataLoader(dataset=train_labeled_dataset, batch_size=cfg_m.training.batch_size, shuffle=True)
        elif cfg_proj.solver == "MixMatch_solver":
            train_labeled_dataset = CustomDataset(train_labeled_dataset, transform=transform_train)
            train_labeled_loader = DataLoader(dataset=train_labeled_dataset, batch_size=cfg_m.training.batch_size, shuffle=True, drop_last=True)
        else:
            train_labeled_dataset = CustomDataset(train_labeled_dataset, transform=transform_train)
            train_labeled_loader = DataLoader(dataset=train_labeled_dataset, batch_size=cfg_m.training.batch_size, shuffle=True, drop_last=False)
        train_labeled_loaders.append(train_labeled_loader)
    
    for train_unlabeled_dataset in train_unlabeled_datasets:
        if cfg_proj.golden_baseline: 
            train_unlabeled_loader = None
        elif cfg_proj.solver == "FixMatch_solver":
            train_unlabeled_dataset = CustomDataset(train_unlabeled_dataset, transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))
            train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, batch_size=9 * cfg_m.training.batch_size, shuffle=False)
        elif cfg_proj.solver == "MixMatch_solver":
            train_unlabeled_dataset = CustomDataset(train_unlabeled_dataset, transform=TransformTwice(transform_train))            
            train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, batch_size=cfg_m.training.batch_size, shuffle=True, drop_last=True)
        else:
            train_unlabeled_dataset = CustomDataset(train_unlabeled_dataset, transform=transform_train)            
            train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, batch_size=cfg_m.training.batch_size, shuffle=True, drop_last=False)
        train_unlabeled_loaders.append(train_unlabeled_loader)
        
    # GOLDEN BASELINE
    if cfg_proj.golden_baseline: 
        test_dataset = CustomDataset(test_dataset, transform=transform_val)
        test_loader = DataLoader(dataset=test_dataset, batch_size=cfg_m.training.batch_size, shuffle=False)
    elif cfg_proj.solver == "FixMatch_solver":
        test_dataset = CustomDataset(test_dataset, transform=transform_val)
        test_loader = DataLoader(dataset=test_dataset, batch_size=cfg_m.training.batch_size, shuffle=False)
    elif cfg_proj.solver == "MixMatch_solver":
        test_dataset = CustomDataset(test_dataset, transform=transform_val)
        test_loader = DataLoader(dataset=test_dataset, batch_size=cfg_m.training.batch_size, shuffle=False)
    else:
        test_dataset = CustomDataset(test_dataset, transform=transform_val)
        test_loader = DataLoader(dataset=test_dataset, batch_size=cfg_m.training.batch_size, shuffle=False)

    return train_labeled_loaders, train_unlabeled_loaders, test_loader
