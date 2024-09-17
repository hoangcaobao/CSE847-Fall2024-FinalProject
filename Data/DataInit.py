from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from Data.DataPreProcessing import get_cifar10, TransformFixMatch, CustomDataset, TransformTwice
import argparse
from sklearn.model_selection import train_test_split

def data_init(cfg_proj, cfg_m):

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    
    if cfg_proj.dataset_name == "MNIST":
        # MNIST dataset
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    if cfg_proj.dataset_name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    transform_train = transforms.Compose([
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
    
    train_labeled_dataset, train_unlabeled_dataset = train_test_split(train_dataset, test_size=0.9, stratify=train_dataset.targets)
    
    # GOLDEN BASELINE
    if cfg_proj.golden_baseline: 
        train_labeled_dataset = CustomDataset(train_dataset, transform=transform_train)
        test_dataset = CustomDataset(test_dataset, transform=transform_val)

        train_labeled_loader = DataLoader(dataset=train_labeled_dataset, batch_size=cfg_m.training.batch_size, shuffle=True, drop_last=False)
        train_unlabeled_loader = None
        test_loader = DataLoader(dataset=test_dataset, batch_size=cfg_m.training.batch_size, shuffle=False)

    elif cfg_proj.solver == "FixMatch_solver":
        train_labeled_dataset = CustomDataset(train_labeled_dataset, transform=transform_train)
        train_unlabeled_dataset = CustomDataset(train_unlabeled_dataset, transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))
        test_dataset = CustomDataset(test_dataset, transform=transform_val)
        
        train_labeled_loader = DataLoader(dataset=train_labeled_dataset, batch_size=cfg_m.training.batch_size, shuffle=True)
        train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, batch_size=8 * cfg_m.training.batch_size, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=cfg_m.training.batch_size, shuffle=False)
        
    elif cfg_proj.solver == "MixMatch_solver":
        train_labeled_dataset = CustomDataset(train_labeled_dataset, transform=transform_train)
        train_unlabeled_dataset = CustomDataset(train_unlabeled_dataset, transform=TransformTwice(transform_train))
        test_dataset = CustomDataset(test_dataset, transform=transform_val)
        
        train_labeled_loader = DataLoader(dataset=train_labeled_dataset, batch_size=cfg_m.training.batch_size, shuffle=True, drop_last=True)
        train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, batch_size=cfg_m.training.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=cfg_m.training.batch_size, shuffle=False)

    else:
        train_labeled_dataset = CustomDataset(train_labeled_dataset, transform=transform_train)
        train_unlabeled_dataset = CustomDataset(train_unlabeled_dataset, transform=transform_train)
        test_dataset = CustomDataset(test_dataset, transform=transform_val)
        
        train_labeled_loader = DataLoader(dataset=train_labeled_dataset, batch_size=cfg_m.training.batch_size, shuffle=True, drop_last=False)
        train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, batch_size=cfg_m.training.batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=cfg_m.training.batch_size, shuffle=False)
    return train_labeled_loader, train_unlabeled_loader, test_loader
