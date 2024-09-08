from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def data_init(cfg_proj, cfg_m):

    if cfg_proj.dataset_name == "MNIST":
        # MNIST dataset
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    if cfg_proj.dataset_name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    # Split dataset into training and validation sets
    labeled_size = int(0.1 * len(train_dataset))
    unlabeled_size = len(train_dataset) - labeled_size

    train_labeled_dataset, train_unlabeled_dataset = random_split(train_dataset, [labeled_size, unlabeled_size])

    train_labeled_loader = DataLoader(dataset=train_labeled_dataset, batch_size=cfg_m.training.batch_size, shuffle=True)
    train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, batch_size=cfg_m.training.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg_m.training.batch_size, shuffle=False)
    
    return train_labeled_loader, train_unlabeled_loader, test_loader


    # train_labeled_loader = DataLoader(dataset=train_dataset, batch_size=cfg_m.training.batch_size, shuffle=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=cfg_m.training.batch_size, shuffle=False)
    
    # return train_labeled_loader, train_unlabeled_loader, test_loader