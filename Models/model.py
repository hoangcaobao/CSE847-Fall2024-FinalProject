import torch
import torch.nn as nn

# Define a simple Conv2D model
class Conv2DModel(nn.Module):
    def __init__(self, dim_out, in_channels = 3, dataset_name = "CIFAR10"):
        super(Conv2DModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dataset_name = dataset_name

        # MNIST
        if self.dataset_name == "MNIST":
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
        
        # CIFAR10
        if self.dataset_name == "CIFAR10":
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
        
        #STL10
        if self.dataset_name == "STL10":
            self.fc1 = nn.Linear(64 * 24 * 24, 128)
        
        self.fc2 = nn.Linear(128, dim_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # MINIST
        if self.dataset_name == "MNIST":
            x = x.view(-1, 64 * 7 * 7)
        
        # CIFAR 10
        if self.dataset_name == "CIFAR10":
            x = x.view(-1, 64 * 8 * 8)
        
        #STL10
        if self.dataset_name == "STL10":
            x = x.view(-1, 64 * 24 * 24)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x