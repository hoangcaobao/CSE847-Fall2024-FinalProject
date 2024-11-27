# Federated Semi-Supervised Learning in Image Classification
Official Code for CSE 847 Fall 2024 Final Project: "Federated Semi-Supervised Learning in Image Classification" Bao Hoang and Manh Tran.

# Overview
Machine learning models often require large-scale labeled datasets, which are scarce in many real-world scenarios. Semi-supervised learning (SSL) addresses this limitation by leveraging a small amount of labeled data alongside a large pool of unlabeled data, enhancing model performance. In this paper, we explore the effectiveness of SSL techniques in improving image classification tasks. Additionally, we tackle the challenges of data privacy in decentralized environments by adapting SSL algorithms to the federated learning framework. Our approach enables privacy-preserving, distributed training across multiple clients, paving the way for robust and secure semi-supervised machine learning algorithms. Our codes are provided in [https://github.com/hoangcaobao/CSE847-Fall2024-FinalProject](https://github.com/hoangcaobao/CSE847-Fall2024-FinalProject).

# Package dependencies

# Data preparation
For STL-10 and CIFAR-10, they already exist in the torchvision dataset library, so no further action is needed. For the Cat and Dog dataset, please download the data from [https://www.kaggle.com/datasets/tongpython/cat-and-dog/data](https://www.kaggle.com/datasets/tongpython/cat-and-dog/data), unzip the folder, and place it in the data folder of the repository.

# Demos

Here we provide several demos of results in the project report.
You can change the arguments from `main.py` to try different settings.

### Arguments of main.py

- `--dataset_name` (string, optional, default: `"Cat_and_Dog"`): 
  - Specifies the dataset. 
  - Options include: `"STL10"`, `"CIFAR10"`, and `"Cat_and_Dog"`.
    
- `--golden_baseline` (flag, optional, default: `False`): 
  - If set, then evaluate the golden baseline which uses all labeled training data. 
  - Options include: `False` and `True`.
  
- `--numberOfClients` (int, optional, default: `5`): 
  - Specifies the number of clients in federated learning (set to 1 means centralized setting).

- `--solver` (string, optional, default = `"SelfTraining_solver"`):
  - Specifies semi-supervised algorithms.
  - Options include: `"Standard_solver"`, `"SelfTraining_solver"`, `"FixMatch_solver"`, `"MeanTeachers_solver"`, and `"MixMatch_solver"`
 
- `--model` (string, optional, default = `"simpleCNN"`):
  - Specifies computer vision models.
  - Options include: `"simpleCNN"`, `"resnet18"`, `"densenet121"`
 
### Example
Run FixMatch algorithm for CIFAR10 using resnet 18 model: ```python main.py --model resnet18 --dataset_name CIFAR10```

  
 
  
