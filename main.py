import os
import argparse
from Data.DataInit import data_init
from configs.cfg import init_cfg
from Solvers.Solver_loader import solver_loader

def main(cfg_proj, cfg_m):
    solver = solver_loader(cfg_proj, cfg_m)
    solver.set_random_seed(cfg_proj.seed)

    train_labeled_loaders, train_unlabeled_loaders, test_loader = data_init(cfg_proj, cfg_m)

    if len(train_unlabeled_loaders) == 1: # Normal Training
        train_labeled_loaders = train_labeled_loaders[0]
        train_unlabeled_loaders = train_unlabeled_loaders[0]

    acc = solver.run(train_labeled_loaders, train_unlabeled_loaders, test_loader)

    print(f"Accuracy: {100*acc:.2f}")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="2", required=False)
    parser.add_argument("--seed", type=int, default = 42, required=False) 
    parser.add_argument("--dataset_name", type = str, default="CIFAR10", required=False) # STL10, CIFAR10
    parser.add_argument("--golden_baseline", type = bool, default=False, required=False)
    
    parser.add_argument("--numberOfClients", type = int, default = 1, required=False) # 1 means normal training

    # Standard_solver, SelfTraining_solver, FixMatch_solver, MeanTeachers_solver, MixMatch_solver
    parser.add_argument("--solver", type = str, default = "SelfTraining_solver", required=False) 

    cfg_proj = parser.parse_args()
    cfg_m = init_cfg(cfg_proj)
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%(cfg_proj.gpu)

    main(cfg_proj, cfg_m)
