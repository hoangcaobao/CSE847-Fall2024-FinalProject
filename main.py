import os
import argparse
from Data.DataInit import data_init
from configs.cfg import init_cfg
from Solvers.Solver_loader import solver_loader

def main(cfg_proj, cfg_m):
    solver = solver_loader(cfg_proj, cfg_m)
    solver.set_random_seed(cfg_proj.seed)

    train_labeled_loader, train_unlabeled_loader, test_loader = data_init(cfg_proj, cfg_m)
    acc = solver.run(train_labeled_loader, train_unlabeled_loader, test_loader)

    print(f"Accuracy: {100*acc:.2f}")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="1", required=False)
    parser.add_argument("--seed", type=int, default = 42, required=False) 
    parser.add_argument("--dataset_name", type = str, default="CIFAR10", required=False)
    parser.add_argument("-golden_baseline", type = bool, default=True, required=False)

    # Standard_solver, SelfTraining_solver, FixMatch_solver, MeanTeachers_solver, MixMatch_solver
    parser.add_argument("--solver", type = str, default = "Standard_solver", required=False) 

    cfg_proj = parser.parse_args()
    cfg_m = init_cfg(cfg_proj)
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%(cfg_proj.gpu)

    main(cfg_proj, cfg_m)
