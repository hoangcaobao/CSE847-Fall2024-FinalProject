import sys

#solver load
from Solvers.Standard_solver import Standard_solver
from Solvers.SelfTraining_solver import SelfTraining_solver
from Solvers.FixMatch_solver import FixMatch_solver
from Solvers.MeanTeachers_solver import MeanTeachers_solver
from Solvers.MixMatch_solver import MixMatch_solver
from Solvers.FedAvg_solver import FedAvg_solver

def solver_loader(cfg_proj, cfg_m, fed = False):
    s = None
    
    if not fed:
        if cfg_proj.solver == "Standard_solver":
            s = Standard_solver(cfg_proj, cfg_m)
        if cfg_proj.solver == "SelfTraining_solver":
            s = SelfTraining_solver(cfg_proj, cfg_m)
        if cfg_proj.solver == "FixMatch_solver":
            s = FixMatch_solver(cfg_proj, cfg_m)
        if cfg_proj.solver == "MeanTeachers_solver":
            s = MeanTeachers_solver(cfg_proj, cfg_m)
        if cfg_proj.solver == "MixMatch_solver":
            s = MixMatch_solver(cfg_proj, cfg_m)
    else:
        if cfg_proj.fedAlgo == "FedAvg":
            s = FedAvg_solver(cfg_proj, cfg_m)
    
    return s

