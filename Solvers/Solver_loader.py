import sys

#solver load
from Solvers.Standard_solver import Standard_solver

def solver_loader(cfg_proj, cfg_m):
    if cfg_proj.solver == "Standard_solver":
        s = Standard_solver(cfg_proj, cfg_m)
    return s

