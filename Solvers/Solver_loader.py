import sys

#solver load
from Solvers.Standard_solver import Standard_solver
from Solvers.SelfTraining_solver import SelfTraining_solver
from Solvers.FixMatch_solver import FixMatch_solver
def solver_loader(cfg_proj, cfg_m):
    s = None
    if cfg_proj.solver == "Standard_solver":
        s = Standard_solver(cfg_proj, cfg_m)
    if cfg_proj.solver == "SelfTraining_solver":
        s = SelfTraining_solver(cfg_proj, cfg_m)
    if cfg_proj.solver == "FixMatch_solver":
        s = FixMatch_solver(cfg_proj, cfg_m)
    return s

