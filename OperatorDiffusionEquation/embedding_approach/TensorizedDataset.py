import sys
sys.path.append('../')

import math
import torch as pt
from torch.utils.data import Dataset

from conditionalGP import gp

from typing import Tuple

class TrunkDataset( Dataset ):
    def __init__(self, N : int,
                       T_max : float,
                       tau_max : float,
                       dtype = pt.float64 ):
        super().__init__()

        self.T_max = T_max
        self.tau_max = tau_max
        self.logk_max = math.log(1e2)

        # Sample log_k and T_s uniformly
        gen = pt.Generator( )
        logk = pt.empty((N,1), dtype=dtype, requires_grad=False).uniform_( -self.logk_max, self.logk_max, generator=gen )
        self.k = pt.exp( logk )
        self.T_s = pt.empty((N,1), dtype=dtype, requires_grad=False).uniform_( -self.T_max, self.T_max, generator=gen )

        # Sample spatial coordinates uniformly
        self.x = pt.rand( (N,1), dtype=dtype, requires_grad=False, generator=gen)

        # Put most tau closer to 0
        tau_min = 1e-2
        gamma = 2.0  # 1 = log-uniform, >1 biases small tau
        u = pt.rand((N, 1), dtype=dtype, generator=gen)
        log_tau = math.log(tau_min) + (math.log(tau_max) - math.log(tau_min)) * (u ** gamma)
        self.tau = pt.exp( log_tau )
        
        # rescale
        self.t = self.tau / self.k

    def __len__( self ):
        return len( self.k )
    
    def __getitem__(self, idx : int) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        return self.x[idx], self.t[idx], pt.cat((self.k[idx], self.T_s[idx]), dim=0)
    
    def all( self ) -> pt.Tensor:
        return pt.cat( (self.x, self.t, self.k, self.T_s), dim=1 )
    
class BranchDataset( Dataset ):
    def __init__( self, N : int,
                        n_grid_points : int):
        super().__init__()

        self.N = N
        self.n_grid_points = n_grid_points

        self.x_grid = pt.linspace(0.0, 1.0, self.n_grid_points)
        self.l = 0.12
        self.data = gp( self.x_grid, self.l, self.N ) # (N, n_grid_points)

    def __len__( self ) -> int:
        return self.N
    
    def __getitem__( self, idx : int ) -> pt.Tensor:
        return self.data[idx,:]
    
    def all( self ) -> pt.Tensor:
        return self.data
    
class TensorizedDataset( Dataset ):
    def __init__(self, N_branch : int,
                       N_trunk : int,
                       n_grid_points : int,
                       T_max : float,
                       tau_max : float,
                       dtype = pt.float64):
        super().__init__()

        self.N_branch = N_branch
        self.N_trunk = N_trunk
        self.branch_dataset = BranchDataset( N_branch, n_grid_points )
        self.trunk_dataset = TrunkDataset( N_trunk, T_max, tau_max, dtype )

    def __len__( self ) -> int:
        return len( self.branch_dataset ) * len( self.trunk_dataset )
    
    def __getitem__( self, idx : int ) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
        branch_idx = idx % self.N_branch
        trunk_idx = idx // self.N_branch
        u0 = self.branch_dataset[branch_idx]
        x, t, params = self.trunk_dataset[trunk_idx]
        return x, t, params, u0
    
    def all( self ) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
        u0 = self.branch_dataset.all()
        trunk_all = self.trunk_dataset.all()
        x, t, params = trunk_all[:,0:1], trunk_all[:,1:2], trunk_all[:,2:]
        return x, t, params, u0
