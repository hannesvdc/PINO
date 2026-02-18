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
                       dtype = pt.float64,
                       plot : bool = False, 
                       tau_sampling : str = "uniform" ):
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
        if tau_sampling == "uniform":
            self.tau = (tau_max - tau_min) * pt.rand( (N,1), generator=gen)
        else:
            gamma = 1.5  # 1 = log-uniform, >1 biases small tau
            u = pt.rand(( int(0.7*N), 1), dtype=dtype, generator=gen)
            log_tau = math.log(tau_min) + (math.log(tau_max) - math.log(tau_min)) * (u ** gamma)

            # Also sample some uniformly
            tau_extra = 1.0 + pt.rand( (N-log_tau.shape[0],1), dtype=dtype, generator=gen ) * (self.tau_max - 1.0)
            self.tau = pt.cat( (pt.exp( log_tau ), tau_extra), dim=0)

        if plot:
            import matplotlib.pyplot as plt
            plt.hist(self.tau)
            plt.show()
        
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
                        n_grid_points : int,
                        l : float,
                        plot : bool = False):
        super().__init__()

        self.N = N
        self.n_grid_points = n_grid_points

        self.x_grid = pt.linspace(0.0, 1.0, self.n_grid_points)
        self.l = l
        self.scale = 2.0
        self.data = gp( self.x_grid, self.l, self.N ) / self.scale # (N, n_grid_points)

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(self.x_grid.detach().cpu().numpy(), self.data.detach().cpu().numpy().T, label="u0")
            plt.xlabel(r"$x$")
            plt.show()

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
                       l : float,
                       T_max : float,
                       tau_max : float,
                       dtype = pt.float64,
                       plot : bool = False,
                       tau_sampling : str = "uniform"):
        super().__init__()

        self.N_branch = N_branch
        self.N_trunk = N_trunk
        self.branch_dataset = BranchDataset( N_branch, n_grid_points, l, plot)
        self.trunk_dataset = TrunkDataset( N_trunk, T_max, tau_max, dtype, plot, tau_sampling )

    def __len__( self ) -> int:
        return len( self.branch_dataset ) * len( self.trunk_dataset )
    
    def __getitem__( self, idx : int ) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
        branch_idx = idx % self.N_branch
        trunk_idx = idx // self.N_branch
        u0 = self.branch_dataset[branch_idx]
        x, t, params = self.trunk_dataset[trunk_idx]
        return x, t, params, u0
    
    def all( self ) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
        B = self.__len__()
        x = pt.zeros( (B,1) )
        t = pt.zeros( (B,1) )
        params = pt.zeros( (B,2) )
        u0 = pt.zeros( (B, self.branch_dataset.n_grid_points) )
        for b in range( B ):
            xb, tb, pb, u0b = self.__getitem__(b)
            x[b,:] = xb
            t[b,:] = tb
            params[b,:] = pb
            u0[b,:] = u0b
        return x, t, params, u0
    
class TestTensorizedDataset( TensorizedDataset ):
    def __init__(self, N_branch : int,
                       N_trunk : int,
                       n_grid_points : int,
                       l : float,
                       T_max : float,
                       tau_max : float,
                       dtype = pt.float64):
        super().__init__(N_branch, N_trunk, n_grid_points, l, T_max, tau_max, dtype)
        tau = 0.1
        t = tau / self.trunk_dataset.k
        self.trunk_dataset.t = t
