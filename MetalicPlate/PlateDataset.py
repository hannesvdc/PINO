import math
import torch as pt
from torch.utils.data import Dataset

from gp import gp

from typing import Tuple

class TrunkDataset( Dataset ):
    def __init__(self, N : int,
                       nu_max : float,
                       dtype = pt.float64 ):
        super().__init__()

        self.N = N
        self.nu_max = nu_max

        # nu uniformly
        gen = pt.Generator( )
        self.nu = pt.empty((self.N,1), dtype=dtype, requires_grad=False).uniform_( 0.0, self.nu_max, generator=gen )

        # Sample spatial coordinates.
        #       x: 60% uniform in [0,1], 20% in [0,0.2], 20% in [0.8,1]
        #       y: uniform in [0,1]
        x_mid = pt.rand( (int(0.6*self.N),1), dtype=dtype, requires_grad=False, generator=gen)
        x_left = 0.2 * pt.rand( (int(0.2*self.N),1), dtype=dtype, requires_grad=False, generator=gen)
        x_right = 0.8 + 0.2 * pt.rand( (int(0.2*self.N),1), dtype=dtype, requires_grad=False, generator=gen)
        self.x = pt.cat( (x_left, x_mid, x_right), dim=0 )
        self.y = pt.rand( (self.N,1), dtype=dtype, requires_grad=False, generator=gen)

    def __len__( self ) -> int:
        return self.N
    
    def __getitem__(self, idx : int) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        return self.x[idx], self.y[idx], self.nu[idx]
    
    def all( self ) -> pt.Tensor:
        return pt.cat( (self.x, self.y, self.nu), dim=1 )
    
class BranchDataset( Dataset ):
    def __init__( self, N : int,
                        n_grid_points : int,
                        l : float,
                        plot : bool = False):
        super().__init__()

        self.N = N
        self.n_grid_points = n_grid_points

        self.y_grid = pt.linspace(0.0, 1.0, self.n_grid_points)
        self.l = l
        self.scale = 2.0
        gx, gy = gp( self.y_grid, self.l, self.N ) # (N, n_grid_points)
        self.gx = gx / self.scale
        self.gy = gy / self.scale

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(self.y_grid.detach().cpu().numpy(), self.gx.detach().cpu().numpy().T, label=r"$g_x(y)$")
            plt.xlabel(r"$y$")
            plt.show()

    def __len__( self ) -> int:
        return self.N
    
    def __getitem__( self, idx : int ) -> Tuple[pt.Tensor, pt.Tensor]:
        return self.gx[idx,:], self.gy[idx,:]
    
    def all( self ) -> Tuple[pt.Tensor, pt.Tensor]:
        return self.gx, self.gy
    
class PlateDataset( Dataset ):
    def __init__(self, N_branch : int,
                       N_trunk : int,
                       n_grid_points : int,
                       l : float,
                       nu_max : float,
                       dtype = pt.float64,
                       plot : bool = False):
        super().__init__()

        self.N_branch = N_branch
        self.N_trunk = N_trunk
        self.branch_dataset = BranchDataset( N_branch, n_grid_points, l, plot)
        self.trunk_dataset = TrunkDataset( N_trunk, nu_max, dtype )

    def __len__( self ) -> int:
        return len( self.branch_dataset ) * len( self.trunk_dataset )
    
    def __getitem__( self, idx : int ) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
        branch_idx = idx % self.N_branch
        trunk_idx = idx // self.N_branch
        gx, gy = self.branch_dataset[branch_idx]
        x, y, nu = self.trunk_dataset[trunk_idx]
        return x, y, nu, gx, gy
    
    # Totally inefficient implementation, but only called once in the training script.
    def all( self ) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
        B = self.__len__()
        x = pt.zeros( (B,1) )
        y = pt.zeros( (B,1) )
        nu = pt.zeros( (B,2) )
        gx = pt.zeros( (B, self.branch_dataset.n_grid_points) )
        gy = pt.zeros( (B, self.branch_dataset.n_grid_points) )
        for b in range( B ):
            xb, yb, nub, gxb, gyb = self.__getitem__(b)
            x[b,:] = xb
            y[b,:] = yb
            nu[b,:] = nub
            gx[b,:] = gxb
            gy[b,:] = gyb
        return x, y, nu, gx, gy