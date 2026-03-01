import math
import torch as pt
from torch.utils.data import Dataset

from gp import gp

from typing import Tuple

class TrunkDataset( Dataset ):
    def __init__(self, N : int,
                       gen : pt.Generator,
                       dtype = pt.float64, ):
        super().__init__()

        self.N = N

        # Sample spatial coordinates.
        #       x: 60% uniform in [0,1], 20% in [0,0.2], 20% in [0.8,1]
        #       y: uniform in [0,1]
        # The probability distribution is 
        # sample x from your mixture
        x_mid  = pt.rand((int(0.6*self.N), 1), dtype=dtype, requires_grad=False, generator=gen)            # U(0,1)
        x_left = 0.2 * pt.rand((int(0.2*self.N), 1), dtype=dtype, requires_grad=False, generator=gen)      # U(0,0.2)
        x_right = 0.8 + 0.2 * pt.rand((int(0.2*self.N), 1), dtype=dtype, requires_grad=False, generator=gen) # U(0.8,1)

        self.x = pt.cat((x_left, x_mid, x_right), dim=0)
        self.y = pt.rand((self.N, 1), dtype=dtype, requires_grad=False, generator=gen)

        # importance weights based on *location* (because x_mid can fall in the edge bands)
        x_flat = self.x[:, 0]
        q = pt.where((x_flat < 0.2) | (x_flat > 0.8),
                    pt.full_like(x_flat, 1.6),   # edge density
                    pt.full_like(x_flat, 0.6))   # middle density

        self.w = (1.0 / q).unsqueeze(1)           # (N,1)

    def __len__( self ) -> int:
        return self.N
    
    def __getitem__(self, idx : int) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        return self.x[idx], self.y[idx], self.w[idx]
    
    def all( self ) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        return self.x, self.y, self.w
    
class BoundaryDataset( Dataset ):
    def __init__(self, N : int,
                       gen : pt.Generator,
                       dtype = pt.float64 ):
        super().__init__()

        self.N = N

        # Sample spatial coordinates.
        #       y: uniform in [0,1]
        self.y_bc = pt.rand( (self.N,1), dtype=dtype, requires_grad=False, generator=gen)

    def __len__( self ) -> int:
        return self.N
    
    def __getitem__(self, idx : int) -> pt.Tensor:
        return self.y_bc[idx]
    
    def all( self ) -> pt.Tensor:
        return self.y_bc
    
class BranchDataset( Dataset ):
    def __init__( self, N : int,
                        n_grid_points : int,
                        l : float,
                        nu_max : float,
                        gen : pt.Generator,
                        dtype = pt.float64,
                        plot : bool = False):
        super().__init__()

        self.N = N
        self.n_grid_points = n_grid_points
        self.nu_max = nu_max

        # nu uniformly
        self.nu = pt.empty((self.N,1), dtype=dtype, requires_grad=False).uniform_( 0.1, self.nu_max, generator=gen )

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
    
    def __getitem__( self, idx : int ) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        return self.gx[idx,:], self.gy[idx,:], self.nu[idx]
    
    def all( self ) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        return self.gx, self.gy, self.nu
