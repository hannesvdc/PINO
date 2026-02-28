import torch as pt
import numpy as np

from typing import Tuple

def computeCovariance(y_grid : pt.Tensor, 
                      l : float,
                      jitter : float):
    kernel = lambda y, yp: pt.exp(-0.5*(y - yp)**2 / l**2) # Covariance kernel
    grid_1, grid_2 = pt.meshgrid( y_grid, y_grid, indexing="ij" )
    K = kernel(grid_1, grid_2)
    
    return K + jitter * pt.eye( y_grid.numel(), dtype=y_grid.dtype, device=y_grid.device )

def gp(y_grid : pt.Tensor,
       l : float,
       N_samples : int,
       jitter : float = 1e-6 ) -> Tuple[pt.Tensor, pt.Tensor]:
    """
    Gaussian Process

    g_x, g_y \\sim \\mathcal{N}(0, K)
    """
    y_grid = y_grid.flatten() # (N,)

    # Compute the covariance matrix
    K = computeCovariance( y_grid, l, jitter)

    # Sample the GP independently for g_x and g_y
    normal = pt.distributions.MultivariateNormal( 0.0*y_grid, covariance_matrix=K )
    g_x = normal.rsample( sample_shape=(N_samples,) ) # (N_samples, N)
    g_y = normal.rsample( sample_shape=(N_samples,) ) # (N_samples, N)

    # Add Dirichlet zero boundary conditions
    return g_x, g_y

def gp_numpy( y_grid : np.ndarray,
              l : float,
              jitter : float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    y_grid_pt = pt.tensor( y_grid )
    g_x, g_y = gp( y_grid_pt, l, 1, jitter )
    return g_x.flatten().numpy(), g_y.flatten().numpy()