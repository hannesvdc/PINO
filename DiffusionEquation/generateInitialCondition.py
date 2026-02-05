import os
import torch as pt
import matplotlib.pyplot as plt

from typing import Callable, Tuple

"""
Conditional Gaussian Process

(u_b, u_i) \\sim \\mathcal{N}(0, K)
->
u_i | u_b = \\sim \\mathcal{N}(0, K_{ii} - K_{ib} K_{bb}^{-1} K_{bi})

I might write a short blog post about this.
"""

def precompute_covariance(x_grid : pt.Tensor, 
                          l : float):
    kernel = lambda y, yp: pt.exp(-0.5*(y - yp)**2 / l**2) # Covariance kernel
    grid_1, grid_2 = pt.meshgrid( x_grid, x_grid, indexing="ij" )
    K = kernel(grid_1, grid_2)
    
    return K

def gp(x_grid : pt.Tensor,
       l : float):
    N = x_grid.numel()

    # Compute the conditional covariance matrix
    K = precompute_covariance( x_grid, l )
    K_bb = K[[0,-1],:][:,[0,-1]] # (2,2)
    K_ii = K[1:-1,:][:,1:-1] #( N-2,N-2)
    K_ib = K[1:-1,[0,-1]] # (N-2,2)
    K_bi = K[[0,-1],1:-1] # (2,N-2)
    K_bb_inv = pt.linalg.inv(K_bb)
    K_cond = K_ii - K_ib @ K_bb_inv @ K_bi + 1e-6 * pt.eye(N-2)

    # Sample the conditional GP
    conditional_normal = pt.distributions.MultivariateNormal( 0.0*x_grid[1:-1], K_cond )
    u0 = conditional_normal.rsample( )
    return pt.cat(( pt.tensor([0.0]), u0, pt.tensor([0.0]) ), dim=0)

def build_rbf_interpolator( u0 : pt.Tensor,
                            x_grid : pt.Tensor,
                            l : float ) -> pt.Tensor:
    N_grid = x_grid.shape[0]
    grid = pt.flatten( x_grid )

    kernel = lambda y, yp: pt.exp(-0.5 * (y - yp)**2 / l**2) # Covariance kernel
    grid_1, grid_2 = pt.meshgrid( grid, grid, indexing="ij" )
    K = kernel(grid_1, grid_2) + 1e-4 * pt.eye( N_grid, device=grid.device, dtype=grid.dtype ) 

    L = pt.linalg.cholesky( K )
    alpha = pt.cholesky_solve( u0[:,None], L ) # shape (N_grid,)
    return alpha[:,0]

def generateInitial( x_grid : pt.Tensor,
                     l : float ):
    u0 = gp( x_grid, l )
    ci_95 = 1.96
    u0 = u0 / ci_95

    alpha = build_rbf_interpolator( u0, x_grid, l )

    # Store the initial, the grid and the interpolation coefficients
    pt.save( pt.cat( (u0[:,None], x_grid[:,None], alpha[:,None]), dim=1), 'initial.pth')

def build_u0_evaluator( l : float,
                        device : pt.device,
                        dtype : pt.dtype ) -> Tuple[Callable[[pt.Tensor], pt.Tensor], pt.Tensor]:
    path_of_file = os.path.dirname( os.path.abspath( __file__ ) )
    initial = pt.load( os.path.join( path_of_file, 'initial.pth' ), map_location=device, weights_only=True ).to( dtype=dtype )
    u0 = initial[:,0]
    x_grid = initial[:,1]
    alpha = initial[:,2]

    x_grid_eval = x_grid[None,:]
    print(x_grid_eval.shape)
    print((x_grid - x_grid_eval).shape)
    def evaluate_u0(x : pt.Tensor ) -> pt.Tensor:        
        K_x_grid = pt.exp( -0.5 * (x - x_grid[None,:])**2 / l**2 )
        u0_at_x = K_x_grid @ alpha
        return u0_at_x[:,None]
    
    # Show the fit for good measure
    plt.plot(x_grid.cpu().numpy(), u0.cpu().numpy())
    plt.plot(x_grid.cpu().numpy(), evaluate_u0(x_grid[:,None]).cpu().numpy())
    plt.show()

    return evaluate_u0, initial

if __name__ == '__main__':
    N_grid_points = 51
    l = 0.12 # GP correlation length
    x_grid = pt.linspace(0.0, 1.0, N_grid_points)
    generateInitial( x_grid, l )
