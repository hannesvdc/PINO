import torch as pt

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