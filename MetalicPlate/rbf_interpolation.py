import torch as pt

from typing import Tuple

@pt.no_grad()
def buildCholeskyMatrix( y_grid : pt.Tensor,
                         l : float ) -> pt.Tensor:
    if y_grid.ndim > 1:
        print( f"`y_grid` must have shape `(n_grid_points, )` but got {y_grid.shape}. Flattening `y_grid` but proceed with caution." )
        y_grid = y_grid.flatten()
    n_grid_points = len( y_grid )

    # Build the kernel matrix
    jitter = 1e-4 # for numerical stability
    grid_1, grid_2 = pt.meshgrid( y_grid, y_grid, indexing="ij" )
    kernel = pt.exp(-0.5 * (grid_1 - grid_2)**2 / l**2)
    K = kernel + jitter * pt.eye( n_grid_points, device=y_grid.device, dtype=y_grid.dtype )

    # Factor to Cholesky and return. This is a square matrix, no vectorization happens here.
    L = pt.linalg.cholesky( K )
    return L

def jointIndexingRBFInterpolator( L : pt.Tensor,
                                  y_grid : pt.Tensor,
                                  l : float, 
                                  y : pt.Tensor,
                                  gx : pt.Tensor,
                                  gy : pt.Tensor, ) -> Tuple[pt.Tensor,pt.Tensor]:
    # Make sure the shapes and sizes match
    assert L.ndim == 2 and L.shape[0] == L.shape[1], f"`L` must be a square matrix, got shape {L.shape}."
    assert gx.ndim == 2 and gx.shape[1] == L.shape[0], f"`gx` must have shape `(B, n_grid_points)` but got {gx.shape}."
    assert gy.ndim == 2 and gy.shape[1] == L.shape[0], f"`gy` must have shape `(B, n_grid_points)` but got {gy.shape}."
    if y.ndim == 1:
        y = y[:,None]
    assert y.ndim == 2 and y.shape[0] == gx.shape[0] and y.shape[1] == 1, f"`y` must have shape (B,1) with `B = gx.shape[0]` but got {y.shape}."

    # Solve the Cholesky system in vectorized form.
    with pt.no_grad():
        z_gx = pt.linalg.solve_triangular(L, gx.T, upper=False)
        alpha_gx = pt.linalg.solve_triangular(L.transpose(-1, -2), z_gx, upper=True)
        z_gy = pt.linalg.solve_triangular(L, gy.T, upper=False)
        alpha_gy = pt.linalg.solve_triangular(L.transpose(-1, -2), z_gy, upper=True)

    # Build a joint-indexing RBF interpolator
    if y_grid.ndim == 1:
        y_grid_eval = y_grid[None,:]
    else:
        y_grid_eval = y_grid.T # Shape (1, n_grid_points)
    
    # Actual interpolation
    K_y_grid = pt.exp( -0.5 * (y - y_grid_eval)**2 / l**2 ) # (B, n_grid_points)
    gx_eval = K_y_grid * alpha_gx.T
    gy_eval = K_y_grid * alpha_gy.T
    return pt.sum( gx_eval, dim=1, keepdim=True ), pt.sum( gy_eval, dim=1, keepdim=True )

def tensorizedRBFInterpolator( L : pt.Tensor,
                               y_grid : pt.Tensor,
                               l : float, 
                               y : pt.Tensor, # (Bt, 1)
                               gx : pt.Tensor, # (Bb, n_grid_points)
                               gy : pt.Tensor, # (Bb, n_grid_points)
                            ) -> Tuple[pt.Tensor,pt.Tensor]:
    # Make sure the shapes and sizes match
    assert L.ndim == 2 and L.shape[0] == L.shape[1], f"`L` must be a square matrix, got shape {L.shape}."
    assert gx.ndim == 2 and gx.shape[1] == L.shape[0], f"`gx` must have shape `(B, n_grid_points)` but got {gx.shape}."
    assert gy.ndim == 2 and gy.shape[1] == L.shape[0], f"`gy` must have shape `(B, n_grid_points)` but got {gy.shape}."
    if y.ndim == 1:
        y = y[:,None]
    assert y.ndim == 2 and y.shape[1] == 1, f"`y` must have shape (Bt_bc,1) but got {y.shape}."

    # Solve the Cholesky system in vectorized form.
    with pt.no_grad():
        z_gx = pt.linalg.solve_triangular(L, gx.T, upper=False)
        alpha_gx = pt.linalg.solve_triangular(L.transpose(-1, -2), z_gx, upper=True)
        z_gy = pt.linalg.solve_triangular(L, gy.T, upper=False)
        alpha_gy = pt.linalg.solve_triangular(L.transpose(-1, -2), z_gy, upper=True)

    # Build a joint-indexing RBF interpolator
    if y_grid.ndim == 1:
        y_grid_eval = y_grid[None,:]
    else:
        y_grid_eval = y_grid.T # Shape (1, n_grid_points)
    
    # Actual interpolation
    K_y_grid = pt.exp( -0.5 * (y - y_grid_eval)**2 / l**2 ) # (B, n_grid_points)
    gx_eval = K_y_grid @ alpha_gx
    gy_eval = K_y_grid @ alpha_gy
    return gx_eval.T, gy_eval.T