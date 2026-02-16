import torch as pt

@pt.no_grad()
def buildCholeskyMatrix( x_grid : pt.Tensor,
                         l : float ) -> pt.Tensor:
    if x_grid.ndim > 1:
        print( f"`x_grid` must have shape `(n_grid_points, )` but got {x_grid.shape}. Flatting `x_grid` but proceed with caution." )
        x_grid = x_grid.flatten()
    n_grid_points = len( x_grid )

    # Build the kernel matrix
    jitter = 1e-4 # for numerical stability
    grid_1, grid_2 = pt.meshgrid( x_grid, x_grid, indexing="ij" )
    kernel = pt.exp(-0.5 * (grid_1 - grid_2)**2 / l**2)
    K = kernel + jitter * pt.eye( n_grid_points, device=x_grid.device, dtype=x_grid.dtype )

    # Factor to Cholesky and return. This is a square matrix, no vectorization happens here.
    L = pt.linalg.cholesky( K )
    return L

def jointIndexingRBFInterpolator( L : pt.Tensor,
                                  x_grid : pt.Tensor,
                                  l : float, 
                                  x : pt.Tensor,
                                  u0 : pt.Tensor ) -> pt.Tensor:
    # Make sure the shapes and sizes match
    assert L.ndim == 2 and L.shape[0] == L.shape[1], f"`L` must be a square matrix, got shape {L.shape}."
    assert u0.ndim == 2 and u0.shape[1] == L.shape[0], f"`u0` must have shape `(B, n_grid_points)` but got {u0.shape}."
    if x.ndim == 1:
        x = x[:,None]
    assert x.ndim == 2 and x.shape[0] == u0.shape[0] and x.shape[1] == 1, f"`x` must have shape (B,1) with `B = u0.shape[0]` but got {x.shape}."

    # Solve the Cholesky system in vectorized form.
    with pt.no_grad():
        z = pt.linalg.solve_triangular(L, u0.T, upper=False)
        alpha = pt.linalg.solve_triangular(L.transpose(-1, -2), z, upper=True)

    # Build a joint-indexing RBF interpolator
    if x_grid.ndim == 1:
        x_grid_eval = x_grid[None,:]
    else:
        x_grid_eval = x_grid.T # Shape (1, n_grid_points)
    
    # Actual interpolation
    K_x_grid = pt.exp( -0.5 * (x - x_grid_eval)**2 / l**2 ) # (B, n_grid_points)
    u0_eval = K_x_grid * alpha.T
    return pt.sum( u0_eval, dim=1, keepdim=True )

def tensorizedRBFInterpolator( L : pt.Tensor, # (n_grid_points, n_grid_points)
                               x_grid : pt.Tensor, # (n_grid_points,)
                               l : float, 
                               x : pt.Tensor, # (Bt, 1)
                               u0 : pt.Tensor, # (Bb, n_grid_points)
                              ) -> pt.Tensor:
    # Make sure the shapes and sizes match
    assert L.ndim == 2 and L.shape[0] == L.shape[1], f"`L` must be a square matrix, got shape {L.shape}."
    assert u0.ndim == 2 and u0.shape[1] == L.shape[0], f"`u0` must have shape `(B, n_grid_points)` but got {u0.shape}."
    if x.ndim == 1:
        x = x[:,None]
    assert x.ndim == 2 and x.shape[0] == u0.shape[0] and x.shape[1] == 1, f"`x` must have shape (B,1) with `B = u0.shape[0]` but got {x.shape}."

    # Solve the Cholesky system in vectorized form.
    with pt.no_grad():
        z = pt.linalg.solve_triangular(L, u0.T, upper=False)
        alpha = pt.linalg.solve_triangular(L.transpose(-1, -2), z, upper=True) # (n_grid_points, Bb)

    # Build a joint-indexing RBF interpolator
    if x_grid.ndim == 1:
        x_grid_eval = x_grid[None,:]
    else:
        x_grid_eval = x_grid.T # Shape (1, n_grid_points)
    
    # Actual interpolation
    K_x_grid = pt.exp( -0.5 * (x - x_grid_eval)**2 / l**2 ) # (Bt, n_grid_points)
    u0_eval = K_x_grid @ alpha # (Bt, Bb)
    return u0_eval.T # (Bb, Bt)