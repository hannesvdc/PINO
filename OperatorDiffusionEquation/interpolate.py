import torch as pt

def interpolateInitialsTensorized(x: pt.Tensor, 
                                  u0: pt.Tensor,
                                  x_min: float = 0.0, 
                                  x_max: float = 1.0) -> pt.Tensor:
    """
    Piecewise-linear interpolation of initial conditions u0(x) on a uniform grid.

    Args:
        x:  (Bt, 1) or (Bt,) query points in [x_min, x_max]
        u0: (Bb, n_grid_points) initial conditions on uniform grid

    Returns:
        u0_at_x: (Bb, Bt)
    """
    if x.ndim == 2:
        x = x.squeeze(1)  # (Bt,)
    Bb, n = u0.shape
    Bt = x.shape[0]

    # Uniform grid spacing
    dx = (x_max - x_min) / (n - 1)

    # Clamp x to domain for sanity
    x = x.clamp(x_min, x_max)
    s = (x - x_min) / dx                     # (Bt,)

    # Compute the left and right indices of the
    # interpolation interval with boundary checks
    i0 = pt.floor(s).to(pt.long)             # (Bt,)
    i0 = pt.clamp(i0, 0, n - 2)
    i1 = i0 + 1

    # Broadcast these indices
    idx0 = i0[None, :].expand(Bb, Bt)        # (Bb, Bt)
    idx1 = i1[None, :].expand(Bb, Bt)

    # Evaluate using torch's `gather`. Extremely efficient by vectorization.
    u0_0 = pt.gather(u0, dim=1, index=idx0)  # (Bb, Bt)
    u0_1 = pt.gather(u0, dim=1, index=idx1)

    # Monotone weights between 0 and 1
    w = (s - i0.to(s.dtype)).to(u0.dtype)   # (Bt,)
    w = w[None, :]                          # (1, Bt)
    return (1.0 - w) * u0_0 + w * u0_1      # (Bb, Bt)