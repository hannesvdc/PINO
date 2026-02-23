import math
import torch as pt

from typing import Tuple

def finiteDifferences( x_grid : pt.Tensor, T0 : pt.Tensor, p : pt.Tensor, T_f : float ) -> Tuple[pt.Tensor, pt.Tensor]:
    # Extract boundary temperature (Dirichlet value)
    T_s = p[1]

    # Flatten shapes defensively
    xg = x_grid.reshape(-1)
    T = T0.reshape(-1).clone()

    # Basic grid info
    dx = float(xg[1] - xg[0])
    N = xg.numel()

    # Choose a stable explicit timestep in tau:
    dtau = 0.5 * dx * dx / 100.0
    N_tau = max(2, int(math.ceil(T_f / dtau)) + 1)  # include tau=0
    dtau = T_f / (N_tau - 1)

    # Proper tau grid consistent with dtau and including tau=0
    tau_grid = 1.e-2 + dtau * pt.arange(N_tau, dtype=xg.dtype, device=xg.device)
    T_sol = pt.zeros((N, N_tau), dtype=T.dtype, device=T.device)

    # Enforce BCs at tau=0 as well
    T[0] = T_s
    T[-1] = T_s
    T_sol[:, 0] = T

    # Time stepping
    for n in range(1, N_tau):
        # Second derivative with *Dirichlet* boundaries (no pt.roll / no periodicity)
        T_xx = pt.zeros_like(T)
        T_xx[1:-1] = (T[2:] - 2.0 * T[1:-1] + T[:-2]) / (dx * dx)

        # Forward Euler step in tau
        T = T + dtau * T_xx

        # Re-impose Dirichlet BCs
        T[0] = T_s
        T[-1] = T_s

        # Store
        T_sol[:, n] = T

    return T_sol, tau_grid