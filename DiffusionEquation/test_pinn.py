import sys
sys.path.append('../')

import math
import torch as pt
import torch.nn as nn
import matplotlib.pyplot as plt
from generateInitialCondition import build_u0_evaluator

from typing import Tuple

T_max = 10.0
tau_max = 8.0 # train to exp( -tau_max )
logk_max = math.log( 1e2 )

def evaluatePINO( model : nn.Module, x_grid : pt.Tensor, T_f : float, p : pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
    B = x_grid.shape[0]
    k = p[0]
    p_batch = pt.unsqueeze(p, 0).expand( [B, len(p)] )

    N_tau = 1001
    tau_grid = pt.linspace( 1e-2, T_f, N_tau )

    # Evaluate the network in a fixed grid of tau-values
    N_grid_points = len( x_grid )
    T_sol = pt.zeros( (N_grid_points, N_tau) )
    for t_idx in range( 0, N_tau ):
        t = tau_grid[t_idx] / k
        T_xt = model( x_grid[:,None], t * pt.ones([B,1]), p_batch )
        T_sol[:,t_idx] = T_xt[:,0]
    
    return T_sol, tau_grid

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

def test_pinn( model, test_dataset ):

    pt.set_default_device( pt.device("cpu") )
    pt.set_default_dtype( pt.float64 )

    N_grid_points = 51
    x_grid = pt.linspace(0.0, 1.0, N_grid_points)

    # Load the initial condition
    l = 0.12
    u0_fcn, ic = build_u0_evaluator( l, pt.device("cpu"), pt.float64 )
    u0 = u0_fcn( x_grid[:,None] )

    # Take one test value and propagate it through the network
    print('Evaluating PINO')
    T_f = 1.0
    idx = 25
    _, _, p = test_dataset[idx]
    T_sol_pinn, tau_grid_pinn = evaluatePINO( model, x_grid, T_f, p )

    # Calculate the finite differences ('analytic') solution
    print('Running Finite Differences')
    T_sol_fd, tau_grid_fd = finiteDifferences( x_grid, T_sol_pinn[:,0], p, T_f)

    # Compute the leading Fourier mode. It should decay exponentially.
    T_s = p[1]
    a1 = 2.0 * pt.trapezoid( (T_sol_pinn - T_s) * pt.sin( math.pi * x_grid[:,None] ), x=x_grid, dim=0 )
    a1_ratio = a1 / a1[0]
    a2 = 2.0 * pt.trapezoid( (T_sol_pinn - T_s) * pt.sin( 2.0 * math.pi * x_grid[:,None] ), x=x_grid, dim=0 )
    a2_ratio = a2 / a2[0]
    a1_ratio_analytic = pt.exp(-math.pi**2 * tau_grid_pinn)
    a1_ratio_analytic = a1_ratio_analytic / a1_ratio_analytic[0]
    a2_ratio_analytic = pt.exp(-2.0**2 * math.pi**2 * tau_grid_pinn)
    a2_ratio_analytic = a2_ratio_analytic / a2_ratio_analytic[0]

    # Make the decay
    plt.plot( tau_grid_pinn.numpy(), pt.abs(a1_ratio).detach().numpy(), label=r'PINO $a_1$ Ratio')
    plt.plot( tau_grid_pinn.numpy(), pt.abs(a1_ratio_analytic).numpy(), label=r'Analytic $a_1$ Ratio', linestyle='dashed')
    plt.plot( tau_grid_pinn.numpy(), pt.abs(a2_ratio).detach().numpy(), label=r'PINO $a_2$ Ratio')
    plt.plot( tau_grid_pinn.numpy(), pt.abs(a2_ratio_analytic).numpy(), label=r'Analytic $a_2$ Ratio', linestyle='dashed')
    plt.xlabel(r'$\tau$')
    plt.title(r"$\frac{a_n(\tau)}{a_n(0)}$")
    plt.legend()

    plt.figure()
    plt.semilogy( tau_grid_pinn.numpy(), pt.abs(a1_ratio).detach().numpy(), label=r'PINO $a_1(\tau)$')
    plt.semilogy( tau_grid_pinn.numpy(), pt.abs(a1_ratio_analytic).numpy(), label=r'Analytic $a_1$ Ratio', linestyle='dashed')
    plt.semilogy( tau_grid_pinn.numpy(), pt.abs(a2_ratio).detach().numpy(), label=r'PINO $a_2(\tau)$')
    plt.semilogy( tau_grid_pinn.numpy(), pt.abs(a2_ratio_analytic).numpy(), label=r'Analytic $a_2$ Ratio', linestyle='dashed')
    plt.xlabel(r'$\tau$')
    plt.title(r"$|a_n(\tau)|$")
    plt.legend()

    # Plot PINN and FD side by side
    fig, (ax1, ax2) = plt.subplots(1,2)
    vmin = min( pt.min(T_sol_pinn), pt.min(T_sol_fd))
    vmax = max( pt.max(T_sol_pinn), pt.max(T_sol_fd))
    idx_pinn = ( 0.01 <= tau_grid_pinn) & (tau_grid_pinn <= 0.2)
    X, Y = pt.meshgrid(x_grid, tau_grid_pinn[idx_pinn], indexing="ij")
    ax1.pcolormesh( X.detach().numpy(), Y.detach().numpy(), T_sol_pinn[:,idx_pinn].detach().numpy(), vmin=float(vmin), vmax=float(vmax), cmap='jet')
    ax1.set_title("PINO")
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$\tau$", rotation=0)
    idx_fd = (0.01 <= tau_grid_fd) & (tau_grid_fd <= 0.2)
    X, Y = pt.meshgrid(x_grid, tau_grid_fd[idx_fd], indexing="ij")
    ax2.pcolormesh( X.detach().numpy(), Y.detach().numpy(), T_sol_fd[:,idx_fd].detach().numpy(), vmin=float(vmin), vmax=float(vmax), cmap='jet')
    ax2.set_title("PDE")
    ax2.set_xlabel(r"$x$")
    ax2.set_yticks([])
    plt.show()