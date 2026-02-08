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
    T_s = p[1]
    p_batch = pt.unsqueeze(p, 0).expand( [B, len(p)] )

    N_tau = 1001
    tau_grid = pt.linspace( 0, T_f, N_tau )

    # Evaluate the network in a fixed grid of tau-values
    N_grid_points = len( x_grid )
    T_sol = pt.zeros( (N_grid_points, N_tau) )
    u0 = model.u0_fcn( x_grid[:,None] )
    T0 = pt.flatten(T_max * u0 + T_s)
    T_sol[:,0] = T0
    for t_idx in range( 1, N_tau ):
        t = tau_grid[t_idx] / k
        T_xt = model( x_grid[:,None], t * pt.ones([B,1]), p_batch )
        T_sol[:,t_idx] = T_xt[:,0]
    
    return T_sol, tau_grid

def finiteDifferences( x_grid : pt.Tensor, u0 : pt.Tensor, p : pt.Tensor, T_f : float ) -> Tuple[pt.Tensor, pt.Tensor]:
    T_s = p[1]
    T0 = pt.flatten(T_max * u0 + T_s)

    # Make the time step sufficiently small
    dx = float( x_grid[1] - x_grid[0] )
    dtau_max = 0.5 * dx**2
    dtau = dtau_max / 10.0
    N_tau = math.ceil( T_f / dtau )
    dtau = T_f / N_tau
    tau_grid = pt.linspace(0.0, tau_max, N_tau)

    # Storage
    N_grid_points = len( x_grid )
    T_sol = pt.zeros( (N_grid_points, N_tau) )
    T_sol[:,0] = T0
    T = pt.clone( T0 )
    for n in range(1, N_tau): 
        T_xx = (pt.roll(T, -1) - 2.0 * T + pt.roll(T, 1)) / dx**2
        T = T + dtau * T_xx
        T[0] = T_s
        T[-1] = T_s
        T_sol[:,n] = T
    
    # Return the solution and time grid
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

    # General PINO evaluation script
    N_tau = 1001
    T_f = 4.0
    tau_grid = pt.linspace( 1e-2, T_f, N_tau )

    # Take one test value and propagate it through the network
    print('Evaluating PINO')
    idx = 25
    _, _, p = test_dataset[idx]
    T_sol_pinn, tau_grid_pinn = evaluatePINO( model, x_grid, T_f, p )

    # Calculate the finite differences ('analytic') solution
    print('Running Finite Differences')
    T_sol_fd, tau_grid_fd = finiteDifferences( x_grid, u0, p, T_f)

    # Compute the leading Fourier mode. It should decay exponentially.
    T_s = p[1]
    a1 = 2.0 * pt.trapezoid( (T_sol_pinn - T_s) * pt.sin( math.pi * x_grid[:,None] ), x=x_grid, dim=0 )
    a1_ratio = a1 / a1[0]
    a2 = 2.0 * pt.trapezoid( (T_sol_pinn - T_s) * pt.sin( 2.0 * math.pi * x_grid[:,None] ), x=x_grid, dim=0 )
    a2_ratio = a2 / a2[0]
    a1_ratio_analytic = pt.exp(-math.pi**2 * tau_grid)
    a1_ratio_analytic = a1_ratio_analytic / a1_ratio_analytic[0]
    a2_ratio_analytic = pt.exp(-2.0**2 * math.pi**2 * tau_grid)
    a2_ratio_analytic = a2_ratio_analytic / a2_ratio_analytic[0]

    # Make the decay
    plt.plot( tau_grid.numpy(), pt.abs(a1_ratio).detach().numpy(), label=r'PINO $a_1$ Ratio')
    plt.plot( tau_grid.numpy(), pt.abs(a1_ratio_analytic).numpy(), label=r'Analytic $a_1$ Ratio')
    plt.plot( tau_grid.numpy(), pt.abs(a2_ratio).detach().numpy(), label=r'PINO $a_2$ Ratio')
    plt.plot( tau_grid.numpy(), pt.abs(a2_ratio_analytic).numpy(), label=r'Analytic $a_2$ Ratio')
    plt.xlabel(r'$\tau$')
    plt.title(r"$\frac{a_n(t)}{a_n(0)}$")
    plt.legend()

    plt.figure()
    plt.semilogy( tau_grid.numpy(), pt.abs(a1).detach().numpy(), label=r'PINO $a_1(\tau)$')
    plt.semilogy( tau_grid.numpy(), pt.abs(a2).detach().numpy(), label=r'PINO $a_2(\tau)$')
    plt.xlabel(r'$\tau$')
    plt.title(r"$|a_n(t)|$")
    plt.legend()

    # Plot PINN and FD side by side
    fig, (ax1, ax2) = plt.subplots(1,2)
    vmin = min( pt.min(T_sol_pinn), pt.min(T_sol_fd))
    vmax = max( pt.max(T_sol_pinn), pt.max(T_sol_fd))
    idx_pinn = (tau_grid_pinn <= 1.0)
    X, Y = pt.meshgrid(x_grid, tau_grid_pinn[idx_pinn], indexing="ij")
    ax1.pcolormesh( X.detach().numpy(), Y.detach().numpy(), T_sol_pinn[:,idx_pinn].detach().numpy(), vmin=float(vmin), vmax=float(vmax))
    ax1.set_title("PINO")
    idx_fd = (tau_grid_fd <= 1.0)
    X, Y = pt.meshgrid(x_grid, tau_grid_fd[idx_fd], indexing="ij")
    ax2.pcolormesh( X.detach().numpy(), Y.detach().numpy(), T_sol_fd[:,idx_fd].detach().numpy(), vmin=float(vmin), vmax=float(vmax))
    ax2.set_title("Finite Differences")

    plt.show()