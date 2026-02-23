import sys
sys.path.append('../')

import math
import torch as pt
import matplotlib.pyplot as plt

from fd import finiteDifferences
from evaluatePINO import evaluatePINO

def test_pinn( model, test_dataset ):

    pt.set_default_device( pt.device("cpu") )
    pt.set_default_dtype( pt.float64 )

    N_grid_points = 51
    x_grid = pt.linspace(0.0, 1.0, N_grid_points)

    # Take one test value and propagate it through the network
    print('Evaluating PINO')
    T_f = 1.0
    idx = 25
    _, _, p, u0 = test_dataset[idx]
    T_sol_pinn, tau_grid_pinn = evaluatePINO( model, x_grid, T_f, p, u0 )

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