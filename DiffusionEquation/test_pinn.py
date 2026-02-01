import math
import torch as pt
from PINNDataset import PINNDataset
from FixedInitialPINN import FixedInitialPINN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create the initial condition. We want T_max to be ~95th percentile, so std=T_max/1.96
T_max = 10.0
tau_max = 8.0 # train to exp( -tau_max )
logk_max = math.log( 1e2 )
N_grid_points = 51
x_grid = pt.linspace(0.0, 1.0, N_grid_points)
dx = x_grid[1] - x_grid[0]
l = 0.12 # GP correlation length
N_test = 100
test_dataset = PINNDataset( N_test, T_max, tau_max, test=True )

# Create the training and validation datasets
store_directory = './Results/pinn/'
u0 = pt.load(store_directory + 'initial.pth', weights_only=True)
z = 64
n_hidden_layers = 2
model = FixedInitialPINN( n_hidden_layers, z, T_max, tau_max, logk_max, u0, x_grid, l )
model.load_state_dict( pt.load(store_directory + '/model_adam.pth', weights_only=True, map_location=pt.device("cpu")) )

# General PINO evaluation script
N_tau = 1001
tau_grid = pt.linspace( 1e-2, tau_max, N_tau )
def evaluatePINO(x_grid : pt.Tensor, p : pt.Tensor) -> pt.Tensor:
    B = x_grid.shape[0]
    k = p[0]
    p_batch = pt.unsqueeze(p, 0).expand( [B, p.shape[1]] )
    
    # Evaluate the network in a fixed grid of tau-values
    T_sol = pt.zeros( (N_grid_points, N_tau) )
    for t_idx in range( N_tau ):
        t = tau_grid[t_idx] / k
        T_xt = model( x_grid, t * pt.ones_like(x_grid), p_batch )
        T_sol[:,t_idx] = T_xt
    
    return T_sol

# Take one test value and propagate it through the network
idx = 25
_, _, p = test_dataset[idx]
T_sol_pinn = evaluatePINO( x_grid, p )

# Compute the leading Fourier mode. It should decay exponentially.
k = p[0]
T_s = p[1]
a1 = 2.0 * pt.trapezoid( (T_sol_pinn - T_s) * pt.sin( math.pi * x_grid ), x=x_grid, dim=0 )
a1_ratio = a1 / a1[0]
a1_ratio_analytic = pt.exp(-math.pi**2 * k * tau_grid)

# Compute the 'analytic' solution via finite differences
T0 = pt.flatten(T_max * u0 + T_s)
d_tau = 0.01
N_tau = int(tau_max / d_tau) + 1
tau_grid_fd = pt.linspace(0.0, tau_max, N_tau)
T_sol_fd = pt.zeros( (N_grid_points, N_tau) )
T_sol_fd[:,0] = T0
T = pt.clone( T0 )
for n in range(1, N_tau):
    dt = ( tau_grid_fd[n] - tau_grid_fd[n-1] ) / k
    T_xx = (pt.roll(T, -1) - 2.0 * T + pt.roll(T, 1)) / dx**2
    T = T + dt * k * T_xx
    T[[0,-1]] = T_s
    T_sol_fd[:,n] = T

# Make the decay
plt.plot( tau_grid.numpy(), a1_ratio.numpy(), label='PINO Ratio')
plt.plot( tau_grid.numpy(), a1_ratio_analytic.numpy(), label='Analytic Ratio')
plt.xlabel(r'$\tau$')
plt.title(r"$\frac{a_1(t)}{a_1(0)}$")
plt.legend()
plt.show()

# Compare the PINN and FD Surface plots.
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
X, Y = pt.meshgrid( x_grid, tau_grid, indexing="ij" )
ax.plot_surface( X, Y, T_sol_pinn, label="PINN" ) # pyright: ignore[reportAttributeAccessIssue]
ax.plot_surface( X, Y, T_sol_fd, label="Finite Differences" ) # pyright: ignore[reportAttributeAccessIssue]
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$t$")
ax.legend()
plt.show()