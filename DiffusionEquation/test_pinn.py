import math
import torch as pt
from PINNDataset import PINNDataset
from FixedInitialPINN import FixedInitialPINN
import matplotlib.pyplot as plt

pt.set_default_device( pt.device("cpu") )
pt.set_default_dtype( pt.float64 )

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
model.load_state_dict( pt.load(store_directory + '/model_lbfgs.pth', weights_only=True, map_location=pt.device("cpu")) )

# General PINO evaluation script
N_tau = 10001
tau_grid = pt.linspace( 1e-2, tau_max, N_tau )
def evaluatePINO(x_grid : pt.Tensor, p : pt.Tensor) -> pt.Tensor:
    B = x_grid.shape[0]
    k = p[0]
    p_batch = pt.unsqueeze(p, 0).expand( [B, len(p)] )

    # Evaluate the network in a fixed grid of tau-values
    T_sol = pt.zeros( (N_grid_points, N_tau) )
    for t_idx in range( N_tau ):
        t = tau_grid[t_idx] / k
        T_xt = model( x_grid[:,None], t * pt.ones([B,1]), p_batch )
        T_sol[:,t_idx] = T_xt[:,0]
    
    return T_sol

# Take one test value and propagate it through the network
idx = 25
_, _, p = test_dataset[idx]
T_sol_pinn = evaluatePINO( x_grid, p )
T_pinn_avg = pt.mean(T_sol_pinn, dim=0)

# Compute the leading Fourier mode. It should decay exponentially.
k = p[0]
T_s = p[1]
a1 = 2.0 * pt.trapezoid( (T_sol_pinn - T_s) * pt.sin( math.pi * x_grid[:,None] ), x=x_grid, dim=0 )
a1_ratio = a1 / a1[0]
a2 = 2.0 * pt.trapezoid( (T_sol_pinn - T_s) * pt.sin( 2.0 * math.pi * x_grid[:,None] ), x=x_grid, dim=0 )
a2_ratio = a2 / a2[0]
a1_ratio_analytic = pt.exp(-math.pi**2 * tau_grid)
a1_ratio_analytic = a1_ratio_analytic / a1_ratio_analytic[0]
a2_ratio_analytic = pt.exp(-2.0**2 * math.pi**2 * tau_grid)
a2_ratio_analytic = a2_ratio_analytic / a2_ratio_analytic[0]

# Compute the 'analytic' solution via finite differences
T0 = pt.flatten(T_max * u0 + T_s)
N_tau_fd = 200001
d_tau = tau_max / (N_tau_fd - 1)
tau_grid_fd = pt.linspace(0.0, tau_max, N_tau_fd)
T_sol_fd = pt.zeros( (N_grid_points, N_tau_fd) )
T_sol_fd[:,0] = T0
T = pt.clone( T0 )
for n in range(1, N_tau_fd):
    dt = ( tau_grid_fd[n] - tau_grid_fd[n-1] ) / k
    T_xx = (pt.roll(T, -1) - 2.0 * T + pt.roll(T, 1)) / dx**2
    T = T + dt * k * T_xx
    T[[0,-1]] = T_s
    T_sol_fd[:,n] = T
T_fd_avg = pt.mean(T_sol_fd,dim=0)

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

plt.figure()
plt.plot( tau_grid, T_pinn_avg.detach().numpy())
plt.plot( tau_grid_fd, T_fd_avg.detach().numpy())
plt.show()
# Compare the PINN and FD Surface plots.
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# X, Y = pt.meshgrid( x_grid, tau_grid, indexing="ij" )
# X_fd, Y_fd = pt.meshgrid( x_grid, tau_grid_fd, indexing="ij" )
# print(X.shape, T_sol_pinn.shape, T_sol_fd.shape)
# ax.plot_surface( X, Y, T_sol_pinn.detach().numpy(), label="PINN" ) # pyright: ignore[reportAttributeAccessIssue]
# #ax.plot_surface( X_fd, Y_fd, T_sol_fd.detach().numpy(), label="Finite Differences" ) # pyright: ignore[reportAttributeAccessIssue]
# ax.set_xlabel(r"$x$")
# ax.set_ylabel(r"$\tau$")
# ax.legend()
# plt.show()