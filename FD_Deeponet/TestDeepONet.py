import torch as pt
import json
import gc

import numpy as np
import numpy.linalg as lg
import scipy as sc
import numpy.random as rd
import matplotlib.pyplot as plt

from DeepONet import DeepONet
from Dataset import DeepONetDataset
import ElasticPDE as pde
import GaussianProces as gp
    
# No need for gradients in test script
pt.set_grad_enabled(False)
device = pt.device('cpu')
dtype = pt.float32

# Load the data configuration
config_file = 'DataConfig.json'
config = json.load(open(config_file))
store_directory = config['Store Directory']
dataset = DeepONetDataset(config, device, dtype)
N = dataset.N

# Initialize the Network and the Optimizer (Adam)
print('\nLoading the DeepONet...')
p = 300
branch_layers = [202, 512, 512, 512, 512, 2*p]
trunk_layers = [2, 512, 512, 512, 512, 2*p]
network = DeepONet(branch_layers=branch_layers, trunk_layers=trunk_layers)
network.load_state_dict(pt.load(store_directory + 'model_vanilla.pth', weights_only=True))

# Generate a proper forcing, on average 3 standard deviations away from the mean
l = 0.12
y_points = np.linspace(0.0, 1.0, N+1)
K = gp.precompute_covariance(y_points, l)
R = np.real(sc.linalg.sqrtm(K))
rng = rd.RandomState()
z1 = 2.0 * rng.multivariate_normal(np.zeros(N+1), np.eye(N+1)) / np.sqrt(N+1)
z2 = 2.0 * rng.multivariate_normal(np.zeros(N+1), np.eye(N+1)) / np.sqrt(N+1)
forcing = (np.dot(R, z1), np.dot(R, z2))
print('Number of Stdevs:', lg.norm(z1), lg.norm(z2))

# Forward the forcing through the network
branch_input = pt.tensor(np.concatenate(forcing), device=device, dtype=dtype)[None,:]
nn_output_u, nn_output_v = network.forward(branch_input, dataset.trunk_input_data)
nn_output_u = nn_output_u * dataset.scale_u
nn_output_v = nn_output_v * dataset.scale_v

# Solve the Elastostatic PDE with given forcing
print('\nComputing Solution to Elastostatic PDE...')
N = 100
E = 410.0 * 1.e3
mu = 0.3
matrix_filename = 'FD_Matrix_mu=' + str(mu) + '.npy'
A = np.load(config['Matrix Directory'] + matrix_filename)
forcing = np.zeros((N+1, 2))
forcing[:,0] = branch_input[0,:(N+1)].numpy()
forcing[:,1] = branch_input[0,(N+1):].numpy()
lu, pivot = sc.linalg.lu_factor(A)
(u, v) = pde.solveElasticPDE(lu, pivot, E, mu, forcing, N)

del A
del dataset
del lu
del pivot
gc.collect()

# Plot result - Everything are numpy arrays from here on out
print('\nPlotting...')
x_space = np.linspace(0.0, 1.0, N+1)
y_space = np.linspace(0.0, 1.0, N+1)
X, Y = np.meshgrid(x_space, y_space)
u_nn = nn_output_u.numpy().reshape((N+1, N+1)).transpose()
v_nn = nn_output_v.numpy().reshape((N+1, N+1)).transpose()
u_min = min(np.min(u), np.min(u_nn))
u_max = max(np.max(u), np.max(u_nn))
v_min = min(np.min(v), np.min(v_nn))
v_max = max(np.max(v), np.max(v_nn))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True, sharex=True, sharey=True)
fig.set_constrained_layout_pads(wspace=0.05)
m1 = ax1.pcolormesh(X, Y, u, shading='auto', cmap='jet', vmin=u_min, vmax=u_max)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
ax1.set_title(r'FEM Displacement $u(x,y)$')

m2 = ax2.pcolormesh(X, Y, u_nn, shading='auto', cmap='jet', vmin=u_min, vmax=u_max)
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$y$')
ax2.set_title(r'DeepONet Displacement $u(x,y)$')

for ax in (ax1, ax2):
    ax.set_aspect('equal', adjustable='box')  # lock aspect, resize box if needed
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
fig.colorbar(m2, ax=[ax1, ax2], 
                    location='right', 
                    shrink=0.75, 
                    pad=0.02)
plt.savefig('./Results/u_displacement.png', transparent=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True, sharex=True, sharey=True)
fig.set_constrained_layout_pads(wspace=0.05)
m1 = ax1.pcolormesh(X, Y, v, shading='auto', cmap='jet', vmin=u_min, vmax=u_max)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
ax1.set_title(r'FEM Displacement $v(x,y)$')

m2 = ax2.pcolormesh(X, Y, v_nn, shading='auto', cmap='jet', vmin=u_min, vmax=u_max)
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$y$')
ax2.set_title(r'DeepONet Displacement $v(x,y)$')

for ax in (ax1, ax2):
    ax.set_aspect('equal', adjustable='box')  # lock aspect, resize box if needed
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
fig.colorbar(m2, ax=[ax1, ax2], 
                    location='right', 
                    shrink=0.75, 
                    pad=0.02)

plt.show()