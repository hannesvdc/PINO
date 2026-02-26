import json
import torch as pt
import numpy as np
import matplotlib.pyplot as plt

from model import ElasticityFiLMPINN
from dataset import ElasticityDataset

# Just some sanity pytorch settings
pt.set_grad_enabled(True)
pt.set_default_dtype(pt.float32)
device = pt.device('cpu')
dtype = pt.float32

# Load the data configuration
config_file = 'DataConfig.json'
config = json.load(open(config_file))
store_directory = config['Store Directory']
data_directory = config['Data Directory']

# Load the data in memory
batch_size = 128
m_boundary = 101
forcing_dataset = ElasticityDataset(config, device, dtype, reduce=True)
forcing_data = forcing_dataset.branch_input_data # shape (N_fcn, 2 * m_boundary)
gx = forcing_data[:, :m_boundary]
gy = forcing_data[:, m_boundary:]
g_batch = pt.stack([gx, gy], dim=1) # (N, 2, m_boundary)
xy_all = forcing_dataset.trunk_input_data
B = xy_all.shape[0]
print(xy_all.shape)

# Create and initialize the model
m_boundary = 101
n_features = 64
trunk_hidden = 128
trunk_depth = 8
network = ElasticityFiLMPINN(m_boundary, n_features, trunk_hidden, trunk_depth)
network.to(device)
print('Number of Trainable Parameters: ', network.getNumberOfParameters())

# Load the network state corresponding to this optimal physics weight.
optimal_weight_state = pt.load( store_directory + f"post_bfgs_training/epoch_0023.pth", map_location=device, weights_only=True)
network.load_state_dict(optimal_weight_state["model_state_dict"])

# Physics parameters
E_train = 1.0
E_material = 300 * 1.e6
nu = 0.3

# Take one random f_sample, replicate it and push through the network
rand_idx = 98
g_batch = g_batch[rand_idx:rand_idx+1,:,:].repeat(B, 1, 1)
print(g_batch.shape)
uv = network.forward(xy_all, g_batch)

u = uv[:,0] / (E_material / E_train)
v = uv[:,1] / (E_material / E_train)
u = u.reshape((forcing_dataset.grid_points, forcing_dataset.grid_points))
v = v.reshape((forcing_dataset.grid_points, forcing_dataset.grid_points))

# Plot the displacement field
X_lin = np.linspace(0.0, 1.0, forcing_dataset.grid_points)
Y_lin = np.linspace(0.0, 1.0, forcing_dataset.grid_points)
X, Y = np.meshgrid(X_lin, Y_lin)
plt.pcolor(X, Y, u.detach().numpy().T, cmap='jet')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r'$x$-displacements $u(x, y)$')
plt.colorbar()
plt.figure()
plt.pcolor(X, Y, v.detach().numpy().T, cmap='jet')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r'$y$-displacements $v(x, y)$')
plt.colorbar()

# plt.figure()
# plt.pcolor(X, Y, u_reference.T, cmap='jet')
# plt.xlabel(r"$x$")
# plt.ylabel(r"$y$")
# plt.title(r'$x$-displacements REF $u(x, y)$')
# plt.colorbar()
# plt.figure()
# plt.pcolor(X, Y, v_reference.T, cmap='jet')
# plt.xlabel(r"$x$")
# plt.ylabel(r"$y$")
# plt.title(r'$y$-displacements REF $v(x, y)$')
# plt.colorbar()

# Plot the forcing vector f
plt.figure()
plt.plot(Y_lin, g_batch[0,0,:], label=r'$f_1$')
plt.plot(Y_lin, g_batch[0,1,:], label=r'$f_2$')
plt.legend()
plt.xlabel(r'$y$')
plt.show()