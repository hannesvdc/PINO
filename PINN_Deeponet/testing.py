import json
import torch as pt
import numpy as np
import matplotlib.pyplot as plt

from model import ConvDeepONet
from dataset import DeepONetDataset

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
index = 101
forcing_dataset = DeepONetDataset(config, device, dtype)
f_sample = pt.unsqueeze(forcing_dataset.branch_input_data[index,:], dim=0) # Take a random forcing from the dataset, doesn't matter
xy_all = forcing_dataset.trunk_input_data
u_reference = np.load(config["Data Directory"] + "output_data_u.npy")[index,:].reshape(101, 101)
v_reference = np.load(config["Data Directory"] + "output_data_v.npy")[index,:].reshape(101, 101)
print(u_reference.shape)

# Create and initialize the model
optimal_posttrain_epoch = 460
optimal_network_weights = pt.load(data_directory + f'pino_epoch_results/posttrain_model_epoch={optimal_posttrain_epoch}.pth', map_location=device, weights_only=True)
network = ConvDeepONet(n_branch_conv=5, n_branch_channels=8, kernel_size=7, n_branch_nonlinear=3, n_trunk_nonlinear=5, p=100)
network.load_state_dict(optimal_network_weights)
network.to(device)

# Push the data through the network
E_train = 1.0
E_material = 300 * 1.e6
nu = 0.3
u, v = network.forward(f_sample, xy_all)
u /= (E_material / E_train)
v /= (E_material / E_train)
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

plt.figure()
plt.pcolor(X, Y, u_reference.T, cmap='jet')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r'$x$-displacements REF $u(x, y)$')
plt.colorbar()
plt.figure()
plt.pcolor(X, Y, v_reference.T, cmap='jet')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r'$y$-displacements REF $v(x, y)$')
plt.colorbar()

# Plot the forcing vector f
plt.figure()
plt.plot(Y_lin, f_sample[0,0:101], label=r'$f_1$')
plt.plot(Y_lin, f_sample[0,101:], label=r'$f_2$')
plt.legend()
plt.xlabel(r'$y$')
plt.show()