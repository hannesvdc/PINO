import sys
sys.path.append('../')

from TensorizedDataset import TensorizedDataset
from rbf_interpolation import buildCholeskyMatrix, jointIndexingRBFInterpolator

import torch as pt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


dtype = pt.float64
pt.set_grad_enabled( True )
pt.set_default_dtype( dtype )
store_directory = './Results/'

# Create the training and validation datasets
T_max = 10.0
tau_max = 8.0 # train to exp( -tau_max )
n_grid_points = 51
x_grid = pt.linspace(0.0, 1.0, n_grid_points)
l = 0.2

N_train_branch = 101
N_train_trunk = 10_000
B = N_train_branch * N_train_trunk
train_dataset = TensorizedDataset( N_train_branch, N_train_trunk, n_grid_points, l, T_max, tau_max, dtype, plot=False )
train_dataloader = DataLoader( train_dataset, B, shuffle=True )

# Plot the dataset
for batch_idx, (x, t, params, u0) in enumerate( train_dataloader ):
    k = params[:,0]

    # Plot x-values
    plt.hist( x.flatten().detach().cpu().numpy(), density=True, bins=100 )
    plt.xlabel(r"$x$")

    plt.figure()
    plt.hist( (t.flatten()*k).detach().cpu().numpy(), density=True, bins=100 )
    plt.xlabel(r"$\tau$")

    plt.figure()
    plt.hist( pt.log(k).flatten().detach().cpu().numpy(), density=True, bins=100 )
    plt.xlabel(r"$k$")

    plt.show()

u0 = train_dataset.branch_dataset.data
plt.figure()
plt.plot(x_grid.numpy(), u0.T.numpy())
plt.show()

# Take one u0, interpolate and calculate derivatives
x_values = pt.linspace(0.0, 1.0, 1001, requires_grad=True)[:,None]
u0 = u0[0:1,:]
u0_repmat = u0.repeat(len(x_values),1)

# Interpolate
l = train_dataset.branch_dataset.l
L = buildCholeskyMatrix( x_grid, l )
u0_int = jointIndexingRBFInterpolator( L, x_grid, l, x_values, u0_repmat)

# Differentiate
dudx = pt.autograd.grad(outputs=u0_int, inputs=x_values, grad_outputs=pt.ones_like(x_values), create_graph=True)[0]
dduddx = pt.autograd.grad(outputs=dudx, inputs=x_values, grad_outputs=pt.ones_like(dudx), create_graph=True)[0]

# Plot everyghint
plt.figure()
plt.plot( x_values.detach().flatten().numpy(), u0_int.detach().flatten().numpy(), label=r"$u_0$")
plt.plot( x_values.detach().flatten().numpy(), dudx.detach().flatten().numpy(), label=r"$du/dx$")
plt.plot( x_values.detach().flatten().numpy(), dduddx.detach().flatten().numpy(), label=r"$d^2u/dx^2$")
plt.xlabel(r"$x$")
plt.legend()
plt.show()