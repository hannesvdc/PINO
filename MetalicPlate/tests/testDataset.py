import sys
sys.path.append('../')

from PlateDataset import PlateDataset
from rbf_interpolation import buildCholeskyMatrix, jointIndexingRBFInterpolator

import torch as pt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dtype = pt.float64
pt.set_grad_enabled( True )
pt.set_default_dtype( dtype )
store_directory = './Results/'

# Create the training and validation datasets
nu_max = 0.45
n_grid_points = 101
y_grid = pt.linspace(0.0, 1.0, n_grid_points)
l = 0.2

N_train_branch = 101
N_train_trunk = 10_000
B = N_train_branch * N_train_trunk
train_dataset = PlateDataset( N_train_branch, N_train_trunk, n_grid_points, l, nu_max, dtype, plot=False )
train_dataloader = DataLoader( train_dataset, B, shuffle=True )

# Plot the dataset
for batch_idx, (x, y, nu, gx, gy) in enumerate( train_dataloader ):

    # Plot x-values
    plt.hist( x.flatten().detach().cpu().numpy(), density=True, bins=100 )
    plt.xlabel(r"$x$")

    # Plot y-values
    plt.figure()
    plt.hist( y.flatten().detach().cpu().numpy(), density=True, bins=100 )
    plt.xlabel(r"$y$")

    plt.figure()
    plt.hist( nu.flatten().detach().cpu().numpy(), density=True, bins=100 )
    plt.xlabel(r"$\nu$")

    plt.show()