import sys
sys.path.append('../')

import math
import numpy as np
import torch as pt
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import getGradientNorm

import matplotlib.pyplot as plt

from PlateDataset import BranchDataset, TrunkDataset, BoundaryDataset
from TrunkFiLMNetwork import TrunkFilmNetwork
from EnergyLoss import EnergyLoss

from typing import List

# Do everything on the CPU in double precision first
dtype = pt.float64
pt.set_default_dtype( dtype )

# Physics parameters
n_grid_points = 101
nu_max = 0.45
l = 0.2

# Create a training and validation dataset
B = 512
N_train_branch = 500
N_train_trunk = 10_000
N_train_bc = 10_000
N_validation_branch = 10
N_validation_trunk = 1000
N_validation_bc = 1000
validation_branch_dataset = BranchDataset( N_validation_branch, n_grid_points, l, nu_max, dtype, plot=False )
validation_trunk_dataset = TrunkDataset( N_validation_trunk)
validation_bc_dataset = BoundaryDataset( N_validation_bc, dtype )

# Setup the network
n_hidden_layers = 4
z = 128
film_channels = [2, 8, 16, 32, 32, 32, 32, 32, 32, 32, 32] # increase gradually
y_grid = validation_branch_dataset.y_grid
model = TrunkFilmNetwork( film_channels, n_grid_points, n_hidden_layers, z, nu_max )
print('Number of Trainable Parameters: ', sum( [ p.numel() for p in model.parameters() if p.requires_grad ]))

# Heat loss fcn
loss_fcn = EnergyLoss( y_grid, l )

# Translate the model to GPU
device = pt.device( "mps" )
dtype = pt.float32
model = model.to(device=device, dtype=dtype)
loss_fcn.to( device=device, dtype=dtype )

# Setup the optimizer and learning rate scheduler
lr = 1e-4
optimizer = optim.Adam( model.parameters(), lr )

# Scheduler: constant for the first `n_epochs` epochs, decrease by cosine for `annealing_epochs` later.
# min_lr  = 1e-6
# warm_epochs = 1000
# anneal_epochs = 10000
# scheduler = optim.lr_scheduler.CosineAnnealingLR(
#     optimizer,
#     T_max=anneal_epochs,
#     eta_min=min_lr
# )

# Load the validation dataset all at once
x_val, y_val = validation_trunk_dataset.all()
x_val = x_val.to(device=device, dtype=dtype)
y_val = y_val.to(device=device, dtype=dtype)
gx_val, gy_val, nu_val = validation_branch_dataset.all()
gx_val = gx_val.to(device=device, dtype=dtype)
gy_val = gy_val.to(device=device, dtype=dtype)
nu_val = nu_val.to(device=device, dtype=dtype)
y_bc_val = validation_bc_dataset.all()
y_bc_val = y_bc_val.to(device=device, dtype=dtype)

# Main training routine
train_counter : List = []
train_losses : List = []
train_grads : List = []
def train_epoch( epoch : int ):
    model.train( )

    # Sample a new dataset every time
    branch_dataset = BranchDataset( N_validation_branch, n_grid_points, l, nu_max, dtype, plot=False )
    trunk_dataset = TrunkDataset( N_validation_trunk)
    bc_dataset = BoundaryDataset( N_validation_bc, dtype )
    branch_loader = DataLoader( branch_dataset, batch_size=B, shuffle=True )

    max_batches = 50
    for _batch_idx, (gx_b, gy_b, nu_b) in enumerate( branch_loader ):
        batch_idx = _batch_idx+1
        if batch_idx > max_batches: # avoid early overtraining by stopping soon enough
            return
        
        # Fetch the trunk inputs
        x_b, y_b = trunk_dataset.all()
        y_bc_b = bc_dataset.all()
        
        optimizer.zero_grad( set_to_none=True )
        
        # Move to the GPU
        x_b = x_b.to( device=device, dtype=dtype )
        y_b = y_b.to( device=device, dtype=dtype )
        nu_b = nu_b.to( device=device, dtype=dtype )
        y_bc_b = y_bc_b.to( device=device, dtype=dtype )
        gx_b = gx_b.to( device=device, dtype=dtype )
        gy_b = gy_b.to( device=device, dtype=dtype )

        # Compute the loss and its gradient
        loss, loss_info = loss_fcn( model, x_b, y_b, nu_b, y_bc_b, gx_b, gy_b )
        loss.backward()
        loss_grad = getGradientNorm( model )

        # Update the weights internally
        pt.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step( )

        # Keep track of important metrics
        energy = loss_info["energy"]
        boundary = loss_info["boundary"]
        if batch_idx == 1:
            train_counter.append( epoch-1 + batch_idx / max_batches)
            train_losses.append( float(loss.item()) )
            train_grads.append( float(loss_grad.item()) )

            progress = 100.0 * batch_idx / max_batches
            print_str = (
                f"\nEpoch {epoch:04d} "
                f"[{batch_idx:03d}/{max_batches:03d} | {progress:6.2f}%]  "
                f"Loss: {loss.item():.3e}  "
                f"Grad: {loss_grad.item():.3e}  "
                f"Lr: {optimizer.param_groups[0]['lr']:.3e}"
            )
            print(print_str)

            info_str = (
                f"Energy = {energy:.3e}   "
                f"Boundary = {boundary:.3e}   "
            )
            print(info_str)

# Validation function
validation_counter : List = []
validation_losses : List = []
def validate_epoch( epoch : int ):
    model.eval( )

    # Compute the loss and its gradient
    loss, loss_info = loss_fcn( model, x_val, y_val, nu_val, y_bc_val, gx_val, gy_val )

    # Store
    validation_counter.append( epoch )
    validation_losses.append( float(loss.item()) )

    # Print and done.
    print_str = f'\nValidation Epoch {epoch:03d}: \tLoss: {loss.item():.3e}'
    print(print_str)

# Main training loop
store_directory = './Results/'
n_epochs = 10_000
try:
    for epoch in range( 1, n_epochs+1 ):
        # Train using the new dataset
        train_epoch( epoch )

        # Validate on independent but fixed data
        validate_epoch( epoch )

        # Store the current model and optimizer weights.
        pt.save( model.state_dict(), store_directory + 'model_adam.pth')
        pt.save( optimizer.state_dict(), store_directory + 'optimizer_adam.pth')
except KeyboardInterrupt:
    print( 'Aborting Training.')
    pass

# Make numpy arrays from the training data
train_counter = np.array( train_counter ) # type: ignore
train_losses = np.array( train_losses ) # type: ignore
train_grads = np.array( train_grads ) # type: ignore
train_data = np.stack( (train_counter[:,np.newaxis], train_losses[:,np.newaxis], train_grads[:,np.newaxis]), axis=1) # type: ignore
validation_counter = np.array( validation_counter ) # type: ignore
validation_losses = np.array( validation_losses ) # type: ignore
validation_data = np.stack( (validation_counter[:,np.newaxis], validation_losses[:,np.newaxis]), axis=1) # type: ignore
np.save( store_directory + 'train_data.npy', train_data)
np.save( store_directory + 'validation_data.npy', validation_data)

# Make a plot of the training progress

fig, ax1 = plt.subplots()
ax1.plot(train_counter, train_losses, color="tab:blue", alpha=0.7, label="Train Loss")
ax1.plot(validation_counter, validation_losses, color="tab:orange", alpha=0.7, label="Validation Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Energy", color="black")
ax1.tick_params(axis="y")
ax2 = ax1.twinx()
ax2.semilogy(train_counter, train_grads, color="tab:red", alpha=0.7, label="Grad Norm")
ax2.set_ylabel("Gradient Norm", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
plt.tight_layout()
plt.savefig(store_directory + 'convergence_adam.png', transparent=True)
plt.show()