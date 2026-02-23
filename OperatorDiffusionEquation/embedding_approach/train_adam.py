import sys
sys.path.append('../')

import math
import numpy as np
import torch as pt
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import getGradientNorm

import matplotlib.pyplot as plt

from TensorizedDataset import TensorizedDataset
from BranchEmbeddingNetwork import BranchEmbeddingNetwork
from Loss import HeatLoss

from typing import List

# Do everything on the CPU in double precision first
dtype = pt.float64
pt.set_default_dtype( dtype )

# Physics parameters
n_grid_points = 51
T_max = 10.0
tau_max = 8.0
logk_max = math.log(1e2)
l = 0.5

# Create a training and validation dataset
sampling_strat = "initial_bias"
B = 512
N_train_branch = 500
N_train_trunk = 10_000
N_validation_branch = 10
N_validation_trunk = 1000
validation_dataset = TensorizedDataset( N_validation_branch, N_validation_trunk, n_grid_points, 
                                       l, T_max, tau_max, logk_max, dtype, plot=False, tau_sampling=sampling_strat)

# Setup the network
n_embedding_hidden_layers = 4
n_hidden_layers = 4
z = 64
q = 32
x_grid = validation_dataset.branch_dataset.x_grid
model = BranchEmbeddingNetwork( n_embedding_hidden_layers, n_hidden_layers, z, q, x_grid, l, T_max, tau_max, logk_max)
print('Number of Trainable Parameters: ', sum( [ p.numel() for p in model.parameters() if p.requires_grad ]))

# Translate the model to GPU
device = pt.device( "mps" )
dtype = pt.float32
model = model.to(device=device, dtype=dtype)

# Heat loss fcn
loss_fcn = HeatLoss( )

# Setup the optimizer and learning rate scheduler
lr = 1e-4
optimizer = optim.Adam( model.parameters(), lr )

# Scheduler: constant for the first `n_epochs` epochs, decrease by cosine for `annealing_epochs` later.
min_lr  = 1e-6
warm_epochs = 1000
anneal_epochs = 10000
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=anneal_epochs,
    eta_min=min_lr
)

# Load the validation dataset all at once
x_val, t_val, params_val, u0_val = validation_dataset.all()
x_val = x_val.to(device=device, dtype=dtype)
t_val = t_val.to(device=device, dtype=dtype)
params_val = params_val.to(device=device, dtype=dtype)
u0_val = u0_val.to(device=device, dtype=dtype)

# Main training routine
train_counter : List = []
train_losses : List = []
train_grads : List = []
train_rms : List = []
def train_epoch( epoch : int ):
    model.train( )

    max_batches = 50
    for _batch_idx, (x_b, t_b, params_b, u0_b) in enumerate( training_loader ):
        batch_idx = _batch_idx+1
        if batch_idx > max_batches: # avoid early overtraining by stopping soon enough
            return
        
        optimizer.zero_grad( set_to_none=True )
        
        # Move to the GPU
        x_b = x_b.to( device=device, dtype=dtype )
        t_b = t_b.to( device=device, dtype=dtype )
        params_b = params_b.to( device=device, dtype=dtype )
        u0_b = u0_b.to( device=device, dtype=dtype )

        # Compute the loss and its gradient
        loss, loss_info = loss_fcn( model, x_b, t_b, params_b, u0_b )
        loss.backward()
        loss_grad = getGradientNorm( model )

        # Update the weights internally
        pt.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step( )

        # Keep track of important metrics
        T_t_rms = loss_info["T_t_rms"]
        T_xx_rms = loss_info["T_xx_rms"]
        rel_rms = loss_info["rms"] / (T_t_rms + T_xx_rms)
        if batch_idx == 1:
            train_counter.append( epoch-1 + batch_idx / max_batches)
            train_losses.append( float(loss.item()) )
            train_grads.append( float(loss_grad.item()) )
            train_rms.append( float(rel_rms) )

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
                f"    RMS:  T_t = {T_t_rms:.3e}   "
                f"T_xx = {T_xx_rms:.3e}   "
                f"Rel = {rel_rms:.3e}"
            )
            print(info_str)

# Validation function
validation_counter : List = []
validation_losses : List = []
validation_rms : List = []
def validate_epoch( epoch : int ):
    model.eval( )

    # Compute the loss and its gradient
    loss, loss_info = loss_fcn( model, x_val, t_val, params_val, u0_val )
    rel_rms = loss_info["rms"] / (loss_info["T_t_rms"] + loss_info["T_xx_rms"])

    # Store
    validation_counter.append( epoch )
    validation_losses.append( float(loss.item()) )
    validation_rms.append( float(rel_rms) )

    # Print and done.
    print_str = f'\nValidation Epoch {epoch:03d}: \tLoss: {loss.item():.3e} \tRelative RMS: {rel_rms:.3e}'
    print(print_str)

# Main training loop
store_directory = './Results/'
try:
    n_epochs = warm_epochs + anneal_epochs
    for epoch in range( 1, n_epochs+1 ):
        # Regenerate the training dataset every epoch to avoid overtraining the branch.
        # Can be slow, we will find out.
        train_dataset = TensorizedDataset(
            N_train_branch, N_train_trunk, n_grid_points,
            l, T_max, tau_max, logk_max, dtype, plot=False, tau_sampling=sampling_strat
        )
        training_loader = DataLoader(train_dataset, batch_size=B, shuffle=True)

        # Train using the new dataset
        train_epoch( epoch )

        # Validate on independent but fixed data
        validate_epoch( epoch )

        # Update the learning rate for the next epoch.
        if epoch > warm_epochs:
            scheduler.step()

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
train_rms = np.array( train_rms ) # type: ignore
train_data = np.stack( (train_counter[:,np.newaxis], train_losses[:,np.newaxis], train_grads[:,np.newaxis], train_rms[:,np.newaxis]), axis=1) # type: ignore
validation_counter = np.array( validation_counter ) # type: ignore
validation_losses = np.array( validation_losses ) # type: ignore
validation_rms = np.array( validation_rms ) # type: ignore
validation_data = np.stack( (validation_counter[:,np.newaxis], validation_losses[:,np.newaxis], validation_rms[:,np.newaxis]), axis=1) # type: ignore
np.save( store_directory + 'train_data.npy', train_data)
np.save( store_directory + 'validation_data.npy', validation_data)

# Make a plot of the training progress
plt.figure()
plt.semilogy( train_counter, train_losses, alpha=0.5, label="Training Loss" )
plt.semilogy( train_counter, train_grads, alpha=0.5, label="Training Loss Gradient" )
plt.semilogy( train_counter, train_rms, alpha=0.5, label="Training Relative RMS" )
plt.semilogy( validation_counter, validation_losses, alpha=0.5, label="Validation Loss" )
plt.semilogy( validation_counter, validation_rms, alpha=0.5, label="Validation Relative RMS" )
plt.xlabel( "Epoch" )
plt.legend()
plt.tight_layout()
plt.savefig( store_directory + 'convergence_mlp_adam.png', transparent=True )
plt.show()