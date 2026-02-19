import sys
sys.path.append('../')

import numpy as np
import torch as pt
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import getGradientNorm

import matplotlib.pyplot as plt

from TensorizedDataset import TensorizedDataset
from BranchEmbeddingNetwork import BranchEmbeddingNetwork
from Loss import HeatLoss

# Do everything on the CPU in double precision first
dtype = pt.float64
pt.set_default_device( pt.device("cpu") )
pt.set_default_dtype( dtype )

# Physics parameters
n_grid_points = 51
T_max = 10.0
tau_max = 8.0
l = 0.2

# Create a training and validation dataset
sampling_strat = "uniform"
N_train_branch = 101
N_train_trunk = 10_000
train_dataset = TensorizedDataset( N_train_branch, N_train_trunk, n_grid_points, 
                                  l, T_max, tau_max, dtype, plot=False, tau_sampling=sampling_strat)
N_validation_branch = 10
N_validation_trunk = 1000
validation_dataset = TensorizedDataset( N_validation_branch, N_validation_trunk, n_grid_points, 
                                       l, T_max, tau_max, dtype, plot=False, tau_sampling=sampling_strat)

# Create a training loader only
B = 2048
training_loader = DataLoader( train_dataset, B, shuffle=True )

# Setup the network
n_embedding_hidden_layers = 4
n_hidden_layers = 4
z = 64
q = 10
x_grid = train_dataset.branch_dataset.x_grid
logk_max = train_dataset.trunk_dataset.logk_max
model = BranchEmbeddingNetwork( n_embedding_hidden_layers, n_hidden_layers, z, q, x_grid, l, T_max, tau_max, logk_max)

# Translate the model to GPU
device = pt.device( "mps" )
dtype = pt.float32
pt.set_default_device( device )
pt.set_default_dtype( dtype )
model = model.to(device=device, dtype=dtype)

# Heat loss fcn
loss_fcn = HeatLoss( )

# Setup the optimizer and learning rate scheduler
lr = 0.01
n_epochs = 1000
optimizer = optim.Adam( model.parameters(), lr )
scheduler = optim.lr_scheduler.CosineAnnealingLR( optimizer, n_epochs, 1e-6 )

# Load the validation dataset all at once
x_val, t_val, params_val, u0_val = validation_dataset.all()
x_val = x_val.to(device=device, dtype=dtype)
t_val = t_val.to(device=device, dtype=dtype)
params_val = params_val.to(device=device, dtype=dtype)
u0_val = u0_val.to(device=device, dtype=dtype)

# Main training routine
train_counter = []
train_losses = []
train_grads = []
train_rms = []
def train_epoch( epoch : int ):
    model.train( )

    for batch_idx, (x_b, t_b, params_b, u0_b) in enumerate( training_loader ):
        optimizer.zero_grad( set_to_none=True )
        
        # Move to the GPU
        x_b = x_b.to( device=device, dtype=dtype )
        t_b = t_b.to( device=device, dtype=dtype )
        params_b = params_b.to( device=device, dtype=dtype )
        u0_b = u0_b.to( device=device, dtype=dtype )

        # Compute the loss and its gradient
        loss, loss_info = loss_fcn( x_b, t_b, params_b, u0_b )
        loss.backward()
        loss_grad = getGradientNorm( model )

        # Update the weights internally
        optimizer.step( )

        # Keep track of important metrics
        T_t_rms = loss_info["T_t_rms"]
        T_xx_rms = loss_info["T_xx_rms"]
        rel_rms = loss_info["rms"] / (T_t_rms + T_xx_rms)
        train_counter.append( epoch + batch_idx / len(training_loader))
        train_losses.append( float(loss.item()) )
        train_grads.append( float(loss_grad.item()) )
        train_rms.append( float(rel_rms) )

        if batch_idx % 100 == 0:
            print_str = f'Epoch {epoch} [{batch_idx}/{len(training_loader)} ({batch_idx/len(training_loader)*100.0}%)]:'
            print_str += "\tLoss: {:4d} \tGradient: {:4d}".format( loss.item(), loss_grad.item())
            print( print_str )

            info_str = f"T_t RMS: {T_t_rms} \tT_xx RMS: {T_xx_rms} \tRelative RMS: {rel_rms}"
            print( info_str )

# Validation function
validation_counter = []
validation_losses = []
def validate_epoch( epoch : int ):
    model.eval( )

    # Compute the loss and its gradient
    loss, loss_info = loss_fcn( x_val, t_val, params_val, u0_val )
    rel_rms = loss_info["rms"] / (loss_info["T_t_rms"] + loss_info["T_xx_rms"])

    # Store
    validation_counter.append( epoch )
    validation_losses.append( float(loss.item()) )

    # Print and done.
    print_str = f'Validation Epoch {epoch}: \tLoss: {loss.item()} \tRelative RMS: {rel_rms}'
    print(print_str)

# Main training loop
store_directory = './Results/'
try:
    for epoch in range( 1, n_epochs+1 ):
        train_epoch( epoch )
        validate_epoch( epoch )
        scheduler.step()

        # Store the current model and optimizer weights.
        pt.save( model.state_dict(), store_directory + 'model_adam.pth')
        pt.save( optimizer.state_dict(), store_directory + 'optimizer_adam.pth')
except KeyboardInterrupt:
    # Do nothing
    pass

# Make numpy arrays from the training data
train_counter = np.array( train_counter )
train_losses = np.array( train_losses )
train_grads = np.array( train_grads )
train_rms = np.array( train_rms )
train_data = np.stack( (train_counter[:,np.newaxis], train_losses[:,np.newaxis], train_grads[:,np.newaxis], train_rms[:,np.newaxis]), axis=1)
validation_counter = np.array( validation_counter )
validation_losses = np.array( validation_losses )
validation_data = np.stack( (validation_counter[:,np.newaxis], validation_losses[:,np.newaxis]), axis=1)
np.save( store_directory + 'train_data.npy', train_data)
np.save( store_directory + 'validation_data.npy', validation_data)

# Make a plot of the training progress
plt.figure()
plt.semilogy( train_counter, train_losses, alpha=0.5, label="Training Loss" )
plt.semilogy( train_counter, train_grads, alpha=0.5, label="Training Loss Gradient" )
plt.semilogy( train_counter, train_rms, alpha=0.5, label="Training Relative RMS" )
plt.semilogy( validation_counter, validation_losses, alpha=0.5, label="Validation Loss" )
plt.xlabel( "Epoch" )
plt.legend()
plt.show()