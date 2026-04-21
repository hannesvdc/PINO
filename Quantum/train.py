import math
import numpy as np
import torch as pt
import torch.optim as optim
from utils import getGradientNorm

import matplotlib.pyplot as plt

from QuantumNetwork import QuantumNetwork
from QuantumLoss import QuantumLoss
from sampleBatch import sampleBatch

from typing import List

# Do everything on the CPU in double precision.
dtype = pt.float64
pt.set_default_dtype( dtype )
pt.set_default_device( 'cpu' )

gen = pt.Generator()

# Create a training and validation dataset
B = 256
N_train = 5000
B_val = 16
N_validation = 1000

# Setup the network
R_cutoff = 5.0
z = 64
neurons_per_layer = [ 4, z, z, z, z, 1]
model = QuantumNetwork( neurons_per_layer, R_cutoff)
print('Number of Trainable Parameters: ', sum( [ p.numel() for p in model.parameters() if p.requires_grad ]))

# Loss function.
chunk_size = 4
loss_fcn = QuantumLoss( chunk_size=chunk_size )

# Setup the optimizer and learning rate scheduler
lr = 1e-3
optimizer = optim.Adam( model.parameters(), lr, amsgrad=True )

# Scheduler: constant for the first `n_epochs` epochs, decrease by cosine for `annealing_epochs` later.
step_size = 500
gamma = 0.1
n_steps = 5
scheduler = optim.lr_scheduler.StepLR( optimizer, step_size, gamma )

# Main training routine
train_counter : List = []
train_losses : List = []
train_grads : List = []

# Sample a new dataset every time
def train_epoch( epoch : int ):
    model.train( )
    optimizer.zero_grad( set_to_none=True )

    # Sample new training points every epoch.
    R, xyz, mc_weights = sampleBatch( B, N_train, R_cutoff, gen)
        
    # Compute the loss (backward is called per-chunk inside loss_fcn)
    loss = loss_fcn( model, R, xyz, mc_weights, training=True )
    loss_grad = getGradientNorm( model )

    # Update the weights internally
    pt.nn.utils.clip_grad_norm_( model.parameters(), max_norm=1.0 )
    optimizer.step( )

    # Keep track of important metrics
    train_counter.append( epoch-1)
    train_losses.append( loss )
    train_grads.append( float(loss_grad.item()) )

    # Print some diagnostics
    print_str = (
        f"\nEpoch {epoch:04d} "
        f"Loss: {loss:.3e}  "
        f"Grad: {loss_grad.item():.3e}  "
        f"Lr: {optimizer.param_groups[0]['lr']:.3e}"
    )
    print(print_str)

# Validation function
val_R = pt.tensor( [1.0], dtype=dtype )
_, val_xyz, val_mc_weights = sampleBatch( B_val, N_validation, R_cutoff, gen )
validation_counter : List = []
validation_losses : List = []
def validate_epoch( epoch : int ) -> float:

    # Compute the loss (no gradients needed for validation)
    loss = loss_fcn( model, val_R, val_xyz, val_mc_weights, training=False )

    # Log some interesting info
    proton_energy = 1.0 / (2.0 * float(val_R.item()) )
    total_energy = proton_energy + loss

    # Store
    validation_counter.append( epoch )
    validation_losses.append( float(loss) )

    # Print and done.
    print_str = f'\nValidation Epoch {epoch:03d}: \tElectron Energy: {loss:.3e} \tTotal Energy {total_energy:.3e}'
    print(print_str)

    return loss

# Main training loop
store_directory = './Results/'
n_epochs = step_size * n_steps
best_val_loss = math.inf
try:
    for epoch in range( 1, n_epochs+1 ):
        # Train using the new dataset
        train_epoch( epoch )

        # Validate on independent but fixed data
        val_loss = validate_epoch( epoch )

        scheduler.step( )

        # Store the current model and optimizer weights.
        pt.save( model.state_dict(), store_directory + 'model_adam.pth')
        pt.save( optimizer.state_dict(), store_directory + 'optimizer_adam.pth')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Storing the best model.")
            pt.save( model.state_dict(), store_directory + 'best_model.pth')
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
fig = plt.figure()
ax1 = fig.gca()
ax1.plot(train_counter, train_losses, color="tab:blue", alpha=0.7, label="Train Loss")
ax1.plot(validation_counter, validation_losses, color="tab:orange", alpha=0.7, label="Validation Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Electron Energy", color="black")
ax1.tick_params(axis="y")
plt.legend()

fig = plt.figure()
ax2 = fig.gca()
ax2.semilogy(train_counter, train_grads, color="tab:red", alpha=0.7, label="Grad Norm")
ax2.set_ylabel("Gradient Norm", color="tab:red")
ax2.set_xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.show()