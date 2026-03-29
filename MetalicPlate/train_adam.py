import math
import numpy as np
import torch as pt
import torch.optim as optim
from utils import getGradientNorm

import matplotlib.pyplot as plt

from PlateDataset import BranchDataset, TrunkDataset, BoundaryDataset
from TrunkFiLMNetwork import TrunkFilmNetwork
from EnergyLoss import EnergyLoss

from typing import List

# Do everything on the CPU in double precision first
dtype = pt.float64
pt.set_default_dtype( dtype )
gen = pt.Generator()

# Physics parameters
n_grid_points = 101
y_grid = pt.linspace(0.0, 1.0, n_grid_points)
nu_max = 0.45
l = 0.2

# Create a training and validation dataset
B = 256
N_train_branch = B
N_train_trunk = 5000
N_train_bc = 1000
N_validation_branch = 10
N_validation_trunk = 5000
N_validation_bc = 1000
trunk_dataset = TrunkDataset( N_train_trunk, gen )
bc_dataset = BoundaryDataset( N_train_bc, gen, dtype )
x_b, y_b, w_b = trunk_dataset.all()
y_bc_b = bc_dataset.all()
np.save( './data/trunk_dataset.npy', pt.cat( (x_b, y_b, w_b), dim=1).detach().numpy() )
np.save( './data/bc_dataset.npy', y_bc_b.detach().numpy() )

# Setup the network
n_hidden_layers = 4
z = 128
film_channels = [2, 32, 32, 64, 64, 64, 64, 64, 64 ] # increase gradually
model = TrunkFilmNetwork( film_channels, n_grid_points, n_hidden_layers, z, nu_max )
print('Number of Trainable Parameters: ', sum( [ p.numel() for p in model.parameters() if p.requires_grad ]))

# Heat loss fcn
lambda_trac = 1.0
loss_fcn = EnergyLoss( y_grid, l, lambda_trac=lambda_trac )

# Translate the model to GPU
device = pt.device("cpu")
dtype = pt.float64
model = model.to(device=device, dtype=dtype)
loss_fcn.to( device=device, dtype=dtype )

# Setup the optimizer and learning rate scheduler
lr = 1e-3
optimizer = optim.Adam( model.parameters(), lr, amsgrad=True )

# Scheduler: constant for the first `n_epochs` epochs, decrease by cosine for `annealing_epochs` later.
step_size = 1000
gamma = 0.1
n_steps = 5
scheduler = optim.lr_scheduler.StepLR( optimizer, step_size, gamma )

# Main training routine
train_counter : List = []
train_losses : List = []
train_grads : List = []
train_tractions : List = []
# Sample a new dataset every time
branch_dataset = BranchDataset( N_train_branch, n_grid_points, l, nu_max, gen, dtype=dtype, plot=False )
def train_epoch( epoch : int ):
    model.train( )
    optimizer.zero_grad( set_to_none=True )
        
    # Fetch the trunk inputs
    x_b, y_b, w_b = trunk_dataset.all()
    gx_b, gy_b, nu_b = branch_dataset.all()
    y_bc_b = bc_dataset.all()

    # Move to the GPU
    x_b = x_b.to( device=device, dtype=dtype )
    y_b = y_b.to( device=device, dtype=dtype )
    w_b = w_b.to( device=device, dtype=dtype )
    nu_b = nu_b.to( device=device, dtype=dtype )
    y_bc_b = y_bc_b.to( device=device, dtype=dtype )
    gx_b = gx_b.to( device=device, dtype=dtype )
    gy_b = gy_b.to( device=device, dtype=dtype )

    # Compute the loss (backward is called per-chunk inside loss_fcn)
    loss, loss_info = loss_fcn( model, x_b, y_b, w_b, nu_b, y_bc_b, gx_b, gy_b )
    loss_grad = getGradientNorm( model )

    # Update the weights internally
    pt.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step( )

    # Keep track of important metrics
    energy = loss_info["energy"]
    boundary = loss_info["boundary"]
    traction = loss_info["traction"]
    train_counter.append( epoch-1)
    train_losses.append( float(loss) )
    train_grads.append( float(loss_grad.item()) )
    train_tractions.append( traction )

    # Print some diagnostics
    print_str = (
        f"\nEpoch {epoch:04d} "
        f"Loss: {loss:.3e}  "
        f"Grad: {loss_grad.item():.3e}  "
        f"Lr: {optimizer.param_groups[0]['lr']:.3e}"
    )
    print(print_str)
    info_str = (
        f"Energy = {energy:.3e}   "
        f"Boundary = {boundary:.3e}   "
        f"Traction = {traction:.3e}.  "
    )
    print(info_str)

# Validation function
validation_branch_dataset = BranchDataset( N_validation_branch, n_grid_points, l, nu_max, gen, dtype, plot=False )
validation_counter : List = []
validation_losses : List = []
validation_tractions : List = []
def validate_epoch( epoch : int ) -> float:

    x_val, y_val, w_val = trunk_dataset.all()
    x_val = x_val.to(device=device, dtype=dtype)
    y_val = y_val.to(device=device, dtype=dtype)
    w_val = w_val.to(device=device, dtype=dtype)
    gx_val, gy_val, nu_val = validation_branch_dataset.all()
    gx_val = gx_val.to(device=device, dtype=dtype)
    gy_val = gy_val.to(device=device, dtype=dtype)
    nu_val = nu_val.to(device=device, dtype=dtype)
    y_bc_val = bc_dataset.all()
    y_bc_val = y_bc_val.to(device=device, dtype=dtype)

    # Compute the loss (no gradients needed for validation)
    loss, loss_info = loss_fcn( model, x_val, y_val, w_val, nu_val, y_bc_val, gx_val, gy_val, training=False )

    # Store
    validation_counter.append( epoch )
    validation_losses.append( float(loss) )
    validation_tractions.append( float(loss_info["traction"]) )

    # Print and done.
    print_str = f'\nValidation Epoch {epoch:03d}: \tLoss: {loss:.3e} \tTraction: {loss_info["traction"]:.3e}'
    print(print_str)

    return float( loss )

# Main training loop
store_directory = './Results/'
warm_epochs = 0
# n_epochs = warm_epochs + anneal_epochs
n_epochs = step_size * n_steps
best_val_loss = math.inf
try:
    for epoch in range( 1, n_epochs+1 ):
        # Train using the new dataset
        train_epoch( epoch )

        # Validate on independent but fixed data
        val_loss = validate_epoch( epoch )

        if epoch > warm_epochs:
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
ax1.set_ylabel("Energy", color="black")
ax1.tick_params(axis="y")
plt.legend()
fig = plt.figure()
plt.semilogy( train_counter, np.array( train_tractions), label="Training boundary traction loss" )
plt.semilogy( validation_counter, np.array( validation_tractions), label="Validation boundary traction loss" )
plt.xlabel( "Epoch" )
plt.legend()
fig = plt.figure()
ax2 = fig.gca()
ax2.semilogy(train_counter, train_grads, color="tab:red", alpha=0.7, label="Grad Norm")
ax2.set_ylabel("Gradient Norm", color="tab:red")
ax2.set_xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.show()