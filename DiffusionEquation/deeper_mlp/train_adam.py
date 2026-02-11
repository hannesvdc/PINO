import sys
sys.path.append('../')

import torch as pt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import PINNDataset
from FixedInitialPINN import FixedInitialPINN
from Loss import HeatLoss
from generateInitialCondition import build_u0_evaluator
from optimizeAdam import optimizeWithAdam

dtype = pt.float64
pt.set_grad_enabled( True )
pt.set_default_dtype( dtype )

# Create the training and validation datasets
T_max = 10.0
tau_max = 8.0 # train to exp( -tau_max )
N_train = 10_000
N_validation = 5_000
train_dataset = PINNDataset( N_train, T_max, tau_max, dtype, test=False)
validation_dataset = PINNDataset( N_validation, T_max, tau_max, dtype, test=False)
B = 128
train_loader = DataLoader( train_dataset, batch_size=B, shuffle=True )
validation_loader = DataLoader( validation_dataset, batch_size=N_validation, shuffle=False )

# Also store the dataset for later use
import os
store_directory = os.getcwd() + '/Results/'
pt.save( train_dataset.all().cpu(), store_directory + 'train_data.pth' )
pt.save( validation_dataset.all().cpu(), store_directory + 'validation_data.pth' )

# Now do everything on the GPU
device = pt.device("mps")
dtype = pt.float32

# Load the initial condition
l = 0.12
u0_fcn, ic = build_u0_evaluator( l, device, dtype )

# Create the PINO
z = 64
n_hidden_layers = 4
model = FixedInitialPINN( n_hidden_layers, z, T_max, tau_max, train_dataset.logk_max, u0_fcn, ic_time_factor=True)
model = model.to( device=device, dtype=dtype )
print('Number of Trainable Parameters: ', sum([ p.numel() for p in model.parameters() ]))

# Create the Loss function
loss_fn = HeatLoss()

# Optimize using Adam
lr = 1e-3
n_steps = 5
step_size = 1000
train_counter, train_losses, train_grads, validation_counter, validation_losses, rel_rmss \
    = optimizeWithAdam( model, loss_fn, train_loader, validation_loader, device, dtype, lr, n_steps, step_size, store_directory)

# Store the per-epoch convergence results
import numpy as np
np.save( store_directory + 'Adam_Training_Convergence.npy', np.hstack( (train_counter, train_losses, train_grads) ) )
np.save( store_directory + 'Adam_Validation_Convergence.npy', np.hstack( (validation_counter, validation_losses) ) )

# Show the training results
plt.semilogy(train_counter, train_losses, label='Training Loss', alpha=0.5)
plt.semilogy(train_counter, train_grads, label='Loss Gradient', alpha=0.5)
plt.semilogy(validation_counter, validation_losses, label='Validation Loss', alpha=0.5)
plt.semilogy(train_counter, rel_rmss, label='Relative Loss RMS')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Adam')
plt.show()