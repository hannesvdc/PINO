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
from ConvBranchEmbeddingNetwork import ConvBranchEmbeddingNetwork
from Loss import HeatLoss

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
B = 2**14
N_train_branch = 200
N_train_trunk = 2500
train_dataset = TensorizedDataset( N_train_branch, N_train_trunk, n_grid_points, 
                                       l, T_max, tau_max, logk_max, dtype, plot=False, tau_sampling=sampling_strat)
train_loader = DataLoader( train_dataset, batch_size=B, shuffle=False, drop_last=True )
N_validation_branch = 10
N_validation_trunk = 1000
validation_dataset = TensorizedDataset( N_validation_branch, N_validation_trunk, n_grid_points, 
                                       l, T_max, tau_max, logk_max, dtype, plot=False, tau_sampling=sampling_strat)

# Setup the network
n_hidden_layers = 4
z = 64
channels = [1, 8, 16, 32, 32, 32, 32, 32, 32, 32] # increase gradually
q = 32
x_grid = validation_dataset.branch_dataset.x_grid
model = ConvBranchEmbeddingNetwork( channels, n_hidden_layers, z, q, x_grid, l, T_max, tau_max, logk_max)
print('Number of Trainable Parameters: ', sum( [ p.numel() for p in model.parameters() if p.requires_grad ]))

# Load the Adam weights
store_directory = "./Results/"
model.load_state_dict( pt.load( store_directory + 'conv_model_adam.pth', weights_only=True, map_location="cpu" ) )

# Heat loss fcn
loss_fcn = HeatLoss( )

# Setup the optimizer
max_iter = 50
history_size = 50
lr = 1.0
optimizer = optim.LBFGS( model.parameters(), lr, max_iter=max_iter, line_search_fn="strong_wolfe", history_size=history_size )

# The L-BFGS closure
last_state = {"loss" : None, "grad_norm" : None, "T_t_rms" : None, "T_xx_rms": None, "rel_rms" : None}
all_loss_evalutaions = []
all_grad_evaluations = []
def closure():
    model.train()
    optimizer.zero_grad( set_to_none=True )

    # for logging
    loss_sum = 0.0
    n_batches = len( train_loader )

    # Mini batch due to limited memory.
    for _batch_idx, (x_b, t_b, params_b, u0_b) in enumerate( train_loader ):

        # Batched but Deterministic Loss for memory constraints
        batch_loss, loss_info = loss_fcn( model, x_b, t_b, params_b, u0_b )
        batch_loss = batch_loss / n_batches
        batch_loss.backward() # gradient per batch to release memory early

        # Store some diagnostic information
        loss_sum += float( batch_loss.item() )
    gradnorm = float(getGradientNorm(model).item())

    # Store averaged metrics
    last_state["loss"] = loss_sum
    last_state["grad_norm"] = gradnorm
    last_state["T_t rms"] = loss_info["T_t_rms"]
    last_state["T_xx rms"] = loss_info["T_xx_rms"]
    last_state["rel_rms"] = loss_info["rms"] / (loss_info["T_t_rms"] + loss_info["T_xx_rms"])
    all_loss_evalutaions.append( loss_sum )
    all_grad_evaluations.append( gradnorm )

    return pt.tensor(loss_sum, dtype=pt.get_default_dtype())

x_val, t_val, params_val, u0_val = validation_dataset.all()
def validate():
    model.eval()

    # Just the one validation batch
    epoch_loss, _ = loss_fcn( model, x_val, t_val, params_val, u0_val )
    
    return epoch_loss

# Actual training and validation
print('\nStarting Training...')
learning_rates = []
train_counter = []
train_losses = []
train_grads = []
validation_counter = []
validation_losses = []
rel_rmss = []

epoch = 1
try:
    while lr > 1.e-6:
        w0 = pt.cat([p.detach().flatten() for p in model.parameters()])
        training_loss = optimizer.step( closure )
        w1 = pt.cat([p.detach().flatten() for p in model.parameters()])
        step_norm = (w1 - w0).norm().item()
        print('\nTrain Epoch: {} \tLoss: {:.10E} \tLoss Gradient: {:.10E} \tStep Norm: {:.10E} \t Lr: {:.4E}'.format( 
            epoch, last_state["loss"], last_state["grad_norm"], step_norm, optimizer.param_groups[0]["lr"] ))
        print('T_t RMS: {:.4E} \tT_xx RMS: {:.4E} \tLoss Relative RMS: {:.4E}'.format(
            last_state["T_t rms"], last_state["T_xx rms"], last_state["rel_rms"]))
        
        validation_loss = validate()
        print('Validation Epoch: {} \tLoss: {:.10E}'.format( epoch, validation_loss.item() ))
        
        # Store the pretrained state
        pt.save( model.state_dict(), store_directory + 'conv_model_lbfgs.pth')
        pt.save( optimizer.state_dict(), store_directory + 'conv_optimizer_lbfgs.pth')

        train_counter.append( epoch )
        train_losses.append( last_state["loss"] )
        train_grads.append( last_state["grad_norm"] )
        rel_rmss.append( last_state["rel_rms"] )
        validation_counter.append( epoch )
        validation_losses.append( validation_loss.item() )
        learning_rates.append( optimizer.param_groups[0]["lr"] )

        if step_norm < 1e-8:
            lr = 0.3 * lr
            print('Lowering L-BFGS Learning Rate to', lr)
            for g in optimizer.param_groups:
                g["lr"] = lr
        
        # Update to the next epoch.
        epoch += 1
except KeyboardInterrupt:
    pass


# Store the per-epoch convergence results
import numpy as np
np.save( store_directory + 'LBFGS_Training_Convergence.npy', np.hstack( (train_counter, train_losses, train_grads) ) )
np.save( store_directory + 'LBFGS_Validation_Convergence.npy', np.hstack( (validation_counter, validation_losses, learning_rates) ) )

# Show the training results
plt.semilogy(train_counter, train_losses, label='Training Loss', alpha=0.5)
plt.semilogy(train_counter, train_grads, label='Loss Gradient', alpha=0.5)
plt.semilogy(validation_counter, validation_losses, label='Validation Loss', alpha=0.5)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('L-BFGS')
plt.show()