import sys
sys.path.append('../')

import math
import torch as pt
import matplotlib.pyplot as plt

from ResidualPINN import ResidualPINN
from Loss import HeatLoss
from generateInitialCondition import build_u0_evaluator
from optimizeLBFGS import optimizeWithLBFGS
from posttrain_dataset import PINNDataset

dtype = pt.float64
device = pt.device("cpu")
pt.set_grad_enabled( True )
pt.set_default_dtype( dtype )
pt.set_default_device( device )

# Create the initial condition. We want T_max to be ~95th percentile, so std=T_max/1.96
T_max = 10.0
tau_max = 8.0 # train to exp( -tau_max )
logk_max = math.log( 1e2 )

# Load the initial condition
l = 0.12
u0_fcn, ic = build_u0_evaluator( l, device, dtype )

# Create the training and validation datasets
N = 10_000
store_directory = './Results/'
train_dataset = PINNDataset( N, T_max, tau_max )
train_data = train_dataset.all()
x_train = train_data[:,0:1]
t_train = train_data[:,1:2]
p_train = train_data[:,2:]
validation_dataset = PINNDataset( 5_000, T_max, tau_max )
validation_data = validation_dataset.all()
x_validation = validation_data[:,0:1]
t_validation = validation_data[:,1:2]
p_validation = validation_data[:,2:]

# Create the PINO
z = 64
n_hidden_layers = 2
model = ResidualPINN( n_hidden_layers, z, T_max, tau_max, logk_max, u0_fcn, ic_time_factor=True)
model.load_state_dict( pt.load(store_directory + '/model_lbfgs.pth', weights_only=True, map_location=device) )
model = model.to( device=device, dtype=dtype )
print('Number of Trainable Parameters: ', sum([ p.numel() for p in model.parameters() ]))

# Create the Loss function
loss_fn = HeatLoss()
lr = 1.0
train_counter, train_losses, train_grads, validation_counter, validation_losses, rel_rmss, learning_rates \
    = optimizeWithLBFGS( model, loss_fn, x_train, t_train, p_train, x_validation, t_validation, p_validation, device, dtype, lr, store_directory, post=True)

# Store the per-epoch convergence results
import numpy as np
np.save( store_directory + 'Post_LBFGS_Training_Convergence.npy', np.hstack( (train_counter, train_losses, train_grads) ) )
np.save( store_directory + 'Post_LBFGS_Validation_Convergence.npy', np.hstack( (validation_counter, validation_losses, learning_rates) ) )

# Show the training results
plt.semilogy(train_counter, train_losses, label='Training Loss', alpha=0.5)
plt.semilogy(train_counter, train_grads, label='Loss Gradient', alpha=0.5)
plt.semilogy(validation_counter, validation_losses, label='Validation Loss', alpha=0.5)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Posttraining L-BFGS')
plt.show()