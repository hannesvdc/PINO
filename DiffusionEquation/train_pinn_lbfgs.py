import math
import torch as pt
from torch.utils.data import DataLoader
from torch.optim import LBFGS
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

from PINNDataset import PINNDataset
from FixedInitialPINN import FixedInitialPINN
from Loss import HeatLoss
from sampleInitialGP import gp

dtype = pt.float64
pt.set_grad_enabled( True )
pt.set_default_dtype( dtype )

# Create the initial condition. We want T_max to be ~95th percentile, so std=T_max/1.96
T_max = 10.0
tau_max = 8.0 # train to exp( -tau_max )
logk_max = math.log( 1e2 )
N_grid_points = 51
x_grid = pt.linspace(0.0, 1.0, N_grid_points)
l = 0.12 # GP correlation length

# Create the training and validation datasets
store_directory = './Results/'
u0 = pt.load(store_directory + 'initial.pth', weights_only=True)
train_data = pt.load( store_directory + 'train_data.pth' ).to( dtype=dtype )
x_train = train_data[:,0:1]
t_train = train_data[:,1:2]
p_train = train_data[:,2:]
validation_data = pt.load( store_directory + 'validation_data.pth' ).to( dtype=dtype )
x_validation = validation_data[:,0:1]
t_validation = validation_data[:,1:2]
p_validation = validation_data[:,2:]

# Create the PINO
z = 64
n_hidden_layers = 2
model = FixedInitialPINN( n_hidden_layers, z, T_max, tau_max, logk_max, u0, x_grid, l )
plot_grid = pt.linspace(0.0, 1.0, 1001)
u0_int = model.evaluate_u0( plot_grid[:,None] )
plt.plot( x_grid.cpu().numpy(), u0.cpu().numpy(), label="Exact Initial Condition")
plt.plot(plot_grid.cpu().numpy(), u0_int.cpu().numpy(), linestyle='--', label="RBF Interpolation")
plt.legend()
plt.xlabel(r"$x$")
plt.ylabel(r"$u_0(x)$")
plt.show( )

# Move the model to the GPU (and single precision dtype )
device = pt.device("mps")
dtype = pt.float32
model = model.to( device=device, dtype=dtype )
print('Number of Trainable Parameters: ', sum([ p.numel() for p in model.parameters() ]))

# Create the Loss function
loss_fn = HeatLoss()
lr = 1.0
max_iter = 50
history_size = 50
optimizer = LBFGS( model.parameters(), lr, max_iter=max_iter, line_search_fn="strong_wolfe", history_size=history_size )
def getGradientNorm():
    grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    return pt.norm(pt.cat(grads))

# The L-BFGS closure
last_state = {"loss" : None, "grad_norm" : None}
all_loss_evalutaions = []
all_grad_evaluations = []
def closure():
    model.train()
    optimizer.zero_grad( set_to_none=True )

    # Batched but Deterministic Loss for memory constraints
    epoch_loss = loss_fn( model, x_train, t_train, p_train )
    epoch_loss.backward( )

    # Store some diagnostic information
    last_gradnorm = getGradientNorm().item()
    last_state["loss"] = epoch_loss.item()
    last_state["grad_norm"] = last_gradnorm # type: ignore
    all_loss_evalutaions.append( epoch_loss.item() )
    all_grad_evaluations.append( last_gradnorm )

    # LBFGS only needs a scalar loss value back.
    # It does NOT need the returned tensor to carry a graph, since grads are already in .grad.
    return epoch_loss
def validate():
    model.eval()

    # Just the one validation batch
    epoch_loss = loss_fn( model, t_validation, p_validation )
    
    return epoch_loss

# Actual training and validation
print('\nStarting Training...')
learning_rates = []
train_counter = []
train_losses = []
train_grads = []
validation_counter = []
validation_losses = []

epoch = 1
try:
    while lr > 1.e-6:
        w0 = pt.cat([p.detach().flatten() for p in model.parameters()])
        training_loss = optimizer.step( closure )
        w1 = pt.cat([p.detach().flatten() for p in model.parameters()])
        step_norm = (w1 - w0).norm().item()
        print('\nTrain Epoch: {} \tLoss: {:.10E} \tLoss Gradient: {:.10E} \tStep Norm: {:.10E} \t Lr: {:.4E}'.format( 
            epoch, last_state["loss"], last_state["grad_norm"], step_norm, optimizer.param_groups[0]["lr"] ))
        
        validation_loss = validate()
        print('Validation Epoch: {} \tLoss: {:.10E}'.format( epoch, validation_loss.item() ))
        
        # Store the pretrained state
        pt.save( model.state_dict(), store_directory + 'model_lbfgs.pth')
        pt.save( optimizer.state_dict(), store_directory + 'optimizer_lbfgs.pth')

        train_counter.append( epoch )
        train_losses.append( last_state["loss"] )
        train_grads.append( last_state["grad_norm"] )
        validation_counter.append( epoch )
        validation_losses.append( validation_loss.item() )
        learning_rates.append( optimizer.param_groups[0]["lr"] )

        if step_norm < 1e-8:
            print('Lowering L-BFGS Learning Rate to', 0.3 * lr)
            lr = 0.3 * lr
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

# Show the training results
plt.figure()
plt.semilogy(all_loss_evalutaions, label='Training Loss', alpha=0.5)
plt.semilogy(all_grad_evaluations, label='Loss Gradient', alpha=0.5)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('L-BFGS: Detailed Information')
plt.show()