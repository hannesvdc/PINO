import torch as pt
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from TensorizedDataset import TensorizedDataset
from EmbeddingNetwork import EmbeddingNetwork
from Loss import HeatLoss
from utils import getGradientNorm

import matplotlib.pyplot as plt

dtype = pt.float64
pt.set_grad_enabled( True )
pt.set_default_dtype( dtype )
store_directory = './Results/'

# Create the training and validation datasets
T_max = 10.0
tau_max = 8.0 # train to exp( -tau_max )
n_grid_points = 51

B = 512
N_train_branch = 128
N_train_trunk = 5_000
N_validation_branch = 32
N_validation_trunk = 512

train_dataset = TensorizedDataset( N_train_branch, N_train_trunk, n_grid_points, T_max, tau_max, dtype)
validation_dataset = TensorizedDataset( N_validation_branch, N_validation_trunk, n_grid_points, T_max, tau_max, dtype)
train_loader = DataLoader( train_dataset, batch_size=B, shuffle=True )

# Also store the dataset for later use
data_store_directory = './data/'
pt.save( train_dataset.branch_dataset.all().cpu(), data_store_directory + 'train_branch_data.pth' )
pt.save( train_dataset.trunk_dataset.all().cpu(), data_store_directory + 'train_trunk_data.pth' )
pt.save( validation_dataset.branch_dataset.all().cpu(), data_store_directory + 'validation_branch_data.pth' )
pt.save( validation_dataset.trunk_dataset.all().cpu(), data_store_directory + 'validation_trunk_data.pth' )

# Now do everything on the GPU
device = pt.device("mps")
dtype = pt.float32

# Create the DeepONet PINO
embedding_setup = { "n_grid_points" : n_grid_points, "n_hidden_layers" : 3, "kernel_size" : 5, "q": 10 }
n_hidden_layers = 4
z = 64
x_grid = pt.linspace( 0.0, 1.0, n_grid_points, device=device, dtype=dtype )
model = EmbeddingNetwork( embedding_setup, n_hidden_layers, z, x_grid, T_max, tau_max, train_dataset.trunk_dataset.logk_max )
model = model.to( device=device, dtype=dtype )
print('Number of Trainable Parameters: ', sum([ p.numel() for p in model.parameters() ]))

# Create the (tensorized) loss function
loss_fn = HeatLoss()

# Build the Adam optimizer
lr = 1e-2
n_steps = 5
step_size = 1000
n_epochs = n_steps * step_size
optimizer = Adam( model.parameters(), lr, amsgrad=True )
scheduler = StepLR( optimizer, step_size=step_size, gamma=0.1 )

# Bookkeeping and storage
train_counter = []
train_losses = []
train_grads = []
validation_counter = []
validation_losses = []
T_t_rmss = []
T_xx_rmss = []
rel_rmss = []
def train( epoch : int ):
    model.train()

    epoch_loss = float( 0.0 )
    for batch_idx, (x, t, params, u0) in enumerate( train_loader ):
        optimizer.zero_grad( set_to_none=True )

        x = x.to(device=device, dtype=dtype)
        t = t.to(device=device, dtype=dtype)
        params = params.to(device=device, dtype=dtype)
        u0 = u0.to(device=device, dtype=dtype)

        # Compute the loss and its gradient
        loss, loss_dict = loss_fn( model, x, t, params, u0 )
        loss.backward()
        epoch_loss += float( loss.item() )
        grad = getGradientNorm( model )

        rms = loss_dict["rms"]
        T_t_rms = loss_dict["T_t_rms"]
        T_xx_rms = loss_dict["T_xx_rms"]
        rel_rms = rms / (T_t_rms.item() + T_xx_rms.item() + 1e-12)
        print(T_xx_rms)

        # Update the weights
        optimizer.step()

        # Bookkeeping
        train_counter.append( (1.0*batch_idx) / len(train_loader) + epoch)
        train_losses.append( loss.item())
        train_grads.append( grad.cpu() )
        T_t_rmss.append( T_t_rms.item() )
        T_xx_rmss.append( T_xx_rms.item() )
        rel_rmss.append( rms.item() / (T_t_rms.item() + T_xx_rms.item() + 1e-12) )

    # Update
    epoch_loss /= len( train_loader )
    print('\nTrain Epoch: {} \tLoss: {:.4E} \tLoss Gradient: {:.4E} \tlr: {:.2E}'.format(
            epoch, epoch_loss, grad.item(), optimizer.param_groups[0]['lr']))
    print('T_t RMS: {:.4E} \tT_xx RMS: {:.4E} \tTotal RMS: {:.4E} \tLoss Relative RMS: {:.4E}'.format(
        T_t_rms, T_xx_rms, rms, rel_rms ))
    
    # Store the pretrained state
    pt.save( model.state_dict(), store_directory + 'model_adam.pth')
    pt.save( optimizer.state_dict(), store_directory + 'optimizer_adam.pth')

def validate( epoch : int ):
    model.eval()

    x, t, params, u0 = validation_dataset.all()
    x = x.to(device=device, dtype=dtype)
    t = t.to(device=device, dtype=dtype)
    params = params.to(device=device, dtype=dtype)
    u0 = u0.to(device=device, dtype=dtype)
    print(x.shape, t.shape, params.shape, u0.shape)

    # Compute the loss and its gradient
    loss, _ = loss_fn( model, x, t, params, u0 )

    # Bookkeeping
    validation_counter.append( epoch)
    validation_losses.append( loss.item() )

    # Update
    print( 'Validation Epoch: {} \tLoss: {:.4E}'.format( epoch, loss.item() ) )

# Actual training and validation
try:
    for epoch in range( n_epochs ):
        train( epoch )
#        validate( epoch )
        scheduler.step( )
except KeyboardInterrupt:
    pass

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