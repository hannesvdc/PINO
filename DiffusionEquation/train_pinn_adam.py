import torch as pt
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

from PINNDataset import PINNDataset
from FixedInitialPINN import FixedInitialPINN
from Loss import HeatLoss
from sampleInitialGP import gp

# Train on the GPU
dtype = pt.float64
pt.set_grad_enabled( True )
pt.set_default_dtype( dtype )

# Create the initial condition. We want T_max to be ~95th percentile, so std=T_max/1.96
T_max = 10.0
N_grid_points = 51
l = 0.12 # GP correlation length
x_grid = pt.linspace(0.0, 1.0, N_grid_points)
u0 = gp( x_grid, l )

# Normalize so that ~95% of T0 lives in [-T_max, T_max]
ci_95 = 1.96
u0 = u0 / ci_95

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
store_directory = './Results/pinn/'
pt.save( train_dataset.all().cpu(), store_directory + 'train_data.pth' )
pt.save( validation_dataset.all().cpu(), store_directory + 'validation_data.pth' )

# Create the PINO
z = 64
n_hidden_layers = 2
model = FixedInitialPINN( n_hidden_layers, z, T_max, tau_max, u0, x_grid, l )
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

# Create the adam optimizer with learning rate scheduler
lr = 1e-2
n_steps = 5
step_size = 1000
n_epochs = n_steps * step_size
optimizer = Adam( model.parameters(), lr )
scheduler = StepLR( optimizer, step_size=step_size, gamma=0.1 )
def getGradientNorm():
    grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    return pt.norm(pt.cat(grads))

# Bookkeeping and storage
train_counter = []
train_losses = []
train_grads = []
validation_counter = []
validation_losses = []
T_t_rmss = []
T_xx_rmss = []
rel_rmss = []

# Train Function
def train( epoch ):
    model.train()
    clip_level = 5e4

    epoch_loss = float( 0.0 )
    for batch_idx, (x, t, p) in enumerate( train_loader ):
        optimizer.zero_grad( set_to_none=True )
        x = x.to(device=device, dtype=dtype)
        t = t.to(device=device, dtype=dtype)
        p = p.to(device=device, dtype=dtype)

        # Compute the loss and its gradient
        loss, rms, T_t_rms, T_xx_rms = loss_fn( model, x, t, p )
        loss.backward()
        epoch_loss += float( loss.item() )
        pre_grad = getGradientNorm()
        pt.nn.utils.clip_grad_norm_( model.parameters(), max_norm=clip_level )
        grad = getGradientNorm()

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
    print('\nTrain Epoch: {} \tLoss: {:.4E} \tPre-Clip Loss Gradient: {:.4E} \tLoss Gradient: {:.4E} \tlr: {:.2E}'.format(
            epoch, epoch_loss, pre_grad.item(), grad.item(), optimizer.param_groups[0]['lr']))
    print('T_t RMS: {:.4E} \tT_xx RMS: {:.4E} \tTotal RMS: {:.4E} \tLoss Relative RMS: {:.4E}'.format(
        T_t_rms, T_xx_rms, rms, rms / (T_t_rms + T_xx_rms + 1e-12) ))
    
    # Store the pretrained state
    pt.save( model.state_dict(), store_directory + 'model_adam.pth')
    pt.save( optimizer.state_dict(), store_directory + 'optimizer_adam.pth')

# Validation Function 
def validate( epoch ):
    model.eval()

    for batch_idx, (x, t, p) in enumerate( validation_loader ):
        x = x.to(device=device, dtype=dtype)
        t = t.to(device=device, dtype=dtype)
        p = p.to(device=device, dtype=dtype)

        # Compute the loss and its gradient
        loss, rms, T_t_rms, T_xx_rms = loss_fn( model, x, t, p )

        # Bookkeeping
        validation_counter.append( (1.0*batch_idx) / len(validation_loader) + epoch)
        validation_losses.append( loss.item())

        # Update
        print( 'Validation Epoch: {} \tLoss: {:.4E}'.format( epoch, loss.item() ) )

# Actual training and validation
try:
    for epoch in range( n_epochs ):
        train( epoch )
        validate( epoch )
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
plt.semilogy(train_counter, T_t_rmss, label=r'RMS $T_t$')
plt.semilogy(train_counter, T_xx_rmss, label=r'RMS $T_{xx}$')
plt.semilogy(train_counter, rel_rmss, label='Relative Loss RMS')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Adam')
plt.show()