import torch as pt
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

from dataset import NewtonDataset
from model import PINO, AdvanedPhysicsPINO
from loss import PINOLoss

import argparse
def parseArguments( ):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_type', dest='model_type')
    return arg_parser.parse_args( )
args = parseArguments()

# Train on the GPU
device = pt.device("mps")
dtype = pt.float32
pt.set_default_dtype( dtype )
pt.set_grad_enabled( True )

# Create the training and validation datasets
T_max = 10.0
tau_max = 8.0 # train to exp( -tau_max )
N_train = 10_000
N_validation = 5_000
train_dataset = NewtonDataset( N_train, T_max, tau_max, device, dtype )
validation_dataset = NewtonDataset( N_validation, T_max, tau_max, device, dtype )
B = 128
train_loader = DataLoader( train_dataset, batch_size=B, shuffle=True )
validation_loader = DataLoader( validation_dataset, batch_size=N_validation, shuffle=False )

# Also store the dataset for later use
store_directory = './Results/'
pt.save( train_dataset.all().cpu(), store_directory + 'train_data.pth' )
pt.save( validation_dataset.all().cpu(), store_directory + 'validation_data.pth' )

# Create the PINO
z = 32
if args.model_type == 'advanced':
    n_hidden_layers = 1
    model = AdvanedPhysicsPINO( n_hidden_layers, z, T_max, tau_max, train_dataset.logk_min, train_dataset.logk_max ).to( device=device )
    step_size = 100
elif args.model_type == 'initial':
    n_hidden_layers = 2
    model = PINO( n_hidden_layers, z, T_max, tau_max, train_dataset.logk_min, train_dataset.logk_max ).to( device=device )
    step_size = 1000
elif args.model_type == "simple":
    n_hidden_layers = 1
    model = PINO( n_hidden_layers, z, T_max, tau_max, train_dataset.logk_min, train_dataset.logk_max ).to( device=device )
    step_size = 1000
else:
    print('This model type is not supported.')
    exit()
print('Number of Trainable Parameters: ', sum([ p.numel() for p in model.parameters() ]))
loss_fn = PINOLoss()

# Create the adam optimizer with learning rate scheduler
lr = 1e-3
n_steps = 4
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

# Train Function
def train( epoch ):
    model.train()
    clip_level = 5e2

    epoch_loss = float( 0.0 )
    for batch_idx, (t, p) in enumerate( train_loader ):
        optimizer.zero_grad( set_to_none=True )

        # Compute the loss and its gradient
        loss = loss_fn( model, t, p )
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

    # Update
    epoch_loss /= len( train_loader )
    print('\nTrain Epoch: {} \tLoss: {:.4E} \tPre-Clip Loss Gradient: {:.4E} \tLoss Gradient: {:.4E} \tlr: {:.2E}'.format(
            epoch, epoch_loss, pre_grad.item(), grad.item(), optimizer.param_groups[0]['lr']))
    
    # Store the pretrained state
    pt.save( model.state_dict(), store_directory + args.model_type + '_model_adam.pth')
    pt.save( optimizer.state_dict(), store_directory + args.model_type + '_optimizer_adam.pth')

# Validation Function 
def validate( epoch ):
    model.eval()

    for batch_idx, (t, p) in enumerate( validation_loader ):

        # Compute the loss and its gradient
        loss = loss_fn( model, t, p )

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
np.save( store_directory + args.model_type + '_Adam_Training_Convergence.npy', np.hstack( (train_counter, train_losses, train_grads) ) )
np.save( store_directory + args.model_type + '_Adam_Validation_Convergence.npy', np.hstack( (validation_counter, validation_losses) ) )

# Show the training results
plt.semilogy(train_counter, train_losses, label='Training Loss', alpha=0.5)
plt.semilogy(train_counter, train_grads, label='Loss Gradient', alpha=0.5)
plt.semilogy(validation_counter, validation_losses, label='Validation Loss', alpha=0.5)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Adam')
plt.show()