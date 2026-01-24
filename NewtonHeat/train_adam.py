import torch as pt
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

from dataset import NewtonDataset
from model import PINO, PINOLoss

# Train on the GPU
device = pt.device("mps")
dtype = pt.float32
pt.set_default_dtype( dtype )
pt.set_grad_enabled(True)

# Create the training and validation datasets
T_max = 10.0
tau_max = 4.0
N_train = 100_000
N_validation = 5_000
train_dataset = NewtonDataset( N_train, T_max, tau_max, device, dtype )
validation_dataset = NewtonDataset( N_validation, T_max, tau_max, device, dtype )
B = 128
train_loader = DataLoader( train_dataset, batch_size=B, shuffle=True )
validation_loader = DataLoader( validation_dataset, batch_size=N_validation, shuffle=False )

# Create the PINO
n_hidden_layers = 2
z = 64
model = PINO( n_hidden_layers, z, T_max ).to( device=device )
loss_fn = PINOLoss()
print('Number of Trainable Parameters: ', sum([ p.numel() for p in model.parameters() ]))

# Create the adam optimizer with learning rate scheduler
lr = 1e-3
n_epochs = 500
optimizer = Adam( model.parameters(), lr)
scheduler = StepLR( optimizer, step_size=100, gamma=0.1 )
def getGradientNorm():
    grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    return pt.norm(pt.cat(grads))

# Bookkeeping and storage
train_counter = []
train_losses = []
train_grads = []
validation_counter = []
validation_losses = []
store_directory = './Results/'

# Train Function
def train( epoch ):
    model.train()
    clip_level = 5.0

    for batch_idx, (t, p) in enumerate( train_loader ):
        optimizer.zero_grad()

        # Compute the loss and its gradient
        loss = loss_fn( model, t, p )
        loss.backward()
        pt.nn.utils.clip_grad_norm_( model.parameters(), max_norm=clip_level )
        grad = getGradientNorm()

        # Update the weights
        optimizer.step()

        # Bookkeeping
        train_counter.append( (1.0*batch_idx) / len(train_loader) + epoch)
        train_losses.append( loss.item())
        train_grads.append( grad.cpu() )

        # Update
        print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.4E} \tLoss Gradient: {:.4E} \tlr: {:.2E}'.format(
                        epoch, batch_idx , len( train_loader ), 100. * batch_idx / len( train_loader ), 
                        loss.item(), grad.item(), optimizer.param_groups[0]['lr']))
    
    # Store the pretrained state
    pt.save( model.state_dict(), store_directory + 'model_adam.pth')
    pt.save( optimizer.state_dict(), store_directory + 'optimizer_adam.pth')

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
        print('Validation Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.4E}'.format(
                        epoch, batch_idx , len( validation_loader ), 100. * batch_idx / len( validation_loader ), loss.item()))

# Actual training and validation    
for epoch in range( n_epochs ):
    train( epoch )
    validate( epoch )
    scheduler.step( )

# Show the training results
plt.semilogy(train_counter, train_losses, label='Training Loss', alpha=0.5)
plt.semilogy(train_counter, train_grads, label='Loss Gradient', alpha=0.5)
plt.semilogy(validation_counter, validation_losses, label='Validation Loss', alpha=0.5)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()