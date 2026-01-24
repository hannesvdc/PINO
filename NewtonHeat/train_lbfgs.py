import torch as pt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import LBFGS
import matplotlib.pyplot as plt

from dataset import NewtonDataset
from model import PINO, PINOLoss

# Train on the GPU
device = pt.device("mps")
dtype = pt.float32
pt.set_default_dtype( dtype )
pt.set_grad_enabled(True)

store_directory = './Results/' 
train_data = pt.load( store_directory + 'train_data.pth', weights_only=True, map_location=device )
t_train = train_data[:,0:1]
p_train = train_data[:,1:]
validation_data = pt.load( store_directory + 'validation_data.pth', weights_only=True, map_location=device )
t_validation = validation_data[:,0:1]
p_validation = validation_data[:,1:]

# Create the training and validation datasets
B = 128
N_validation = validation_data.shape[0]
train_dataset = TensorDataset( t_train, p_train )
train_loader = DataLoader( train_dataset, batch_size=B, shuffle=False, drop_last=True )
validation_dataset = TensorDataset( t_validation, p_validation )
validation_loader = DataLoader( validation_dataset, batch_size=N_validation, shuffle=False, drop_last=True )

# Create the PINO model and loss
n_hidden_layers = 2
z = 64
T_max = 10.0
model = PINO( n_hidden_layers, z, T_max ).to( device=device )
model.load_state_dict( pt.load( store_directory + 'model_adam.pth', weights_only=True ) )
loss_fn = PINOLoss()
print('Number of Trainable Parameters: ', sum([ p.numel() for p in model.parameters() ])) 

# Create the adam optimizer with learning rate scheduler
lr = 1.0
max_iter = 50
n_epochs = 100
optimizer = LBFGS( model.parameters(), lr, max_iter=max_iter )
def getGradientNorm():
    grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    return pt.norm(pt.cat(grads))

# The L-BFGS closure
def closure():
    model.train()
    optimizer.zero_grad()

    # Batched but Deterministic Loss for memory constraints
    epoch_loss = 0.0
    n_batches = len( train_loader )
    for batch_idx, (t, p) in enumerate( train_loader ):
        loss = loss_fn( model, t, p ) / n_batches
        epoch_loss += float( loss.item() )
        loss.backward()
    
    # LBFGS only needs a scalar loss value back.
    # It does NOT need the returned tensor to carry a graph, since grads are already in .grad.
    return pt.tensor(epoch_loss, device=device, dtype=dtype)
def validate():
    print('Model Validation...')
    model.eval()

    # Just the one validation batch
    epoch_loss : float = 0.0
    n_batches = len( validation_loader )
    for batch_idx, (t, p) in enumerate( validation_loader ):
        loss = loss_fn( model, t, p ) / n_batches
        epoch_loss += float(loss.item())
    
    return epoch_loss

# Actual training and validation
print('\nStarting Training...')
train_counter = []
train_losses = []
validation_counter = []
validation_losses = []
for epoch in range( n_epochs ):
    training_loss = optimizer.step( closure )
    print('Train Epoch: {} \tLoss: {:.4E}'.format( epoch, training_loss ))
    
    validation_loss = validate()
    print('Validation Epoch: {} \tLoss: {:.4E}'.format( epoch, validation_loss ))
    
    # Store the pretrained state
    pt.save( model.state_dict(), store_directory + 'model_lbfgs.pth')
    pt.save( optimizer.state_dict(), store_directory + 'optimizer_lbfgs.pth')

    train_counter.append( epoch )
    train_losses.append( training_loss )
    validation_losses.append( validation_loss )

# Show the training results
plt.semilogy(train_counter, train_losses, label='Training Loss', alpha=0.5)
plt.semilogy(validation_counter, validation_losses, label='Validation Loss', alpha=0.5)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('L-BFGS')
plt.show()
