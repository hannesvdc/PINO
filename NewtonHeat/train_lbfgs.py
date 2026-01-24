import torch as pt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import LBFGS
import matplotlib.pyplot as plt

from dataset import NewtonDataset
from model import PINO, PINOLoss

# Train on the GPU
device = pt.device("cpu")
dtype = pt.float64
pt.set_default_dtype( dtype )
pt.set_grad_enabled(True)

store_directory = './Results/' 
train_data = pt.load( store_directory + 'train_data.pth', map_location=device ).to( dtype=dtype )
t_train = train_data[:,0:1]
p_train = train_data[:,1:]
validation_data = pt.load( store_directory + 'validation_data.pth', map_location=device ).to( dtype=dtype )
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
model = PINO( n_hidden_layers, z, T_max ).to( dtype=dtype )
model.load_state_dict( pt.load( store_directory + 'model_adam.pth', weights_only=True, map_location=device ) )
loss_fn = PINOLoss()
print('Number of Trainable Parameters: ', sum([ p.numel() for p in model.parameters() ])) 

# Create the adam optimizer with learning rate scheduler
lr = 1.0
max_iter = 100
n_epochs = 250
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
    epoch_loss = loss_fn( model, t_train, p_train )
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
train_counter = []
train_losses = []
train_grads = []
validation_counter = []
validation_losses = []
n_stuck = 0
last_loss = pt.inf
try:
    for epoch in range( n_epochs ):
        w0 = pt.cat([p.detach().flatten() for p in model.parameters()])
        training_loss = optimizer.step( closure )
        w1 = pt.cat([p.detach().flatten() for p in model.parameters()])
        step_norm = (w1 - w0).norm().item()
        print('\nTrain Epoch: {} \tLoss: {:.4E} \tLoss Gradient: {:.4E} \tStep Norm: {:.4E}'.format( epoch, last_state["loss"], last_state["grad_norm"], step_norm ))
        
        validation_loss = validate()
        print('Validation Epoch: {} \tLoss: {:.4E}'.format( epoch, validation_loss.item() ))
        
        # Store the pretrained state
        pt.save( model.state_dict(), store_directory + 'model_lbfgs.pth')
        pt.save( optimizer.state_dict(), store_directory + 'optimizer_lbfgs.pth')

        train_counter.append( epoch )
        train_losses.append( last_state["loss"] )
        train_grads.append( last_state["grad_norm"] )
        validation_losses.append( validation_loss.item() )

        stuck = abs(training_loss - last_loss) < 1e-10 and step_norm < 1e-8
        n_stuck += stuck
        if n_stuck >= 5:
            print('Lowering L-BFGS Learning Rate.')
            n_stuck = 0
            lr = 0.1 * lr
            optimizer = LBFGS( model.parameters(), lr, max_iter=max_iter, line_search_fn="strong_wolfe", history_size=history_size )
        last_loss = float( training_loss )
except KeyboardInterrupt:
    pass

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
plt.show()
