import json
import torch as pt
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from model import ConvDeepONet, PhysicsLoss
from dataset import DeepONetDataset

# Just some sanity pytorch settings
pt.set_grad_enabled(True)
pt.set_default_dtype(pt.float32)
if pt.backends.mps.is_available():
    device = pt.device("mps")
else:
    print('Using CPU because no GPU is available.')
    device = pt.device("cpu")
dtype = pt.float32

# Load the data configuration
config_file = 'DataConfig.json'
config = json.load(open(config_file))
store_directory = config['Store Directory']

# Load the data in memory
batch_size = 64
forcing_dataset = DeepONetDataset(config, device, dtype)
forcing_loader = DataLoader(forcing_dataset, batch_size=batch_size, shuffle=True)
internal_dataset = TensorDataset(forcing_dataset.xy_int)
batch_xy = 1024
internal_loader = DataLoader(internal_dataset, batch_size=batch_xy, shuffle=True)
n_chunks = len(internal_loader)
n_data_points = len(forcing_dataset) * len(internal_dataset)
xy_left = forcing_dataset.xy_left
xy_forcing = forcing_dataset.xy_forcing


# Read the command line arguments
n_branch_conv = 3
kernel_size = 5
p = 25
network = ConvDeepONet(n_branch_conv, kernel_size, p).to(device)
optimizer = optim.Adam(network.parameters(), lr=1.e-3)
step = 250
scheduler = sch.StepLR(optimizer, step_size=step, gamma=0.1)
print('Number of Data Points per Parameter: ', n_data_points / (1.0 * network.getNumberofParameters()))

# Training Routine
E_train = 1.0 
nu = 0.3
w_int = 1.0
w_dirichlet = 1.0
w_forcing = 1.0
loss_fn = PhysicsLoss(E_train, nu, w_int, w_dirichlet, w_forcing)
train_losses = []
train_grads = []
train_counter = []
def getGradient():
    grads = []
    for param in network.parameters():
        grads.append(param.grad.view(-1)) # type: ignore
    grads = pt.cat(grads)
    return pt.norm(grads)

xy_empty = pt.empty((0,2), device=device, dtype=pt.float32)
def train(epoch):
    network.train()

    for f_batch_idx, f_batch in enumerate(forcing_loader):
        optimizer.zero_grad()

        # Do the boundary losses once
        loss_bc = loss_fn(network, f_batch, xy_empty, xy_left, xy_forcing)
        loss_bc.backward()

        physics_loss : float = 0.0
        for xy_batch_idx, (xy_batch,) in enumerate(internal_loader):
            loss_int = loss_fn.forward(network, f_batch, xy_batch, xy_empty, xy_empty) / n_chunks
            loss_int.backward()
            physics_loss += loss_int.item()

        # Do an optimizer step
        grad = getGradient()
        optimizer.step()

        # Some housekeeping
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBoundary Loss: {:.4E} \tPhysics Loss: {:.4E} \tLoss Gradient: {:.4E} \tlr: {:.2E}'.format(
                        epoch, f_batch_idx * len(f_batch), len(forcing_dataset),
                        100. * f_batch_idx / len(forcing_loader), loss_bc.item(), physics_loss, grad, optimizer.param_groups[0]['lr']))
        train_losses.append(loss_bc.item() + physics_loss)
        train_grads.append(grad.cpu())
        train_counter.append((1.0*f_batch_idx)/len(forcing_loader) + epoch-1)

        # Store the temporary state
        pt.save(network.state_dict(), store_directory + 'model.pth')
        pt.save(optimizer.state_dict(), store_directory + 'optimizer.pth')

# Do the actual training
print('\nStarting Training Procedure...')
n_epochs = 4 * step
try:
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        scheduler.step()
except KeyboardInterrupt:
    print('Aborting Training. Plotting Training Convergence.')

# Show the training results
plt.semilogy(train_counter, train_losses, color='tab:blue', label='Training Loss', alpha=0.5)
plt.semilogy(train_counter, train_grads, color='tab:orange', label='Loss Gradient', alpha=0.5)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()