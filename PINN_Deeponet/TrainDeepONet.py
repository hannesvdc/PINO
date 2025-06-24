import json
import torch as pt
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sch
from torch.utils.data import DataLoader
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
batch_size = 128
dataset = DeepONetDataset(config, device, dtype)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
n_data_points = len(dataset) * 101**2

# Read the command line arguments
n_branch_conv = 3
kernel_size = 5
p = 25
network = ConvDeepONet(n_branch_conv, kernel_size, p)
optimizer = optim.Adam(network.parameters(), lr=1.e-3, amsgrad=True)
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

def train(epoch):
    network.train()
    for batch_idx, branch_input in enumerate(train_loader):
        optimizer.zero_grad()

        # Compute Loss
        xy_int = dataset.xy_int
        xy_left = dataset.xy_left
        xy_forcing = dataset.xy_forcing
        loss = loss_fn.forward(network, branch_input, xy_int, xy_left, xy_forcing)

        # Compute loss gradient and do one optimization step
        loss.backward()
        grad = getGradient()
        optimizer.step()

        # Some housekeeping
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6E} \tLoss Gradient: {:.6E} \tlr: {:.4E}'.format(
                        epoch, batch_idx * len(branch_input), len(train_loader),
                        100. * batch_idx / len(train_loader), loss.item(), grad, optimizer.param_groups[0]['lr']))
        train_losses.append(loss.item())
        train_grads.append(grad.cpu())
        train_counter.append((1.0*batch_idx)/len(train_loader) + epoch-1)

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