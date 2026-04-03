import sys
sys.path.append('../')
import json
import torch as pt
import torch.nn as nn

import matplotlib.pyplot as plt

from DeepONet import DeepONet
from Dataset import DeepONetDataset
from SSBroyden import SelfScaledBroyden

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
n_data_points = len(dataset) * 101**2
trunk_input = dataset.trunk_input_data

# Read the command line arguments
p = 200
branch_layers = [202, 128, 2*p]
trunk_layers = [2, 128, 2*p]
network = DeepONet(branch_layers=branch_layers, trunk_layers=trunk_layers).to(device, dtype=dtype)
optimizer = SelfScaledBroyden(network.parameters())
print('Number of Data Points per Parameter: ', n_data_points / (1.0 * network.getNumberofParameters()))

# Training Routine
loss_fn = nn.MSELoss()
def closure():
    optimizer.zero_grad()

    # Compute Loss
    output_u, output_v = network(dataset.branch_input_data, dataset.trunk_input_data)
    loss_u = loss_fn(output_u, dataset.output_data_u)
    loss_v = loss_fn(output_v, dataset.output_data_v)
    loss = loss_u + loss_v

    # Compute loss gradient and do one optimization step
    loss.backward()
    return loss

train_losses = []
train_grads = []
train_counter = []
def train(epoch):
    network.train()

    # Perform one optimization step per epoch
    loss = optimizer.step(closure=closure)
    grad_norm = pt.norm(pt.cat([p.grad.view(-1) for p in network.parameters()]))

    # Print and store the current state.
    print('Train Epoch: {} \tLoss: {:.6E} \tLoss Gradient: {:.6E}'.format(
                    epoch, loss.item(), grad_norm))
    train_losses.append(loss.item())
    train_grads.append(grad_norm.cpu())
    train_counter.append(epoch-1)

    # Store the temporary state
    pt.save(network.state_dict(), store_directory + 'model_broyden.pth')
    pt.save(optimizer.state_dict(), store_directory + 'optimizer_broyden.pth')

# Do the actual training
print('\nStarting Training Procedure...')
n_epochs = 1000
try:
    for epoch in range(1, n_epochs + 1):
        train(epoch)
except KeyboardInterrupt:
    print('Aborting Training. Plotting Training Convergence.')

# Show the training results
plt.semilogy(train_counter, train_losses, color='tab:blue', label='Training Loss', alpha=0.5)
plt.semilogy(train_counter, train_grads, color='tab:orange', label='Loss Gradient', alpha=0.5)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()