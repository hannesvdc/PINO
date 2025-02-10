import json
import torch as pt
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from DeepONet import DeepONet
from MultiIndexDataset import MultiIndexDeepONetDataset

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
dataset = MultiIndexDeepONetDataset(config, device, dtype)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Read the command line arguments
p = 300
branch_layers = [202, 512, 512, 2*p]
trunk_layers = [2, 512, 512, 2*p]
network = DeepONet(branch_layers=branch_layers, trunk_layers=trunk_layers).to(device, dtype=dtype)
optimizer = optim.Adam(network.parameters(), lr=1.e-3, amsgrad=True)
step = 250
scheduler = sch.StepLR(optimizer, step_size=step, gamma=0.1)
print('Number of Data Points per Parameter: ', len(dataset) / (1.0 * network.getNumberofParameters()))

# Training Routine
loss_fn = nn.MSELoss()
train_losses = []
train_grads = []
train_counter = []
def getGradient():
    grads = []
    for param in network.parameters():
        grads.append(param.grad.view(-1))
    grads = pt.cat(grads)
    return pt.norm(grads)
def getTrunkGradient():
    grads = []
    for param in network.trunk_net.parameters():
        grads.append(param.grad.view(-1))
    grads = pt.cat(grads)
    return pt.norm(grads)
def train(epoch):
    network.train()
    for batch_idx, (branch_input, trunk_input, branch_indices, trunk_indices) in enumerate(train_loader):
        optimizer.zero_grad()

        # Get the output data - Should be a 2d tensor
        target_u = dataset.output_data_u[branch_indices,:][:, trunk_indices]
        target_v = dataset.output_data_v[branch_indices,:][:, trunk_indices]

        # Compute Loss
        output_u, output_v = network(branch_input, trunk_input)
        loss_u = loss_fn(output_u, target_u)
        loss_v = loss_fn(output_v, target_v)
        loss = loss_u + loss_v

        # Compute loss gradient and do one optimization step
        loss.backward()
        trunk_grad = getTrunkGradient()
        grad = getGradient()
        optimizer.step()

        # Some housekeeping
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6E} \tLoss Gradient: {:.6E} \tTrunk Gradient: {:.6E} \tlr: {:.4E}'.format(
                        epoch, batch_idx * len(branch_input), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item(), grad, trunk_grad, optimizer.param_groups[0]['lr']))
        train_losses.append(loss.item())
        train_grads.append(grad.cpu())
        train_counter.append((1.0*batch_idx)/len(train_loader) + epoch-1)

        # Store the temporary state
        pt.save(network.state_dict(), store_directory + 'model_trunk_batch.pth')
        pt.save(optimizer.state_dict(), store_directory + 'optimizer_trunk_batch.pth')

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