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
elif pt.cuda.device_count() > 0:
    device = pt.device("cuda:0")
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
batch_xy = 512
internal_loader = DataLoader(internal_dataset, batch_size=batch_xy, shuffle=True)
n_chunks = len(internal_loader)
n_data_points = len(forcing_dataset) * len(internal_dataset)
xy_left = forcing_dataset.xy_left
xy_forcing = forcing_dataset.xy_forcing

# Create and initialize the model
n_branch_conv = 5
n_branch_channels = 8
kernel_size = 7
n_branch_residual = 3
n_trunk_residual = 4
p = 100
network = ConvDeepONet(n_branch_conv, n_branch_channels, kernel_size, n_branch_residual, n_trunk_residual, p)
network.to(device)

# Create the Adam optimizer and LR scheduler
optimizer = optim.Adam(network.parameters(), lr=10**(-2.5), amsgrad=True)
n_epochs = 2500
scheduler = sch.OneCycleLR(optimizer,
                       max_lr=1.e-3,
                       total_steps=n_epochs,
                       pct_start=0.3, div_factor=25, final_div_factor=10)
print('Number of Data Points per Parameter: ', n_data_points / (1.0 * network.getNumberofParameters()))

# Training Routine
E_train = 1.0 
nu = 0.3
w_int = 0.0
w_forcing = 1.0
loss_fn = PhysicsLoss(E_train, nu, w_int, w_forcing)
train_losses = []
train_grads = []
train_counter = []
def getGradient():
    grads = [p.grad.view(-1) for p in network.parameters() if p.grad is not None]
    return pt.norm(pt.cat(grads))

pref = loss_fn.pref
xy_empty = pt.empty((0,2), device=device, dtype=pt.float32)
xy_left = pt.stack((pt.zeros(forcing_dataset.grid_points), pt.linspace(0.0, 1.0, forcing_dataset.grid_points)), dim=1).to(device=device, dtype=pt.float32)
def train(epoch):
    network.train()

    clip_level = 5.0 # only optimizing the forcing loss
    for f_batch_idx, f_batch in enumerate(forcing_loader):
        optimizer.zero_grad()

        # Print some diagnostics
        with pt.no_grad():
            g_x = f_batch[:, :101]
            g_y = f_batch[:, 101:]
            u, v = network.forward(f_batch, xy_left)
            dirichlet_loss = u.square().mean() + v.square().mean()
            J, _ = loss_fn.grads_and_hess(network, f_batch, xy_forcing, needsHessian=False)
            u_x = J[:,:,0,0];  u_y = J[:,:,0,1]
            v_x = J[:,:,1,0];  v_y = J[:,:,1,1]
            tx = (pref * (u_x + nu * v_y) + g_x).abs().mean().item()
            ty = (pref * ((1-nu)/2)*(u_y+v_x) + g_y).abs().mean().item()
            print(f"\n⟨|s_xx+g_x|⟩={tx:.3e}   ⟨|s_xy+g_y|⟩={ty:.3e}")

        # Do the boundary losses once
        loss_forcing = loss_fn(network, f_batch, xy_empty, xy_forcing)
        loss_forcing.backward()

        # Do an optimizer step
        pt.nn.utils.clip_grad_norm_(network.parameters(), max_norm=clip_level)
        grad = getGradient()
        optimizer.step()

        # Some housekeeping
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tDirichlet Loss: {:.4E} \tForcing Loss: {:.4E} \tPhysics Loss: {:.4E} \tLoss Gradient: {:.4E} \tlr: {:.2E}'.format(
                        epoch, f_batch_idx * len(f_batch), len(forcing_dataset),
                        100. * f_batch_idx / len(forcing_loader), dirichlet_loss, loss_forcing.item(), 0.0, grad, optimizer.param_groups[0]['lr']))
        train_losses.append(loss_forcing.item())
        train_grads.append(grad.cpu())
        train_counter.append((1.0*f_batch_idx)/len(forcing_loader) + epoch-1)

    # Store the pretrained state
    pt.save(network.state_dict(), store_directory + 'pretrained_model.pth')
    pt.save(optimizer.state_dict(), store_directory + 'pretrained_optimizer.pth')

# Do the actual training
print('\nStarting Training Procedure...')
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