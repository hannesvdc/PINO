import json
import torch as pt
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from model import ElasticityFiLMPINN
from loss import PhysicsLoss
from dataset import ElasticityDataset

from itertools import cycle

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
batch_size = 128
forcing_dataset = ElasticityDataset(config, device, dtype)
forcing_loader = DataLoader(forcing_dataset, batch_size=batch_size, shuffle=True)
xy_forcing_data = forcing_dataset.xy_forcing
internal_dataset = TensorDataset(forcing_dataset.xy_int)
internal_loader = DataLoader(internal_dataset, batch_size=batch_size, shuffle=True)
internal_iter = cycle(internal_loader)

# Create and initialize the model
m_boundary = 101
n_features = 64
trunk_hidden = 128
trunk_depth = 8
network = ElasticityFiLMPINN(m_boundary, n_features, trunk_hidden, trunk_depth)
network.to(device)
pretrained_state = pt.load(store_directory + 'pretrained_model.pth', map_location=device, weights_only=True)
network.load_state_dict(pretrained_state)
print('Number of Trainable Parameters: ', network.getNumberOfParameters())

# Optimizer
lr = 1e-3
optimizer = optim.Adam(network.parameters(), lr=lr, amsgrad=True)

# Physics loss
E_train = 1.0
nu = 0.3
w_int_init = 1e-3
w_int_max  = 5.0
loss_fn = PhysicsLoss(E_train, nu, w_int=w_int_init, w_forcing=1.0)
pref = loss_fn.pref

# Tracking
train_losses = []
train_grads = []
train_counter = []
physics_losses = []
physics_weights = []
forcing_losses = []
xy_empty = pt.empty((0, 2), device=device, dtype=dtype)

def getGradient():
    return pt.norm(pt.cat([p.grad.view(-1) for p in network.parameters() if p.grad is not None]))
def save_checkpoint(epoch, w_int, forcing_loss, physics_loss, total_loss):
    ckpt = {
        "epoch": int(epoch),
        "w_int": float(w_int),
        "forcing_loss" : forcing_loss,
        "physics_loss" : physics_loss,
        "total_loss" : total_loss,
        "model_state_dict": network.state_dict(),
    }
    filename = f'physics_training/epoch_{epoch:04d}.pth'
    pt.save(ckpt, store_directory + filename)
def posttrain(epoch):
    network.train()
    clip_level = 5.0

    for f_batch_idx, f_batch in enumerate(forcing_loader):
        optimizer.zero_grad()

        (xy_batch,) = next(internal_iter)

        # Reshape the forcing inputs
        gx = f_batch[:, :m_boundary]
        gy = f_batch[:, m_boundary:]
        g_batch = pt.stack([gx, gy], dim=1)

        # Forcing loss
        loss_forcing = loss_fn(network, g_batch, xy_empty, xy_forcing_data)

        # Interior Loss
        loss_int = loss_fn(network, g_batch, xy_batch, xy_empty)

        # Compute the combined loss and backprop
        total = loss_forcing + loss_int
        total.backward()

        # Step optimizer and scheduler
        pt.nn.utils.clip_grad_norm_(network.parameters(), max_norm=clip_level)
        grad = getGradient()
        optimizer.step()

        # Some housekeeping
        print('Train Epoch: {} [{}/{} ({:.0f}%)] \tForcing Loss: {:.4E} (w = {:.1E}) \tPhysics Loss: {:.4E} (w = {:.1E}) \tLoss Gradient: {:.4E} \tlr: {:.2E}'.format(
                        epoch, f_batch_idx * len(f_batch), len(forcing_dataset),
                        100. * f_batch_idx / len(forcing_loader), loss_forcing.item(), loss_fn.w_forcing, loss_int.item(), loss_fn.w_int, grad, optimizer.param_groups[0]['lr']))
        forcing_losses.append(loss_forcing.item())
        physics_losses.append(loss_int.item())
        physics_weights.append(loss_fn.w_int)
        train_losses.append(total.item())
        train_grads.append(grad.cpu().item())
        train_counter.append(epoch + f_batch_idx / len(forcing_loader))

    # Store the current checkpoint
    save_checkpoint(epoch, loss_fn.w_int, loss_forcing.item(), loss_int.item(), total.item())

# Main loop
n_epochs = 300
try:
    for epoch in range(1, n_epochs + 1):
        # Exponentially ramp w_int from init to max
        w_int = w_int_init * (w_int_max / w_int_init) ** (epoch / n_epochs)
        loss_fn.setWeights(w_int=w_int, w_forcing=1.0)
        print(f"\n>>> Epoch {epoch:3d} | w_int = {w_int:.2e}")
        posttrain(epoch)
except KeyboardInterrupt:
    print('Aborting Training. Plotting Training Convergence.')

# Plot
plt.figure()
plt.semilogy(train_counter, forcing_losses, label="Forcing Loss", alpha=0.6)
plt.semilogy(train_counter, physics_losses, label="Weighted Physics Loss", alpha=0.6)
plt.semilogy(train_counter, np.array(physics_losses) / np.array(physics_weights), label="Weighted Physics Loss", alpha=0.6)
plt.semilogy(train_counter, train_losses, label="Combined Loss", alpha=0.6)
plt.semilogy(train_counter, train_grads, label="Weighted Gradient Norm", alpha=0.6)
plt.xlabel("Epoch")
plt.ylabel("Loss / Gradient")
plt.legend()
plt.title("Determining the Optimal Physics Weights")
plt.tight_layout()
plt.show()
