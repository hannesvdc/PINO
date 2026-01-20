import json
import torch as pt
import numpy.linalg as lg
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from model import ElasticityFiLMPINN
from loss import PhysicsLoss
from dataset import ElasticityDataset

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
m_boundary = 101
forcing_dataset = ElasticityDataset(config, device, dtype, reduce=True)
forcing_loader = DataLoader(forcing_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
fixed_forcing_sample = next(iter(forcing_loader))
gx = fixed_forcing_sample[:, :m_boundary]
gy = fixed_forcing_sample[:, m_boundary:]
g_batch = pt.stack([gx, gy], dim=1)

xy_forcing_data = forcing_dataset.xy_forcing
internal_dataset = TensorDataset(forcing_dataset.xy_int)
internal_loader = DataLoader(internal_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
n_chunks = len(internal_loader)

# Create and initialize the model
n_features = 64
trunk_hidden = 128
trunk_depth = 8
network = ElasticityFiLMPINN(m_boundary, n_features, trunk_hidden, trunk_depth)
network.to(device)
print('Number of Trainable Parameters: ', network.getNumberOfParameters())

# Load the network state corresponding to this optimal physics weight.
optimal_weight_state = pt.load( store_directory + f"post_training/epoch_0100.pth", map_location=device, weights_only=True)
network.load_state_dict(optimal_weight_state["model_state_dict"])

# Optimizer
max_iter = 20
n_epochs = 10
optimizer = optim.LBFGS(network.parameters(), lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

# Physics loss
E_train = 1.0
nu = 0.3
w_int = 1.02
loss_fn = PhysicsLoss(E_train, nu, w_int=w_int, w_forcing=1.0)
pref = loss_fn.pref

# Tracking
train_losses = []
train_grads = []
train_counter = []
xy_empty = pt.empty((0, 2), device=device, dtype=dtype)

def getGradient():
    grads = [p.grad.view(-1) for p in network.parameters() if p.grad is not None]
    return pt.cat(grads).cpu().numpy()
def save_checkpoint(epoch, total_loss):
    ckpt = {
        "epoch": int(epoch),
        "total_loss" : total_loss,
        "model_state_dict": network.state_dict(),
    }
    filename = f'post_bfgs_training/epoch_{epoch:04d}.pth'
    pt.save(ckpt, store_directory + filename)
def closure():
    print('Model Evaluation...')

    network.train()
    optimizer.zero_grad()

    # Forcing loss
    loss_forcing = loss_fn(network, g_batch, xy_empty, xy_forcing_data)
    loss_forcing.backward()
    epoch_loss = float(loss_forcing.detach())

    # Interior Loss
    for _, (xy_batch,) in enumerate(internal_loader):
        loss_int = loss_fn(network, g_batch, xy_batch, xy_empty) / n_chunks
        loss_int.backward()
        epoch_loss += float(loss_int.detach())
    
    # LBFGS only needs a scalar loss value back.
    # It does NOT need the returned tensor to carry a graph, since grads are already in .grad.
    print('...Done')
    return pt.tensor(epoch_loss, device=device, dtype=dtype)

# Main loop
try:
    for epoch in range(1, n_epochs + 1):
        print(f"\n>>> Epoch {epoch:3d}")
        loss_val = optimizer.step(closure)
        loss_grad = getGradient()

        # Some housekeeping
        print('Epoch {}: \tTotal Loss: {:.4E} \tTotal Loss Gradient: {:.4E}'.format(
                        epoch, loss_val.item(), lg.norm(loss_grad)))
        train_losses.append(loss_val.item())
        train_grads.append(lg.norm(loss_grad))
        train_counter.append(epoch)

        save_checkpoint(epoch, loss_val.item())
        
except KeyboardInterrupt:
    print('Aborting Training. Plotting Training Convergence.')

# Plot
plt.figure()
plt.semilogy(train_counter, train_losses, label="Total Loss", alpha=0.6)
plt.semilogy(train_counter, train_grads, label="Gradient Norm", alpha=0.6)
plt.xlabel("Epoch")
plt.ylabel("Loss / Gradient")
plt.legend()
plt.title("Post Training with L-BFGS")
plt.tight_layout()
plt.show()
