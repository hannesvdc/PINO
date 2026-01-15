import json
import torch as pt
import torch.optim as optim
from torch.utils.data import DataLoader
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
forcing_dataset = ElasticityDataset(config, device, dtype)
forcing_loader = DataLoader(forcing_dataset, batch_size=batch_size, shuffle=True)
xy_forcing = forcing_dataset.xy_forcing

# Create and initialize the model
m_boundary = 101
n_features = 64
trunk_hidden = 128
trunk_depth = 8
network = ElasticityFiLMPINN(m_boundary, n_features, trunk_hidden, trunk_depth)
network.to(device)
print('Number of Trainable Parameters: ', network.getNumberOfParameters())

# Create the Adam optimizer and LR scheduler
lr = 1e-4
optimizer = optim.Adam(network.parameters(), lr=lr, amsgrad=True)
n_epochs = 2500

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

        # Reshape the forcing inputs
        gx = f_batch[:, :m_boundary]
        gy = f_batch[:, m_boundary:2*m_boundary]
        g_batch = pt.stack([gx, gy], dim=1)

        # Print some diagnostics
        with pt.no_grad():
            uv = network.forward(g_batch, xy_left)
            dirichlet_loss = uv[...,0].square().mean() + uv[...,1].square().mean()
        J, _ = loss_fn.grads_and_hess(network, g_batch, xy_forcing, needsHessian=False)
        u_x = J[:,:,0,0];  u_y = J[:,:,0,1]
        v_x = J[:,:,1,0];  v_y = J[:,:,1,1]
        tx = (pref * (u_x + nu * v_y) + gx).abs().mean().item()
        ty = (pref * ((1-nu)/2)*(u_y+v_x) + gy).abs().mean().item()
        print(f"\n⟨|s_xx+g_x|⟩={tx:.3e}   ⟨|s_xy+g_y|⟩={ty:.3e}")

        # Do the boundary losses once
        loss_forcing = loss_fn(network, g_batch, xy_empty, xy_forcing)
        loss_forcing.backward()

        # Do an optimizer step
        pt.nn.utils.clip_grad_norm_(network.parameters(), max_norm=clip_level)
        grad = getGradient()
        optimizer.step()

        # Some housekeeping
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tDirichlet Loss: {:.4E} \tForcing Loss: {:.4E} \tPhysics Loss: {:.4E} \tLoss Gradient: {:.4E} \tlr: {:.2E}'.format(
                        epoch, f_batch_idx * len(g_batch), len(forcing_dataset),
                        100. * f_batch_idx / len(forcing_loader), dirichlet_loss, loss_forcing.item(), 0.0, grad.item(), optimizer.param_groups[0]['lr']))
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
except KeyboardInterrupt:
    print('Aborting Training. Plotting Training Convergence.')

# Show the training results
plt.semilogy(train_counter, train_losses, color='tab:blue', label='Training Loss', alpha=0.5)
plt.semilogy(train_counter, train_grads, color='tab:orange', label='Loss Gradient', alpha=0.5)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()