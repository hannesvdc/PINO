import json
import torch as pt
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from model import ConvDeepONet, PhysicsLoss
from dataset import DeepONetDataset

# PyTorch setup
pt.set_grad_enabled(True)
pt.set_default_dtype(pt.float32)
dtype = pt.float32
if pt.backends.mps.is_available():
    device = pt.device("mps")
    batch_size = 64
    batch_xy = 512
elif pt.cuda.device_count() > 0:
    device = pt.device("cuda:0")
    batch_size = 64
    batch_xy = 4096
else:
    device = pt.device("cpu")
    batch_size = 64
    batch_xy = 512

# Load config and data
config = json.load(open('DataConfig.json'))
data_directory = config['Data Directory']
store_directory = config['Store Directory']
forcing_dataset = DeepONetDataset(config, device, dtype)
forcing_loader = DataLoader(forcing_dataset, batch_size=batch_size, shuffle=True)
internal_dataset = TensorDataset(forcing_dataset.xy_int)
internal_loader = DataLoader(internal_dataset, batch_size=batch_xy, shuffle=True)
xy_forcing = forcing_dataset.xy_forcing
n_chunks = len(internal_loader)
grid_size = forcing_dataset.grid_points

# Load model
network = ConvDeepONet(n_branch_conv=5, n_branch_channels=8, kernel_size=7, n_branch_nonlinear=3, n_trunk_nonlinear=5, p=100)
network.load_state_dict(pt.load(store_directory + 'pretrained_model.pth', map_location=device, weights_only=True))
network.to(device)
for p in network.branch_net.parameters():
    p.requires_grad = False
last_layer = network.branch_net.layers[-1]
if isinstance(last_layer, pt.nn.Linear):
    for p in last_layer.parameters():
        p.requires_grad = True

# Optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=1e-4, amsgrad=True)

# Physics loss
E_train = 1.0
nu = 0.3
w_int_init = 1e-8 # Start at the physics_training value
w_int_max  = 1.0
loss_fn = PhysicsLoss(E_train, nu, w_int=w_int_init, w_forcing=1.0)

# Tracking
train_losses = []
train_grads = []
train_counter = []
physics_losses = []
physics_weights = []
forcing_losses = []
disp_x = []
disp_y = []
xy_empty = pt.empty((0, 2), device=device, dtype=pt.float32)

def getGradient():
    grads = [p.grad.view(-1) for p in network.parameters() if p.grad is not None]
    return pt.norm(pt.cat(grads))

pref = loss_fn.pref
def posttrain(epoch):
    network.train()
    clip_level = 5.0

    for f_batch_idx, f_batch in enumerate(forcing_loader):
        optimizer.zero_grad()

        with pt.no_grad():
            g_x = f_batch[:, :grid_size]
            g_y = f_batch[:, grid_size:]
            J, _ = loss_fn.grads_and_hess(network, f_batch, xy_forcing, needsHessian=False)
            u_x = J[:,:,0,0];  u_y = J[:,:,0,1]
            v_x = J[:,:,1,0];  v_y = J[:,:,1,1]

            # raw traction residuals (no weights, no squares, *plane-stress prefactor*)
            tx = (pref * (u_x + nu * v_y) + g_x).abs().mean().item()
            ty = (pref * ((1-nu)/2)*(u_y+v_x) + g_y).abs().mean().item()
            disp_x.append(tx)
            disp_y.append(ty)
            print(f"\n⟨|s_xx+g_x|⟩={tx:.3e}   ⟨|s_xy+g_y|⟩={ty:.3e}")

        # Forcing loss
        loss_forcing = loss_fn(network, f_batch, xy_empty, xy_forcing)
        loss_forcing.backward()
        forcing_losses.append(loss_forcing.item())

        # Physics loss
        physics_loss = 0.0
        for (xy_batch,) in internal_loader:
            loss_int = loss_fn(network, f_batch, xy_batch, xy_empty) / n_chunks
            loss_int.backward()
            physics_loss += loss_int.item()

        # Step optimizer and scheduler
        pt.nn.utils.clip_grad_norm_(network.parameters(), max_norm=clip_level)
        grad = getGradient()
        optimizer.step()

        # Some housekeeping
        total = loss_forcing.item() + physics_loss
        physics_losses.append(physics_loss)
        physics_weights.append(loss_fn.w_int)
        print('Train Epoch: {} [{}/{} ({:.0f}%)] \tForcing Loss: {:.4E} (w = {:.1E}) \tPhysics Loss: {:.4E} (w = {:.1E}) \tLoss Gradient: {:.4E} \tlr: {:.2E}'.format(
                        epoch, f_batch_idx * len(f_batch), len(forcing_dataset),
                        100. * f_batch_idx / len(forcing_loader), loss_forcing.item(), loss_fn.w_forcing, physics_loss, loss_fn.w_int, grad, optimizer.param_groups[0]['lr']))

        train_losses.append(total)
        train_grads.append(grad.cpu().item())
        train_counter.append(epoch + f_batch_idx / len(forcing_loader))

    pt.save(network.state_dict(), data_directory + f"pino_epoch_results/posttrain_model_epoch={epoch}.pth")
    pt.save(optimizer.state_dict(), data_directory + f"pino_epoch_results/posttrain_optimizer_epoch={epoch}.pth")

# Main loop
n_epochs = 500
for epoch in range(1, n_epochs + 1):
    # Exponentially ramp w_int from init to max
    w_int = w_int_init * (w_int_max / w_int_init) ** (epoch / n_epochs)
    loss_fn.setWeights(w_int=w_int, w_forcing=1.0)
    print(f"\n>>> Epoch {epoch:3d} | w_int = {w_int:.2e}")
    posttrain(epoch)

# Save all loss and mean-displacement arrays
np.save(store_directory + "physics_losses.npy", np.array(physics_losses))
np.save(store_directory + "forcing_losses.npy", np.array(forcing_losses))
np.save(store_directory + "disp_x.npy", np.array(disp_x))
np.save(store_directory + "disp_y.npy", np.array(disp_y))
np.save(store_directory + "epochs.npy", np.array(train_counter))
np.save(store_directory + "physics_weights.npy", np.array(physics_weights))

# Plot
plt.figure()
plt.semilogy(train_counter, train_losses, label="Loss", alpha=0.6)
plt.semilogy(train_counter, train_grads, label="Gradient Norm", alpha=0.6)
plt.xlabel("Epoch")
plt.ylabel("Loss / Gradient")
plt.legend()
plt.title("Determining the Optimal Physics Weights")
plt.tight_layout()
plt.show()