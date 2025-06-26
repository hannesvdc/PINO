import json
import torch as pt
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR

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
store_directory = config['Store Directory']
forcing_dataset = DeepONetDataset(config, device, dtype)
forcing_loader = DataLoader(forcing_dataset, batch_size=batch_size, shuffle=True)
internal_dataset = TensorDataset(forcing_dataset.xy_int)
internal_loader = DataLoader(internal_dataset, batch_size=batch_xy, shuffle=True)
xy_forcing = forcing_dataset.xy_forcing
n_chunks = len(internal_loader)

# Load model
network = ConvDeepONet(n_branch_conv=5, n_branch_channels=8, kernel_size=7, n_branch_nonlinear=3, p=100)
network.load_state_dict(pt.load(store_directory + 'physics_model.pth', map_location=device, weights_only=True))
network.to(device)

# Optimizer
optimizer = optim.Adam(network.parameters(), lr=1e-5, amsgrad=True)
n_epochs = 250
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=n_epochs,
    pct_start=0.3,
    div_factor=25,
    final_div_factor=10
)

# Physics loss
E_train = 1.0
nu = 0.3
final_w_int = 1e-6  # Use your final w_int here
loss_fn = PhysicsLoss(E_train, nu, w_int=final_w_int, w_forcing=1.0)

# Tracking
train_losses = []
train_grads = []
train_counter = []
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
            g_x = f_batch[:, :101]
            g_y = f_batch[:, 101:]
            J, _ = loss_fn.grads_and_hess(network, f_batch, xy_forcing, needsHessian=False)
            u_x = J[:,:,0,0];  u_y = J[:,:,0,1]
            v_x = J[:,:,1,0];  v_y = J[:,:,1,1]

            # raw traction residuals (no weights, no squares, *plane-stress prefactor*)
            tx = (pref * (u_x + nu * v_y) + g_x).abs().mean().item()
            ty = (pref * ((1-nu)/2)*(u_y+v_x) + g_y).abs().mean().item()
            print(f"\n⟨|s_xx+g_x|⟩={tx:.3e}   ⟨|s_xy+g_y|⟩={ty:.3e}")

        # Forcing loss
        loss_forcing = loss_fn(network, f_batch, xy_empty, xy_forcing)
        loss_forcing.backward()

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

        total = loss_forcing.item() + physics_loss
        # Some housekeeping
        print('Train Epoch: {} [{}/{} ({:.0f}%)] \tForcing Loss: {:.4E} (w = {:.1E}) \tPhysics Loss: {:.4E} (w = {:.1E}) \tLoss Gradient: {:.4E} \tlr: {:.2E}'.format(
                        epoch, f_batch_idx * len(f_batch), len(forcing_dataset),
                        100. * f_batch_idx / len(forcing_loader), loss_forcing.item(), loss_fn.w_forcing, physics_loss, loss_fn.w_int, grad, optimizer.param_groups[0]['lr']))

        train_losses.append(total)
        train_grads.append(grad.cpu().item())
        train_counter.append(epoch + f_batch_idx / len(forcing_loader))

    pt.save(network.state_dict(), store_directory + "posttrain_model.pth")
    pt.save(optimizer.state_dict(), store_directory + "posttrain_optimizer.pth")

# Main loop
for epoch in range(1, n_epochs + 1):
    posttrain(epoch)
    scheduler.step()

# Plot
plt.figure()
plt.semilogy(train_counter, train_losses, label="Loss", alpha=0.6)
plt.semilogy(train_counter, train_grads, label="Gradient Norm", alpha=0.6)
plt.xlabel("Epoch")
plt.ylabel("Loss / Gradient")
plt.legend()
plt.title("Post-Training with OneCycleLR")
plt.tight_layout()
plt.show()