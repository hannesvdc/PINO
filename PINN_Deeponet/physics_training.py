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
n_branch_nonlinear = 3
p = 100
network = ConvDeepONet(n_branch_conv, n_branch_channels, kernel_size, n_branch_nonlinear, p)
network.load_state_dict(pt.load(store_directory + 'pretrained_model.pth', weights_only=True))
network.to(device)

# Create the Adam optimizer and LR scheduler
optimizer = optim.Adam(network.parameters(), lr=1.e-3, amsgrad=True)
print('Number of Data Points per Parameter: ', n_data_points / (1.0 * network.getNumberofParameters()))

# Training Routine
E_train = 1.0 
nu = 0.3
w_int = 0.0
w_dirichlet = 0.0
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

    clip_level = 5.0 # only optimizing the forcing loss
    for f_batch_idx, f_batch in enumerate(forcing_loader):
        optimizer.zero_grad()

        with pt.no_grad():
            # one mini-batch: f_batch (B,202)  |  xy_forcing  (101,2)
            g_x = f_batch[:, :101]                       # (B,101)
            g_y = f_batch[:, 101:]

            # forward pass on the right boundary only
            J, _ = loss_fn.grads_and_hess(network, f_batch, xy_forcing, needsHessian=False)
            u_x = J[:,:,0,0];  u_y = J[:,:,0,1]
            v_x = J[:,:,1,0];  v_y = J[:,:,1,1]

            # raw traction residuals (no weights, no squares, *plane-stress prefactor*)
            pref = loss_fn.pref          # = E_train/(1-ν²)
            tx = (pref * (u_x + nu * v_y) + g_x).abs().mean().item()
            ty = (pref * ((1-nu)/2)*(u_y+v_x) + g_y).abs().mean().item()
            print(f"\n⟨|s_xx+g_x|⟩={tx:.3e}   ⟨|s_xy+g_y|⟩={ty:.3e}")
            sig_x = loss_fn.pref * (u_x + nu * v_y)      # (B,101)
            print("corr(σ_xx, g_x) =", pt.corrcoef(pt.stack((sig_x.flatten(), g_x.flatten())))[0,1].item())

        # Do the boundary losses once
        loss_dirichlet = loss_fn(network, f_batch, xy_empty, xy_left, xy_empty)
        loss_dirichlet.backward()
        loss_forcing = loss_fn(network, f_batch, xy_empty, xy_empty, xy_forcing)
        loss_forcing.backward()

        physics_loss : float = 0.0
        for xy_batch_idx, (xy_batch,) in enumerate(internal_loader):
            loss_int = loss_fn.forward(network, f_batch, xy_batch, xy_empty, xy_empty) / n_chunks
            loss_int.backward()
            physics_loss += loss_int.item()

        # Do an optimizer step
        pt.nn.utils.clip_grad_norm_(network.parameters(), max_norm=clip_level)
        grad = getGradient()
        optimizer.step()

        # Some housekeeping
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tDirichlet Loss: {:.4E} (w = {:.1E}) \tForcing Loss: {:.4E} (w = {:.1E}) \tPhysics Loss: {:.4E} (w = {:.1E}) \tLoss Gradient: {:.4E} \tlr: {:.2E}'.format(
                        epoch, f_batch_idx * len(f_batch), len(forcing_dataset),
                        100. * f_batch_idx / len(forcing_loader), loss_dirichlet.item(), w_dirichlet, loss_forcing.item(), w_forcing, physics_loss, w_int, grad, optimizer.param_groups[0]['lr']))
        train_losses.append(0.0 + loss_forcing.item() + physics_loss)
        train_grads.append(grad.cpu())
        train_counter.append((1.0*f_batch_idx)/len(forcing_loader) + epoch-1)

        # Store the temporary state
        pt.save(network.state_dict(), store_directory + 'physics_model.pth')
        pt.save(optimizer.state_dict(), store_directory + 'physics_optimizer.pth')

# Increase the physics weights first with a constant learning rate.
print('\nStarting Training Procedure...')
# ----------------------------------------------
# curriculum hyper-parameters
# ----------------------------------------------
w_forc   = 1.0                     # always 1
w_dir_0  = 1e-4                    # start value
w_pde_0  = 1e-5
w_dir_max = 1.0                    # final value you want
w_pde_max = 1e-1

warm_epochs   = 10                 # low-LR period just after the switch
ramp_step     = 50                 # how often to multiply by 10
base_lr       = 1e-3               # your usual LR
low_lr_factor = 0.2                # LR multiplier during warm-up
max_epochs    = 300                # run as long as you like
clip_val      = 5.0                # keep your current clip
def ramp_weight(init, max_, epoch, step):
    """log-ramp: multiply by 10 every <step> epochs until max is reached"""
    k = epoch // step           # how many steps have passed
    w = init * (10.0 ** k)
    return min(w, max_)
def set_lr(optim, factor):
    for g in optim.param_groups:
        g['lr'] = base_lr * factor
print("\nStarting curriculum-phase training...")
try:
    for epoch in range(1, max_epochs + 1):

        # --- LR handling ---------------------------------------------
        if epoch <= warm_epochs:
            set_lr(optimizer, low_lr_factor)
        else:
            set_lr(optimizer, 1.0)

        # --- task weights --------------------------------------------
        w_dirichlet = ramp_weight(w_dir_0, w_dir_max, epoch, ramp_step)
        w_int = ramp_weight(w_pde_0, w_pde_max, epoch, ramp_step)

        loss_fn.setWeights(w_int = w_int,
                           w_dirichlet = w_dirichlet,
                           w_forcing = w_forcing)
        train(epoch)
except KeyboardInterrupt:
    print('Moving on to post training.')

# Post-training: slowly decrease the learing rate to obtain the optimal fit.
step = 10
scheduler = sch.StepLR(optimizer, step_size=step, gamma=0.5)
n_epochs = 10 * step # Do a max of 100 epochs post-training. Should be enough convergence.
try:
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        scheduler.step()
except KeyboardInterrupt:
    print('Stopping Post-Trainig. Plotting Training Convergence.')

# Show the training results
plt.semilogy(train_counter, train_losses, color='tab:blue', label='Training Loss', alpha=0.5)
plt.semilogy(train_counter, train_grads, color='tab:orange', label='Loss Gradient', alpha=0.5)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()