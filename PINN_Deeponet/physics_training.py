import json
import torch as pt
import torch.optim as optim
import torch.optim.lr_scheduler as sch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from model import ConvDeepONet, PhysicsLoss
from dataset import DeepONetDataset

# Just some sanity pytorch settings
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
    print('Using CPU because no GPU is available.')
    device = pt.device("cpu")
    batch_size = 64
    batch_xy = 512

# Load the data configuration
config_file = 'DataConfig.json'
config = json.load(open(config_file))
store_directory = config['Store Directory']

# Load the data in memory
forcing_dataset = DeepONetDataset(config, device, dtype)
forcing_loader = DataLoader(forcing_dataset, batch_size=batch_size, shuffle=True)
internal_dataset = TensorDataset(forcing_dataset.xy_int)
internal_loader = DataLoader(internal_dataset, batch_size=batch_xy, shuffle=True)
n_chunks = len(internal_loader)
n_data_points = len(forcing_dataset) * len(internal_dataset)
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
optimizer = optim.Adam(network.parameters(), lr=1.3e-5, amsgrad=True)
print('Number of Data Points per Parameter: ', n_data_points / (1.0 * network.getNumberofParameters()))

# Training Routine
E_train = 1.0 
nu = 0.3
w_forcing = 1.0
loss_fn = PhysicsLoss(E_train, nu, w_int=0.0, w_forcing=w_forcing)
train_losses = []
train_grads = []
train_counter = []
def getGradient():
    grads = [p.grad.view(-1) for p in network.parameters() if p.grad is not None]
    return pt.norm(pt.cat(grads))

pref = loss_fn.pref
xy_empty = pt.empty((0,2), device=device, dtype=pt.float32)
def train(epoch):
    network.train()

    forcing_sum = 0.0
    physics_sum = 0.0

    clip_level = 5.0 # only optimizing the forcing loss
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

        # Do the boundary losses once
        loss_forcing = loss_fn(network, f_batch, xy_empty, xy_forcing)
        loss_forcing.backward()
        forcing_sum += loss_forcing.item()

        physics_loss : float = 0.0
        for xy_batch_idx, (xy_batch,) in enumerate(internal_loader):
            loss_int = loss_fn.forward(network, f_batch, xy_batch, xy_empty) / n_chunks
            loss_int.backward()
            physics_loss += loss_int.item()
        physics_sum += physics_loss

        # Do an optimizer step
        pt.nn.utils.clip_grad_norm_(network.parameters(), max_norm=clip_level)
        grad = getGradient()
        optimizer.step()

        # Some housekeeping
        print('Train Epoch: {} [{}/{} ({:.0f}%)] \tForcing Loss: {:.4E} (w = {:.1E}) \tPhysics Loss: {:.4E} (w = {:.1E}) \tLoss Gradient: {:.4E} \tlr: {:.2E}'.format(
                        epoch, f_batch_idx * len(f_batch), len(forcing_dataset),
                        100. * f_batch_idx / len(forcing_loader), loss_forcing.item(), w_forcing, physics_loss, loss_fn.w_int, grad, optimizer.param_groups[0]['lr']))
        train_losses.append(0.0 + loss_forcing.item() + physics_loss)
        train_grads.append(grad.cpu())
        train_counter.append((1.0*f_batch_idx)/len(forcing_loader) + epoch-1)

    # Store the temporary state
    pt.save(network.state_dict(), store_directory + 'physics_model.pth')
    pt.save(optimizer.state_dict(), store_directory + 'physics_optimizer.pth')

    # Return the averaged forcing and physics losses for weight calculations
    n_batches = len(forcing_loader)
    return forcing_sum / n_batches, physics_sum / n_batches

# Increase the physics weights first with a constant learning rate.
print('\nStarting Training Procedure...')
w_int_0  = 1e-7 # Very small
w_int_max = 1e-1
current_w_int = w_int_0
warm_epochs   = 5        # low-LR period just after the switch
ramp_step     = 10       # how often to multiply by 10
base_lr       = 1e-3     # your usual LR
low_lr_factor = 0.1      # LR multiplier during warm-up
max_epochs    = 100      # run as long as you like
def set_lr(optim, factor):
    for g in optim.param_groups:
        g['lr'] = base_lr * factor
print("\nStarting curriculum-phase training...")

epoch  = 0
ramping_done = False        # turns true once w_int hits target_band
warm_epochs_after_ramp = 3  # low-LR epochs after each jump
warm_left = warm_epochs     # remaining epochs in low-LR mode
target_band = (0.3, 3.0)    # balance band for weighted losses
try:
    while not ramping_done:
        epoch += 1

        # ---- learning-rate handling ---------------------------------
        if warm_left > 0:
            set_lr(optimizer, low_lr_factor)
            warm_left -= 1
        else:
            set_lr(optimizer, 1.0)

        # Do one epoch with these weights
        loss_fn.setWeights(w_int = current_w_int, w_forcing = w_forcing)
        forcing_term, physics_term = train(epoch)
        weighted_ratio = physics_term / forcing_term
        print(f"epoch {epoch:3d} | ratio={weighted_ratio:.2f} "
          f"| w_int={current_w_int:.1e} | LR={optimizer.param_groups[0]['lr']:.1e}")
        
        # 4. decide whether to ramp
        if warm_left == 0 and current_w_int < w_int_max and weighted_ratio < target_band[0]:
            # ×10 jump
            current_w_int = min(current_w_int*10.0, w_int_max)
            warm_left = warm_epochs_after_ramp             # restart cool-down
            print(f" ↑ increased w_int to {current_w_int:.1e}, "
                f"low-LR for {warm_left} epochs")
        elif target_band[0] <= weighted_ratio <= target_band[1]:
            print(" ✓ losses balanced – stop ramping")
            ramping_done = True
except KeyboardInterrupt:
    print('Moving on to post training.')

# Post-training: slowly decrease the learing rate to obtain the optimal fit.
step = 10
scheduler = sch.StepLR(optimizer, step_size=step, gamma=0.5)
last_epoch = epoch
n_epochs = 10 * step
try:
    for epoch in range(1, n_epochs + 1):
        train(last_epoch + epoch)
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