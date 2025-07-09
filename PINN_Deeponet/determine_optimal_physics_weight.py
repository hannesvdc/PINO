import json
import numpy as np
import matplotlib.pyplot as plt

config_file = 'DataConfig.json'
config = json.load(open(config_file))
store_directory = config["Store Directory"]

# Save all loss and mean-displacement arrays
physics_losses = np.load(store_directory + "physics_losses.npy")
forcing_losses = np.load(store_directory + "forcing_losses.npy")
disp_x = np.load(store_directory + "disp_x.npy")
disp_y = np.load(store_directory + "disp_y.npy")
train_counter = np.load(store_directory + "epochs.npy")
physics_weights = np.load(store_directory + "physics_weights.npy")

# Compute the weighted/unweighted losses
n_epochs = len(physics_losses)
epochs = np.linspace(1, n_epochs, n_epochs)
unweighted_physics_losses = physics_losses / physics_weights
total_losses = forcing_losses + physics_losses

# Determine the optimal w_int
optimal_epoch = np.argmin(unweighted_physics_losses)
optimal_physics_weight = physics_weights[optimal_epoch]
print('Optimal Physics Weight: ', optimal_physics_weight, ' at epoch', optimal_epoch)

# Plot all arrays
plt.semilogy(epochs, unweighted_physics_losses, label='Unweighted Physics Losses')
plt.semilogy(epochs, physics_losses, label='Weighted Physics Losses')
plt.semilogy(epochs, forcing_losses, label='Forcing Losses')
plt.semilogy(epochs, forcing_losses, label='Total Weighted Losses')
plt.semilogy(epochs, disp_x, label=r'Averaged $x$-Displacements')
plt.semilogy(epochs, disp_y, label=r'Averaged $y$-Displacements')
plt.xlabel('Epoch')
plt.ylabel('Log Scale')
plt.legend()
plt.show()
