import numpy as np
import torch as pt

import matplotlib.pyplot as plt

from TrunkFiLMNetwork import TrunkFilmNetwork
from gp import gp

# Do everything on the CPU in double precision first
dtype = pt.float64
pt.set_default_dtype( dtype )
pt.set_grad_enabled( False )

# Physics parameters
n_grid_points = 101
y_grid = pt.linspace(0.0, 1.0, n_grid_points)
nu_max = 0.45
nu = 0.3
l = 0.2
gx, gy = gp(y_grid, l, 1) #(1, N_grid_points)

# Build a grid of xy values and flatten
X, Y = pt.meshgrid( y_grid, y_grid, indexing="ij" )
x = X.flatten()
y = Y.flatten() # (Bt,1)

# Setup the network
n_hidden_layers = 4
z = 128
film_channels = [2, 8, 16, 32, 32, 32, 32, 32, 32, 32, 32] # increase gradually
model = TrunkFilmNetwork( film_channels, n_grid_points, n_hidden_layers, z, nu_max )
model.load_state_dict( pt.load('./Results/model_adam.pth', weights_only=True) )

# Evalute the network in (x,y)
u_pino, v_pino = model( x[:,None], y[:,None], pt.tensor([nu]), gx, gy)

# Plot the solutions
u_plot = pt.reshape( u_pino, (n_grid_points, n_grid_points) )
v_plot = pt.reshape( v_pino, (n_grid_points, n_grid_points) )
val_min = min( u_pino.min(), v_pino.min() )
val_max = max( u_pino.max(), v_pino.max() )
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.pcolormesh( X.numpy(), Y.numpy(), u_plot.numpy(), vmin=val_min, vmax=val_max, cmap='jet')
ax1.set_xlabel( r"$x$" )
ax1.set_ylabel( r"$y$" )
ax2.pcolormesh( X.numpy(), Y.numpy(), v_plot.numpy(), vmin=val_min, vmax=val_max, cmap='jet')
ax2.set_xlabel( r"$x$" )
plt.show()