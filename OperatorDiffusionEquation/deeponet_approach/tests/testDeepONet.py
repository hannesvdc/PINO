import sys
sys.path.append('../')

import torch as pt
from ConvDeepONet import DeepONet

n_grid_points = 51
x_grid = pt.linspace( 0.0, 1.0, n_grid_points )
l = 0.2

Bb = 100
Bt = 97
x = pt.randn((Bt,1))
tau = pt.randn((Bt,1))
params = pt.randn((Bt,2))
params[:,0] = pt.exp( params[:,0] )
u0 = pt.randn( (Bb, n_grid_points) )

# Build the DeepONet
branch_architecture = {"n_grid_points" : n_grid_points,
                       "n_hidden_layers" : 3,
                       "branch_kernel_size" : 5 }
trunk_architecture = { "input_dim" : 3,
                       "n_hidden_layers" : 4,
                       "z" : 64 }

# Physics parameters
T_max = 10.0
tau_max = 8.0
logk_max = 2.0
q = 25

# Propagate the data through the network
deeponet = DeepONet( branch_architecture, trunk_architecture, x_grid, l, T_max, tau_max, logk_max, q)
output = deeponet( x, tau, params, u0 )
print( output.shape )