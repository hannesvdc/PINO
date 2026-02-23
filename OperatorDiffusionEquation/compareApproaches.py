import math
import torch as pt
import matplotlib.pyplot as plt

from TensorizedDataset import TensorizedDataset
from embedding_approach.BranchEmbeddingNetwork import BranchEmbeddingNetwork
from embedding_approach.ConvBranchEmbeddingNetwork import ConvBranchEmbeddingNetwork
from film_appraoch.TrunkFiLMNetwork import TrunkFilmNetwork

from fd import finiteDifferences
from evaluatePINO import evaluatePINOAt

### Load the three models

dtype = pt.float64
pt.set_grad_enabled( True )
pt.set_default_dtype( dtype )
device = pt.device("cpu")

# Physics parameters
n_grid_points = 51
T_max = 10.0
tau_max = 8.0
logk_max = math.log(1e2)
l = 0.5
test_dataset = TensorizedDataset( 1, 1, n_grid_points, 
                                  l, T_max, tau_max, logk_max, 
                                  dtype, plot=False, 
                                  tau_sampling="initial_bias")

# Create the PINOs
n_hidden_layers = 4
z = 64
q = 32
x_grid = test_dataset.branch_dataset.x_grid

n_embedding_hidden_layers = 4
mlp_embed_model = BranchEmbeddingNetwork( n_embedding_hidden_layers, n_hidden_layers, z, q, x_grid, l, T_max, tau_max, logk_max)
mlp_embed_model.load_state_dict( pt.load('./embedding_approach/Results/model_adam.pth', map_location=device, weights_only=True ) )

channels = [1, 8, 16, 32, 32, 32, 32, 32, 32, 32] # increase gradually
conv_embed_model = ConvBranchEmbeddingNetwork( channels, n_hidden_layers, z, q, x_grid, l, T_max, tau_max, logk_max)
conv_embed_model.load_state_dict( pt.load('./embedding_approach/Results/conv_model_adam.pth', map_location=device, weights_only=True) )

film_model = TrunkFilmNetwork( channels, n_hidden_layers, z, x_grid, l, T_max, tau_max, logk_max )
film_model.load_state_dict( pt.load('./film_approach/Results/conv_model_adam.pth', map_location=device, weights_only=True))

# Calculate the finite differences ('analytic') solution

print('Running Finite Differences')
tau_min = 1e-2
T_f = 1.0
_, _, p, u0 = test_dataset[0]
T_0_mlp = evaluatePINOAt( mlp_embed_model, x_grid, pt.tensor([tau_min]), p, u0 )
T_0_conv = evaluatePINOAt( conv_embed_model, x_grid, pt.tensor([tau_min]), p, u0 )
T_0_film = evaluatePINOAt( film_model, x_grid, pt.tensor([tau_min]), p, u0 )

T_fd_mlp, tau_fd_mlp = finiteDifferences( x_grid, T_0_mlp, p, T_f)
T_fd_conv, tau_fd_conv = finiteDifferences( x_grid, T_0_conv, p, T_f)
T_fd_film, tau_fd_film = finiteDifferences( x_grid, T_0_film, p, T_f)

print('Evaluating PINOs')
T_sol_mlp, tau_grid_mlp = evaluatePINOAt( mlp_embed_model, x_grid, tau_fd_mlp, p, u0 )
T_sol_conv, tau_grid_conv = evaluatePINOAt( conv_embed_model, x_grid, tau_fd_conv, p, u0 )
T_sol_film, tau_grid_film = evaluatePINOAt( film_model, x_grid, tau_fd_film, p, u0 )

# Calculate the relative errors
mlp_rel_err = pt.abs( T_sol_mlp - T_fd_mlp ) / pt.abs( T_fd_mlp )
conv_rel_err = pt.abs( T_sol_conv - T_fd_conv ) / pt.abs( T_fd_conv )
film_rel_err = pt.abs( T_sol_film - T_fd_film ) / pt.abs( T_fd_film )

# Make side-by-side plots of the solution and errors
f, (ax1, ax2, ax3) = plt.subplots(1, 3)
X_mlp, Y_mlp = pt.meshgrid( x_grid, tau_fd_mlp, indexing="ij" )
ax1.pcolormesh( X_mlp.numpy(), Y_mlp.numpy(), T_sol_mlp.T.numpy())
ax1.set_title("MLP Embedding")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$x$")
X_conv, Y_conv = pt.meshgrid( x_grid, tau_fd_conv, indexing="ij" )
ax2.pcolormesh( X_conv.numpy(), Y_conv.numpy(), T_sol_conv.T.numpy())
ax2.set_title("Convolution Embedding")
ax2.set_xlabel(r"$x$")
X_film, Y_film = pt.meshgrid( x_grid, tau_fd_film, indexing="ij" )
ax3.pcolormesh( X_film.numpy(), Y_film.numpy(), T_sol_film.T.numpy())
ax3.set_title("FiLM")
ax3.set_xlabel(r"$x$")

f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.pcolormesh( X_mlp.numpy(), Y_mlp.numpy(), mlp_rel_err.T.numpy())
ax1.set_title("MLP Embedding")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$x$")
ax2.pcolormesh( X_conv.numpy(), Y_conv.numpy(), conv_rel_err.T.numpy())
ax2.set_title("Convolution Embedding")
ax2.set_xlabel(r"$x$")
ax3.pcolormesh( X_film.numpy(), Y_film.numpy(), film_rel_err.T.numpy())
ax3.set_title("FiLM")
ax3.set_xlabel(r"$x$")

plt.show()