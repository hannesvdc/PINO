import math
import torch as pt
import matplotlib.pyplot as plt

from TensorizedDataset import TensorizedDataset
from embedding_approach.BranchEmbeddingNetwork import BranchEmbeddingNetwork
from embedding_approach.ConvBranchEmbeddingNetwork import ConvBranchEmbeddingNetwork
from film_approach.TrunkFiLMNetwork import TrunkFilmNetwork

from fd import finiteDifferences
from evaluatePINO import evaluatePINOAt
from rbf_interpolation import tensorizedRBFInterpolator

### Load the three models

# Physics parameters
n_grid_points = 51
T_max = 10.0
tau_max = 8.0
logk_max = math.log(1e2)
l = 0.5
test_dataset = TensorizedDataset( 1, 1, n_grid_points, 
                                  l, T_max, tau_max, logk_max, 
                                  pt.float64, plot=False, 
                                  tau_sampling="initial_bias")


# Create the PINOs
n_hidden_layers = 4
z = 64
q = 32
x_grid = test_dataset.branch_dataset.x_grid

n_embedding_hidden_layers = 4
mlp_embed_model = BranchEmbeddingNetwork( n_embedding_hidden_layers, n_hidden_layers, z, q, x_grid, l, T_max, tau_max, logk_max)
mlp_embed_model.load_state_dict( pt.load('./embedding_approach/Results/model_adam.pth', weights_only=True ) )

channels = [1, 8, 16, 32, 32, 32, 32, 32, 32, 32, 32] # increase gradually
conv_embed_model = ConvBranchEmbeddingNetwork( channels, n_hidden_layers, z, q, x_grid, l, T_max, tau_max, logk_max)
conv_embed_model.load_state_dict( pt.load('./embedding_approach/Results/conv_model_adam.pth', weights_only=True) )

film_model = TrunkFilmNetwork( channels, n_hidden_layers, z, x_grid, l, T_max, tau_max, logk_max )
film_model.load_state_dict( pt.load('./film_approach/Results/conv_model_adam.pth', weights_only=True))

_, _, p, u0 = test_dataset[0]
u0 = tensorizedRBFInterpolator( film_model.cholesky_L, x_grid, l, x_grid[:,None], u0.unsqueeze(0))
u0 = u0.flatten()

dtype = pt.float32
pt.set_grad_enabled( False )
pt.set_default_dtype( dtype )
device = pt.device("mps")
mlp_embed_model.to( device=device )
conv_embed_model.to( device=device )
film_model.to( device=device )

# Calculate the finite differences ('analytic') solution
print('Running Finite Differences')
tau_min = 1e-2
T_f = 1.0

p = p.to(device=device, dtype=dtype)
u0 = u0.to(device=device, dtype=dtype)
T_0_mlp = evaluatePINOAt( mlp_embed_model, x_grid, pt.tensor([tau_min]), p, u0 )
T_0_conv = evaluatePINOAt( conv_embed_model, x_grid, pt.tensor([tau_min]), p, u0 )
T_0_film = evaluatePINOAt( film_model, x_grid, pt.tensor([tau_min]), p, u0 )

T_fd_mlp, tau_fd_mlp = finiteDifferences( x_grid, T_0_mlp, p, T_f)
T_fd_conv, tau_fd_conv = finiteDifferences( x_grid, T_0_conv, p, T_f)
T_fd_film, tau_fd_film = finiteDifferences( x_grid, T_0_film, p, T_f)

print('Evaluating PINOs')
T_sol_mlp = evaluatePINOAt( mlp_embed_model, x_grid, tau_fd_mlp, p, u0 )
T_sol_conv = evaluatePINOAt( conv_embed_model, x_grid, tau_fd_conv, p, u0 )
T_sol_film = evaluatePINOAt( film_model, x_grid, tau_fd_film, p, u0 )

# Calculate the relative errors
mlp_rel_err = pt.abs( T_sol_mlp - T_fd_mlp ) / pt.abs( T_fd_mlp )
conv_rel_err = pt.abs( T_sol_conv - T_fd_conv ) / pt.abs( T_fd_conv )
film_rel_err = pt.abs( T_sol_film - T_fd_film ) / pt.abs( T_fd_film )
print( 'Maximum Relative Erors:', mlp_rel_err.max(), conv_rel_err.max(), film_rel_err.max() )

# Make side-by-side plots of the solution and errors
print( 'Plotting' )
f, (ax1, ax2, ax3) = plt.subplots(1, 3)
X_mlp, Y_mlp = pt.meshgrid( x_grid, tau_fd_mlp, indexing="ij" )
v_min = min( float(pt.min(T_sol_mlp)), float(pt.min(T_sol_conv)), float(pt.min(T_sol_film)) )
v_max = max( float(pt.max(T_sol_mlp)), float(pt.max(T_sol_conv)), float(pt.max(T_sol_film)) )
ax1.pcolormesh( X_mlp.cpu().numpy(), Y_mlp.cpu().numpy(), T_sol_mlp.cpu().numpy(), vmin=v_min, vmax=v_max , cmap='jet', shading="nearest")
ax1.set_title("MLP Embedding")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$t$")
X_conv, Y_conv = pt.meshgrid( x_grid, tau_fd_conv, indexing="ij" )
ax2.pcolormesh( X_conv.cpu().numpy(), Y_conv.cpu().numpy(), T_sol_conv.cpu().numpy(), vmin=v_min, vmax=v_max , cmap='jet', shading="nearest")
ax2.set_title("Convolution Embedding")
ax2.set_xlabel(r"$x$")
X_film, Y_film = pt.meshgrid( x_grid, tau_fd_film, indexing="ij" )
ax3.pcolormesh( X_film.cpu().numpy(), Y_film.cpu().numpy(), T_sol_film.cpu().numpy(), vmin=v_min, vmax=v_max , cmap='jet', shading="nearest")
ax3.set_title("FiLM")
ax3.set_xlabel(r"$x$")

f, (ax1, ax2, ax3) = plt.subplots(1, 3)
v_min = min( float(pt.min(mlp_rel_err)), float(pt.min(conv_rel_err)), float(pt.min(film_rel_err)) )
v_max = max( float(pt.max(mlp_rel_err)), float(pt.max(conv_rel_err)), float(pt.max(film_rel_err)) )
ax1.pcolormesh( X_mlp.cpu().numpy(), Y_mlp.cpu().numpy(), mlp_rel_err.cpu().numpy(), vmin=v_min, vmax=v_max , cmap='jet', shading="nearest")
ax1.set_title("MLP Embedding")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$t$")
ax2.pcolormesh( X_conv.cpu().numpy(), Y_conv.cpu().numpy(), conv_rel_err.cpu().numpy(), vmin=v_min, vmax=v_max , cmap='jet', shading="nearest")
ax2.set_title("Convolution Embedding")
ax2.set_xlabel(r"$x$")
m = ax3.pcolormesh( X_film.cpu().numpy(), Y_film.cpu().numpy(), film_rel_err.cpu().numpy(), vmin=v_min, vmax=v_max , cmap='jet', shading="nearest")
ax3.set_title("FiLM")
ax3.set_xlabel(r"$x$")
plt.colorbar( m )
plt.show()