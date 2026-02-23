import sys
sys.path.append('../')

import math
import torch as pt

from TensorizedDataset import TensorizedDataset
from ConvBranchEmbeddingNetwork import ConvBranchEmbeddingNetwork
from BranchEmbeddingNetwork import BranchEmbeddingNetwork
from test_pinn import test_pinn

import argparse
def parseArguments( ):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?', dest='model', default='mlp')
    return parser.parse_args()
args = parseArguments( )

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

N_test_branch = 100
N_test_trunk = 2000
test_dataset = TensorizedDataset( N_test_branch, N_test_trunk, n_grid_points, 
                                       l, T_max, tau_max, logk_max, dtype, plot=False, tau_sampling="initial_bias")

# Create the PINO
n_hidden_layers = 4
z = 64
q = 32
x_grid = test_dataset.branch_dataset.x_grid
if args.model == 'mlp':
    n_embedding_hidden_layers = 4
    model = BranchEmbeddingNetwork( n_embedding_hidden_layers, n_hidden_layers, z, q, x_grid, l, T_max, tau_max, logk_max)
    model.load_state_dict( pt.load('./Results/model_adam.pth', map_location=device, weights_only=True ) )
elif args.model == 'conv':
    channels = [1, 8, 16, 32, 32, 32, 32, 32, 32, 32] # increase gradually
    model = ConvBranchEmbeddingNetwork( channels, n_hidden_layers, z, q, x_grid, l, T_max, tau_max, logk_max)
    model.load_state_dict( pt.load('./Results/conv_model_adam.pth', map_location=device, weights_only=True) )
else:
    print('This model is not supported. Returning.')
    exit()

# General testing routine.
test_pinn( model, test_dataset )