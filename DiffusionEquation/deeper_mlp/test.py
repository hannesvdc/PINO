import sys
sys.path.append('../')

import torch as pt

from dataset import PINNDataset
from FixedInitialPINN import FixedInitialPINN
from test_pinn import test_pinn
from generateInitialCondition import build_u0_evaluator

dtype = pt.float64
pt.set_grad_enabled( True )
pt.set_default_dtype( dtype )
device = pt.device("cpu")

# Create the training and validation datasets
T_max = 10.0
tau_max = 8.0 # train to exp( -tau_max )
N_test = 100
N_validation = 5_000
test_dataset = PINNDataset( N_test, T_max, tau_max, dtype, test=True)

# Load the initial condition
l = 0.12
u0_fcn, ic = build_u0_evaluator( l, pt.device("cpu"), pt.float64 )

# Create the PINO
z = 64
n_hidden_layers = 4
model = FixedInitialPINN( n_hidden_layers, z, T_max, tau_max, test_dataset.logk_max, u0_fcn, ic_time_factor=True)
model.load_state_dict( pt.load('./Results/model_lbfgs.pth', map_location=device, weights_only=True) )

test_pinn( model, test_dataset )