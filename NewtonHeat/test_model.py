import torch as pt
import matplotlib.pyplot as plt

from model import PINO, AdvanedPhysicsPINO
from dataset import NewtonDataset

import argparse
def parseArguments( ):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_type', dest='model_type')
    return arg_parser.parse_args( )
args = parseArguments()

# Test on the CPU
device = pt.device("cpu")
dtype = pt.float64
pt.set_default_dtype( dtype )
pt.set_grad_enabled(True)

# Create the test
T_max = 10.0
tau_max = 10.0
N_test = 100
test_dataset = NewtonDataset( N_test, T_max, tau_max, device, dtype, test=True )

# Load the model
z = 32
if args.model_type == 'advanced':
    n_hidden_layers = 1
    model = AdvanedPhysicsPINO( n_hidden_layers, z, T_max, tau_max, test_dataset.logk_min, test_dataset.logk_max ).to( device=device )
    step_size = 100
elif args.model_type == 'initial':
    n_hidden_layers = 2
    model = PINO( n_hidden_layers, z, T_max, tau_max, test_dataset.logk_min, test_dataset.logk_max ).to( device=device )
    step_size = 1000
elif args.model_type == "simple":
    n_hidden_layers = 1
    model = PINO( n_hidden_layers, z, T_max, tau_max, test_dataset.logk_min, test_dataset.logk_max ).to( device=device )
    step_size = 1000
else:
    print('This model type is not supported.')
    exit()
store_directory = './Results/'
model.load_state_dict( pt.load( store_directory + args.model_type + '_model_lbfgs.pth', weights_only=True, map_location=device ) )

# Evaluate the model for every parameter combination in the dataset. I know for-loops are inefficient but IDC
N_t_grid = 100
tau_grid = pt.linspace(0.0, tau_max, N_t_grid)
T_evaluations = pt.zeros( (N_t_grid, N_test))
def evaluatePINO(t_grid, p):
    B = t_grid.shape[0]
    p_batch = pt.unsqueeze(p, 0).expand( [B,3] )
    return model( t_grid, p_batch )
for n in range( N_test ):
    _, p = test_dataset[n]
    T0 = p[0]
    k = p[1]
    T_inf = p[2]
    t_grid = tau_grid / k
    T_t = evaluatePINO( t_grid[:,None], p )
    T_evaluations[:,n] = (T_t[:,0] - T_inf) / (T0 - T_inf)

# Plot the master curve
plt.figure()
plt.plot( tau_grid.numpy(), pt.mean(T_evaluations, dim=1).detach().numpy(), color='tab:blue', label='Averaged PINO Master Curve')
plt.plot( tau_grid.numpy(), pt.exp(-tau_grid).detach().numpy(), color='tab:orange', label=r"$\exp(-kt)$")
plt.xlabel(r"$k t$")
plt.ylabel(r"$\frac{T(kt)-T_{\infty}}{T_0 - T_{\infty}}$", rotation=0)
plt.legend()

# Plot realizations of the master curve
plt.figure()
for n in range( 10 ):
    _, p = test_dataset[n]
    T0 = p[0]
    T_inf = p[2]
    T_diff = float(T0 - T_inf)
    plt.plot( tau_grid.numpy(), T_evaluations[:,n].detach().numpy(), label=r"$T_0-T_{\infty} =$"  + str(round(T_diff,2)))
plt.plot( tau_grid.numpy(), pt.exp(-tau_grid).detach().numpy(), color='tab:orange', label=r"$\exp(-kt)$")
plt.xlabel(r"$k t$")
plt.ylabel(r"$\frac{T(kt)-T_{\infty}}{T_0 - T_{\infty}}$", rotation=0)
plt.legend()
plt.show()