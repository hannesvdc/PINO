import torch as pt
import matplotlib.pyplot as plt

from model import PINO
from dataset import NewtonDataset

import argparse
def parseArguments( ):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--tau', dest='tau', default="unbiased")
    arg_parser.add_argument('--time_factor', dest='time_factor', default="linear")
    return arg_parser.parse_args( )
args = parseArguments()

# Test on the CPU
device = pt.device("cpu")
dtype = pt.float64
pt.set_default_dtype( dtype )
pt.set_grad_enabled(True)

# Create the test
T_max = 10.0
tau_max = 8.0
N_test = 10000
test_dataset = NewtonDataset( N_test, T_max, tau_max, device, dtype, test=True )

# Load the model
z = 32
n_hidden_layers = 2
model = PINO( n_hidden_layers, z, T_max, tau_max, test_dataset.logk_min, test_dataset.logk_max, time_factor=args.time_factor ).to( device=device )
store_directory = './Results/'
# model.load_state_dict( pt.load( store_directory + args.time_factor + '_' + args.tau + '_model_adam.pth', weights_only=True, map_location=device ) )
model.load_state_dict( pt.load( store_directory + 'model_lbfgs.pth', weights_only=True, map_location=device ) )

# Evaluate the model for every parameter combination in the dataset. I know for-loops are inefficient but IDC
N_t_grid = 100
tau_grid = pt.linspace(0.0, tau_max, N_t_grid)
T_evaluations = pt.zeros( (N_t_grid, N_test))
def evaluatePINO( _t_grid, _p, verbose=False):
    B = _t_grid.shape[0]
    if len(_t_grid.shape) == 1:
        _t_grid = _t_grid.unsqueeze(1)
    _p_batch = pt.unsqueeze(_p, 0).expand( [B,3] )
    if verbose:
        print(_p_batch)
    _output = model( _t_grid, _p_batch )
    if verbose:
        print(_output)
    return _output
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
plt.plot( tau_grid.numpy(), pt.mean(T_evaluations, dim=1).detach().numpy(), color='tab:blue', label='PINO Master Curve')
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

# Compare one profile to the analytic solution
idx = 25
_, p = test_dataset[idx]
T0 = p[0]
k = p[1]
T_inf = p[2]
t_grid = tau_grid / k
T_t = evaluatePINO( t_grid, p )
plt.figure()
plt.plot(t_grid.detach().numpy(), T_t.detach().numpy(), label="PINO Evolution")
plt.plot(t_grid.detach().numpy(), T_inf + (T0 - T_inf)*pt.exp(-k*t_grid.detach().numpy()), label="Analytic Solution")
plt.xlabel(r"$t$")
plt.title(rf"$T_0 = {round(float(T0),1)}, \ k = {round(float(k),1)}, \ T_s = {round(float(T_inf),1)}$")
plt.legend()
plt.show()