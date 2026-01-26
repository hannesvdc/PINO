import torch as pt
import matplotlib.pyplot as plt

from model import PINO
from dataset import NewtonDataset

# Test on the CPU
device = pt.device("cpu")
dtype = pt.float64
pt.set_default_dtype( dtype )
pt.set_grad_enabled(True)

# Create the test
T_max = 10.0
tau_max = 4.0
N_test = 100
train_dataset = NewtonDataset( N_test, T_max, tau_max, device, dtype )

# Load the model
n_hidden_layers = 2
z = 32
model = PINO( n_hidden_layers, z, T_max )
model.load_state_dict( pt.load( './Results/model_lbfgs.pth', weights_only=True, map_location=device )  )

# Compare PINO with the exact solution of the ODE
idx = 25
_, p = train_dataset[idx]
T0 = p[0]
k = p[1]
T_inf = p[2]
t_grid = pt.linspace(0.0, tau_max, 100) / k
def evaluatePINO(t_grid, p):
    B = t_grid.shape[0]
    p_batch = pt.unsqueeze(p, 0).expand( [B,3] )
    return model( t_grid, p_batch )
T_t = evaluatePINO( t_grid[:,None], p )

# Compare to the analytical solution
T_t_analytic = T_inf + (T0 - T_inf) * pt.exp( -k * t_grid )

# Plot both
plt.plot( t_grid.numpy(), T_t.detach().numpy(), label="PINO" )
plt.plot( t_grid.numpy(), T_t_analytic.detach().numpy(), label="Analytic Solution")
plt.xlabel(r"$t$")
plt.ylabel(r"$T(t)$")
plt.legend()
plt.show()