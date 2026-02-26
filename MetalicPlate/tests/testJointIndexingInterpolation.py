import sys
sys.path.append('../')
sys.path.append('../../')

import torch as pt
import matplotlib.pyplot as plt

from gp import gp
from rbf_interpolation import buildCholeskyMatrix, jointIndexingRBFInterpolator

n_grid_points = 101
y_grid = pt.linspace( 0.0, 1.0, n_grid_points )
l = 0.12
gx, gy = gp(y_grid, l, 1)

# Build the Cholesky factorization
L = buildCholeskyMatrix( y_grid, l )

device = pt.device("mps")
dtype = pt.float32
pt.set_default_device(device)
pt.set_default_dtype(dtype)
L = L.to(device=device, dtype=dtype)
gx = gx.to(device=device, dtype=dtype)
gy = gy.to(device=device, dtype=dtype)
y_grid = y_grid.to(device=device, dtype=dtype)

# Do joint indexig interpolation. This means replicating u0 for all `x`.
n_plot_points = 10001
y = pt.linspace( 0.0, 1.0, n_plot_points, requires_grad=True )
gx_repmat = gx.repeat( n_plot_points, 1)
gy_repmat = gy.repeat( n_plot_points, 1)
gx_interpolated, gy_interpolated = jointIndexingRBFInterpolator( L, y_grid, l, y, gx_repmat, gy_repmat )

# Calculate the first and second derivative
gx_int_der = gx_interpolated[:,0]
gy_int_der = gy_interpolated[:,0]
dgx_dy = pt.autograd.grad(outputs=gx_int_der, inputs=y, create_graph=True, grad_outputs=pt.ones_like(y))[0]
dgy_dy = pt.autograd.grad(outputs=gy_int_der, inputs=y, create_graph=True, grad_outputs=pt.ones_like(y))[0]

# Plot for a visual comparison
plt.plot( y_grid.flatten().cpu().numpy(), gx.flatten().cpu().numpy(), label="Original GP for g_x")
plt.plot( y.flatten().detach().cpu().numpy(), gx_interpolated.flatten().detach().cpu().numpy(), linestyle='--', label="RBF Interpolator")
plt.plot( y.flatten().detach().cpu().numpy(), dgx_dy.flatten().detach().cpu().numpy(), linestyle='--', label="First Derivative")
plt.xlabel( r"$y$" )
plt.legend()
plt.show()