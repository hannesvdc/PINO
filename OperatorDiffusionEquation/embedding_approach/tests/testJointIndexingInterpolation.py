import sys
sys.path.append('../')
sys.path.append('../../')

import torch as pt
import matplotlib.pyplot as plt

from conditionalGP import gp
from rbf_interpolation import buildCholeskyMatrix, jointIndexingRBFInterpolator

N_grid_points = 51
l = 0.2 # GP correlation length. Make large enough to avoid blow-up in the second derivative.
x_grid = pt.linspace(0.0, 1.0, N_grid_points)
u0 = gp( x_grid, l, 1 ) / 2.0 # scale
u0.requires_grad_( False )

# Build the Cholesky factorization
L = buildCholeskyMatrix( x_grid, l )

# Do joint indexig interpolation. This means replicating u0 for all `x`.
n_plot_points = 10001
x = pt.linspace( 0.0, 1.0, n_plot_points, requires_grad=True )
u0_repmat = u0.repeat( n_plot_points, 1)
u0_interpolated = jointIndexingRBFInterpolator( L, x_grid, l, x, u0_repmat )

# Calculate the first and second derivative
u0_int_der = u0_interpolated[:,0]
du0_dx = pt.autograd.grad(outputs=u0_int_der, inputs=x, create_graph=True, grad_outputs=pt.ones_like(x))[0]
du0_dxx = pt.autograd.grad(outputs=du0_dx, inputs=x, create_graph=True, grad_outputs=pt.ones_like(du0_dx))[0]

# Plot for a visual comparison
plt.plot( x_grid.flatten().numpy(), u0.flatten().numpy(), label="Original cGP")
plt.plot( x.flatten().detach().numpy(), u0_interpolated.flatten().detach().numpy(), linestyle='--', label="RBF Interpolator")
plt.plot( x.flatten().detach().numpy(), du0_dx.flatten().detach().numpy(), linestyle='--', label="First Derivative")
plt.plot( x.flatten().detach().numpy(), du0_dxx.flatten().detach().numpy(), linestyle='--', label="Second Derivative")
plt.xlabel( r"$x$" )
plt.legend()
plt.show()