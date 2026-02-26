import sys
sys.path.append( '../')

import torch as pt
import matplotlib.pyplot as plt

from gp import gp
from rbf_interpolation import buildCholeskyMatrix, jointIndexingRBFInterpolator

n_grid_points = 101
y_grid = pt.linspace( 0.0, 1.0, n_grid_points )

# Sample the GP
l = 0.12
gx, gy = gp(y_grid, l, 1)

# Interpolate
n_int = 201
y_int = pt.linspace( 0.0, 1.0, n_int)
gx_repmat = gx.repeat( n_int, 1 )
gy_repmat = gy.repeat( n_int, 1 )
L = buildCholeskyMatrix(y_grid, l)
gx_int, gy_int = jointIndexingRBFInterpolator( L, y_grid, l, y_int, gx_repmat, gy_repmat )

# Original Dataset
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot( y_grid, gx.flatten(), label='GP Sample' )
ax1.plot( y_int, gx_int, '--', label='Interpolation' )
ax1.set_title(r"$g_x(y)$")
ax2.plot( y_grid, gy.flatten(), label='GP Sample' )
ax2.plot( y_int, gy_int, '--', label='Interpolation' )
ax2.set_title(r"$g_y(y)$")
ax1.set_xlabel(r"$y$")
ax2.set_xlabel(r"$y$")
plt.legend()
plt.show()