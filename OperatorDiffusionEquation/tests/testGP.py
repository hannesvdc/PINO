import sys
sys.path.append( '../')

import torch as pt
import matplotlib.pyplot as plt

from conditionalGP import gp
from interpolate import evaluateInterpolatingSpline

n_grid_points = 51
x_grid = pt.linspace( 0.0, 1.0, n_grid_points )

# Sample the GP
l = 0.12
N_samples = 10
u0 = gp(x_grid, l, N_samples)

# Interpolate
n_int = 101
x_int = pt.linspace( 0.0, 1.0, n_int)
u0_int = evaluateInterpolatingSpline( x_int, x_grid, u0 )

# Original Dataset
plt.plot( x_grid, u0.T )
plt.xlabel(r"$x$")

# Interpolated Dataset
plt.figure()
plt.plot( x_int, u0_int.T )
plt.xlabel(r"$x$")
plt.ylabel(r"$u_0$")
plt.title(r"Conditional Gaussian Process for $u_0$")
plt.show()