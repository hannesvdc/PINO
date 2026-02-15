import sys
sys.path.append( '../')

import torch as pt
import matplotlib.pyplot as plt

from conditionalGP import gp

n_grid_points = 51
x_grid = pt.linspace( 0.0, 1.0, n_grid_points )

l = 0.12
N_samples = 100
u0 = gp(x_grid, l, N_samples)

print( u0.shape )

plt.plot( x_grid, u0.T )
plt.xlabel(r"$x$")
plt.show()