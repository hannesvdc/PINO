import sys
sys.path.append('../')

import torch as pt
from ConvDeepONet import DeepONet

Bb = 100
Bt = 97
x = pt.randn((Bt,1))
tau = pt.randn((Bt,1))
params = pt.randn((Bt,2))
params[:,0] = pt.exp( params[:,0] )
n_grid_points = 51
u0 = pt.randn( (Bb, n_grid_points) )