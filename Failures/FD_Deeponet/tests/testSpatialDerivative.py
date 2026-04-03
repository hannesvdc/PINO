import sys
sys.path.append('../')

import torch as pt
import numpy.linalg as lg

from Navier_Stokes.BlackBoxDataset import NSDataSet

L = 95.0
M = 1000
dx = L / M
dataset = NSDataSet()
y = dataset.y_data[10,:]
fft_dydx = dataset.dydx_data[10,:]
df_dydx = (pt.roll(y, -1) - pt.roll(y, 1)) / (2.0 * dx)


print(lg.norm(fft_dydx - df_dydx) / lg.norm(fft_dydx)) # Gives a relative error of 0.11 (dx = 95/1000 = 0.095 \approx 0.1)