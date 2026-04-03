import sys
import time
sys.path.append('../')

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

import ElasticPDE as pde

def testPDE():
    N = 100           # N is the number of intervals
    E = 410.0 * 1.e3 # This is really, really large?
    mu = 0.3

    rng = rd.RandomState(seed=int(time.time()))
    f = rng.normal(0.0, 1.0, size=(N+1,2))
    
    (u, v) = pde.solveElasticPDE(E, mu, f, N)
    min_u = np.min(u)
    max_u = np.max(u)
    min_v = np.min(v)
    max_v = np.max(v)

    # Plotting a heatmap of the result
    print('\nPlotting...')
    x = np.linspace(0.0, 1.0, N+1)
    y = np.linspace(0.0, 1.0, N+1)
    X, Y = np.meshgrid(x, y)
    plt.pcolormesh(X, Y, u, shading='auto', vmin=min_u, vmax=max_u, cmap='jet')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'$u(x,y)$')
    plt.colorbar()

    plt.figure()
    plt.pcolormesh(X, Y, v, shading='auto', vmin=min_v, vmax=max_v, cmap='jet')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'$v(x,y)$')
    plt.colorbar()
    plt.show()

testPDE()