import sys
sys.path.append('../')

import autograd.numpy as np
import autograd.numpy.linalg as lg

import GaussianProces as GP

def test_mean_and_covariance():
    N = 101
    M = 100000
    y_points = np.linspace(0.0, 1.0, N)
    l = 0.12
    K_kernel = GP.precompute_covariance(y_points, l)

    samples = np.zeros((N, M))
    for k in range(M):
        print(k)
        f = GP.gp(y_points, K_kernel)
        samples[:,k] = f[:,0] # first column

    # Compute mean and covariance
    m = np.mean(samples, axis=1)
    c = np.cov(samples)
    print(m)

    # Print
    print('L_2 error on mean:',  lg.norm(m)/M)
    print('L_2 error on covariance:', lg.norm(c - K_kernel)/M**2)

test_mean_and_covariance()
