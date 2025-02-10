import autograd.numpy as np
import autograd.numpy.random as rd

rng = rd.RandomState()

def precompute_covariance(y_points, l):
    kernel = lambda y, yp: np.exp(-0.5*(y - yp)**2 / l**2) # Covariance kernel
    grid_1, grid_2 = np.meshgrid(y_points, y_points)
    K = kernel(grid_1, grid_2)
    
    return K

def gp(y_points, K):
    f_1 = rng.multivariate_normal(mean=np.zeros_like(y_points), cov=K)
    f_2 = rng.multivariate_normal(mean=np.zeros_like(y_points), cov=K)
    return f_1, f_2


