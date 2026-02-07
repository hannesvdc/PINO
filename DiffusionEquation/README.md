# Physics-Informed Neural Operator for the 1D heat equation with fixed inital condition and Dirichlet boundary conditions.

## biased_tau/
Contains my first experiment with tau-samples biased close to 0.

## extra_uniform_tau/
Tries to remove the large-tau bias by including uniform large-tau samples.

## ic_time_factor/
Another attempt at remove large-tau oscillations by dampening the initial condition term by 1/(1+tau). Trains much slower.

## residual
Attempt at getting the physics more accurately by using residual (skip) layers instead of regular forward layers. Typically works better with PINNs.