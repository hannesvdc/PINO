# PINO - Physics-Informed Neural Operators
This repository contains my experiments regarding Physics-Informed Neural Operators. Phyiscs-Informed learning has great potential, but for some reason I have never gotten it to work for a real differential equation or other physical model. The goal of this repository is to change that, starting small but with this express end goal of learning the 3D solution to a second-order PDE (TBD later)/

There are currently two projects:
-  NewtonHeat contains a simple PINO that learns the solution T(t) to Newton's heat law dT/dt = -k * (T(t) - T_inf) with initial condition T(0) = T0. (T0, k, T_inf) are three model parameters and the neural network outputs T(t).
-  2D Steady-State Linear Elasticity equation. The code is spread over `PINN_DeepONet`, `Residual_PINN_Deeponet` and `FiLM_PINN`. This is mostly a failed project for now, which prompted me to go back to basics in the first example.

More about these projects below. I will keep this repo and README updated as much as possible.

Note: An operator is a mathematical object that maps functions to functions. The term 'Physics-Informed Neural Operator' is therefore not always correct. For example, in Newton's Heat Law example, we map a discrete state of three parameters onto a scalar variable. The NN is a function, not an operator, but I will mostly ignore this abuse of notation.

## Newton's Heat Law
