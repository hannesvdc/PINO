# PINO - Physics-Informed Neural Operators
This repository contains my experiments with Physics-Informed Neural Operators (PINOs). Physics-Informed learning has great potential, but for some reason I have never gotten it to work for a real physical model. My goal with this repository is to change that; starting small to master the basics but with the express end goal of learning the 3D solution to a second-order PDE. The precise PDE will be decided later, but it will be a real challenge! 

Stay tuned and star this repo if you're interested in following my journey!

There are currently two projects:
-  `NewtonHeat` contains a simple PINO that learns the solution $T(t)$ to Newton's heat law $\frac{dT(t)}{dt} = -k * (T(t) - T_{\infty})$ with initial condition $T(0) = T0$. The PINO learns the temperature evolution for any combination of the three model parameteres $(T0, k, T_{\infty})$.

-  2D Steady-State Linear Elasticity equation. The code is spread over `PINN_DeepONet`, `Residual_PINN_Deeponet` and `FiLM_PINN`. This is mostly a learning project for now. None of my approaches have worked yet, which prompted me to go back to basics in the first example.

I explain these projects and my approach in much more detail below. I will keep the repo and README updated as much as possible.

**Just me being pedantic** An operator is a mathematical object that maps functions to functions. The term 'Physics-Informed Neural Operator' is therefore not always correct. For example, in Newton's Heat Law example, we map a discrete state of three parameters onto a scalar variable. The NN is a function, not an operator, but I will mostly ignore this abuse of notation.


## Newton's Heat Law
