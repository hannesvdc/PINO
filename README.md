# PINO – Physics-Informed Neural Operators

This repository contains my experiments with Physics-Informed Neural Operators (PINOs).  
Physics-informed learning has huge potential to transform how we do scientific modeling and discovery — but so far, I’ve personally struggled to make these methods work on even moderately realistic physical systems.

The goal of this repo is to change that. I’m starting from very simple models and gradually building toward learning the solution operator of a 3D second-order PDE with non-trivial boundary conditions. The exact PDE is still TBD and will evolve with my interests — but two outcomes are guaranteed: lots of learning, and plenty of frustration.

If you’re interested in PINNs, PINOs, or scientific ML more broadly, feel free to star the repo and follow along.

## What’s in this repository?

There are currently three main projects:

- **`NewtonHeat`**  
  A simple PINO for Newton’s law of cooling  
  $$
  \frac{dT(t)}{dt} = -k (T(t) - T_s), \quad T(0) = T_0.
  $$  
  The model learns the temperature evolution $T(t)$ for arbitrary parameter triples $(T_0, k, T_s)$.

- **`DiffusionEquation`**  
  A PINO for the 1D heat equation  
  $$
  \frac{\partial T}{\partial t} = \kappa \frac{\partial^2 T}{\partial x^2},
  $$  
  with Dirichlet boundary conditions $(0,t) = T(1,t) = T_s$ and initial condition $T(x,0)=T_0(x)$.  
  At the moment, I have a working model for a *fixed* initial condition; extending this to a true operator over random initial conditions is ongoing work.

- **2D Steady-State Linear Elasticity**  
  Experiments using PINNs / DeepONets / FiLM-based PINNs (see `PINN_DeepONet`, `Residual_PINN_DeepONet`, `FiLM_PINN`).  
  This is currently exploratory and largely unsuccessful — which is what motivated me to go back to much simpler test cases.

I describe each experiment and my design choices in more detail in the individual project folders. Also check out my [blog](https://hvandecasteele.com/blog/) and [substack](https://hannesvdc.substack.com) where I regularly post about the things I learned, some results for each project, and the mathematics behind each problem (and many other things.)

## A small terminological nitpick

An operator maps functions to functions. Strictly speaking, not all examples in this repository qualify as “neural operators” (e.g. Newton’s law maps a finite-dimensional parameter vector to a scalar trajectory). Some of these models are therefore just physics-informed neural networks in disguise. I’ll mostly ignore this distinction for readability.
