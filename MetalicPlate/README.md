# Physics-Informed Neural Operator for predicing the elastic deformation of a 2D metalic plate.

This folder contains the code and experiments associated to [my blog post](https://www.hvandecasteele.com/blog/pino-for-elasticity/) and [Substack article](https://hannesvdc.substack.com/p/from-heat-to-elasticity-the-next). It is my first example of a PINO for an equation other than the heat or diffusion equation. The goal is to solve the linear elasticity equations in two dimensions

$$
\nabla \cdot \sigma\left(u, v\right) + F(x, y) = 0 \quad \tag{1}
$$

where $u, v$ are the displacements of the plate in the $x$ and $y$ directions, respectively, $\sigma(u,v)$ is the stess tensor, and $F(x,y)$ is the bulk forcing. The latter function is zero in our case. The plate is subject to clamped (zero Dirichlet) boundary conditions on the left, $u(0,y)=v(0,y) = 0$, and a traction force is applied on the right boundary

$$
\sigma(u(x=1, y), v(x=1,y)) n = (g_x(y), g_y(y)) = g(y)
$$ 

with $n$ the boundary normal vector. The upper and lower boundaries are *free* meaning $\sigma(u,v) n = 0$ at $y=0$ and $y=1$.

The PINO learns the displacement field for any right-boundary forcing functions $g(y)$, Poission ratio $\nu$ and Young's modulus $E$. The neural network is unaware of $E$ because the physics scales linearly with $1/E$. As a result, the raw boundary forcing is $E g(y)$. The network architecture is based on a branch-trunk approach. The branch consists of 8 convolution layers to process $g_x$ and $g_y$ as separate channels. The branch layer uses a FiLM appraoch; it's outputs are linear coefficients $\gamma_i$ and $\beta_i$ that attenuate the hidden layers $h_i$ of the trunk

$$
\gamma_i(g_x, g_y) h_i(x, y, \nu) + \beta_i(g_x, g_y)
$$

The trunk network is a simple MLP that takes $(x, y, \nu)$ as inputs and outputs $\left(u(x,y), v(x,y)\right)$. The activation functions are GELU.

The training procedure does not use the residual of equation (1) as loss, but uses an enery loss together with Monte Carlo quadrature to speed up training. The main benefit of the energy loss is that it only requires first derivatives of the network output, making training significantly faster and less memory intense. Read more on the blog or Substack if you want to know more about the energy loss.

## Code Structure

The code is structured as follows:

│   EnergyLoss.py                  # Implements the energy-based loss function with Monte Carlo quadrature. Also uses fast-autograd.			
│   FEM.py      		   		   # Simple finite-element implementation to compare PINO output with.
|	FiLM.py    		    		   # Branch-FiLM network implementation.
|	gp.py				           # Gaussian-Process code for sampling the boundary forcings $g(y) = (g_x(y), g_y(y))$.
|	PlateDataset.py          	   # Main dataset used for training. Loads GP for the branch, samples $(x,y)$ coordinates for the trunk and combines both.
|	rbf_interpolation.py           # Global radial-basis function interplation of the boundary forcing, used for loss and energy evaluations. Requires double precision.
|   test_model.py                  # Main testing script. Compares PINO output with FEM simulation.
|   train_adam.py                  # Main training script. Implements the Adam optimizer with a simple step learning rate scheduler. No L-BFGS.
|   TrunkFiLMNetwork.py            # Complete architecture. Combines the branch-FiLM network with an MLP for the trunk.
|	data/						   # Stores the dataset
|	Results/					   # Optimal weights, some images and logs of training progress. I recommend re-generating these yourself.
|	tests/						   # Some tests I did along the way, mostly about correctness of the RBF interpolation and its derivatives.

Let me know if you're interested in running this code yourself. It is mostly functional but could use some more cleanup. Happy to chat!