# *Actual* Physics-Informed Neural Operator for the 1D heat equation with given inital condition and Dirichlet boundary conditions.

This folder contains the follow-up experiments after those in /PINO/DiffusionEquation/. The goal is still to solve the one-dimensional heat equation

$$
\frac{dT}{dt}(x,t) = \kappa \frac{dT}{dx^2}(x,t)
$$

for all rate constants $\kappa$ and Dirichlet boundary conditions $T(x=0,1, t) = T_s$ using physics-informed neural operators, but now the initial condition $T(x,t=0)$ is also an input to the model, not kept fixed. The full solution map is therefore

$$
\left(x, t, \kappa, T_s, T_0\right) \longrightarrow T(x,t).
$$

Since $T_0$ is function of $x \in [0,1]$, simply appending it as extra inputs to the network is very unlikely to work well. Instead, we explore two approaches in this folder

- Train an embedding $T_0 \to z \in \mathbb{R}^q$ consisting of $q$ features via a 'branch' network, and append $z$ as extra input features to the 'trunk' network (alongside $x$, $t$, $\kappa$ and $T_s$);
- Build a Feature-wise Linear Modulation (FiLM) that learns vectors $\gamma_i$, $\beta_i$ that modulate the pre-activation feature vectors $z_i$ via
 $\sigma\left(\gamma_i z_i + \beta_i\right)$ for every layer ($i$) of the trunk network. $\sigma$ is the trunk activation function (Tanh here).

The latter approach is more expressive, has a smoother training curve, but uses slightly more parameters. The embedding approach contains two alternatives: an MLP-only branch layer, and a convolutional branch layer. The FiLM network only uses convolutions (and pooling).

The root folder contains the following helper files:

│   conditionalGP.py                    # To sample initial conditions u_0 that satisfy zero Dirichlet boundary conditions				
│   rbf_interpolation.py      		    # General global interpolation of functions defined on a grid. Has a tensorized and joint indexing implementation.
|	TensorizedDataset.py    		    # Torch dataset class used to generate (x, t, \kappa, T_s, u_0) training and testing data.
|	MLP.py				                # General implementation of Multilayer Perceptrons.
|	ConvNet.py          			    # General (enough) implementation of convolution networks for this project.
|	compareApproaches.py                # Testing script to compare MLP-embedding, Conv-embedding and FiLM networks.
|   fd.py                               # Finite differences simulator of the heat equation.
|   evaluatePINO.py                     # Contains two functions to evaluate the trained PINO. Used in compareApproaches.py
|   Loss.py                             # Defines the physics-informed loss function.

## embedding_approach/
The first alternative to encoding functional inputs into the PINO. It contains the following files:
│   BranchEmbeddingNetwork.py           # Defines the branch-MLP embedding network.
│   ConvBranchEmbeddingNetwork.py       # Defines the convolution-branch embedding network.
|   train_adam.py                       # Training script for the branch-MLP approach.
|   train_conv_adam.py                  # Training script for the conv-branch approach.
|   train_conv_lbfgs.py                 # Attempt to further optimize PINO. Blew up my Macbook (quite literally - I have picturs).

## film_approach
Builds a full PINO using the Feature-wise Linear Modulation framework for processing initial conditions $T_0$. Contains the following files:
|   FiLM.py                             # Defines the convolutional FiLM network that produces vectors $\gamma_i$ and $\beta_i$.
|   TrunkFiLMNetwork.py                 # Trunk network that takes in the FiLM modulation vectors.
|   train_adam.py                       # Training script of the Film approach using the Adam optimizer.

## deeponet_approach/
Unfinished experiments with vanilla deeponet.

Enjoy!