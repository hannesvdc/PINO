import torch as pt
import torch.nn as nn

from collections import OrderedDict

from rbf_interpolation import buildCholeskyMatrix, jointIndexingRBFInterpolator

from typing import Dict, Callable
    
class MLP( nn.Module ):
    def __init__( self, input_dim : int,
                        n_hidden_layers : int,
                        z : int,
                        output_dim : int,
                        act : Callable = nn.GELU(),
                        name : str = ""):
        super().__init__()
        self.act = act

        neurons_per_layer = [input_dim] + [z]*n_hidden_layers + [output_dim]
        layers = []
        for n in range( 1, len(neurons_per_layer) ):
            n_in = neurons_per_layer[n-1]
            n_out = neurons_per_layer[n]
            layers.append( (f"{name}linear {n}", nn.Linear(n_in, n_out, bias=True)) )
            if n < len(neurons_per_layer)-1: # no activation after the last layer
                layers.append( (f"{name}act {n}", self.act))
        self.layers = nn.Sequential( OrderedDict( layers ) )

    def getNumberOfTrainableParameters( self ):
        return sum( [param.numel() for param in self.parameters() if param.requires_grad] )
    
    def forward(self, x : pt.Tensor ) -> pt.Tensor:
        return self.layers( x )
    
class InitialEmbeddingMLP( nn.Module ):
    def __init__(self, embedding_setup : Dict, 
                       n_hidden_layers : int,
                       z : int,
                       x_grid : pt.Tensor,
                       l : float,
                       T_max : float,
                       tau_max : float,
                       logk_max : float):
        super().__init__()
        
        # Store network parameters
        self.l = l
        self.T_max = T_max
        self.tau_max = tau_max
        self.logk_max = logk_max
        self.z = z

        # Precompute the Cholesky factorization
        self.register_buffer( "x_grid", x_grid )
        L = buildCholeskyMatrix( self.x_grid, self.l )
        self.register_buffer( "L", L)

        # Setup the branch convolutional network with some good default values.
        n_grid_points = embedding_setup[ "n_grid_points" ] # number of discretization points, must be passed explicitly
        n_branch_layers = embedding_setup.get( "n_hidden_layers", 3 )
        q = embedding_setup.get( "q", 10 )
        self.embedding = MLP( n_grid_points, n_branch_layers, z, q, name="embedding_")
        self.embed_norm = nn.LayerNorm(q)

        # Setup the trunk network with some good default values.
        self.n_input_dim = 3 + q
        self.n_hidden_layers = n_hidden_layers
        self.z = z
        self.mlp = MLP( self.n_input_dim, self.n_hidden_layers, self.z, 1, name="main_" )

    def getNumberOfTrainableParameters( self ):
        return sum( [param.numel() for param in self.parameters() if param.requires_grad] )
    
    def forward(self, x : pt.Tensor, # (B, 1)
                      t : pt.Tensor, # (B, 1)
                      params : pt.Tensor, # (B, 2)
                      u0 : pt.Tensor, # (B, n_grid_points)
                      ) -> pt.Tensor: # (B, 1)
        # Make sure all inputs have the required shape
        if x.ndim == 1:
            x = x[:,None]
        if t.ndim == 1:
            t = t[:,None]
        assert x.ndim == 2 and x.shape[1] == 1, "`x` must have shape (B,1)"
        assert t.ndim == 2 and t.shape[1] == 1, "`t` must have shape (B,1)"
        assert params.ndim == 2 and params.shape[1] == 2, "`params` must have shape (B,2)"
        assert x.shape[0] == t.shape[0] and x.shape[0] == params.shape[0], "`x`, `t` and `params` must have the same batch size."

        # Pre-process the parameters and input
        k = params[:,0:1] 
        logk_hat = pt.log(k) / self.logk_max # scale between -1 and 1
        tau = t * k
        tau_hat = tau / self.tau_max

        # Calculate the embedding and normalize using LayerNorm
        u0_embed = self.embedding( u0 )
        u0_embed = self.embed_norm(u0_embed)

        # Propagate everything through the MLP
        input = pt.cat((x, tau_hat, logk_hat, u0_embed), dim=1)
        g = self.mlp( input ) # (B, 1)

        # Evaluate all initial conditions at all spatial points
        u0_at_x = jointIndexingRBFInterpolator( self.L, self.x_grid, self.l, x, u0 )

        # Bring back to the physics
        alpha_tau = pt.exp( -2.0 * tau )
        beta_tau = 1.0 - alpha_tau
        u_hat = u0_at_x * alpha_tau + beta_tau * x * (1.0 - x) * g

        # Return actual temperatures
        T_s = params[:,1:2] # (Bt, 1)
        return T_s + u_hat * self.T_max