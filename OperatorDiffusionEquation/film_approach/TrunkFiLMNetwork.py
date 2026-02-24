import sys
sys.path.append('./')
sys.path.append('../')

import math
import torch as pt
import torch.nn as nn

from collections import OrderedDict
from typing import List

try:
    from FiLM import ConvFiLMNet
except:
    from film_approach.FiLM import ConvFiLMNet
from rbf_interpolation import buildCholeskyMatrix, jointIndexingRBFInterpolator

class TrunkFilmNetwork( nn.Module ):

    def __init__( self, film_channels : List[int],
                        n_hidden_layers : int,
                        z : int,
                        x_grid : pt.Tensor,
                        l : float,
                        T_max : float,
                        tau_max : float,
                        logk_max : float ):
        super().__init__()

        self.l = l
        self.T_max = T_max
        self.tau_max = tau_max
        self.logk_max = logk_max

        # Build the interpolating cholesky matrix.
        self.register_buffer( "x_grid", x_grid )
        cholesky_L = buildCholeskyMatrix( self.x_grid, self.l )
        self.register_buffer( "cholesky_L", cholesky_L )

        # Construct the FiLM embedding with GELU actication functions
        self.n_grid_points = len( self.x_grid )
        self.n_hidden_layers = n_hidden_layers
        self.film = ConvFiLMNet( self.n_grid_points, z, self.n_hidden_layers, film_channels, kernel_size=5, act=nn.GELU(), film_hidden_dim=128)

        # Setup the forward network. Tanh has more stable second derivatives, no blow-up.
        forward_layers = [4] + [z] * self.n_hidden_layers + [1]
        layers = []
        for n in range(1, len(forward_layers)-1 ):
            n_in = forward_layers[n-1]
            n_out = forward_layers[n]
            layers.append( ( f"linear_{n}", nn.Linear(n_in, n_out, bias=True) ) )
        self.hidden_layers = nn.Sequential( OrderedDict(layers) )
        self.output_layer = nn.Linear(n_out, forward_layers[-1], bias=True)
        self.act = nn.Tanh()

        # Initialize the last layer at zero.
        nn.init.zeros_( self.output_layer.weight )
        nn.init.zeros_( self.output_layer.bias )
        
    def forward( self, x : pt.Tensor, # (B,1)
                       t : pt.Tensor, # (B,1)
                       params : pt.Tensor, # (B,2)
                       u0 : pt.Tensor, # (B, n_grid_points)
                       ) -> pt.Tensor:
        # Checks on the input
        if x.ndim == 1:
            x = x[:,None]
        if t.ndim == 1:
            t = t[:,None]
        assert x.ndim == 2 and x.shape[1] == 1, f"`x` Must have shape (B,1) but got {x.shape}"
        assert t.ndim == 2 and t.shape[1] == 1, f"`t` Must have shape (B,1) but got {t.shape}"
        assert params.ndim == 2 and params.shape[1] == 2, f"`params` must have shape (B,2) but got {params.shape}"
        assert u0.ndim == 2 and u0.shape[1] == self.n_grid_points, f"`u0` must have shape (B, n_grid_points) but got {u0.shape}"
        assert x.shape[0] == t.shape[0] and t.shape[0] == params.shape[0] and params.shape[0] == u0.shape[0], \
                f"`x`, `t`, `params`, `u0` must have the same batch size but got {x.shape}, {t.shape}, {params.shape}, {u0.shape}"
        
        # Extract parameters and process the input
        k = params[:,0:1]
        logk_hat = pt.log( k ) / self.logk_max
        Ts = params[:,1:2]
        Ts_hat = Ts / self.T_max
        tau = t * k
        tau_hat = tau / self.tau_max

        # Calculate the embedding
        gamma_raw, beta_raw = self.film( u0 )

        # Main prediction layer
        y = pt.cat( (x, tau_hat, logk_hat, Ts_hat ), dim=1)
        for i in range( len( self.hidden_layers ) ):
            y = self.hidden_layers[i](y)
            y = (1.0 + gamma_raw[:,i,:] ) * y + beta_raw[:,i,:]
            y = self.act( y )
        y = self.output_layer( y )

        # Interpolate the initial condition to be evaluated at every input `x`.
        u0_at_x = jointIndexingRBFInterpolator( self.cholesky_L, self.x_grid, self.l, x, u0 )

        # Include the Dirichlet boundary condition
        c = math.pi**2 #F.softplus( self.rate_c )
        beta = 1.0 - pt.exp( -c * tau )
        u_xt = u0_at_x + beta * x * (1.0 - x) * y

        # Go back to the physics
        T_xt = Ts + self.T_max * u_xt
        return T_xt
