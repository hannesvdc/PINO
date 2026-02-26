import math
import torch as pt
import torch.nn as nn

from collections import OrderedDict
from typing import List, Tuple

from FiLM import ConvFiLMNet

class TrunkFilmNetwork( nn.Module ):

    def __init__( self, film_channels : List[int],
                        n_grid_points : int,
                        n_hidden_layers : int,
                        z : int,
                        nu_max : float, ):
        super().__init__()

        self.nu_max = nu_max

        # Construct the FiLM embedding with GELU actication functions
        self.n_grid_points = n_grid_points
        self.n_hidden_layers = n_hidden_layers
        self.film = ConvFiLMNet( self.n_grid_points, z, self.n_hidden_layers, film_channels, kernel_size=5, act=nn.GELU(), film_hidden_dim=128)

        # Setup the forward network. Tanh has more stable derivatives, no blow-up.
        forward_layers = [3] + [z] * self.n_hidden_layers + [2]
        layers = []
        for n in range(1, len(forward_layers)-1 ):
            n_in = forward_layers[n-1]
            n_out = forward_layers[n]
            layers.append( ( f"linear_{n}", nn.Linear(n_in, n_out, bias=True) ) )
        self.hidden_layers = nn.Sequential( OrderedDict(layers) )
        self.output_layer = nn.Linear(n_out, forward_layers[-1], bias=True)
        self.act = nn.Tanh()

        # Initialize the last layer at zero.
        #nn.init.zeros_( self.output_layer.weight )
        #nn.init.zeros_( self.output_layer.bias )
        
    def forward( self, x : pt.Tensor,  # (Bt,1)
                       y : pt.Tensor,  # (Bt,1)
                       nu : pt.Tensor, # (Bb,1)
                       gx : pt.Tensor, # (Bb, n_grid_points)
                       gy : pt.Tensor, # (Bb, n_grid_points)
                       ) -> Tuple[pt.Tensor, pt.Tensor]:
        # Checks on the input
        if x.ndim == 1:
            x = x[:,None]
        if y.ndim == 1:
            y = y[:,None]
        if nu.ndim == 1:
            nu = nu[:,None]
        assert x.ndim == 2 and x.shape[1] == 1, f"`x` Must have shape (Bt,1) but got {x.shape}"
        assert y.ndim == 2 and y.shape[1] == 1, f"`y` Must have shape (Bt,1) but got {y.shape}"
        assert nu.ndim == 2 and nu.shape[1] == 1, f"`nu` must have shape (Bb,1) but got {nu.shape}"
        assert gx.ndim == 2 and gx.shape[1] == self.n_grid_points, f"`gx` must have shape (Bb, n_grid_points) but got {gx.shape}"
        assert gy.ndim == 2 and gy.shape[1] == self.n_grid_points, f"`gy` must have shape (Bb, n_grid_points) but got {gy.shape}"
        assert x.shape[0] == y.shape[0], f"`x` and `y` must have the same batch size but got {x.shape, y.shape}"
        assert nu.shape[0] == gx.shape[0] and gx.shape[0] == gy.shape[0], f"`nu`, `gx` and `gy` must have the same batch size but got {x.shape}, {y.shape}, {nu.shape}, {gx.shape}, {gy.shape}."
        
        # Process the parameters. x and y are naturally normalized.
        Bb = gx.shape[0]
        Bt = x.shape[0]
        nu_hat = nu / self.nu_max
        nu_hat_repmat = nu_hat[:,None,:].expand(Bb, Bt, 1)
        x_repmat = x[None,:,:].expand(Bb, Bt, 1)
        y_repmat = y[None,:,:].expand(Bb, Bt, 1)

        # Calculate the embedding
        film_input = pt.cat( (gx[:,None,:], gy[:,None,:]), dim=1 )
        gamma_raw, beta_raw = self.film( film_input )

        # Main prediction layer
        w = pt.cat( (x_repmat, y_repmat, nu_hat_repmat), dim=2) # (Bb, Bt, 3)
        for i in range( len( self.hidden_layers ) ):
            w = self.hidden_layers[i](w)  # (Bb, Bt, z)
            gamma = gamma_raw[:,i,:]
            beta = beta_raw[:,i,:]
            w = (1.0 + gamma[:,None,:] ) * w + beta[:,None,:]
            w = self.act( w )
        w = self.output_layer( w ) # (Bb, Bt, 2)
        
        # Extract the displacement fields
        u = w[:,:,0]
        v = w[:,:,1]

        # Enforce the Dirichlet boundary condition
        beta_x = x_repmat[:,:,0]**2
        u = beta_x * u
        v = beta_x * v

        return u, v # Shape (Bb, Bt) each