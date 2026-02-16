import sys
sys.path.append('../')
sys.path.append('../../')

import torch as pt
import torch.nn as nn

from collections import OrderedDict

from rbf_interpolation import buildCholeskyMatrix, tensorizedRBFInterpolator

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
    
class ConvNet( nn.Module ):
    """
    1D Conv encoder for an initial condition u0(x) sampled on a grid.

    Input:
        x: (B, n_grid_points)
    Output:
        z: (B, q)

    Default behavior:
      - Keep spatial length fixed (stride=1, same padding)
      - Increase channels across layers
      - Global average pool over space
      - Linear projection to q
    """
    def __init__(self, n_grid_point : int, # 51 in our case
                       n_conv_layers : int,
                       kernel_size : int,
                       q : int):
        super().__init__()
        assert kernel_size % 2 == 1, "Use odd kernel_size for same padding."

        self.n_grid_point = n_grid_point
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.q = q

        self.act = nn.GELU()

        # Setup the convolutional layers
        padding = kernel_size // 2
        self.conv_layers = []
        in_channels = 1
        for n in range( self.n_conv_layers ):
            out_channels = min( max( 8, 2*in_channels ), 64 )
            self.conv_layers.append( nn.Conv1d(in_channels, out_channels, self.kernel_size, stride=1, padding=padding, bias=True))
            in_channels = out_channels
        self.conv_layers = nn.ModuleList( self.conv_layers )
        self.c_out = out_channels

        # Do a global average pooling operation from shape (B, self.c_out, n_grid_points) to (B, self.c_out)
        self.pool = nn.AdaptiveAvgPool1d(output_size=1) # Shape (B, self.c_out)
        self.head = nn.Linear( self.c_out, self.q, bias=True )

    def getNumberOfTrainableParameters( self ):
        return sum( [param.numel() for param in self.parameters() if param.requires_grad] )

    def forward( self, x : pt.Tensor ) -> pt.Tensor:
        if x.ndim == 2:
            x = x[:,None,:] # Shape (B, 1, n_grid_points)
        assert x.shape[2] == self.n_grid_point, "Last dimension of `x` must have size `n_grid_points`"

        for n in range( self.n_conv_layers ):
            x = self.conv_layers[n]( x )
            x = self.act( x )

        # Average pooling
        x = self.pool( x )[:,:,0] # (B, self.c_out)
        x = self.act( x )

        # Map to p dimensions for Deeponet
        x = self.head( x )
        
        return x
        
class DeepONet( nn.Module ):
    def __init__(self, branch_setup : Dict, 
                       trunk_setup : Dict,
                       x_grid : pt.Tensor,
                       l : float,
                       T_max : float,
                       tau_max : float,
                       logk_max : float,
                       q : int):
        super().__init__()
        
        self.l = l
        self.T_max = T_max
        self.tau_max = tau_max
        self.logk_max = logk_max
        self.q = q

        # Precompute the Cholesky factorization
        self.register_buffer( "x_grid", x_grid )
        L = buildCholeskyMatrix( self.x_grid, self.l )
        self.register_buffer( "L", L )

        # Setup the branch convolutional network with some good default values.
        n_grid_points = branch_setup[ "n_grid_points" ] # number of discretization points, must be passed explicitly
        n_branch_layers = branch_setup.get( "n_hidden_layers", 3 )
        branch_kernel_size = branch_setup.get( "kernel_size", 5 )
        self.branch = ConvNet( n_grid_points, n_branch_layers, branch_kernel_size, q)

        # Setup the trunk network with some good default values.
        trunk_input_dim = trunk_setup.get( "input_dim", 3 )
        trunk_n_hidden = trunk_setup.get( "n_hidden_layers", 4)
        trunk_z = trunk_setup.get( "z", 64 )
        self.trunk = MLP( trunk_input_dim, trunk_n_hidden, trunk_z, q )

    def getNumberOfTrainableParameters( self ):
        return sum( [param.numel() for param in self.parameters() if param.requires_grad] )
    
    def forward(self, x : pt.Tensor, # (Bt, 1)
                      t : pt.Tensor, # (Bt, 1)
                      params : pt.Tensor, # (Bt, 2)
                      u0 : pt.Tensor, # (Bb, n_grid_points)
                ) -> pt.Tensor: # (Bb, Bt)
        # Make sure all inputs have the required shape
        if x.ndim == 1:
            x = x[:,None]
        if t.ndim == 1:
            t = t[:,None]
        assert x.ndim == 2 and x.shape[1] == 1, "`x` must have shape (Bt,1)"
        assert t.ndim == 2 and t.shape[1] == 1, "`t` must have shape (Bt,1)"
        assert params.ndim == 2 and params.shape[1] == 2, "`params` must have shape (Bt,2)"
        assert x.shape[0] == t.shape[0] and x.shape[0] == params.shape[0], "`x`, `t` and `params` must have the same batch size."

        # Pre-process the parameters and input
        k = params[:,0:1] 
        logk_hat = pt.log(k) / self.logk_max # scale between -1 and 1
        tau = t * k
        tau_hat = tau / self.tau_max

        # Evaluate all initial conditions at all spatial points
        u0_at_x = tensorizedRBFInterpolator( self.L, self.x_grid, self.l, x, u0 ) # (Bb, Bt)

        # Propagate the branch and trunk
        branch_output = self.branch( u0 ) # (Bb, self.q)
        trunk_input = pt.cat((x, tau_hat, logk_hat), dim=1)
        trunk_output = self.trunk( trunk_input ) # (Bt, self.q)

        # Expand, multiply and reduce
        g = branch_output @ trunk_output.T # (Bb, q) @ (q, Bt) = (Bb, Bt)

        # Bring back to the physics
        alpha_tau = 1.0 / (1.0 + tau.T) # (1, Bt)
        beta_tau = 1.0 - alpha_tau
        u_hat = u0_at_x * alpha_tau + beta_tau * x.T * (1.0 - x.T) * g

        # Return actual temperatures
        T_s = params[:,1] # (Bt, )
        return T_s[None,:] + u_hat * self.T_max