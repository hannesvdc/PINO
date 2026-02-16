import torch as pt
import torch.nn as nn

from interpolate import evaluateInterpolatingSpline

from typing import Dict
    
class MLP( nn.Module ):
    def __init__( self, input_dim : int,
                        n_hidden_layers : int,
                        z : int,
                        output_dim : int):
        super().__init__()

        self.stem = nn.Linear( input_dim, z, bias=True )
        self.act = nn.Tanh()

        self.n_hidden_layers = n_hidden_layers
        layers = []
        for _ in range( self.n_hidden_layers ):
            hidden_layer = nn.Linear( z, z, bias=True )
            layers.append( hidden_layer )
        self.hidden_layers = nn.ModuleList( layers )

        self.head = nn.Linear( z, output_dim, bias=True )
        nn.init.zeros_( self.head.weight )
        nn.init.zeros_( self.head.bias )

    def getNumberOfTrainableParameters( self ):
        return sum( [param.numel() for param in self.parameters() if param.requires_grad] )
    
    def forward(self, x : pt.Tensor ) -> pt.Tensor:
        # Project to z dimensions
        x = self.act( self.stem(x) )

        for n in range( self.n_hidden_layers ):
            x = self.hidden_layers[n] (x)
            x = self.act( x )
        
        # Project back to `output_dim` dimensions
        y = self.head( x )

        return y
    
class EmbeddingConvNet( nn.Module ):
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

        # Map to p dimensions for Deeponet
        x = self.head( x )
        
        return x
    
class EmbeddingNetwork( nn.Module ):
    def __init__(self, embedding_setup : Dict, 
                       n_hidden_layers : int,
                       z : int,
                       x_grid : pt.Tensor,
                       T_max : float,
                       tau_max : float,
                       logk_max : float):
        super().__init__()
        
        # Store network parameters
        self.x_grid = x_grid
        self.T_max = T_max
        self.tau_max = tau_max
        self.logk_max = logk_max
        self.z = z

        # Setup the branch convolutional network with some good default values.
        n_grid_points = embedding_setup[ "n_grid_points" ] # number of discretization points, must be passed explicitly
        n_branch_layers = embedding_setup.get( "n_hidden_layers", 3 )
        q = embedding_setup.get( "q", 10 )
        self.embedding = MLP( n_grid_points, n_branch_layers, z, q)

        # Setup the trunk network with some good default values.
        self.n_input_dim = 3 + q
        self.n_hidden_layers = n_hidden_layers
        self.z = z
        self.mlp = MLP( self.n_input_dim, self.n_hidden_layers, self.z, 1 )

        # Initialize the head layer with zeros to avoid T_xx from blowing up
        nn.init.zeros_( self.mlp.head.weight )
        nn.init.zeros_( self.mlp.head.bias )

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

        # Calculate the embedding
        u0_embed = self.embedding( u0 )

        # Propagate everything through the MLP
        input = pt.cat((x, tau_hat, logk_hat, u0_embed), dim=1)
        g = self.mlp( input ) # (B, 1)

        # Evaluate all initial conditions at all spatial points
        u0_at_x_tensor = evaluateInterpolatingSpline( x[:,0], self.x_grid, u0 ) # check this shape.
        u0_at_x = pt.diag( u0_at_x_tensor )[:,None]

        # Bring back to the physics
        alpha_tau = 1.0 / (1.0 + tau)
        beta_tau = tau * alpha_tau
        u_hat = u0_at_x * alpha_tau + beta_tau * x * (1.0 - x) * g

        # Return actual temperatures
        T_s = params[:,1:2] # (Bt, 1)
        return T_s + u_hat * self.T_max