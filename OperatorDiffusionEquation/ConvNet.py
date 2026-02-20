import torch as pt
import torch.nn as nn

from collections import OrderedDict

from typing import List, Callable

class ConvolutionalNetwork( nn.Module ):

    def __init__( self, n_grid_points : int,
                        output_dim : int,
                        channels : List[int],
                        kernel_size : int,
                        act : Callable[[], nn.Module], 
                        padding_mode : str = "zeros", 
                        init_zeros : bool = True ):
        super().__init__()

        assert kernel_size % 2 == 1, "`kernel_size` must be odd."

        # Store some variables
        self.n_grid_points = n_grid_points
        self.output_dim = output_dim
        
        # Build the convolutional layers
        layers = []
        for n in range( 1, len(channels) ):
            channels_in = channels[n-1]
            channels_out = channels[n]

            layers.append( ( f"conv_{n}", nn.Conv1d(channels_in, channels_out, kernel_size, stride=1,
                                                    padding=kernel_size//2, padding_mode=padding_mode, bias=True)))
            if n < len(channels)-1:
                layers.append( ( f"act_{n}", act() ) )
        self.layers = nn.Sequential( OrderedDict(layers) )

        # Average pooling (B, C, n_grid_points) -> (B, C, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Final layer (B, C) -> (B, output_dim)
        self.head = nn.Linear(channels[-1], output_dim, bias=True)

        if init_zeros:
            self.init_last_layer_zero( )

    def init_last_layer_zero(self ):
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward( self, x : pt.Tensor ) -> pt.Tensor:
        if x.ndim == 2:
            x = x[:,None,:] # (B,1,n_grid_points)
        x = self.layers( x )

        x = self.pool( x ) # (B, C, 1)
        x = x[:,:,0] # (B, C)

        x = self.head( x ) # (B, output_dim)
        return x
