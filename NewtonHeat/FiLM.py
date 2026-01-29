import torch as pt
import torch.nn as nn

from typing import Tuple

class FiLMLayer( nn.Module ):

    def __init__( self, z : int ):
        super().__init__()

        self.z = z
        self.hidden_layer = nn.Linear(3, 2*z, bias=True)
        self.act = nn.ReLU()
        self.output_layer = nn.Linear(2*z, 2*z, bias=True)

        # Initialize near gamma = 1, beta = 0
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    # Assumes the model forward parameters are p = (T0 - T_inf, ln k, T_inf )
    def forward(self, p : pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        p = self.act( self.hidden_layer( p ) )
        p = self.output_layer( p )
        
        gamma_raw = p[:,0:self.z]
        beta = p[:,self.z:]
        return 1.0 + gamma_raw, beta