import torch as pt
import torch.nn as nn

from collections import OrderedDict

from typing import List, Callable

class MultiLayerPerceptron( nn.Module ):

    def __init__(self, neurons_per_layer : List[int],
                       act : Callable[[], nn.Module],
                       init_zero=True):
        super().__init__()

        assert len(neurons_per_layer) >= 2, "`neurons_per_layer` must contain at least two elements."

        layers = []
        for n in range(1, len(neurons_per_layer) ):
            n_in = neurons_per_layer[n-1]
            n_out = neurons_per_layer[n]

            layers.append( ( f"linear_{n}", nn.Linear(n_in, n_out, bias=True) ) )
            if n < len(neurons_per_layer)-1:
                layers.append( ( f"act_{n}", act() ) )
        self.layers = nn.Sequential( OrderedDict(layers) )

        if init_zero:
            self.apply( self.init_last_layer_zero )

    def init_last_layer_zero(self, m):
        if isinstance(m, nn.Linear) and m.out_features == 1:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x : pt.Tensor ) -> pt.Tensor:
        return self.layers(x)