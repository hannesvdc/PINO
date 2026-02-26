"""
This file contains the architecture for the FiLM-PINN neural network.

"""

import torch as pt
import torch.nn as nn

from typing import Tuple, Optional

class FourierFeatureEmbedding ( nn.Module ):
    """
    Frequency encoding (x,y) -> [x,y, sin(B[x,y]), cos(B[x,y])] in the trunk network
    with fixed but random B. This step alone improves PINN training for spatial fields.
    """
    def __init__(self, in_dim: int = 2, n_frequencies: int = 16, scale: float = 5.0):
        super().__init__()

        self.in_dim = in_dim
        self.n_frequencies = n_frequencies
        B = pt.randn(n_frequencies, in_dim) * scale
        self.register_buffer("B", B)  # fixed (not trained)

    def forward(self, 
                xy: pt.Tensor, # (B, 2)
                ) -> pt.Tensor:
        assert xy.shape[1] == 2

        # xy: (B, 2)
        freq = (xy @ self.B.t())  # (B, n_freq)
        return pt.cat([xy, pt.sin(freq), pt.cos(freq)], dim=-1)
    
class BoundaryConvEncoder(nn.Module):
    """
    Branch Conv1D encoder for the boundary forcing f(x=1, y) = (f_x(x=1,y), f_y(x=1,y)).
    
    Training data are `f_samples: (B, 2, m) -> z: (B, dz)`

    Internally: Conv1D over length m.
    """
    def __init__( self,
                  m: int, # number of grid points on the right boundary (m = 100)
                  n_features: int = 64, # number of latent output features
                  hidden_channels: Tuple[int, int, int] = (16, 32, 64) ):
        super().__init__()
        self.m = m
        self.n_features = n_features

        c1, c2, c3 = hidden_channels
        # (B, 2, m) -> (B, c1, m)
        self.conv1 = nn.Conv1d(2, c1, kernel_size=5, padding=2)
        # downsample length by 2: m -> m/2
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=5, padding=2, stride=2)
        # downsample again: m/2 -> m/4
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=5, padding=2, stride=2)
        self.act = nn.ReLU()

        # Global average pooling over length -> (B, c3)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Project to n_features
        self.proj = nn.Sequential( nn.Linear(c3, n_features), nn.Tanh())

    def forward(self, 
                f_samples: pt.Tensor) -> pt.Tensor:
        """
        f_samples: (B, 2, m)

        returns z: (B, n_features)
        """
        assert len(f_samples.shape) == 3

        x = self.conv1(f_samples)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.act(x)

        x = self.pool(x).squeeze(2)  # (B, c3)
        z = self.proj(x)              # (B, dz)
        return z
    
class FiLM(nn.Module):
    """
    The FiLM MLP that maps the output of BoundaryConvEncoder to multiplication coefficients.

    Given z: (B, n_features), produce gamma,beta: (B, H) and modulate h: (B, H).
    """
    def __init__(self, n_features: int, H: int, init_gamma_bias: float = 1.0):
        super().__init__()
        self.to_gb = nn.Linear(n_features, 2 * H)

        # Initialize FiLM near identity: gamma ~ 1, beta ~ 0
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)
        with pt.no_grad():
            self.to_gb.bias[:H].fill_(init_gamma_bias)  # gamma factor
            self.to_gb.bias[H:].zero_()                 # beta bias

    def forward(self, 
                h: pt.Tensor, # shape 
                z: pt.Tensor) -> pt.Tensor:
        assert len(h.shape) == 2

        gb = self.to_gb(z)  # (B, 2H)
        H = h.shape[1]
        gamma, beta = gb[:, :H], gb[:, H:]
        return gamma * h + beta
    
class FiLMTrunkMLP(nn.Module):
    """
    Trunk network that takes a new position (x, y) and predicts
    the displacement field (u(x,y), v(x,y)).
    
    This network applies a separate FiLM scaling after every linear + activation layer.
    """
    def __init__( self,
                  n_features: int,
                  hidden: int = 128,
                  depth: int = 8,
                  fourier_freqs: int = 16,
                  fourier_scale: float = 5.0):
        super().__init__()

        self.hidden = hidden
        self.depth = depth

        self.ff = FourierFeatureEmbedding(in_dim=2, n_frequencies=fourier_freqs, scale=fourier_scale)
        real_in_dim = 2 + (2 * fourier_freqs)

        layers = []
        films = []
        for i in range(depth):
            layers.append(nn.Linear(real_in_dim if i == 0 else hidden, hidden))
            films.append(FiLM(n_features, hidden)) # One FiLM head per layer (cheap and effective)
        self.layers = nn.ModuleList(layers)
        self.films = nn.ModuleList(films)

        # The trunk network needs a smooth activation function for backward differentiation.
        self.act = nn.Tanh()
        self.output_layer = nn.Linear(hidden, 2)  # -> (u(x,y), v(x,y))

        # Small init helps stability for PDE residual training
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, 
                xy: pt.Tensor, # (B, 2)
                z: pt.Tensor, # (B, n_features)
                ) -> pt.Tensor:
        """
        xy: (B, 2)
        z:  (B, n_features)

        returns uv: (B, 2)
        """
        # (x,y) embedding
        h = self.ff(xy) # (B, in_dim)
        for i, (lin, film) in enumerate(zip(self.layers, self.films)):
            h = lin(h) # linear
            h = self.act(h) # activation
            h = film(h, z) #FiLM
        
        uv = self.output_layer(h)
        return uv
    
class ElasticityFiLMPINN(nn.Module):
    """
    The complete physics-informed neural operator network that
    inputs f(y) through the branch (i.e. FiLM networks) and (x, y)
    through the trunk and outputs the predicted displacements (u, v).
    """
    def __init__( self,
                  m_boundary: int,
                  n_features: int = 64,
                  trunk_hidden: int = 128,
                  trunk_depth: int = 8,):
        super().__init__()
        self.m = m_boundary

        self.branch = BoundaryConvEncoder(m=self.m, n_features=n_features)
        self.trunk = FiLMTrunkMLP(n_features=n_features,
                                  hidden=trunk_hidden,
                                  depth=trunk_depth)
        
    def getNumberOfParameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def forward( self,
                 xy: pt.Tensor,         # (B, 2)
                 f_samples: pt.Tensor,  # (B, 2, m)
                 ) -> pt.Tensor:
        """
        Returns uv: (B,2) = (u,v) at the query points xy.

        This network explicitly encodes zero Dirichlet boundary conditions at x=0
        by multiplying the output of the trunk network by x.
        """
        z = self.branch(f_samples)  # (B, n_features)
        uv = self.trunk(xy, z)      # (B, 2)

        x = xy[:, 0:1]  # (B,1)
        return x * uv
