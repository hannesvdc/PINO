import torch as pt
import torch.nn as nn

from collections import OrderedDict
from typing import List

class ConvFiLMNet(nn.Module):
    """
    Convolutional FiLM generator:
      (gx, gy)) -> conv features -> pooled vector -> (gamma_raw, beta) per trunk layer.

    Outputs:
      gamma_raw: (B, L, z)
      beta:      (B, L, z)
    """
    def __init__(
        self,
        n_grid_points: int,
        trunk_width: int,        # z
        n_trunk_layers: int,     # L (hidden layers to modulate)
        channels : List[int],
        kernel_size: int = 5,
        act: nn.Module = nn.GELU(),
        film_hidden_dim: int = 128,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd to keep length with symmetric padding."
        self.n_grid_points = n_grid_points
        self.trunk_width = trunk_width
        self.n_trunk_layers = n_trunk_layers
        self.act = act

        # Conv stack (keeps spatial length fixed)
        conv_layers = []
        for i in range(1, len(channels)):
            conv_layers.append( ( f"conv_{i}", nn.Conv1d(
                in_channels=channels[i-1],
                out_channels=channels[i],
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                padding_mode="replicate",
                bias=True ) ) )
            if i < len(channels) - 1:
                conv_layers.append( (f"act_{i}", act) )
        self.conv_layers = nn.Sequential( OrderedDict(conv_layers) )

        # Global pooling over x: (B, C, n_grid) -> (B, C)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Head produces 2*L*z outputs
        out_dim = 2 * n_trunk_layers * trunk_width
        self.head = nn.Sequential(
            nn.Linear(channels[-1], film_hidden_dim, bias=True),
            act,
            nn.Linear(film_hidden_dim, out_dim, bias=True),
        )

        # Start with gamma_raw=0, beta=0 => gamma≈1, beta=0
        #nn.init.zeros_(self.head[-1].weight)
        #nn.init.zeros_(self.head[-1].bias)

    def forward(self, g: pt.Tensor) -> tuple[pt.Tensor, pt.Tensor]:
        # g: (B, 2, self.n_grid_points)
        assert g.ndim == 3 and g.shape[1] == 2 and g.shape[2] == self.n_grid_points, \
            f"`g` must have shape (B, 2, n_grid_points) but got {g.shape}"

        # Push through the convolution layers.
        x = self.conv_layers(g)  # (B, C, n_grid)
        x = self.pool(x)         # (B, C, 1)
        x = x[:, :, 0]           # (B, C)

        # Apply the head layers
        film = self.head(x)      # (B, 2*L*z)

        # Reshape to gamma and beta per trunk layer
        B = film.shape[0]
        L = self.n_trunk_layers
        z = self.trunk_width
        film = film.view(B, L, 2, z)
        gamma_raw = film[:, :, 0, :]  # (B, L, z)
        beta = film[:, :, 1, :]  # (B, L, z)
        return gamma_raw, beta