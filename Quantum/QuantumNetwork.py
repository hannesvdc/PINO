import torch as pt
import torch.nn as nn

from typing import List

from MLP import MultiLayerPerceptron


class QuantumNetwork( nn.Module ):
    """
    The main trainable neural network to solve the Schrodinger equation
    for the (unnormalized) wave function $\psi(r)$.
    """

    def __init__( self, 
                  neurons_per_layer : List[int],
                  r_cutoff : float ) -> None:
        super().__init__()

        if neurons_per_layer[-1] != 1:
            raise ValueError("QuantumNetwork output dimension should be 1.")

        self.r_cutoff = r_cutoff
        self.mlp = MultiLayerPerceptron( neurons_per_layer, nn.GELU, init_zero=True )

    def forward(self, R : pt.Tensor,
                      xyz : pt.Tensor ) -> pt.Tensor:
        assert xyz.ndim == 2 and xyz.shape[1] == 3, f"`xyz` must have shape (B, 3) but got {xyz.shape}"
        N = xyz.shape[0]
        R = R.flatten()
        B = len( R )

        # Repmat and put all in one big tensor of shape (B, N, 4)
        xyz = xyz[None,:,:] * pt.ones( (B, 1, 1), device=R.device, dtype=R.dtype )
        log_R = pt.log(R)[:,None,None] * pt.ones( (1, N, 1), device=R.device, dtype=R.dtype )
        mlp_input = pt.cat( (xyz, log_R), dim=2 )

        mlp_output = self.mlp( mlp_input ) # shape (B, N, 1)
        mlp_output = pt.squeeze( mlp_output, -1 )

        P1 = pt.stack((-R, pt.zeros_like(R), pt.zeros_like(R)), dim=-1)[:, None, :] # (B, 1, 3)
        P2 = pt.stack(( R, pt.zeros_like(R), pt.zeros_like(R)), dim=-1)[:, None, :]
        r1 = pt.linalg.norm(xyz - P1, dim=2).clamp_min( 1e-8 )
        r2 = pt.linalg.norm(xyz - P2, dim=2).clamp_min( 1e-8 )
        envelope = pt.exp( -r1 ) + pt.exp( -r2 )

        r_sq = pt.sum( xyz * xyz, dim=2 ) # (B, N)
        dirichlet_gate = 1.0 - r_sq / self.r_cutoff**2
        psi = dirichlet_gate * envelope * (1.0 + mlp_output) # (B, N)

        return psi