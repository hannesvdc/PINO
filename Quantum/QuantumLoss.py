import torch as pt
import torch.nn as nn

from typing import Tuple, Dict

class QuantumLoss( nn.Module ):
    """
    Rayleigh-energy loss for the Schrodinger equation

    Model signature:
        psi = model( input )
    """
    def __init__( self ):
        super().__init__()

    def forward(self, model : nn.Module,
                      R : pt.Tensor, # (B,)
                      xyz : pt.Tensor, # (N,3)
                      mc_weights : pt.Tensor, # (N,)
                ) -> pt.Tensor:
        R = R.flatten()
        R = R.requires_grad_(False)
        P1 = pt.stack((-R, pt.zeros_like(R), pt.zeros_like(R)), dim=-1)[:, None, :] # (B, 1, 3)
        P2 = pt.stack(( R, pt.zeros_like(R), pt.zeros_like(R)), dim=-1)[:, None, :]
        if mc_weights.ndim > 1: mc_weights = mc_weights.flatten()

        # Evaluate the model
        psi, d_psi = fcn_and_grad( model, R, xyz ) # (B, N) and (B, N, 3)

        # Compute the loss
        xyz_ext = xyz[None,:,:] # (1, N, 3)
        r1 = pt.linalg.norm(xyz_ext - P1, dim=2).clamp_min( 1e-8 )
        r2 = pt.linalg.norm(xyz_ext - P2, dim=2).clamp_min( 1e-8 )
        d_psi_term = 0.5 * pt.sum( d_psi * d_psi, dim=2 )
        V_term = ( 1.0 / r1 + 1.0 / r2 ) * psi**2  #(B, N)
        
        numerator = pt.sum( (d_psi_term - V_term) * mc_weights[None,:], dim=1 )
        denominator = pt.sum( psi**2 * mc_weights[None,:], dim=1 )
        electron_energy = numerator / denominator

        loss_val = electron_energy.mean()

        # Log some things to the training routine
        return loss_val

def fcn_and_grad(model, R : pt.Tensor, 
                        xyz: pt.Tensor ):
    """
    R: (B,)
    xyz: (N,3)
    
    Returns:
        psi:      (B, N)
        grad_psi: (B, N, 3)
                  grad_psi[b, n, :] = d psi(R_b, xyz_n) / d xyz_n
    """
    # Make `xyz` a leaf tensor.
    xyz = xyz.detach().clone().requires_grad_(True)

    B = R.shape[0]

    # psi[b, n] = psi_Rb(xyz_n)
    psi = model(R, xyz)  # (B, N)

    grad_psi_all = []
    for b in range(B):
        grad_b = pt.autograd.grad( outputs=psi[b].sum(), inputs=xyz, 
                                   create_graph=True, retain_graph=True )[0]  # (N, 3)
        grad_psi_all.append(grad_b)
    grad_psi = pt.stack(grad_psi_all, dim=0)  # (B, N, 3)

    return psi, grad_psi