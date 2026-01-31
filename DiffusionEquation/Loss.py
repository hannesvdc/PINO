import torch as pt
import torch.nn as nn

from typing import Tuple

class HeatLoss( nn.Module ):
    def __init__( self, eps=1e-12 ):
        super().__init__()
        self.eps = eps

    def forward(self,
                model : nn.Module,
                x : pt.Tensor,
                t : pt.Tensor,
                p : pt.Tensor) -> Tuple[pt.Tensor,pt.Tensor,pt.Tensor,pt.Tensor]:
        k = p[:,0:1]

        # Propagate through the model
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)
        p = p.requires_grad_(False)
        T_t = model( x, t, p )

        # Calculate the loss
        dT_t = pt.autograd.grad( outputs=T_t, 
                                 inputs=t, 
                                 grad_outputs=pt.ones_like(T_t),
                                 create_graph=True )[0]
        dT_x = pt.autograd.grad( outputs=T_t,
                                 inputs=x,
                                 grad_outputs=pt.ones_like(T_t),
                                 create_graph=True)[0]
        dT_xx = pt.autograd.grad(outputs=dT_x,
                                 inputs=x,
                                 grad_outputs=pt.ones_like(dT_x),
                                 create_graph=True)[0]
        eq = dT_t / (k + 1e-8) -  dT_xx # Not sure if this is the best formulation, perhaps we need to divide by k to regularize
        loss = pt.mean( eq**2 )

        # also return some diagnostics
        rms = pt.mean( eq**2 ).sqrt()
        T_t_rms = pt.mean( (dT_t / k)**2 ).sqrt()
        T_xx_rms = pt.mean( dT_xx**2 ).sqrt()

        return loss, rms, T_t_rms, T_xx_rms