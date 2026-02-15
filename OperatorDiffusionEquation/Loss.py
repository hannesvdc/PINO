import torch as pt
import torch.nn as nn

from ConvDeepONet import DeepONet

from typing import Tuple, Dict

class HeatLoss( nn.Module ):
    """
    PDE residual loss for tensorized DeepONet output.

    Model signature:
        T = model(x, t, params, u0)
    where:
        x:      (Bt, 1)  requires_grad True
        t:      (Bt, 1)  requires_grad True
        params:  (Bt, 2)  [k, T_s]
        u0:     (Bb, n_grid)

    Model output:
        T: (Bb, Bt)
    """
    def __init__( self, eps=1e-12 ):
        super().__init__()
        self.eps = eps

    def forward(self,
                model : DeepONet,
                x : pt.Tensor,
                t : pt.Tensor,
                p : pt.Tensor,
                u0 : pt.Tensor) -> Tuple[pt.Tensor, Dict]:
        k = p[:,0]

        # Propagate through the model
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)
        p = p.requires_grad_(False)
        u0 = u0.requires_grad_(False)
        T_t = model( x, t, p, u0 ) # Shape (Bb, Bt)

        # Pre-sum over branch to keep per-trunk-point derivatives
        # Tmean: (Bt,) This is mainly for efficiency reasons.
        Tsum = T_t.sum(dim=0) # shape (Bt,)

        # All shapes (Bt,1)
        dT_t = pt.autograd.grad( outputs=Tsum, 
                                 inputs=t, 
                                 grad_outputs=pt.ones_like(Tsum),
                                 create_graph=True,
                                 retain_graph=True)[0] # (Bt,)
        dT_x = pt.autograd.grad( outputs=Tsum,
                                 inputs=x,
                                 grad_outputs=pt.ones_like(Tsum),
                                 create_graph=True,
                                 retain_graph=True)[0] # (Bt,)
        dT_xx = pt.autograd.grad(outputs=dT_x,
                                 inputs=x,
                                 grad_outputs=pt.ones_like(dT_x),
                                 create_graph=True,
                                 retain_graph=True)[0] # (Bt,)

        # Broadcast to (Bb, Bt)
        dT_dt_mat  = dT_t[None, :]    # (1,Bt)
        dT_dxx_mat = dT_xx[None, :]   # (1,Bt)
        k_mat = k[None, :]        # (1,Bt)

        # Compute the PDE residual and average
        eq = dT_dt_mat / (k_mat + 1e-8) -  dT_dxx_mat
        loss = pt.mean( eq**2 )

        # also return some diagnostics
        rms = pt.mean( eq**2 ).sqrt()
        T_t_rms = pt.mean( (dT_dt_mat / k_mat)**2 ).sqrt()
        T_xx_rms = pt.mean( dT_dxx_mat**2 ).sqrt()
        return_dict = {"rms": rms, "T_t_rms" : T_t_rms, "T_xx_rms": T_xx_rms}

        return loss, return_dict