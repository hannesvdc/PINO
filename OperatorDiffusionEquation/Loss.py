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
    def __init__( self, eps=1e-6 ):
        super().__init__()
        self.eps = eps

    def forward(self, model, x, t, params, u0):
        if x.ndim == 1: x = x[:, None]
        if t.ndim == 1: t = t[:, None]

        x = x.requires_grad_(True)
        t = t.requires_grad_(True)

        # params: (Bt,2), u0: (Bb,N)
        T = model(x, t, params, u0)  # (Bb,Bt)

        Bb, Bt = T.shape
        dT_dt_list = []
        dT_dxx_list = []

        for b in range(Bb):
            Tb = T[b, :].unsqueeze(1)  # (Bt,1)

            dTb_dt = pt.autograd.grad(
                outputs=Tb,
                inputs=t,
                grad_outputs=pt.ones_like(Tb),
                create_graph=True,
                retain_graph=True,
            )[0]  # (Bt,1)

            dTb_dx = pt.autograd.grad(
                outputs=Tb,
                inputs=x,
                grad_outputs=pt.ones_like(Tb),
                create_graph=True,
                retain_graph=True,
            )[0]  # (Bt,1)

            dTb_dxx = pt.autograd.grad(
                outputs=dTb_dx,
                inputs=x,
                grad_outputs=pt.ones_like(dTb_dx),
                create_graph=True,
                retain_graph=True,
            )[0]  # (Bt,1)

            dT_dt_list.append(dTb_dt.squeeze(1))     # (Bt,)
            dT_dxx_list.append(dTb_dxx.squeeze(1))   # (Bt,)

        dT_dt = pt.stack(dT_dt_list, dim=0)    # (Bb,Bt)
        dT_dxx = pt.stack(dT_dxx_list, dim=0)  # (Bb,Bt)

        k = params[:, 0].unsqueeze(0)  # (1,Bt)

        eq = dT_dt / (k + self.eps) - dT_dxx  # (Bb,Bt)
        loss = (eq**2).mean()

        with pt.no_grad():
            rms = (eq**2).mean().sqrt()
            Tt_rms = ((dT_dt / (k + self.eps))**2).mean().sqrt()
            Txx_rms = (dT_dxx**2).mean().sqrt()

        return loss, {"rms": rms, "T_t_rms": Tt_rms, "T_xx_rms": Txx_rms}