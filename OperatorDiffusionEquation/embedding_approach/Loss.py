import torch as pt
import torch.nn as nn

class HeatLoss( nn.Module ):
    """
    PDE residual loss for tensorized DeepONet output.

    Model signature:
        T = model(x, t, params, u0)
    where:
        x:  (B,)  or  (B, 1)  requires_grad True
        t:  (B,)  or  (B, 1)  requires_grad True
        params:  (B, 2)  [k, T_s]
        u0:   (B, n_grid_points)

    Model output:
        T: (B,)
    """
    def __init__( self, eps=1e-10 ):
        super().__init__()
        self.eps = eps

    def forward(self, model, x, t, params, u0):
        if x.ndim == 1: x = x[:, None]
        if t.ndim == 1: t = t[:, None]

        x = x.requires_grad_(True)
        t = t.requires_grad_(True)
        params = params.requires_grad_(True)
        u0 = u0.requires_grad_(True)

        # params: (B,2), u0: (B, n_grid_points)
        T = model(x, t, params, u0)  # (B,1)

        dT_dt = pt.autograd.grad(
                outputs=T,
                inputs=t,
                grad_outputs=pt.ones_like(T),
                create_graph=True,
            )[0]  # (B,1)

        dT_dx = pt.autograd.grad(
                outputs=T,
                inputs=x,
                grad_outputs=pt.ones_like(T),
                create_graph=True,
            )[0]  # (B,1)

        dT_dxx = pt.autograd.grad(
                outputs=dT_dx,
                inputs=x,
                grad_outputs=pt.ones_like(dT_dx),
                create_graph=True,
            )[0]  # (B,1)

        k = params[:, 0:1]
        eq = dT_dt / (k + self.eps) - dT_dxx  # (B,1)
        # eq = dT_dt - k * dT_dxx
        loss = ( eq**2 ).mean()

        with pt.no_grad():
            rms = (eq**2).mean().sqrt()
            Tt_rms = ( (dT_dt / k)**2 ).mean().sqrt()
            Txx_rms = ( dT_dxx**2 ).mean().sqrt()

        return loss, {"rms": float(rms), "T_t_rms": float(Tt_rms), "T_xx_rms": float(Txx_rms)}