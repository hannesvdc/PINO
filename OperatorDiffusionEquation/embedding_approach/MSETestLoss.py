import torch as pt
import torch.nn as nn

class MSETestLoss( nn.Module ):
   
    def __init__( self, eps=1e-6 ):
        super().__init__()
        self.eps = eps

    def forward(self, model : nn.Module, 
                      x : pt.Tensor, 
                      t : pt.Tensor, 
                      params : pt.Tensor, 
                      u0 : pt.Tensor ):
        if x.ndim == 1: x = x[:, None]
        if t.ndim == 1: t = t[:, None]

        x = x.requires_grad_(True)
        t = t.requires_grad_(True)

        # params: (Bt,2), u0: (Bb,N)
        T = model(x, t, params, u0)  # (Bb,Bt)

        reference = x**2
        loss = pt.mean( (T - reference)**2 )

        return loss, {"rms": None, "T_t_rms": None, "T_xx_rms": None}