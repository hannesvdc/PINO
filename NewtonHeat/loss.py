import torch as pt
import torch.nn as nn

class PINOLoss( nn.Module ):
    def __init__( self, eps=1e-12 ):
        super().__init__()
        self.eps = eps

    def forward(self,
                model : nn.Module,
                t : pt.Tensor,
                p : pt.Tensor) -> pt.Tensor:
        k = p[:,1:2]
        T_inf = p[:,2:]

        # Propagate through the model
        t = t.requires_grad_(True)
        T_t = model( t, p )

        # Calculate the loss
        dT_t = pt.autograd.grad( outputs=T_t, 
                                 inputs=t, 
                                 grad_outputs=pt.ones_like(T_t),
                                 create_graph=True )[0]
        eq = dT_t / (k + self.eps) + (T_t - T_inf)
        loss = pt.mean( eq**2 )

        return loss