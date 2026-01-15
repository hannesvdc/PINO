import torch as pt
import torch.nn as nn

from torch.func import jacrev, vmap # type: ignore
from typing import Tuple

class PhysicsLoss(nn.Module):
    """
    Plane-stress formulation using prefactor = E/(1-nu^2).

    Inputs:
      model: ElasticityFiLMPINN that returns actual uv (Dirichlet already enforced)
      g_batch: (B, 2, m)  where channel 0 = g_x(y_i), channel 1 = g_y(y_i)
      xy_int: (N_int, 2)
      xy_forcing: (m, 2) points on x=1, ordered to match y_i used in g_batch
    """
    def __init__(self, 
                 E: float, 
                 nu: float, 
                 w_int: float = 1.0, 
                 w_forcing: float = 1.0):
        super().__init__()

        self.E = float(E)
        self.nu = float(nu)
        self.pref = self.E / (1.0 - self.nu**2)
        self.w_int = float(w_int)
        self.w_forcing = float(w_forcing)

    def setWeights(self, w_int: float, w_forcing: float):
        self.w_int = float(w_int)
        self.w_forcing = float(w_forcing)

    def forward( self,
                 model: nn.Module,
                 g_batch: pt.Tensor,      # (B, 2, m)
                 xy_int: pt.Tensor,       # (N_int, 2)
                 xy_forcing: pt.Tensor,   # (m, 2)
                 ) -> pt.Tensor:

        device = g_batch.device
        dtype = g_batch.dtype

        # Interior PDE residual
        if xy_int.numel() == 0:
            loss_int = pt.zeros((), device=device, dtype=dtype)
        else:
            _, H = self.grads_and_hess(model, g_batch, xy_int, needsHessian=True)

            # H: (B, N_int, 2, 2, 2) with indices [comp, d1, d2]
            u_xx = H[:, :, 0, 0, 0]
            u_xy = H[:, :, 0, 0, 1]
            u_yy = H[:, :, 0, 1, 1]
            v_xx = H[:, :, 1, 0, 0]
            v_xy = H[:, :, 1, 0, 1]
            v_yy = H[:, :, 1, 1, 1]

            # Strong-form residuals
            res_u = self.pref * (u_xx + 0.5*(1.0 + self.nu)*v_xy + 0.5*(1.0 - self.nu)*u_yy)
            res_v = self.pref * (v_yy + 0.5*(1.0 + self.nu)*u_xy + 0.5*(1.0 - self.nu)*v_xx)

            loss_int = res_u.square().mean() + res_v.square().mean()

        # Right boundary traction BC
        if xy_forcing.numel() == 0:
            loss_forcing = pt.zeros((), device=device, dtype=dtype)
        else:
            # g_x, g_y: (B, m)
            g_x = g_batch[:, 0, :]
            g_y = g_batch[:, 1, :]

            J, _ = self.grads_and_hess(model, g_batch, xy_forcing, needsHessian=False)

            # J: (B, m, 2, 2) with indices [comp, dir]
            u_x = J[:, :, 0, 0]
            u_y = J[:, :, 0, 1]
            v_x = J[:, :, 1, 0]
            v_y = J[:, :, 1, 1]

            # Neumann residuals
            bc_u = self.pref * (u_x + self.nu * v_y) + g_x
            bc_v = self.pref * (1.0 - self.nu) / 2.0 * (u_y + v_x) + g_y

            loss_forcing = bc_u.square().mean() + bc_v.square().mean()

        return self.w_int * loss_int + self.w_forcing * loss_forcing

    def grads_and_hess( self,
                        model: nn.Module,
                        g_batch: pt.Tensor,   # (B, 2, m)
                        xy: pt.Tensor,        # (N, 2)
                        needsHessian: bool = True
                        ) -> Tuple[pt.Tensor, pt.Tensor]:
        """
        Returns:
          J: (B, N, 2, 2) gradients d(u,v)/d(x,y)
          H: (B, N, 2, 2, 2) Hessians if requested
        """

        # Pointwise function for jacrev: inputs are single sample g, single point xy
        def uv_vec(g_single: pt.Tensor, xy_single: pt.Tensor) -> pt.Tensor:
            # g_single: (2, m), xy_single: (2,)
            uv = model(g_single.unsqueeze(0), xy_single.unsqueeze(0))
            # Expect model output shape either (1,1,2) or (1,2).
            if uv.ndim == 3:
                return uv[0, 0]   # (2,)
            elif uv.ndim == 2:
                return uv[0]      # (2,)
            else:
                raise RuntimeError(f"Unexpected model output shape: {uv.shape}")

        J_fn = jacrev(uv_vec, argnums=1)  # (2,2)

        # vmap over xy points (inner), and over batch g_single (outer)
        J = vmap(vmap(J_fn, (None, 0)), (0, None))(g_batch, xy)  # (B, N, 2, 2)

        if needsHessian:
            H_fn = jacrev(J_fn, argnums=1)  # (2,2,2)
            H = vmap(vmap(H_fn, (None, 0)), (0, None))(g_batch, xy)  # (B, N, 2, 2, 2)
        else:
            H = pt.empty(0, device=g_batch.device, dtype=g_batch.dtype)

        return J, H