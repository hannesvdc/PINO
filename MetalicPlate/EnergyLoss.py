import torch as pt
import torch.nn as nn
from torch.func import jacrev, vmap # type: ignore
from rbf_interpolation import buildCholeskyMatrix, tensorizedRBFInterpolator

from typing import Tuple, Dict

class EnergyLoss( nn.Module ):
    """
    PDE residual loss for tensorized DeepONet output.

    Model signature:
        u,v = model(x, y, nu, gx, gy)
    where:
        x:  (Bt,)  or  (Bt, 1)  requires_grad True
        y:  (Bt,)  or  (Bt, 1)  requires_grad True
        nu:  (Bb, 1)
        gx:   (Bb, n_grid_points)
        gy:   (Bb, n_grid_points)

    Model output:
        u, v: (Bb,Bt)
    """
    def __init__( self, y_grid : pt.Tensor,
                        l : float,
                        branch_chunk : int = 4 ):
        super().__init__()

        self.l = l

        # Build the interpolating cholesky matrix.
        self.register_buffer( "y_grid", y_grid )
        cholesky_L = buildCholeskyMatrix( self.y_grid, self.l )
        self.register_buffer( "cholesky_L", cholesky_L )

        # For memory performance
        self.branch_chunk = branch_chunk

    def forward(self, model : nn.Module, 
                      x : pt.Tensor, # (Bt,1)
                      y : pt.Tensor, # (Bt,1)
                      mc_weights : pt.Tensor, # (Bt,1)
                      nu : pt.Tensor, # (Bb,1)
                      bc_y : pt.Tensor, # (Bt_bc,1)
                      gx : pt.Tensor, # (Bb, n_grid_points)
                      gy : pt.Tensor, # (Bb, n_grid_points)
                ) -> Tuple[pt.Tensor, Dict]:
        if x.ndim == 1: x = x[:, None]
        if y.ndim == 1: y = y[:, None]
        if mc_weights.ndim == 2: mc_weights = mc_weights.flatten()
        if nu.ndim == 1: nu = nu[:, None]
        if bc_y.ndim == 1: bc_y = bc_y[:,None]

        Bb = gx.shape[0]
        Bt = x.shape[0]

        # Safety
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        nu = nu.requires_grad_(False)
        gx = gx.requires_grad_(False)
        gy = gy.requires_grad_(False)

        # Accumulators (keep them as tensors so autograd can flow)
        total_energy = x.new_zeros(())
        total_boundary = x.new_zeros(())
        total_weight = 0.0

        # Evaluate the model in the interior points and chunk over the branch inputs for memory reduction
        for s in range(0, Bb, self.branch_chunk):
            e = min(Bb, s + self.branch_chunk)
            w = float(e - s) / float(Bb)  # weight for mean reduction

            nu_c = nu[s:e,:]   # (Bc,1)
            gx_c = gx[s:e,:]   # (Bc,Ny)
            gy_c = gy[s:e,:]   # (Bc,Ny)

            # Evaluate model derivatives on interior points for this branch chunk
            _, _, u_x, u_y, v_x, v_y = grads_uv_jacrev(model, x, y, nu_c, gx_c, gy_c)  # (Bc,Bt)

            # Boundary values only (no spatial derivatives needed)
            bc_x = pt.ones_like( bc_y )
            u_bc, v_bc, u_xb, u_yb, v_xb, v_yb = grads_uv_jacrev(model, bc_x, bc_y, nu_c, gx_c, gy_c)

            # Interior strain energy
            nu_flat = nu_c[:, 0][:, None]            # (Bc,1) broadcasts over Bt
            C11 = 1.0 / (1.0 - nu_flat**2)
            C12 = nu_flat / (1.0 - nu_flat**2)
            C66 = 0.5 / (1.0 + nu_flat)
            strain_energy = 0.5 * ( C11 * (u_x**2 + v_y**2) + 2.0*C12*u_x*v_y + C66 * (u_y + v_x)**2 )
            strain_integral = pt.sum(strain_energy * mc_weights[None,:], dim=1) / pt.sum( mc_weights )

            gx_int, gy_int = tensorizedRBFInterpolator( self.cholesky_L, self.y_grid, self.l, bc_y, gx_c, gy_c )                                                    # (Bc,Bt_bc) each
            bc_integral = (gx_int * u_bc + gy_int * v_bc).mean(dim=1)  # (Bc,)

            # Accumulate means (keep grads)
            total_energy = total_energy + w * strain_integral.mean()
            total_boundary = total_boundary + w * bc_integral.mean()
            total_weight += w

            # Traction loss for reference
            nu_b = nu_c[:, 0][:, None]  # (Bc,1) broadcast over Bt_bc
            t_x = (1.0 / (1.0 - nu_b**2)) * (u_xb + nu_b * v_yb)                 # (Bc,Bt_bc)
            t_y = (1.0 / (2.0 * (1.0 + nu_b))) * (u_yb + v_xb)                   # (Bc,Bt_bc)
            traction_residual = ((t_x - gx_int)**2 + (t_y - gy_int)**2).mean()  / ( gx_int**2 + gy_int**2 ).mean() # (Bc,)

            # free refs early for memory
            del u_x, u_y, v_x, v_y, u_bc, v_bc, gx_int, gy_int, strain_energy

        avg_energy = total_energy / total_weight
        avg_boundary = total_boundary / total_weight
        loss = avg_energy - avg_boundary
        loss_info = {"energy" : float(avg_energy.detach().item()), "boundary" : float(avg_boundary.detach().item()), 
                     "rms" : float(loss.detach().item()), "traction" : float(traction_residual.detach().item())}
        return loss, loss_info

def grads_uv_jacrev(model, x, y, nu, gx, gy):
    """
    x,y: (Bt,1)
    nu: (Bb,1)
    gx,gy: (Bb, n_grid_points)
    Returns: u,v and u_x,u_y,v_x,v_y each (Bb,Bt)
    """
    if x.ndim == 1: x = x[:, None]
    if y.ndim == 1: y = y[:, None]
    if nu.ndim == 1: nu = nu[:, None]

    # Pack points as (Bt,2)
    xy_points = pt.cat([x, y], dim=1)  # (Bt,2)

    # pointwise wrapper
    def uv_at_xy(xy):
        # xy: (2,)
        x1 = xy[0].view(1, 1)  # Bt=1
        y1 = xy[1].view(1, 1)
        u, v = model(x1, y1, nu, gx, gy)  # (Bb,1), (Bb,1) or (Bb,1) each
        return u[:, 0], v[:, 0]          # (Bb,), (Bb,)

    # Jacobian wrt xy for u and v separately: returns (Bb,2)
    Ju_fn = jacrev(lambda xy: uv_at_xy(xy)[0])  # d u_vec / d(xy)
    Jv_fn = jacrev(lambda xy: uv_at_xy(xy)[1])  # d v_vec / d(xy)

    # vmap over Bt points => (Bt,Bb,2)
    Ju = vmap(Ju_fn)(xy_points)  # (Bt,Bb,2)
    Jv = vmap(Jv_fn)(xy_points)  # (Bt,Bb,2)

    # Extract and transpose to (Bb,Bt)
    u_x = Ju[:, :, 0].T
    u_y = Ju[:, :, 1].T
    v_x = Jv[:, :, 0].T
    v_y = Jv[:, :, 1].T

    # Also compute u,v on all Bt points in one call (efficient)
    u_all, v_all = model(x, y, nu, gx, gy)  # (Bb,Bt) each

    return u_all, v_all, u_x, u_y, v_x, v_y