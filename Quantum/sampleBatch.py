import math
import torch as pt

from typing import Tuple

@pt.no_grad()
def sampleBatch( B : int, N : int, R_cutoff : float, gen : pt.Generator ) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
    log_R_min = math.log( 0.1 )
    log_R_max = math.log( 2.0 )
    log_R = log_R_min + (log_R_max - log_R_min) * pt.rand( (B,1), generator=gen )
    R = pt.exp( log_R )

    # Sample (x,y,z) normal with a wider variance on the x-axis. 
    sigma_x = 2.0
    sigma_y = 1.0
    sigma_z = 1.0
    x = pt.normal( pt.zeros((N,)), sigma_x*pt.ones((N,)), generator=gen )
    y = pt.normal( pt.zeros((N,)), sigma_y*pt.ones((N,)), generator=gen )
    z = pt.normal( pt.zeros((N,)), sigma_z*pt.ones((N,)), generator=gen )

    # Reject  at `R_cutoff`. The number of samples is not identical to `N` every run.
    xyz = pt.stack( (x,y,z), dim=1 )
    r_sq = pt.sum( xyz * xyz, dim=1 )
    inside_region = ( r_sq <= R_cutoff**2 )
    xyz = xyz[inside_region]

    # Compute the MC weights
    exponent = -0.5 * ( (xyz[:,0] / sigma_x) ** 2 + (xyz[:,1] / sigma_y) ** 2 + (xyz[:,2] / sigma_z) ** 2 )
    q = pt.exp(exponent)  # proportional to q(x)
    mc_weights = 1.0 / q.clamp_min(1e-12)
    mc_weights = mc_weights / mc_weights.mean()

    return R, xyz, mc_weights