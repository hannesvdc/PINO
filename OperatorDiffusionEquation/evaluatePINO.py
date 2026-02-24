import torch as pt

from typing import Tuple

@pt.no_grad()
def evaluatePINO( model : pt.nn.Module, 
                  x_grid : pt.Tensor, 
                  T_f : float, 
                  p : pt.Tensor, 
                  u0 : pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
    N_tau = 1001
    tau_grid = pt.linspace( 1e-2, T_f, N_tau )
    T_sol = evaluatePINOAt( model, x_grid, tau_grid, p, u0 )

    return T_sol, tau_grid

@pt.no_grad()
def evaluatePINOAt( model : pt.nn.Module, 
                   x_grid : pt.Tensor, 
                   tau_values: pt.Tensor, 
                   p : pt.Tensor, 
                   u0 : pt.Tensor) -> pt.Tensor:
    B = x_grid.shape[0]
    k = p[0]
    p_batch = pt.unsqueeze(p, 0).expand( [B, len(p)] )
    u0_batch = pt.unsqueeze(u0, 0).expand( [B, u0.numel() ])

    # Evaluate the network in a fixed grid of tau-values
    N_tau = tau_values.numel()
    N_grid_points = len( x_grid )
    T_sol = pt.zeros( (N_grid_points, N_tau) )
    for t_idx in range( 0, N_tau ):
        if t_idx % 1000 == 0:
            print(t_idx)
        t = tau_values[t_idx] / k
        T_xt = model( x_grid[:,None].to(device=u0.device,dtype=u0.dtype), t * pt.ones([B,1],device=u0.device,dtype=u0.dtype), p_batch, u0_batch )
        T_sol[:,t_idx] = T_xt[:,0]
    
    return T_sol