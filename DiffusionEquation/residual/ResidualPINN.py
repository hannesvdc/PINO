import torch as pt
import torch.nn as nn

from typing import Callable

class ResidualLayer( nn.Module ):
    def __init__( self, n_in : int, z : int):
        super().__init__()

        self.linear1 = nn.Linear(n_in, z, bias=True)
        self.linear2 = nn.Linear(z, n_in, bias=True)
        self.act = nn.Tanh()

        # Start close to identity: residual branch initially ~0
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        
    def forward(self, x):
        y = self.act( self.linear1(x) )
        y = self.linear2(y)
        return x + y
    
class ResidualPINN( nn.Module ):
    """
    Physics-Informed Neural Operator for predicting the temperature time evolution
    in accordance to the heat eqauation. The equation is

    .. math::
        T_t(x, t) = \\kappa T_{xx}(x, t)
    
    subject to a fixed initial temperature profile :math:`T(x, 0) = T_0(x)` and Dirichlet boundary conditions 
    :math:`T(0,t) = T(1,t) = T_s` where :math:`T_s` is the temperature of the surrounding heat bath.

    This PINO adds in a lot of physics into the model. Given input parameters :math:`(\\kappa, T_s)`,
    the output is :math:`(T(x,t) - T_s) / T_max  =  u_0(x) + t * k * x * (1-x) * NN(t*k, x)` with
    :math:`u_0(x) = (T_0(x) - T_s) / T_max`.
    """
    def __init__( self, 
                  n_hidden_layers : int, 
                  z : int, 
                  T_max : float,
                  tau_max : float,
                  logk_max : float,
                  u0_fcn : Callable[[pt.Tensor], pt.Tensor],
                  ic_time_factor : bool = False):
        super().__init__()
        self.T_max = T_max
        self.tau_max = tau_max
        self.logk_max = logk_max

        # How to treat the initial condition
        self.u0_fcn = u0_fcn
        if ic_time_factor:
            self.ic_time_factor = lambda tau : 1. / (1. + tau)
        else:
            self.ic_time_factor = lambda tau: 1.0

        # Hidden layers with differentiable activation function
        self.act = nn.Tanh()
        self.n_hidden_layers = n_hidden_layers
        self.stem = nn.Linear( 4, z )
        hidden_layers = [ ]
        for _ in range(n_hidden_layers):
            hidden_layers.append( ResidualLayer(z, z) )
        self.head = nn.Linear( z, 1 )
        self.layers = nn.ModuleList( hidden_layers )

    # Assumes the parameters are p = (k, T_s)
    def forward(self, x : pt.Tensor, # (B, 1)
                      t : pt.Tensor, # (B, 1)
                      p : pt.Tensor, # (B, 2)
                ) -> pt.Tensor:
        assert p.shape[1] == 2, "This model expects two parameters (k, T_s)"

        # Pre-process the parameters and input
        k = p[:,0:1]
        logk_hat = pt.log(k) / self.logk_max # scale between -1 and 1
        tau = t * k
        tau_hat = tau / self.tau_max
        T_s = p[:,1:2]
        
        # Pre-process the parameters and input
        u0_at_x = self.u0_fcn( x )

        # Push through the network
        y = pt.cat((x, tau_hat, logk_hat, u0_at_x), dim=1)
        y = self.act( self.stem(y) )
        for n in range( self.n_hidden_layers ): # residual layers
            y = self.layers[n](y)
        g = self.head( y )

        # Form the output 
        beta_tau = tau / (1.0 + tau)
        u_hat = self.ic_time_factor(tau) * u0_at_x + beta_tau * x * (1.0 - x) * g

        return T_s + u_hat * self.T_max