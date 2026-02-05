import torch as pt
import torch.nn as nn

from typing import Callable
    
class FixedInitialPINN( nn.Module ):
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
                  u0_fcn : Callable[[pt.Tensor], pt.Tensor]):
        super().__init__()
        self.T_max = T_max
        self.tau_max = tau_max
        self.logk_max = logk_max

        # u0 evaluation function (should be on the same device as the network)
        self.u0_fcn = u0_fcn

        # Hidden layers with differentiable activation function
        self.n_hidden_layers = n_hidden_layers
        hidden_layers = []
        self.act = nn.Tanh()
        for n in range(n_hidden_layers):
            n_inputs = 4 if n == 0 else z # (x, tau, log k, u0_at_x)
            hidden_layers.append( nn.Linear(n_inputs, z, bias=True) )
        self.layers = nn.ModuleList( hidden_layers )

        # Linear output layer
        self.output_layer = nn.Linear(z, 1, bias=True)

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
        for n in range( self.n_hidden_layers ):
            y = self.act( self.layers[n](y) )

        # Form the output 
        g = self.output_layer( y )
        beta_tau = tau / (1.0 + tau)
        u_hat = u0_at_x + beta_tau * x * (1.0 - x) * g

        return T_s + u_hat * self.T_max