import torch as pt
import torch.nn as nn

from collections import OrderedDict
from typing import Tuple

class FiLMLayer( nn.Module ):

    def __init__( self, z : int ):
        super().__init__()

        self.z = z
        self.hidden_layer = nn.Linear(3, 2*z, bias=True)
        self.act = nn.ReLU()
        self.output_layer = nn.Linear(2*z, 2*z, bias=True)

        # Initialize near gamma = 1, beta = 0
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    # Assumes the model forward parameters are p = (T0 - T_inf, ln k, T_inf )
    def forward(self, p : pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        p = self.act( self.hidden_layer( p ) )
        p = self.output_layer( p )
        
        gamma_raw = p[:,0:self.z]
        beta = p[:,self.z:]
        return 1.0 + gamma_raw, beta
    
class PINO( nn.Module ):
    """
    Physics-Informed Neural Operator for predicting the temperature time evolution
    in accordance to Newton's heat law. The equation is
        dT / dt = -k * (T(t) - T_inf )
    which has an exact solution T(t) = T_inf + exp(-k*t) * (T0 - T_inf)

    The goal of the PINO is to essentially learn the exponential. Given input parameters (T0, k, T_inf),
    the PINO output is T(t) = (T0 - T_inf) + t * NN(t*k; T0, k, T_inf) + T_inf
    So it learns a normalized offset with respect to T_inf
    """
    def __init__( self, 
                  n_hidden_layers : int, 
                  z : int, 
                  T_max : float,
                  tau_max : float,
                  logk_min : float,
                  logk_max : float ):
        super().__init__()
        self.T_max = T_max
        self.tau_max = tau_max
        self.logk_min = logk_min
        self.logk_max = logk_max

        # Different FiLM module per layer
        self.FiLM_layers = nn.ModuleList( FiLMLayer(z) for _ in range(n_hidden_layers) )

        # Hidden layers with differentiable activation function
        self.n_hidden_layers = n_hidden_layers
        hidden_layers = []
        self.act = nn.Tanh()
        for n in range(n_hidden_layers):
            n_inputs = 1 if n == 0 else z
            hidden_layers.append( nn.Linear(n_inputs, z, bias=True) )
        self.layers = nn.ModuleList( hidden_layers )

        # Linear output layer
        self.output_layer = nn.Linear(z, 1, bias=True)

    # Assumes the parameters are p = (T0, k, T_inf)
    def forward(self, t : pt.Tensor,
                      p : pt.Tensor) -> pt.Tensor:
        assert p.shape[1] == 3, "This model expects three parameters (T0, k, T_inf)" 
        
        # Preprocess the parameters
        T0_hat = p[:,0:1] / self.T_max
        T_hat_inf = p[:,2:3] / self.T_max
        k = p[:,1:2].clamp_min(1e-12)
        logk_hat = (pt.log(k) - self.logk_min) / (self.logk_max - self.logk_min)
        p_film = pt.cat( ( T0_hat - T_hat_inf, logk_hat, T_hat_inf ), dim=1 )

        # Pass through the hidden layers
        tau = t * k
        x = tau / self.tau_max
        for n in range( self.n_hidden_layers ):
            gamma, beta = self.FiLM_layers[n]( p_film )
            x = gamma * self.layers[n](x) + beta
            x = self.act( x )

        # Calculate the final output and enforce the Dirichlet boundary conditions.
        x = self.output_layer( x ) # (B,) learns the temperature delta (in T_max units)
        T_hat = (T0_hat - T_hat_inf) + t * x + T_hat_inf
        return T_hat * self.T_max

class PINOLoss( nn.Module ):
    def __init__( self, eps=1e-12 ):
        super().__init__()
        self.eps = eps

    def forward(self,
                model : PINO,
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
        