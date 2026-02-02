import torch as pt
import torch.nn as nn

class LinearTimeFactor( nn.Module ):
    def __init__( self ):
        super().__init__()

    def forward( self, tau : pt.Tensor ) -> pt.Tensor:
        return tau
    
class ExponentialTimeFactor( nn.Module ):
    def __init__( self, alpha : float = 2.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, tau : pt.Tensor ) -> pt.Tensor:
        return 1.0 - pt.exp( -self.alpha * tau )

class PINO( nn.Module ):
    """
    Physics-Informed Neural Operator for predicting the temperature time evolution
    in accordance to Newton's heat law. The equation is
        dT / dt = -k * (T(t) - T_inf )
    which has an exact solution T(t) = T_inf + exp(-k*t) * (T0 - T_inf)

    The goal of this simple PINO is to essentially learn the solution curve T(t). A regular 
    but simple physics-informed network represents the solution as 
         T(t) = T_0 + t * NN(t*k; T0, k, T_inf)
    where multiplying by `t` is required to automatically satisfy the Dirichlet 
    boundary condition. This PINN knows nothing about the decay rate or the 
    steady-state temperature T_inf. The only thing it really does is input and output 
    normalization.
    """
    def __init__( self, 
                  n_hidden_layers : int, 
                  z : int, 
                  T_max : float,
                  tau_max : float,
                  logk_min : float,
                  logk_max : float,
                  time_factor : str = "linear" ):
        super().__init__()
        self.T_max = T_max
        self.tau_max = tau_max
        self.logk_min = logk_min
        self.logk_max = logk_max

        # Build the time factor module
        if time_factor == "linear":
            self.time_factor = LinearTimeFactor()
        elif time_factor == "exponential":
            self.time_factor = ExponentialTimeFactor()
        else:
            print("Time factor type not recognized. Must be 'linear' or 'exponential'.")
            return

        # Hidden layers with differentiable activation function
        self.n_hidden_layers = n_hidden_layers
        hidden_layers = []
        self.act = nn.Tanh()
        for n in range(n_hidden_layers):
            n_inputs = 4 if n == 0 else z # (tau, log k, T0_hat, T_s_hat)
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
        T_s_hat = p[:,2:3] / self.T_max
        k = p[:,1:2].clamp_min(1e-12)
        logk_hat = (pt.log(k) - self.logk_min) / (self.logk_max - self.logk_min)

        # Pass through the hidden layers
        tau = t * k
        tau_hat = tau / self.tau_max
        x = pt.cat( (tau_hat, logk_hat, T0_hat, T_s_hat), dim=1 )
        for n in range( self.n_hidden_layers ):
            x = self.layers[n](x)
            x = self.act( x )

        # Calculate the final output and enforce the Dirichlet boundary conditions.
        x = self.output_layer( x ) # (B,) learns the temperature delta (in T_max units)
        T_hat = T0_hat + self.time_factor( tau ) * x
        return T_hat * self.T_max