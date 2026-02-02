import torch as pt
import torch.nn as nn
    
class AdvanedPhysicsPINO( nn.Module ):
    """
    Physics-Informed Neural Operator for predicting the temperature time evolution
    in accordance to Newton's heat law. The equation is
        dT / dt = -k * (T(t) - T_inf )
    which has an exact solution T(t) = T_inf + exp(-k*t) * (T0 - T_inf)

    This PINO adds in more physics compared to the previous model. Given input parameters (T0, k, T_inf),
    the AdvanedPhysicsPINO output is T(t) =  T_inf + (T0 - T_inf) * (1 + t*k * NN(t*k; T0, k, T_inf))
    So it learns a normalized offset with respect to T_inf and is aware of the decay from T0 to T_inf.
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

        # Hidden layers with differentiable activation function
        self.n_hidden_layers = n_hidden_layers
        hidden_layers = []
        self.act = nn.Tanh()
        for n in range(n_hidden_layers):
            n_inputs = 3 if n == 0 else z
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
        k = p[:,1:2].clamp_min(1e-12)
        T_s_hat = p[:,2:3] / self.T_max
        logk_hat = (pt.log(k) - self.logk_min) / (self.logk_max - self.logk_min)

        # Pass through the hidden layers
        tau = t * k
        tau_hat = tau / self.tau_max
        x = pt.cat( (tau_hat, logk_hat, T0_hat - T_s_hat), dim=1 )
        for n in range( self.n_hidden_layers ):
            x = self.layers[n](x)
            x = self.act( x )

        # Calculate the final output and enforce the Dirichlet boundary conditions.
        x = self.output_layer( x ) # = \tilde{g}( k * t / tau_max )
        time_factor = 1.0 - pt.exp(-tau)
        T_hat = T0_hat + (T0_hat - T_s_hat) * time_factor * x
        return T_hat * self.T_max