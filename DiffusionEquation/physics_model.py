import torch as pt
import torch.nn as nn

    
class AdvanedPhysicsPINO( nn.Module ):
    """
    Physics-Informed Neural Operator for predicting the temperature time evolution
    in accordance to the heat eqauation. The equation is
        T_t(x, t) = \kappa T_[xx}(x, t)
    subject to an initial temperature profile T(x, 0) = T_0(x) and Dirichlet boundary conditions 
    T(0,t) = T(1,t) = T_s where T_s is the temperature of the surrounding heat bath.

    This PINO adds in a lot of physics into the model. Given input parameters (T0(x), \kappa, T_s),
    the output is (T(x,t) - T_s) / T_max  =  u_0(x) + t*k * x * (1-x) * NN(t*k, x; u_0) with
    u_0(x) = (T_0(x) - T_s) / T_max.
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
            n_inputs = 3 if n == 0 else z # (tau, x, u_0)
            hidden_layers.append( nn.Linear(n_inputs, z, bias=True) )
        self.layers = nn.ModuleList( hidden_layers )

        # Linear output layer
        self.output_layer = nn.Linear(z, 1, bias=True)

    # Assumes the parameters are p = (T0, k, T_s)
    def forward(self, x : pt.Tensor,
                      t : pt.Tensor,
                      p : pt.Tensor) -> pt.Tensor:
        assert p.shape[1] == 3, "This model expects three parameters (T0(X), k, T_s)" 
        
        # Pre-process the parameters and input
        T0_at_x = p[:,0:1]
        T_s = p[:,2:3]
        u0 = (T0_at_x - T_s) / self.T_max
        k = p[:,1:2].clamp_min(1e-12)
        tau = t * k
        tau_hat = tau / self.tau_max

        # Push through the network
        y = pt.cat((tau_hat, x, u0), dim=1)
        for n in range( self.n_hidden_layers ):
            y = self.act( self.layers[n](y) )

        g = self.output_layer( y )
        u_hat =  u0 + tau * x * (1.0 - x) * g
        return T_s + u_hat * self.T_max

        # Preprocess the parameters
        # T0_hat = p[:,0:1] / self.T_max
        # T_hat_inf = p[:,2:3] / self.T_max
        # k = p[:,1:2].clamp_min(1e-12)
        # logk_hat = (pt.log(k) - self.logk_min) / (self.logk_max - self.logk_min)
        # p_film = pt.cat( ( T0_hat - T_hat_inf, logk_hat, T_hat_inf ), dim=1 )

        # Pass through the hidden layers
        # tau = t * k
        # tau_hat = tau / self.tau_max
        # x = tau_hat
        # for n in range( self.n_hidden_layers ):
        #     gamma, beta = self.FiLM_layers[n]( p_film )
        #     x = gamma * self.layers[n](x) + beta
        #     x = self.act( x )

        # Calculate the final output and enforce the Dirichlet boundary conditions.
        x = self.output_layer( x ) # = \tilde{g}( k * t / tau_max )
        T_hat = T_hat_inf + (T0_hat - T_hat_inf) * (1.0 + tau_hat * x)
        return T_hat * self.T_max