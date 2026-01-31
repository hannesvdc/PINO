import torch as pt
import torch.nn as nn

    
class FixedInitialPINN( nn.Module ):
    """
    Physics-Informed Neural Operator for predicting the temperature time evolution
    in accordance to the heat eqauation. The equation is

    $$
        T_t(x, t) = \kappa T_{xx}(x, t)
    $$
    
    subject to a fixed initial temperature profile $T(x, 0) = T_0(x)$ and Dirichlet boundary conditions 
    $T(0,t) = T(1,t) = T_s$ where $T_s$ is the temperature of the surrounding heat bath.

    This PINO adds in a lot of physics into the model. Given input parameters $(\kappa, T_s)$,
    the output is $(T(x,t) - T_s) / T_max  =  u_0(x) + t * k * x * (1-x) * NN(t*k, x)$ with
    $u_0(x) = (T_0(x) - T_s) / T_max$.
    """
    def __init__( self, 
                  n_hidden_layers : int, 
                  z : int, 
                  T_max : float,
                  tau_max : float,
                  T0 : pt.Tensor,
                  x_grid : pt.Tensor ):
        super().__init__()
        self.T_max = T_max
        self.tau_max = tau_max

        n_grid_points = pt.numel( T0 )
        self.T0 = pt.reshape( T0, [1, n_grid_points] )
        self.x_grid = pt.reshape( x_grid, [1, n_grid_points] )

        # Hidden layers with differentiable activation function
        self.n_hidden_layers = n_hidden_layers
        hidden_layers = []
        self.act = nn.Tanh()
        for n in range(n_hidden_layers):
            n_inputs = 2 if n == 0 else z # (x, tau)
            hidden_layers.append( nn.Linear(n_inputs, z, bias=True) )
        self.layers = nn.ModuleList( hidden_layers )

        # Linear output layer
        self.output_layer = nn.Linear(z, 1, bias=True)

    def evaluateT0(self, x : pt.Tensor ) -> pt.Tensor:
        diff = pt.abs( x - self.x_grid ) # (B, N_grid) 
        indices = pt.argmin( diff, dim=1) # (B,)
        return self.T0[0,:].gather(0, indices)[:,None]

    # Assumes the parameters are p = (k, T_s)
    def forward(self, x : pt.Tensor, # (B, 1)
                      t : pt.Tensor, # (B, 1)
                      p : pt.Tensor, # (B, 2)
                ) -> pt.Tensor:
        assert p.shape[1] == 2, "This model expects two parameters (k, T_s)"

        # Pre-process the parameters and input
        T_s = p[:,1:2]
        k = p[:,0:1]
        tau = t * k
        tau_hat = tau / self.tau_max
        
        # Pre-process the parameters and input
        T0_at_x = self.evaluateT0( x )
        u0_at_x = (T0_at_x - T_s) / self.T_max # Shape (B, 1)

        # Push through the network
        y = pt.cat((x, tau_hat), dim=1)
        for n in range( self.n_hidden_layers ):
            y = self.act( self.layers[n](y) )

        # Form the output 
        g = self.output_layer( y )
        u_hat = u0_at_x + tau * x * (1.0 - x) * g
        return T_s + u_hat * self.T_max