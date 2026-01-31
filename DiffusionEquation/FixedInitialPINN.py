import torch as pt
import torch.nn as nn
    
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
                  u0 : pt.Tensor,
                  x_grid : pt.Tensor, 
                  l : float):
        super().__init__()
        self.T_max = T_max
        self.tau_max = tau_max

        # Store the initial condition and grid as non-autograd parameters
        n_grid_points = pt.numel( u0 )
        self.register_buffer( "u0", pt.reshape( u0, [1, n_grid_points] ) )
        self.register_buffer( "x_grid", pt.reshape( x_grid, [1, n_grid_points] ) )
        self.l = l
        self.build_rbf_interpolator()

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

    def build_rbf_interpolator( self ):
        N_grid = self.x_grid.shape[1]
        grid = pt.flatten(self.x_grid)

        kernel = lambda y, yp: pt.exp(-0.5 * (y - yp)**2 / self.l**2) # Covariance kernel
        grid_1, grid_2 = pt.meshgrid( grid, grid, indexing="ij" )
        K = kernel(grid_1, grid_2) + 1e-4 * pt.eye( N_grid, device=grid.device, dtype=grid.dtype ) 

        u0 = pt.transpose( self.u0, 0, 1 )
        L = pt.linalg.cholesky( K )
        alpha = pt.cholesky_solve( u0, L ) # shape (N_grid,)
        self.register_buffer("alpha", alpha) # shape (N_grid,1)

    # Need to make sure the PINN output is fully differentiable w.r.t. x and t.
    # This means evaluating u0 through interpolation, even if we only actually
    # evaluate it in grid points (for now). 
    #
    # We use a radial basis interpolator: u0(x) = K(x, X_grid) alpha
    def evaluate_u0(self, x : pt.Tensor ) -> pt.Tensor:        
        K_x_grid = pt.exp( -0.5 * (x - self.x_grid)**2 / self.l**2 )
        return K_x_grid @ self.alpha

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
        u0_at_x = self.evaluate_u0( x )

        # Push through the network
        y = pt.cat((x, tau_hat), dim=1)
        for n in range( self.n_hidden_layers ):
            y = self.act( self.layers[n](y) )

        # Form the output 
        g = self.output_layer( y )
        u_hat = u0_at_x + tau * x * (1.0 - x) * g
        return T_s + u_hat * self.T_max