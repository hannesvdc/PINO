import torch as pt
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim.optimizer import Optimizer

class SelfScaledBroyden(Optimizer):
    def __init__(self, params):
        super().__init__(params, dict())

        self.params = list(params)
        print('len(self.params)', len(self.params))
        self.n_params = sum(p.numel() for group in self.params for p in group['params'])
        self.Hk = pt.eye(self.n_params)
        print('number of params', self.n_params)

        # Line search hyperparameters
        self.c1 = 1.e-4
        self.c2 = 0.9
        self.rho = 0.9

        # Broyden hyperparamters
        self.theta = 0.5

    @pt.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        assert closure is not None, "SSBroyden requires a closure function that re-evaluates the model and returns the loss."

        # Compute the current loss, gradient and search direction
        x_k = parameters_to_vector(self.params).detach()
        f_k = closure() 
        df_k = parameters_to_vector([p.grad for p in self.params]).detach()
        p_k = - pt.mv(self.Hk, df_k)

        # Do Line search
        alpha_k = self.backtracking_wolfe(closure, x_k, f_k, p_k, df_k)
        s_k = - alpha_k * p_k
        x_kp1 = x_k + s_k
        vector_to_parameters(x_kp1, self.params)

        # Compute the new loss and gradient
        f_kp1 = closure()
        df_kp1 = parameters_to_vector([p.grad for p in self.params]).detach()
        y_k = df_kp1 - df_k

        # Some useful variables for later computations
        Bksk = -alpha_k * df_k
        Hkyk = pt.mv(self.Hk, y_k)
        yksk = pt.inner(y_k, s_k)
        ykHkyk = pt.inner(y_k, Hkyk)

        # Calculate the necessary paramater values before the Broyden update
        b_k = pt.inner(s_k, Bksk) / yksk
        h_k = ykHkyk / yksk
        phi_k = (1.0 - self.theta) / (1.0 + self.theta*(h_k*b_k - 1.0))
        tau_k = min(1.0, 1.0 / b_k)
        v_k = s_k / yksk - Hkyk / ykHkyk

        # Update the inverse Hessian approximation
        term1 = pt.outer(Hkyk, pt.mv(self.Hk.T, y_k)) / pt.inner(y_k, Hkyk)
        term2 = phi_k * ykHkyk * pt.outer(v_k, v_k)
        term3 = pt.outer(s_k, s_k) / yksk
        self.Hk = 1.0/tau_k * (self.Hk - term1 + term2) + term3

        # Return the new loss, maybe it is useful
        return f_kp1
    
    def backtracking_wolve(self, closure, xk, f_k, pk, df_k):
        alpha = 1.0
        pkdfk = pt.inner(pk, df_k)

        while True:
            # Compute the current trial
            x_new = xk + alpha * pk
            vector_to_parameters(x_new, self.params)
            f_new = closure()
            df_new = parameters_to_vector([p.grad for p in self.params]).detach()

            # Check if the Wolfe conditions are fulfilled
            cond1 = (f_new < f_k + self.c1 * alpha * pkdfk)
            cond2 = (pt.abs(pt.inner(df_new, pk)) < self.c2 * pkdfk)
            if cond1 and cond2:
                return alpha
            else:
                alpha = self.rho * alpha