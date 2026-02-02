import torch as pt
import math
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from typing import Tuple

class NewtonDataset(Dataset):
    def __init__(self, N : int,
                      T_max : float,
                      tau_max : float,
                      device=pt.device('cpu'), 
                      dtype=pt.float32,
                      biased_tau=False,
                      test=False):
        super().__init__()

        self.T_max = T_max
        self.tau_max = tau_max
        self.logk_min = math.log(1e-2)
        self.logk_max = math.log(1e2)

        # Sample T0, T_inf and log_k unformly
        if test:
            import time
            gen = pt.Generator( device=device )
            gen.manual_seed( int(time.time()) )
        else:
            gen = pt.Generator( device=device )
        self.T0 = pt.empty((N,1), device=device, dtype=dtype, requires_grad=False).uniform_( -self.T_max, self.T_max, generator=gen )
        logk = pt.empty((N,1), device=device, dtype=dtype, requires_grad=False).uniform_( self.logk_min, self.logk_max, generator=gen )
        self.k = pt.exp( logk )
        self.T_inf = pt.empty((N,1), device=device, dtype=dtype, requires_grad=False).uniform_( -self.T_max, self.T_max, generator=gen )

        # Also sample the time-points
        if biased_tau:
            log_tau = pt.randn( (N,1), device=device, dtype=dtype, generator=gen)
            log_tau = math.log(2.0) + 0.5*log_tau

            #tau_min = 1e-2
            #gamma = 2.0  # 1 = log-uniform, >1 biases small tau
            #u = pt.rand((N,1), device=device, dtype=dtype, generator=gen)
            #log_tau = math.log(tau_min) + (math.log(tau_max) - math.log(tau_min)) * (u ** gamma)
            tau = pt.exp(log_tau).clamp_max( tau_max )
            plt.hist(tau.cpu().numpy(), bins=100, density=True)
            plt.show()
        else:
            tau = tau_max * pt.rand( (N,1), device=device, dtype=dtype, requires_grad=False, generator=gen)
        self.t = tau / self.k

    def __len__( self ):
        return len( self.k )
    
    def __getitem__(self, idx : int) -> Tuple[pt.Tensor, pt.Tensor]:
        return self.t[idx], pt.cat((self.T0[idx], self.k[idx], self.T_inf[idx]), dim=0)
    
    def all( self ) -> pt.Tensor:
        return pt.cat( (self.t, self.T0, self.k, self.T_inf), dim=1 )
    
if __name__ == '__main__':
    N = 100_000
    T_max = 10.0
    tau_max = 4.0
    dataset = NewtonDataset( N, T_max, tau_max )
    print(dataset[3], len(dataset))