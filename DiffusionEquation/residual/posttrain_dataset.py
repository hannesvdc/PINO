import torch as pt
import math
from torch.utils.data import Dataset

from typing import Tuple

class PINNDataset(Dataset):
    def __init__(self, N : int,
                       T_max : float,
                       tau_max : float,
                       dtype = pt.float64,
                       test=False):
        super().__init__()

        self.T_max = T_max
        self.tau_max = tau_max
        self.logk_max = math.log(1e2)

        # Sample log_k and T_s uniformly
        gen = pt.Generator( )
        if test:
            import time
            gen.manual_seed( int(time.time()) )
        logk = pt.empty((N,1), dtype=dtype, requires_grad=False).uniform_( -self.logk_max, self.logk_max, generator=gen )
        self.k = pt.exp( logk )
        self.T_s = pt.empty((N,1), dtype=dtype, requires_grad=False).uniform_( -self.T_max, self.T_max, generator=gen )

        # Also sample space and time
        self.x = pt.rand( (N,1), dtype=dtype, requires_grad=False, generator=gen )

        # Put most tau closer to 0
        tau_min = 1e-2
        gamma = 2.0  # 1 = log-uniform, >1 biases small tau
        u = pt.rand((int(0.1*N),1), dtype=dtype, generator=gen)
        log_tau = math.log(tau_min) + (math.log(tau_max) - math.log(tau_min)) * (u ** gamma)
        tau = pt.exp(log_tau)

        # But also add uniform large values
        N_late = int(0.9 * N)  # 90% extra points
        u_late = pt.rand((N_late, 1), dtype=dtype, generator=gen)
        tau_late = 1.0 + (tau_max - 1.0) * u_late
        
        # concatenate
        tau = pt.cat([tau, tau_late], dim=0)
        self.t = tau / self.k

    def __len__( self ):
        return len( self.k )
    
    def __getitem__(self, idx : int) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        return self.x[idx], self.t[idx], pt.cat((self.k[idx], self.T_s[idx]), dim=0)
    
    def all( self ) -> pt.Tensor:
        return pt.cat( (self.x, self.t, self.k, self.T_s), dim=1 )
    
if __name__ == '__main__':
    N = 10_000
    T_max = 10.0
    tau_max = 10.0
    dataset = PINNDataset( N, T_max, tau_max )
    print(dataset[3], len(dataset))
    print(dataset.all())