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
        self.x = pt.rand( (N,1), dtype=dtype, requires_grad=False, generator=gen)

        # Sample tau uniformly
        u = pt.rand((N, 1), dtype=dtype, generator=gen)
        tau = tau_max * u

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