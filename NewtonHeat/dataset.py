import torch as pt
import math
from torch.utils.data import Dataset

from typing import Tuple

class NewtonDataset(Dataset):
    def __init__(self, N : int,
                      T_max : float,
                      tau_max : float,
                      device=pt.device('cpu'), 
                      dtype=pt.float32):
        super().__init__()

        self.device = device
        self.dtype = dtype

        # Sample T0, T_inf and log_k unformly
        gen = pt.Generator( device=device )
        self.T0 = pt.empty((N,1), device=device, dtype=dtype, requires_grad=False).uniform_( -T_max, T_max, generator=gen )
        logk = pt.empty((N,1), device=device, dtype=dtype, requires_grad=False).uniform_( math.log(1e-2), math.log(1e2), generator=gen )
        self.k = pt.exp( logk )
        self.T_inf = pt.empty((N,1), device=device, dtype=dtype, requires_grad=False).uniform_( -T_max, T_max, generator=gen )

        # Also sample the time-points
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
    T_max = 1.0
    tau_max = 4.0
    dataset = NewtonDataset( N, T_max, tau_max )
    print(dataset[3], len(dataset))