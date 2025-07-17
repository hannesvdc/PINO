import json
import math
import torch as pt
from torch.utils.data import Dataset

import numpy as np

class DeepONetDataset(Dataset):
    def __init__(self, config, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype

        directory = config['Data Directory']
        branch_filename = 'branch_data.npy'
        trunk_filename = 'trunk_data.npy'

        self.N = 10_000
        self.lam = 2.0

        print('Loading Data...')
        self.grid_points = 101
        self.branch_input_data = pt.tensor(np.load(directory + branch_filename).transpose(), requires_grad=False, device=self.device, dtype=self.dtype) # Transpose to make data row-major
        self.trunk_input_data = pt.tensor(np.load(directory + trunk_filename).transpose(), requires_grad=True, device=self.device, dtype=self.dtype) # Grads necessary for PINN loss
        self.xy_left = self.trunk_input_data[0:self.grid_points, :]
        self.xy_forcing = self.trunk_input_data[-self.grid_points:, :]
        self.xy_int = self.samplePhysicsSamples(self.N, self.lam)
        print(self.branch_input_data)

        # Print the total memory consumption
        memory_usage = self.branch_input_data.numel() * self.branch_input_data.element_size() \
                     + self.xy_left.numel() * self.xy_left.element_size() \
                     + self.xy_forcing.numel() * self.xy_forcing.element_size() \
                     + self.xy_int.numel() * self.xy_int.element_size()
        print('Total Data Memory Consumption: ', round(memory_usage / 1024.0**2, 2), 'MB')

    def __len__(self):
        return self.branch_input_data.shape[0]
	
    def __getitem__(self, idx):
        return self.branch_input_data[idx,:]

    def samplePhysicsSamples(self, N, lam):
        u = pt.rand(N, device=self.device)
        x_values = 1.0 + pt.log1p(-u * (1.0 - math.exp(-lam))) / lam
        y_values = pt.rand(N, device=self.device)
        return pt.stack((x_values, y_values), dim=1)
    
def plotPhysicsSamplingDistribution(dataset):
    lam = 2.0
    Z = 1.0 - math.exp(-lam)
    N = 1001
    x_values = pt.linspace(0.0, 1.0, N)
    dist = lambda x: lam * pt.exp(-lam * (1.0 - x)) / Z

    N_samples = 100_000
    samples = dataset.samplePhysicsSamples(N_samples, lam)

    import matplotlib.pyplot as plt
    plt.plot(x_values.numpy(), dist(x_values).numpy())
    plt.hist(samples[:,0].numpy(), density=True, bins=int(math.sqrt(samples.shape[0])), alpha=0.5)
    plt.hist(samples[:,1].numpy(), density=True, bins=int(math.sqrt(samples.shape[0])), alpha=0.5)
    plt.show()

if __name__ == '__main__':
    config_file = 'DataConfig.json'
    config = json.load(open(config_file))
    dataset = DeepONetDataset(config, pt.device('cpu'), pt.float32)

    plotPhysicsSamplingDistribution(dataset)
