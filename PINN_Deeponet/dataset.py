import json
import torch as pt
from torch.utils.data import Dataset

import numpy as np

class DeepONetDataset(Dataset):
    def __init__(self, config, device, dtype):
        super().__init__()

        directory = config['Data Directory']
        branch_filename = 'branch_data.npy'
        trunk_filename = 'trunk_data.npy'

        print('Loading Data...')
        self.grid_points = 101
        self.branch_input_data = pt.tensor(np.load(directory + branch_filename).transpose(), requires_grad=False, device=device, dtype=dtype) # Transpose to make data row-major
        trunk_input_data = pt.tensor(np.load(directory + trunk_filename).transpose(), requires_grad=True, device=device, dtype=dtype) # Grads necessary for PINN loss
        self.xy_left = trunk_input_data[0:self.grid_points, :]
        self.xy_forcing = trunk_input_data[-self.grid_points:, :]
        self.xy_int = trunk_input_data[self.grid_points:-self.grid_points, :]

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
    
if __name__ == '__main__':
    config_file = 'DataConfig.json'
    config = json.load(open(config_file))
    DeepONetDataset(config, pt.device('cpu'), pt.float32)