import json
import torch as pt
from torch.utils.data import Dataset

import numpy as np

class MultiIndexDeepONetDataset(Dataset):
    def __init__(self, config, device, dtype):
        super().__init__()

        directory = config['Data Directory']
        branch_filename = 'branch_data.npy'
        trunk_filename = 'trunk_data.npy'
        G_u_filename = 'output_data_u.npy'
        G_v_filename = 'output_data_v.npy'

        print('Loading Data...')
        self.N = 100
        self.branch_input_data = pt.tensor(np.load(directory + branch_filename).transpose(), requires_grad=False, device=device, dtype=dtype) # Transpose to make data row-major
        self.trunk_input_data = pt.tensor(np.load(directory + trunk_filename).transpose(), requires_grad=False, device=device, dtype=dtype)
        self.output_data_u = pt.tensor(np.load(directory + G_u_filename), requires_grad=False, device=device, dtype=dtype)
        self.output_data_v = pt.tensor(np.load(directory + G_v_filename), requires_grad=False, device=device, dtype=dtype)
        self.scale_u = pt.max(pt.abs(pt.std(self.output_data_u , dim=0)))
        self.scale_v = pt.max(pt.abs(pt.std(self.output_data_v , dim=0)))
        self.output_data_u = self.output_data_u / self.scale_u
        self.output_data_v = self.output_data_v / self.scale_v

        # Subsample the data for faster training
        self.N_branch_datapoints = 1000
        self.N_trunk_datapoints = 101**2 // 2
        random_branch_indices = pt.randperm(self.branch_input_data.size(0))[0:self.N_branch_datapoints]
        random_trunk_indices = pt.randperm(self.trunk_input_data.size(0))[:self.N_trunk_datapoints]
        self.branch_input_data = self.branch_input_data[random_branch_indices,:]
        self.trunk_input_data = self.trunk_input_data[random_trunk_indices,:]
        self.output_data_u = self.output_data_u[random_branch_indices,:][:,random_trunk_indices]
        self.output_data_v = self.output_data_v[random_branch_indices,:][:,random_trunk_indices]

        # Print the total memory consumption
        memory_usage = self.branch_input_data.numel() * self.branch_input_data.element_size() \
                     + self.trunk_input_data.numel() * self.trunk_input_data.element_size() \
                     + self.output_data_u.numel() * self.output_data_u.element_size() \
                     + self.output_data_v.numel() * self.output_data_v.element_size()
        print('Total Memory Consumption: ', memory_usage / (1024.0)**2, 'MB')

    def __len__(self):
        return self.N_branch_datapoints * self.N_trunk_datapoints
	
    def __getitem__(self, idx):
        branch_idx = idx % self.N_branch_datapoints
        trunk_idx = idx // self.N_branch_datapoints
        return self.branch_input_data[branch_idx,:], self.trunk_input_data[trunk_idx, :], branch_idx, trunk_idx
    
if __name__ == '__main__':
    config_file = 'DataConfig.json'
    config = json.load(open(config_file))
    dataset = MultiIndexDeepONetDataset(config, pt.device('cpu'), pt.float32)
    print(dataset[54489236], len(dataset))