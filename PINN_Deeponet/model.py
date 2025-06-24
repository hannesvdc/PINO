import torch as pt
import torch.nn as nn
from torch.func import vmap, jacrev

from collections import OrderedDict

class BranchModel(nn.Module):
    """
    The branch network in our DeepONet. The input is a vector of shape (B, 101).
    """
    def __init__(self, n_conv : int, kernel_size : int, p : int) -> None:
        super().__init__()

        # Define the convolutional layers
        layer_list = list()
        for i in range(n_conv):
            layer_list.append((f"conv{i+1}", nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=kernel_size//2)))
            layer_list.append((f"act{i+1}", nn.Tanh()))

        # Add an average-pooling layer
        layer_list.append(("pool", nn.AvgPool1d(kernel_size=2, stride=2)))

        # Flatten and add two more nonlinear layers for good measure
        layer_list.append(("flatten", nn.Flatten()))
        layer_list.append(("linear1", nn.Linear(50, 2*p)))
        layer_list.append((f"act{n_conv+1}", nn.Tanh()))
        layer_list.append(("linear2", nn.Linear(2*p, 2*p)))
        layer_list.append((f"act{n_conv+2}", nn.Tanh()))
        layer_list.append(("linear3", nn.Linear(2*p, 2*p)))
        
        self.layers = nn.Sequential(OrderedDict(layer_list))

    def forward(self, x : pt.Tensor):
        return self.layers(x)
    
class TrunkModel(nn.Module):
    """
    The branch network in our DeepONet. The input is a vector of shape (Nc, 2).
    """
    def __init__(self, p : int):
        super().__init__()

        # Define two linear layers with activation
        layer_list = list()
        layer_list.append(("linear1", nn.Linear(2*p, 2*p)))
        layer_list.append((f"act1", nn.Tanh()))
        layer_list.append(("linear2", nn.Linear(2*p, 2*p)))
        layer_list.append((f"act2", nn.Tanh()))
        layer_list.append(("linear3", nn.Linear(2*p, 2*p)))

        self.layers = nn.Sequential(OrderedDict(layer_list))

    def forward(self, x : pt.Tensor):
        return self.layers(x)
    
class ConvDeepONet(nn.Module):
    def __init__(self, n_branch_conv : int, kernel_size : int, p : int) -> None:
        super(ConvDeepONet, self).__init__()
        self.p = p
        
        self.branch_net = BranchModel(n_branch_conv, kernel_size, p)
        self.trunk_net = TrunkModel(p)
        
        #self.params = []
        #self.params.extend(self.branch_net.parameters())
        #self.params.extend(self.trunk_net.parameters())
        print('Number of DeepONet Parameters:', sum(p.numel() for p in self.parameters()))
    
    def getNumberofParameters(self):
        return sum(p.numel() for p in self.parameters())
    
    # The input data: branch_input with shape (batch_size, 101), and trunk_input with shape (Nc, 2)
    def forward(self, branch_input : pt.Tensor, trunk_input : pt.Tensor):
        branch_input = branch_input.unsqueeze(dim=1)
        branch_output = self.branch_net.forward(branch_input)
        trunk_output = self.trunk_net.forward(trunk_input)

        # Extract the u and v components
        b_u = branch_output[:, :self.p]
        b_v = branch_output[:, self.p:]
        t_u = trunk_output[:, :self.p]
        t_v = trunk_output[:, self.p:]

        # Combine through matrix multiplication
        u = b_u @ t_u.T
        v = b_v @ t_v.T

        # The output are two matrices of shape (B, Nc)
        return u, v
    
class PhysicsLoss(nn.Module):
    def __init__(self, E, nu, w_int=1.0, w_dirichlet=1.0, w_forcing=1.0):
        super().__init__()
        self.E, self.nu = E, nu
        self.w_int, self.w_dirichlet, self.w_forcing = w_int, w_dirichlet, w_forcing

    def forward(self, model : ConvDeepONet, f_batch : pt.Tensor, xy_int : pt.Tensor, xy_diriclet : pt.Tensor, xy_forcing : pt.Tensor):
        pass