import torch as pt
import torch.nn as nn

from collections import OrderedDict

# Class for Dense Neural Networks used for branch and trunk networks.
class DenseNN(nn.Module):
    def __init__(self, layers=[]):
        super(DenseNN, self).__init__()
        
        # Create all feed-forward layers
        self.depth = len(layers) - 1
        self.activation = nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, pt.nn.Linear(layers[i], layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation()))
        layer_list.append(('layer_%d' % (self.depth-1), pt.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)

        # Combine all layers in a single Sequential object to keep track of parameter count
        self.layers = pt.nn.Sequential(layerDict)

    def forward(self, x):
        return self.layers(x)
    
# Class for general DeepONets
class DeepONet(nn.Module):
    def __init__(self, branch_layers, trunk_layers):
        super(DeepONet, self).__init__()
        assert branch_layers[-1] == trunk_layers[-1]
        self.p = branch_layers[1] // 2

        self.branch_net = DenseNN(branch_layers)
        self.trunk_net = DenseNN(trunk_layers)
        
        self.params = []
        self.params.extend(self.branch_net.parameters())
        self.params.extend(self.trunk_net.parameters())
        print('Number of DeepONet Parameters:', sum(p.numel() for p in self.parameters()))
    
    def getNumberofParameters(self):
        return sum(p.numel() for p in self.parameters())

    # The input data: branch_input with shape (batch_size, len(f)), and trunk_input with shape (N^2, 2)
    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net.forward(branch_input)
        trunk_output = self.trunk_net.forward(trunk_input)

        # Increase the dimensions
        branch_output = branch_output[:,None,:] # New shape (batch_size, 1, 2p)
        trunk_output = trunk_output[None,:,:] # New shape (1, N^2, 2p)
        #print((branch_output * trunk_output).shape)

        # Multiply element-wise over the last dimension and reduce
        output_u = pt.sum(branch_output[:,:,:self.p] * trunk_output[:,:,:self.p], dim=2)
        output_v = pt.sum(branch_output[:,:,self.p:] * trunk_output[:,:,self.p:], dim=2)
        return output_u, output_v