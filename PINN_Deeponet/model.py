import torch as pt
import torch.nn as nn
from torch.func import vmap, jacrev

from collections import OrderedDict
from typing import Tuple

class BranchModel(nn.Module):
    """
    The branch network in our DeepONet. The input is a vector of shape (B, 2, 101).
    """
    def __init__(self, n_conv : int, kernel_size : int, p : int) -> None:
        super().__init__()

        # Define the convolutional layers. The forcing has shape (B, 2, 101).
        channels = 2
        layer_list = []
        for i in range(n_conv):
            layer_list.append((f"conv{i+1}", nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=kernel_size//2)))
            layer_list.append((f"act{i+1}", nn.Tanh()))

        # Add an average-pooling layer and flatten
        layer_list.append(("pool", nn.AvgPool1d(kernel_size=2, stride=2))) # output shape (B, 2, 50)
        layer_list.append(("flatten", nn.Flatten())) # output shape (B, 100)

        # Add two more nonlinear layers for good measure
        layer_list.append(("linear1", nn.Linear(100, 2*p)))
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

        print('Number of DeepONet Parameters:', sum(p.numel() for p in self.parameters()))
    
    def getNumberofParameters(self):
        return sum(p.numel() for p in self.parameters())
    
    # The input data: branch_input with shape (batch_size, 101), and trunk_input with shape (Nc, 2)
    def forward(self, branch_input : pt.Tensor, trunk_input : pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        g_x = branch_input[:, :101] # (B, 101)
        g_y = branch_input[:, 101:] # (B, 101)
        branch_2ch = pt.stack((g_x, g_y), dim=1)  # (B, 2, 101)
        branch_output = self.branch_net.forward(branch_2ch)
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
    def __init__(self, E_train, nu, w_int=1.0, w_dirichlet=1.0, w_forcing=1.0):
        super().__init__()
        self.E, self.nu = E_train, nu
        self.pref = self.E / (1.0 - self.nu**2)
        self.w_int, self.w_dirichlet, self.w_forcing = w_int, w_dirichlet, w_forcing

    def forward(self, model : ConvDeepONet, f_batch : pt.Tensor, xy_int : pt.Tensor, xy_diriclet : pt.Tensor, xy_forcing : pt.Tensor):

        # PDE residual inside the domain
        _, H_int = self.grads_and_hess(model, f_batch, xy_int)
        u_xx = H_int[:,:,0,0,0]
        u_xy = H_int[:,:,0,0,1]
        u_yy = H_int[:,:,0,1,1]
        v_xx = H_int[:,:,1,0,0]
        v_xy = H_int[:,:,1,0,1]
        v_yy = H_int[:,:,1,1,1]
        loss_int_u = self.pref * (u_xx + 0.5*(1.0 + self.nu)*v_xy + 0.5*(1.0 - self.nu)*u_yy)
        loss_int_v = self.pref * (v_yy + 0.5*(1.0 + self.nu)*u_xy + 0.5*(1.0 - self.nu)*v_xx)
        loss_int = loss_int_u.square().mean() + loss_int_v.square().mean()
        
        # Dirichlet boundary conditions on the left
        u_left, v_left = model.forward(f_batch, xy_diriclet)
        loss_left = u_left.square().mean() + v_left.square().mean()

        # Forcing (Neumann) boundary conditions on the right
        g_x = f_batch[:, :101]
        g_y = f_batch[:, 101:]
        J_forcing, _ = self.grads_and_hess(model, f_batch, xy_forcing)
        u_x = J_forcing[:,:,0,0] # Shape (B, 101) because Nc = 101 on rhe right boundary
        u_y = J_forcing[:,:,0,1]
        v_x = J_forcing[:,:,1,0]
        v_y = J_forcing[:,:,1,1]
        loss_forcing_u = self.pref * (u_x + self.nu * v_y) + g_x
        loss_forcing_v = self.pref * (1.0 - self.nu) / 2.0 * (u_y + v_x) + g_y
        loss_forcing = loss_forcing_u.square().mean() + loss_forcing_v.square().mean()

        return self.w_int * loss_int + self.w_dirichlet * loss_left + self.w_forcing * loss_forcing

    def grads_and_hess(self, model : ConvDeepONet, f_batch : pt.Tensor, xy : pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        """
        model     : ConvDeepONet
        f_batch   : (B, 101)
        xy        : (Nc, 2)
        returns   : J_v, H_v
                        J_v  (B, Nc, 2, 2)   grad  [comp, dir ]
                        H_v  (B, Nc, 2, 2, 2) Hess  [comp, d1 , d2]
        """
        def uv_vec(forcing : pt.Tensor, xy_single : pt.Tensor): # forcing has shape (101,) and xy_single (2,) -> need to unsqueeze
            u, v = model.forward(forcing.unsqueeze(dim=0), xy_single.unsqueeze(dim=0)) # Both shape (1,1)
            return pt.stack((u[0,0], v[0,0]))
        
        J_fn = jacrev(uv_vec, argnums=1) # Jacobian of (u, v) with shape (2, 2)
        H_fn = jacrev(J_fn , argnums=1) # Hessian of (u, v) with shape (2, 2, 2)

        # Map the Hessian to all inputs in forcing and xy_single
        J = vmap(vmap(J_fn, (None,0)), (0,None))(f_batch, xy) # Shape (B, Nc, 2, 2)
        H = vmap(vmap(H_fn, (None,0)), (0,None))(f_batch, xy) # Shape (B, Nc, 2, 2, 2)

        return J, H