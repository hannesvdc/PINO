import torch as pt
import torch.nn as nn
from torch.func import vmap, jacrev

from collections import OrderedDict
from typing import Tuple

class ResidualFC(nn.Module):
    """Width-preserving fully-connected residual block with tanh activations."""
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        out = self.act(self.fc1(x))      # ϕ(W₁ x)
        out = self.fc2(out)              # W₂ ϕ(W₁ x)
        out = out + x                    # skip connection
        return out 
    
class BranchModel(nn.Module):
    """
    Convolutional branch of the DeepONet with residual FC blocks.
    Input: (B, 3, 101)  ➜  (B, 2 p)
    """
    def __init__(self,
                 n_conv: int,
                 n_channels: int,
                 kernel_size: int,
                 n_res_blocks: int,
                 p: int) -> None:
        super().__init__()

        layers = []
        for i in range(n_conv): # Convolutional Layers
            in_channels = 3 if i == 0 else n_channels
            layers += [
                (f"conv{i+1}", nn.Conv1d(in_channels, n_channels, kernel_size, padding=kernel_size // 2)),
                (f"act_c{i+1}", nn.Tanh())
            ]

        # Add a flattening layer
        layers.append(("flatten", nn.Flatten())) # output shape (B, n_channels * 101)

        # Fully-connected projection to width = 2 p
        in_feats = n_channels * 101
        width = 2 * p
        layers.append(("fc_proj", nn.Linear(in_feats, width)))
        layers.append(("act_proj", nn.Tanh()))

        # Residual fully-connected blocks 
        for k in range(n_res_blocks):
            layers.append((f"res{k+1}", ResidualFC(width)))

        # Final linear head (unchanged)
        layers.append(("fc_out", nn.Linear(width, width)))

        # Register all layers
        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        return self.layers(x)
    
class TrunkModel(nn.Module):
    """
    The branch network in our DeepONet. The input is a vector of shape (Nc, 2).
    """
    def __init__(self, n_residual : int, p : int):
        super().__init__()

        # Define two linear layers with activation
        n_hidden = 256

        layers = OrderedDict()

        # 1) Plain input layer   + activation
        layers["linear_in"] = nn.Linear(2, n_hidden)
        layers["act_in"]    = nn.Tanh()

        # 2) Stack of residual blocks, each keeps dimension n_hidden
        for i in range(n_residual):
            layers[f"res{i+1}"] = ResidualFC(n_hidden)

        # 3) Final projection  (n_hidden → 2p), no activation here
        layers["linear_out"] = nn.Linear(n_hidden, 2 * p)

        self.layers = nn.Sequential(layers)

    def forward(self, x : pt.Tensor) -> pt.Tensor:
        return self.layers(x)
    
class ConvDeepONet(nn.Module):
    def __init__(self, n_branch_conv : int, 
                       n_branch_channels : int, 
                       kernel_size : int, 
                       n_branch_residual : int,
                       n_trunk_residual : int,
                       p : int) -> None:
        super(ConvDeepONet, self).__init__()
        self.p = p
        self.beta = 6.0
        
        self.branch_net = BranchModel(n_branch_conv, n_branch_channels, kernel_size, n_branch_residual, p)
        self.trunk_net = TrunkModel(n_trunk_residual, p)

        print('Number of DeepONet Parameters:', sum(p.numel() for p in self.parameters()))
    
    def getNumberofParameters(self):
        return sum(p.numel() for p in self.parameters())
    
    # The input data: branch_input with shape (batch_size, 101), and trunk_input with shape (Nc, 2)
    def forward(self, branch_input : pt.Tensor, trunk_input : pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        y_coords = pt.linspace(0.0, 1.0, 101, device=branch_input.device, dtype=branch_input.dtype).repeat(branch_input.shape[0], 1) # shape (B, 101,)
        g_x = branch_input[:, :101] # (B, 101)
        g_y = branch_input[:, 101:] # (B, 101)
        branch_2ch = pt.stack((g_x, g_y, y_coords), dim=1)  # (B, 3, 101)
        branch_output = self.branch_net.forward(branch_2ch)
        trunk_output = self.trunk_net.forward(trunk_input)

        # Extract the u and v components
        b_u = branch_output[:, :self.p]
        b_v = branch_output[:, self.p:]
        t_u = trunk_output[:, :self.p]
        t_v = trunk_output[:, self.p:]

        # Combine through matrix multiplication
        u_hat = b_u @ t_u.T
        v_hat = b_v @ t_v.T

        # Multiply by the ramp to enforce the Dirichlet BCs
        x = trunk_input[:, 0]            # (Nc,)
        r = 1.0 - pt.exp(-x / self.beta).unsqueeze(0)
        u = r * u_hat
        v = r * v_hat

        # The output are two matrices of shape (B, Nc)
        return u, v
    
class PhysicsLoss(nn.Module):
    def __init__(self, E_train, nu, w_int=1.0, w_forcing=1.0):
        super().__init__()
        self.E, self.nu = E_train, nu
        self.pref = self.E / (1.0 - self.nu**2)
        self.w_int, self.w_forcing = w_int, w_forcing

    def setWeights(self, w_int : float, w_forcing : float):
        self.w_int = w_int
        self.w_forcing = w_forcing

    def forward(self, model : ConvDeepONet, f_batch : pt.Tensor, xy_int : pt.Tensor, xy_forcing : pt.Tensor):

        # PDE residual inside the domain
        if xy_int.numel() != 0:
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
        else:
            loss_int = pt.zeros(1, device=f_batch.device, dtype=f_batch.dtype, requires_grad=False).squeeze()

        # Forcing (Neumann) boundary conditions on the right
        if xy_forcing.numel() != 0:
            g_x = f_batch[:, :101]
            g_y = f_batch[:, 101:]
            J_forcing, _ = self.grads_and_hess(model, f_batch, xy_forcing, needsHessian=False)
            u_x = J_forcing[:,:,0,0] # Shape (B, 101) because Nc = 101 on rhe right boundary
            u_y = J_forcing[:,:,0,1]
            v_x = J_forcing[:,:,1,0]
            v_y = J_forcing[:,:,1,1]
            loss_forcing_u = self.pref * (u_x + self.nu * v_y) + g_x
            loss_forcing_v = self.pref * (1.0 - self.nu) / 2.0 * (u_y + v_x) + g_y
            loss_forcing = loss_forcing_u.square().mean() + loss_forcing_v.square().mean()
        else:
            loss_forcing = pt.zeros(1, device=f_batch.device, dtype=f_batch.dtype, requires_grad=False).squeeze()

        return self.w_int * loss_int + self.w_forcing * loss_forcing

    def grads_and_hess(self, model : ConvDeepONet, f_batch : pt.Tensor, xy : pt.Tensor, needsHessian : bool = True) -> Tuple[pt.Tensor, pt.Tensor]:
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
        J = vmap(vmap(J_fn, (None,0)), (0,None))(f_batch, xy) # Shape (B, Nc, 2, 2)

        if needsHessian:
            H_fn = jacrev(J_fn , argnums=1) # Hessian of (u, v) with shape (2, 2, 2)
            H = vmap(vmap(H_fn, (None,0)), (0,None))(f_batch, xy) # Shape (B, Nc, 2, 2, 2)
        else:
            H = pt.Tensor((0.0,))

        return J, H