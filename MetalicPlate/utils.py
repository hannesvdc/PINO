import torch as pt

def getGradientNorm( model : pt.nn.Module ) -> pt.Tensor:
    grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    return pt.norm(pt.cat(grads))

def print_gradients( model : pt.nn.Module ):
    print("=== Gradient norms by parameter ===")
    any_nonzero = False
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"{name:50s}  grad=None")
        else:
            gnorm = param.grad.detach().norm().item()
            print(f"{name:50s}  grad_norm={gnorm:.3e}")
            if gnorm > 0:
                any_nonzero = True
    print("Any nonzero grads?", any_nonzero)