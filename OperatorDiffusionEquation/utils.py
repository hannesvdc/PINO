import torch as pt

def getGradientNorm( model : pt.nn.Module ):
    grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    return pt.norm(pt.cat(grads))