import torch as pt
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from typing import Tuple, List

def optimizeWithAdam( model : nn.Module,
                      loss_fn : nn.Module,
                      train_loader : pt.utils.data.DataLoader,
                      validation_loader : pt.utils.data.DataLoader,
                      device : pt.device,
                      dtype : pt.dtype,
                      lr : float,
                      n_steps : int,
                      step_size : int,
                      store_directory : str,
                      ) -> Tuple[List, List, List, List, List, List]:
    n_epochs = n_steps * step_size
    optimizer = Adam( model.parameters(), lr )
    scheduler = StepLR( optimizer, step_size=step_size, gamma=0.1 )

    def getGradientNorm():
        grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        return pt.norm(pt.cat(grads))

    # Bookkeeping and storage
    train_counter = []
    train_losses = []
    train_grads = []
    validation_counter = []
    validation_losses = []
    T_t_rmss = []
    T_xx_rmss = []
    rel_rmss = []

    # Train Function
    def train( epoch ):
        model.train()

        epoch_loss = float( 0.0 )
        for batch_idx, (x, t, p) in enumerate( train_loader ):
            optimizer.zero_grad( set_to_none=True )
            x = x.to(device=device, dtype=dtype)
            t = t.to(device=device, dtype=dtype)
            p = p.to(device=device, dtype=dtype)

            # Compute the loss and its gradient
            loss, loss_dict = loss_fn( model, x, t, p )
            loss.backward()
            epoch_loss += float( loss.item() )
            pre_grad = getGradientNorm()
            grad = getGradientNorm()

            rms = loss_dict["rms"]
            T_t_rms = loss_dict["T_t_rms"]
            T_xx_rms = loss_dict["T_xx_rms"]
            rel_rms = rms / (T_t_rms.item() + T_xx_rms.item() + 1e-12)

            # Update the weights
            optimizer.step()

            # Bookkeeping
            train_counter.append( (1.0*batch_idx) / len(train_loader) + epoch)
            train_losses.append( loss.item())
            train_grads.append( grad.cpu() )
            T_t_rmss.append( T_t_rms.item() )
            T_xx_rmss.append( T_xx_rms.item() )
            rel_rmss.append( rms.item() / (T_t_rms.item() + T_xx_rms.item() + 1e-12) )

        # Update
        epoch_loss /= len( train_loader )
        reg_loss = loss_dict.get("reg_loss", 0.0)
        print('\nTrain Epoch: {} \tLoss: {:.4E} \tPre-Clip Loss Gradient: {:.4E} \tLoss Gradient: {:.4E} \tlr: {:.2E}'.format(
                epoch, epoch_loss, pre_grad.item(), grad.item(), optimizer.param_groups[0]['lr']))
        print('T_t RMS: {:.4E} \tT_xx RMS: {:.4E} \tTotal RMS: {:.4E} \tLoss Relative RMS: {:.4E} \tRegularization: {:.4E}'.format(
            T_t_rms, T_xx_rms, rms, rel_rms, reg_loss ))
        
        # Store the pretrained state
        pt.save( model.state_dict(), store_directory + 'model_adam.pth')
        pt.save( optimizer.state_dict(), store_directory + 'optimizer_adam.pth')

    # Validation Function 
    def validate( epoch ):
        model.eval()

        for batch_idx, (x, t, p) in enumerate( validation_loader ):
            x = x.to(device=device, dtype=dtype)
            t = t.to(device=device, dtype=dtype)
            p = p.to(device=device, dtype=dtype)

            # Compute the loss and its gradient
            loss, _ = loss_fn( model, x, t, p )

            # Bookkeeping
            validation_counter.append( (1.0*batch_idx) / len(validation_loader) + epoch)
            validation_losses.append( loss.item())

            # Update
            print( 'Validation Epoch: {} \tLoss: {:.4E}'.format( epoch, loss.item() ) )

    # Actual training and validation
    try:
        for epoch in range( n_epochs ):
            train( epoch )
            validate( epoch )
            scheduler.step( )
    except KeyboardInterrupt:
        pass

    # Return everything
    return train_counter, train_losses, train_grads, validation_counter, validation_losses, rel_rmss