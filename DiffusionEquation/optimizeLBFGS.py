import torch as pt
import torch.nn as nn
from torch.optim import LBFGS

from typing import Tuple, List

def optimizeWithLBFGS( model : nn.Module,
                      loss_fn : nn.Module,
                      x_train : pt.Tensor,
                      t_train : pt.Tensor,
                      p_train : pt.Tensor,
                      x_validation : pt.Tensor,
                      t_validation : pt.Tensor,
                      p_validation : pt.Tensor,
                      device : pt.device,
                      dtype : pt.dtype,
                      lr : float,
                      store_directory : str,
                      post : bool = False,
                      ) -> Tuple[List, List, List, List, List, List, List]:
    if post:
        prepend = "post_"
    else:
        prepend = ""

    max_iter = 50
    history_size = 50
    optimizer = LBFGS( model.parameters(), lr, max_iter=max_iter, line_search_fn="strong_wolfe", history_size=history_size )
    def getGradientNorm():
        grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        return pt.norm(pt.cat(grads))

    # The L-BFGS closure
    last_state = {"loss" : None, "grad_norm" : None, "rel_rms" : None}
    all_loss_evalutaions = []
    all_grad_evaluations = []
    def closure():
        model.train()
        optimizer.zero_grad( set_to_none=True )

        # Batched but Deterministic Loss for memory constraints
        epoch_loss, loss_dict = loss_fn( model, x_train, t_train, p_train )
        epoch_loss.backward( )
        last_gradnorm = getGradientNorm().item()

        # Store some diagnostic information
        rms = loss_dict["rms"]
        T_t_rms = loss_dict["T_t_rms"]
        T_xx_rms = loss_dict["T_xx_rms"]
        reg_loss = loss_dict.get("reg_loss", 0.0)
        rel_rms = rms / (T_t_rms.item() + T_xx_rms.item() + 1e-12)
        last_state["loss"] = epoch_loss.item()
        last_state["grad_norm"] = last_gradnorm # type: ignore
        last_state["rel_rms"] = rel_rms
        last_state["T_t rms"] = T_t_rms.item()
        last_state["T_xx rms"] = T_xx_rms.item()
        last_state["reg_loss"] = reg_loss
        all_loss_evalutaions.append( epoch_loss.item() )
        all_grad_evaluations.append( last_gradnorm )

        # LBFGS only needs a scalar loss value back.
        # It does NOT need the returned tensor to carry a graph, since grads are already in .grad.
        return epoch_loss
    def validate():
        model.eval()

        # Just the one validation batch
        epoch_loss, _, _, _ = loss_fn( model, x_validation, t_validation, p_validation )
        
        return epoch_loss

    # Actual training and validation
    print('\nStarting Training...')
    learning_rates = []
    train_counter = []
    train_losses = []
    train_grads = []
    validation_counter = []
    validation_losses = []
    rel_rmss = []

    epoch = 1
    try:
        while lr > 1.e-6:
            w0 = pt.cat([p.detach().flatten() for p in model.parameters()])
            training_loss = optimizer.step( closure )
            w1 = pt.cat([p.detach().flatten() for p in model.parameters()])
            step_norm = (w1 - w0).norm().item()
            print('\nTrain Epoch: {} \tLoss: {:.10E} \tLoss Gradient: {:.10E} \tStep Norm: {:.10E} \t Lr: {:.4E}'.format( 
                epoch, last_state["loss"], last_state["grad_norm"], step_norm, optimizer.param_groups[0]["lr"] ))
            print('T_t RMS: {:.4E} \tT_xx RMS: {:.4E} \tLoss Relative RMS: {:.4E} \tRegularization: {:.4E}'.format(
                last_state["T_t rms"], last_state["T_xx rms"], last_state["rel_rms"], last_state["reg_loss"]))
            
            validation_loss = validate()
            print('Validation Epoch: {} \tLoss: {:.10E}'.format( epoch, validation_loss.item() ))
            
            # Store the pretrained state
            pt.save( model.state_dict(), store_directory + prepend + 'model_lbfgs.pth')
            pt.save( optimizer.state_dict(), store_directory + prepend + 'optimizer_lbfgs.pth')

            train_counter.append( epoch )
            train_losses.append( last_state["loss"] )
            train_grads.append( last_state["grad_norm"] )
            rel_rmss.append( last_state["rel_rms"] )
            validation_counter.append( epoch )
            validation_losses.append( validation_loss.item() )
            learning_rates.append( optimizer.param_groups[0]["lr"] )

            if step_norm < 1e-8:
                lr = 0.3 * lr
                print('Lowering L-BFGS Learning Rate to', lr)
                for g in optimizer.param_groups:
                    g["lr"] = lr
            
            # Update to the next epoch.
            epoch += 1
    except KeyboardInterrupt:
        pass

    # return everything
    return train_counter, train_losses, train_grads, validation_counter, validation_losses, rel_rmss, learning_rates