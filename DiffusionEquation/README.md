# Physics-Informed Neural Operator for the 1D heat equation with fixed inital condition and Dirichlet boundary conditions.

## uniform_tau/
Contains the first PINO experiments for the heat equation with $\tau$ sampled uniformly between 0.01 and 8.0.
Training converges but the validation loss stayed stagnant. Did not validate / test the network performance further.

## ic_time_factor/
Biases the sampling of $\tau$ through the power law in log-scale. This pushes most of the samples close to 0 (though gated at 0.01). The validation loss did go down in tandem with the training loss. Solid improvement over 'uniform_tau/'. Also includes 30% of the samples in the [1, tau_max] regime so we remove the large-tau bias by including uniform large-tau samples. Better performance than only including small $\tau$ values, especially at the tail of the Fourier modes. This model serves as a great baseline!

## deeper_mlp/
Increase the number of hidden layers from 2 to 4 at the cost of more parameters. Performance is largely the same but we are able to follow the 'tails' of the Fourier modes ($\tau>1$) for much longer in log space. Use this model if you care about accuracy in the higher-frequency components.

## regularize/
Contains my experiments with adding weight decay to the model. Made the Fourier modes worse as $\tau > 0.8$. Don't use this model.

## residual/
My final experiments with using residual / skip layers instead of making the model deeper. Currently contains a bug but I want to have a closer look at skip layers in the future. Might come back to this model at some point.

Enjoy!