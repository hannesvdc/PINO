import numpy as np
import matplotlib.pyplot as plt

# Load traction-less training
notraction_train = np.load( './Results/resample_1_train_data.npy' )
notraction_valid = np.load( './Results/resample_1_validation_data.npy' )
traction_train = np.load( './Results/resample_10_train_data.npy' )
traction_valid = np.load( './Results/resample_10_validation_data.npy' )

notraction_epochs = notraction_train[:,0]
notraction_loss = notraction_train[:,1]
notraction_grads = notraction_train[:,2]
notraction_validepochs = notraction_valid[:,0]
notraction_validloss = notraction_valid[:,1]

traction_epochs = traction_train[:,0]
traction_loss = traction_train[:,1]
traction_grads = traction_train[:,2]
traction_validepochs = traction_valid[:,0]
traction_validloss = traction_valid[:,1]

fig, (ax1, ax2) = plt.subplots(1,2)
idx = (notraction_epochs < 2500)
ax1.plot( notraction_epochs[idx], notraction_loss[idx], label="Training Loss")
idx = (notraction_validepochs < 2500)
ax1.plot( notraction_validepochs[idx], notraction_validloss[idx], label="Validation Loss")
ax1.set_xlabel( "Epoch" )
ax1.set_ylabel( "Elastic Energy" )
ax1.set_title( "Training Without Traction Loss")
#plt.legend()

idx = (traction_epochs < 2500)
ax2.plot( traction_epochs[idx], traction_loss[idx], label="Training Loss")
idx = (traction_validepochs < 2500)
ax2.plot( traction_validepochs[idx], traction_validloss[idx], label="Validation Loss")
ax2.set_xlabel( "Epoch" )
# plt.ylabel( "Elastic Energy" )
ax2.set_title( "Training With Traction Loss")
plt.legend()
plt.show()