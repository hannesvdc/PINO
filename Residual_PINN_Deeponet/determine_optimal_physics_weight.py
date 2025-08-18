import json
import matplotlib.pyplot as plt
import glob
import re
import torch

config_file = 'DataConfig.json'
config = json.load(open(config_file))
directory = config["Data Directory"] + 'posttraining/'

epochs, physics, forcing, total, unweighted, physics_weights = [], [], [], [], [], []

for fname in sorted(glob.glob(f"{directory}/checkpoint_*.pth")):
    m = re.search(r"checkpoint_(\d+)\.pth", fname)
    if not m:
        continue
    epoch = int(m.group(1))
    if not (1 <= epoch <= 300):
        continue

    ckpt = torch.load(fname, map_location="cpu")
    ep = float(ckpt['epoch'])
    print('ep', ep)
    ploss = float(ckpt['physics_loss'])
    floss = float(ckpt['forcing_loss'])
    tloss = float(ckpt['total_loss'])
    w_int = float(ckpt['w_int'])
    ploss_unw = ploss / w_int if w_int != 0 else None
    print('ep', ep, w_int)

    epochs.append(ep)
    physics.append(ploss)
    forcing.append(floss)
    total.append(tloss)
    unweighted.append(ploss_unw)
    physics_weights.append(w_int)

# Sort by epoch
order = sorted(range(len(epochs)), key=lambda i: epochs[i])
epochs = [epochs[i] for i in order]
physics = [physics[i] for i in order]
forcing = [forcing[i] for i in order]
total = [total[i] for i in order]
unweighted = [unweighted[i] for i in order]
physics_weights = [physics_weights[i] for i in order]

# Plot
plt.figure(figsize=(10,6))
plt.semilogy(epochs, physics, label="Physics loss")
plt.semilogy(epochs, forcing, label="Forcing loss")
plt.semilogy(epochs, total, label="Total loss")
plt.semilogy(epochs, unweighted, label="Unweighted physics loss")
plt.semilogy(epochs, physics_weights, label='Physics (Interior) Weight')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
