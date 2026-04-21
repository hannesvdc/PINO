import math

import torch as pt
import matplotlib.pyplot as plt

from QuantumNetwork import QuantumNetwork
from QuantumLoss import QuantumLoss

from sampleBatch import sampleBatch

# ============================================================
# Config
# ============================================================


r_cutoff = 5.0

# Wavefunction slice plot
R_slice = 1.0
x_min = -5.0
x_max = 5.0
n_x = 1000

# Energy curve
R_min = 0.1
R_max = 2.0
n_R = 100

def load_model( ) -> QuantumNetwork:
    z = 64
    neurons_per_layer = [ 4, z, z, z, z, 1]
    model = QuantumNetwork( neurons_per_layer, r_cutoff)
    model.load_state_dict( pt.load('./Results/best_model.pth') )
    return model

# ---------------------------------------------------------------------
# Wavefunction slice plotting
# ---------------------------------------------------------------------
def evaluate_psi_slice( model, R_value: float,
    y_value: float,
    z_value: float,
    x_min: float,
    x_max: float,
    n_x: int, ):
    """
    Evaluate psi(x, y, z; R) along a 1D x-slice.

    Assumes:
        model(R, xyz) -> (B, N)
    """

    x = pt.linspace(x_min, x_max, n_x )
    y = pt.full_like(x, float(y_value))
    z = pt.full_like(x, float(z_value))

    xyz = pt.stack([x, y, z], dim=1)  # (N, 3)
    xyz.requires_grad_(True)
    R = pt.tensor([R_value] )  # (1,)

    psi = model(R, xyz)[0].detach()  # (N,)
    psi_norm = psi / (psi.abs().max() + 1e-12)

    rho = psi**2
    rho_norm = rho / (rho.max() + 1e-12)

    return x.detach(), psi_norm, rho_norm

def plot_wavefunction_slices(
    model,
    R_value: float = 1.0,
    slices=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
    x_min: float = -5.0,
    x_max: float = 5.0,
    n_x: int = 1000, ):
    """
    Plot normalized psi(x) and normalized |psi(x)|^2 for fixed (y,z) slices.

    The normalization here is only for visualization:
        psi_plot = psi / max(|psi|)
        rho_plot = psi^2 / max(psi^2)
    """

    plt.figure(figsize=(8, 5))
    for y_value, z_value in slices:
        x, psi_norm, _ = evaluate_psi_slice(
            model=model,
            R_value=R_value,
            y_value=y_value,
            z_value=z_value,
            x_min=x_min,
            x_max=x_max,
            n_x=n_x,
        )
        plt.plot(x, psi_norm, label=f"(y,z)=({y_value}, {z_value})")

    plt.axvline(-R_value, linestyle="--", linewidth=1, label="nuclei")
    plt.axvline(R_value, linestyle="--", linewidth=1)

    plt.xlabel("x [Bohr]")
    plt.ylabel("normalized psi(x,y,z)")
    plt.title(f"Wavefunction slices at R = {R_value}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- probability density plot ----
    plt.figure(figsize=(8, 5))
    for y_value, z_value in slices:
        x, _, rho_norm = evaluate_psi_slice(
            model=model,
            R_value=R_value,
            y_value=y_value,
            z_value=z_value,
            x_min=x_min,
            x_max=x_max,
            n_x=n_x, )
        plt.plot(x, rho_norm, label=f"(y,z)=({y_value}, {z_value})")

    plt.axvline(-R_value, linestyle="--", linewidth=1, label="nuclei")
    plt.axvline(R_value, linestyle="--", linewidth=1)

    plt.xlabel("x [Bohr]")
    plt.ylabel("normalized |psi(x,y,z)|²")
    plt.title(f"Probability-density slices at R = {R_value}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# Energy curve plotting
# ---------------------------------------------------------------------
def plot_energy_curve_with_loss(
    model,
    loss_fn,
    xyz_val: pt.Tensor,
    mc_weights_val: pt.Tensor,
    R_min: float = 0.1,
    R_max: float = 2.0,
    n_R: int = 100,
    logspace: bool = True, ):
    """
    Plot electronic and total H2+ energies versus R.

    Requires loss_fn to support:

        E_elec = loss_fn(
            model,
            R_values,
            xyz_val,
            mc_weights_val,
        )

    Important:
        Do NOT wrap this in torch.no_grad(), because the loss needs
        autograd to compute grad_x psi for the kinetic term.
    """
    if logspace:
        R_values = pt.exp( pt.linspace( math.log(R_min), math.log(R_max), n_R, ) )
    else:
        R_values = pt.linspace(R_min, R_max, n_R)

    E_elec = []
    for idx in range(len(R_values)):
        R = R_values[idx]
        en = loss_fn( model, pt.tensor([R]), xyz_val, mc_weights_val, training=False )  # (n_R,)
        E_elec.append( float(en) )
    E_elec = pt.tensor( E_elec )
    E_total = E_elec + 1.0 / (2.0 * R_values)

    plt.figure(figsize=(8, 5))
    plt.plot(R_values, E_elec, label="electronic energy")
    plt.plot(R_values, E_total, label="total energy")
    plt.axvline(1.0, linestyle="--", linewidth=1, label="R = 1")
    plt.xlabel("R [Bohr], nuclei at (-R,0,0), (R,0,0)")
    plt.ylabel("Energy [Hartree]")
    plt.title("H₂⁺ energy curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Useful validation printout near R = 1
    idx_R1 = int( pt.argmin(pt.abs(R_values - 1.0)).item() )
    print()
    print("Validation near R = 1")
    print("---------------------")
    print(f"R                    = {R_values[idx_R1].item():.6f}")
    print(f"E_elec               = {E_elec[idx_R1].item():.8f} Hartree")
    print(f"E_total              = {E_total[idx_R1].item():.8f} Hartree")
    print("Reference rough target at R=1:")
    print("E_elec  ≈ -1.097 Hartree")
    print("E_total ≈ -0.597 Hartree")

def main():

    model = load_model( )
    loss_fn = QuantumLoss()
    plot_wavefunction_slices(
        model=model,
        R_value=R_slice,
        slices=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
        x_min=x_min,
        x_max=x_max,
        n_x=n_x, )

    # -----------------------------------------------------------------
    # Plot energy curve
    # -----------------------------------------------------------------
    gen = pt.Generator()
    _, xyz_val, mc_weights = sampleBatch( 1, n_x, r_cutoff, gen )
    plot_energy_curve_with_loss(
        model=model,
        loss_fn=loss_fn,
        xyz_val=xyz_val,
        mc_weights_val=mc_weights,
        R_min=R_min,
        R_max=R_max,
        n_R=n_R,
        logspace=True, )

if __name__ == "__main__":
    main()