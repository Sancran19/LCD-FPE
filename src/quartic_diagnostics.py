"""
Quartic coalescence model -- potential-landscape snapshots + spectral gap.
============================================================================

Companion diagnostics for free_energy_diff.py's stress-test model
    U(x,zeta) = x^4 - 16(1-zeta)x^2,   zeta(t,tau) = t/tau  (linear ramp)
used to visually/quantitatively show WHY the LCD/spectral method
struggles here: for zeta < 0.5 the potential is a deep symmetric double
well (barrier > 16), and the slowest relaxation mode's rate closes
exponentially (Eyring-Kramers), which is exactly the CD-cost bottleneck
c_1 ~ 1/|lambda_1|.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import Lcd_full_magnus4 as sim   # reuse: biorthogonal_decomp (generic)
import plotter as plot_style     # PRL rcParams (applied on import) + colors/_save

OUT_DATA  = os.path.join(sim.DATA_ROOT, "quartic_diagnostics")
OUT_PLOTS = os.path.join(sim.PLOTS_ROOT, "quartic_diagnostics")
os.makedirs(OUT_DATA, exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)

# ─── Model (identical definitions to free_energy_diff.py) ──────
beta = 1.0
N    = 80
x_np = np.linspace(-4.5, 4.5, N)
dx   = x_np[1] - x_np[0]

def potential_q(xv, zeta):
    """U(x,zeta) = x^4 - 16(1-zeta)x^2"""
    return xv**4 - 16.0 * (1.0 - zeta) * xv**2

def build_L_q(zeta):
    V  = potential_q(x_np, zeta)
    pi = np.exp(-beta * V); pi /= pi.sum()
    Dc = 1.0 / dx**2
    dV = np.diff(V)
    kf = Dc * np.exp(-beta * dV / 2.0)
    kb = Dc * np.exp( beta * dV / 2.0)
    L  = np.zeros((N, N))
    L[np.arange(1, N), np.arange(0, N - 1)] = kf
    L[np.arange(0, N - 1), np.arange(1, N)] = kb
    for i in range(N):
        L[i, i] = -L[:, i].sum()
    return L, pi

TAU = 1.0  # zeta(t,tau) = t/tau is a linear ramp -> t/tau IS zeta directly

# ═══════════════════════════════════════════════════════════════
# 1. Potential-landscape snapshots: 2x3 grid, zeta<0.5 vs zeta>=0.5
# ═══════════════════════════════════════════════════════════════

SNAP_ZETAS = [0.0, 0.2, 0.4, 0.5, 0.7, 1.0]

def plot_potential_snapshots():
    x_plot = np.linspace(-4.2, 4.2, 500)  # wide enough that the x^4 wings
                                           # rising past the zeta=0 wells
                                           # (at +-sqrt(8)~2.83) are unambiguous
    cmap = plt.cm.plasma
    colors = [cmap(i) for i in np.linspace(0.1, 0.85, len(SNAP_ZETAS))]

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.0), sharex=True, sharey=False)
    axes = axes.reshape(-1)
    for ax, zeta, c in zip(axes, SNAP_ZETAS, colors):
        V = potential_q(x_plot, zeta)
        ax.plot(x_plot, V, color=c, lw=3.0)
        regime = "gap closed (metastable)" if zeta < 0.5 else "gap open"
        ax.set_title(rf"$\zeta={zeta:.2f}\,\tau$" + "\n" + regime, fontsize=12)
        ax.tick_params(direction="in")
        ax.grid(True, linestyle=":", alpha=0.3)
    fig.supxlabel(r"$x$", fontsize=20, y=0.02)
    fig.supylabel(r"$U(x,\zeta)$", fontsize=22, y=0.55)
    fig.tight_layout(rect=[0.03, 0.04, 1, 1])
    plot_style._save(fig, OUT_PLOTS, "potential_snapshots")
    print(f"Saved {OUT_PLOTS}/potential_snapshots.{{png,pdf}}")

# ═══════════════════════════════════════════════════════════════
# 2. Inverse spectral gap vs t/tau (= zeta, linear ramp)
# ═══════════════════════════════════════════════════════════════

N_GAP_MODES  = 5
N_SCAN_TIMES = 300
_GAP_COLORS  = ['mediumvioletred', 'darkorange', 'gold', 'seagreen', 'dodgerblue']

def spectral_scan():
    ts = np.linspace(0, TAU, N_SCAN_TIMES)   # t/tau = zeta directly
    eigval_traj = np.zeros((N_SCAN_TIMES, N_GAP_MODES))
    for i, t in enumerate(ts):
        zeta = t / TAU
        L, _ = build_L_q(zeta)
        eigvals, R, l0, Linv = sim.biorthogonal_decomp(L, N)
        eigval_traj[i] = eigvals[1:N_GAP_MODES + 1]
    return ts, eigval_traj

def plot_spectral_gap(ts, eigval_traj):
    fig, ax = plt.subplots()
    for n in range(N_GAP_MODES):
        rate = 1.0 / np.abs(eigval_traj[:, n])
        ax.semilogy(ts / TAU, rate, color=_GAP_COLORS[n % len(_GAP_COLORS)],
                    lw=2.2 if n == 0 else 1.4, ls='-' if n == 0 else '--',
                    label=rf"$1/|\lambda_{{{n+1}}}(t)|$")
    ax.axvline(0.5, color='gray', ls=':', lw=1.2)
    ax.text(0.505, ax.get_ylim()[0], r"$\zeta=0.5$", fontsize=8, color='gray',
            rotation=90, va='bottom')
    ax.set_xlabel(r"Normalized time, $t/\tau$")
    ax.set_ylabel(r"Inverse relaxation rate, $1/|\lambda_n(t)|$")
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()
    plot_style._save(fig, OUT_PLOTS, "spectral_gap")
    print(f"Saved {OUT_PLOTS}/spectral_gap.{{png,pdf}}")

if __name__ == "__main__":
    print("Quartic coalescence model: potential snapshots + spectral gap")
    plot_potential_snapshots()

    ts, eigval_traj = spectral_scan()
    np.save(os.path.join(OUT_DATA, "spectral_gap.npy"),
            dict(ts=ts, eigvals=eigval_traj, n_modes=N_GAP_MODES), allow_pickle=True)
    print(f"Saved {OUT_DATA}/spectral_gap.npy")
    plot_spectral_gap(ts, eigval_traj)

    print("\nDONE.")
