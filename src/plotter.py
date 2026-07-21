"""
PRL-style plotting for the LCD Magnus4 analysis.
====================================================
Two families of figures are produced per run folder:
  1. TVD / D_KL / |W_diss| vs t/T   (moved here from Lcd_full_magnus4.py)
  2. Potential-landscape snapshots with an annealing-protocol inset
     (PRL single-column style, matching the reference figure format)

The double-well annealing parameter is denoted zeta (not lambda) in all
labels/legends below, per the current convention; the harmonic trap's
annealing parameter remains kappa_0.
"""

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

import Lcd_full_magnus4 as sim

# -----------------------------------------------------------------------------
# 1. PRL Publication Style Configuration (applied to every figure below)
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "figure.figsize": (4.2, 3.5),   # PRL single-column size
    "figure.dpi": 300,
    "axes.labelsize": 16,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 14,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

C_BARE = "tomato"   # red    — bare FP
C_LCD  = "darkslategrey"   # blue   — LCD (Magnus4 spectral)
C_AN   = "springgreen"   # green  — analytic CD (harmonic trap only)

DATA_ROOT  = sim.DATA_ROOT
PLOTS_ROOT = sim.PLOTS_ROOT

# ═══════════════════════════════════════════════════════════════
# Run-folder discovery + loading
# ═══════════════════════════════════════════════════════════════

def discover_folders(pattern):
    paths = sorted(glob.glob(os.path.join(DATA_ROOT, pattern)))
    return [os.path.basename(p) for p in paths if os.path.isdir(p)]

def load_run(folder):
    """Load every .npy in results/data/<folder>/ into a dict keyed by stem."""
    out_dir = os.path.join(DATA_ROOT, folder)
    data = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "*.npy"))):
        key = os.path.splitext(os.path.basename(path))[0]
        data[key] = np.load(path, allow_pickle=True)
    if "meta" in data:
        data["meta"] = data["meta"].item()
    return data

def _plots_dir(folder):
    out_dir = os.path.join(PLOTS_ROOT, folder)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _save(fig, out_dir, name):
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"), format="pdf")
    fig.savefig(os.path.join(out_dir, f"{name}.png"), format="png")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════
# 1. TVD / KL / |W_diss| line plots (moved from Lcd_full_magnus4.py)
# ═══════════════════════════════════════════════════════════════

def plot_dw_timeseries(folder):
    d = load_run(folder)
    T = d["meta"]["T"]
    out_dir = _plots_dir(folder)
    t_frac = d["ts"] / T

    fig, ax = plt.subplots()
    ax.semilogy(t_frac, d["tvd_b"], color=C_BARE, lw=4.0, label="Bare")
    ax.semilogy(t_frac, d["tvd_l"], color=C_LCD, lw=4.0, ls="--", label="LCD")
    ax.set_xlabel(r"$t/\tau$", fontsize=18)
    ax.set_ylabel(r"$\mathrm{TVD}(\rho,\pi_{\rm eq})$")
    ax.legend(frameon=False)
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "tvd")

    fig, ax = plt.subplots()
    ax.semilogy(t_frac, np.maximum(d["kl_b"], 1e-16), color=C_BARE, lw=4.0, label="Bare")
    ax.semilogy(t_frac, np.maximum(d["kl_l"], 1e-16), color=C_LCD, lw=4.0, ls="--", label="LCD")
    ax.set_xlabel(r"$t/\tau$", fontsize=18)
    ax.set_ylabel(r"$D_{\rm KL}(\rho\|\pi_{\rm eq})$")
    ax.legend(frameon=False)
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "kl")

    fig, ax = plt.subplots()
    ax.semilogy(t_frac, np.abs(d["Wd_b"]), color=C_BARE, lw=4.0, label="Bare")
    ax.semilogy(t_frac, np.abs(d["Wd_l"]), color=C_LCD, lw=4.0, ls="--", label="LCD")
    ax.set_xlabel(r"$t/\tau$", fontsize=18)
    ax.set_ylabel(r"$|W_{\rm diss}(t)|$")
    ax.legend(frameon=False)
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "wdiss")

    print(f"  [DW]  tvd/kl/wdiss -> results/plots/{folder}/")

def plot_harm_timeseries(folder):
    d = load_run(folder)
    T = d["meta"]["T"]
    out_dir = _plots_dir(folder)
    t_frac = d["ts"] / T

    fig, ax = plt.subplots()
    ax.semilogy(t_frac, d["tvd_b"], color=C_BARE, lw=4.0, label="Bare")
    ax.semilogy(t_frac, d["tvd_a"], color=C_AN, ls="solid", lw=4.0, label="Analytic CD")
    ax.semilogy(t_frac, d["tvd_l"], color=C_LCD, ls="--", lw=4.0, label="LCD (spectral)")
    ax.set_xlabel(r"$t/\tau$", fontsize=18)
    ax.set_ylabel(r"$\mathrm{TVD}(\rho,\pi_{\rm eq})$")
    ax.legend(frameon=False, loc='center right')
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "tvd")

    fig, ax = plt.subplots()
    ax.semilogy(t_frac, np.maximum(d["kl_b"], 1e-16), color=C_BARE, lw=4.0, label="Bare")
    ax.semilogy(t_frac, np.maximum(d["kl_a"], 1e-16), color=C_AN, ls="solid", lw=4.0, label="Analytic CD")
    ax.semilogy(t_frac, np.maximum(d["kl_l"], 1e-16), color=C_LCD, ls="--", lw=4.0, label="LCD (spectral)")
    ax.set_xlabel(r"$t/\tau$", fontsize=18)
    ax.set_ylabel(r"$D_{\rm KL}(\rho\|\pi_{\rm eq})$")
    ax.legend(frameon=False)
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "kl")

    fig, ax = plt.subplots()
    ax.semilogy(t_frac, np.abs(d["Wd_b"]), color=C_BARE, lw=4.0, label="Bare FP")
    ax.semilogy(t_frac, np.abs(d["Wd_a"]), color=C_AN, ls="solid", lw=4.0, label="Analytic CD")
    ax.semilogy(t_frac, np.abs(d["Wd_l"]), color=C_LCD, ls="--", lw=4.0, label="LCD (spectral)")
    ax.set_xlabel(r"$t/\tau$", fontsize=18)
    ax.set_ylabel(r"$|W_{\rm diss}(t)|$")
    ax.legend(frameon=False, loc='best')
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "wdiss")

    print(f"  [Harm] tvd/kl/wdiss -> results/plots/{folder}/")

# ═══════════════════════════════════════════════════════════════
# 2. Potential-landscape + annealing-protocol inset (PRL example format)
# ═══════════════════════════════════════════════════════════════

def _landscape_plot(x_vals, t_vals, t_snapshots, protocol_vals, protocol_snapshots,
                     dprotocol_vals, protocol_i, protocol_f, potential_fn,
                     xlabel, ylabel, protocol_symbol, dprotocol_symbol,
                     protocol_label_fmt, out_dir, name):
    fig, ax = plt.subplots()

    cmap = plt.cm.plasma
    colors = [cmap(i) for i in np.linspace(0.1, 0.85, len(t_snapshots))]

    for idx, t_snap in enumerate(t_snapshots):
        p = protocol_snapshots[idx]
        label = protocol_label_fmt(t_snap, p)
        v_vals = potential_fn(x_vals, p)
        ax.plot(x_vals, v_vals, color=colors[idx], lw=1.5, label=label)

    ax.set_xlabel(xlabel,fontsize=20)
    ax.set_ylabel(ylabel)
    ax.set_xlim(x_vals.min(), x_vals.max())
    all_v = np.array([potential_fn(x_vals, p) for p in protocol_snapshots])
    pad = 0.1 * (all_v.max() - all_v.min())
    ax.set_ylim(all_v.min() - pad, all_v.max() + pad)
    # ax.legend(loc="lower center", frameon=True, facecolor="white",
    #           edgecolor="none", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)

    ax_inset = ax.inset_axes([0.30, 0.56, 0.40, 0.38])

    color_p = "#1f77b4"
    ax_inset.plot(t_vals, protocol_vals, color=color_p, lw=1.2)
    ax_inset.set_xlabel(r"$t$", labelpad=1)
    ax_inset.set_ylabel(protocol_symbol, color=color_p, labelpad=1)
    ax_inset.tick_params(axis="y", labelcolor=color_p, labelsize=7)
    ax_inset.tick_params(axis="x", labelsize=7)
    ax_inset.set_yticks(sorted({protocol_i, protocol_f}))

    ax_inset_twin = ax_inset.twinx()
    color_dp = "#d62728"
    ax_inset_twin.plot(t_vals, dprotocol_vals, color=color_dp, linestyle="--", lw=1.2)
    ax_inset_twin.set_ylabel(dprotocol_symbol, color=color_dp, labelpad=4)
    ax_inset_twin.tick_params(axis="y", labelcolor=color_dp, labelsize=7)

    for t_snap in t_snapshots:
        ax_inset.axvline(x=t_snap, color="gray", linestyle=":", lw=0.5, alpha=0.6)

    ax_inset.tick_params(direction="in")
    ax_inset_twin.tick_params(direction="in")

    fig.tight_layout()
    _save(fig, out_dir, name)

def plot_landscape_dw(folder, n_snapshots=5):
    """
    Double-well potential V(x,zeta) at n_snapshots equally time-spaced
    points along the annealing protocol, with a zeta(t)/zeta_dot(t)
    inset. The annealing parameter is labelled zeta (not lambda).
    """
    d = load_run(folder)
    T = d["meta"]["T"]
    out_dir = _plots_dir(folder)

    # Zoomed in relative to the simulation grid (x_np spans +-2.5) so the
    # double-well dip is visible instead of being washed out by the x^4 wings.
    x_vals = np.linspace(-1.8, 1.8, 400)
    t_vals = np.linspace(0, T, 200)
    t_snapshots = np.linspace(0, T, n_snapshots)
    zeta_snapshots = sim.lam_dw(t_snapshots, T)

    _landscape_plot(
        x_vals=x_vals, t_vals=t_vals, t_snapshots=t_snapshots,
        protocol_vals=sim.lam_dw(t_vals, T), protocol_snapshots=zeta_snapshots,
        dprotocol_vals=sim.lam_dw_dot(t_vals, T),
        protocol_i=sim.lam_i_dw, protocol_f=sim.lam_f_dw,
        potential_fn=sim.potential_dw,
        xlabel=r"$x$",
        ylabel=r"Potential profile, $V(x, \zeta(t))$",
        protocol_symbol=r"$\zeta$", dprotocol_symbol=r"$\dot{\zeta}$",
        protocol_label_fmt=lambda t_snap, p: rf"$t = {t_snap:.2f}\,\tau$ ($\zeta = {p:.2f}$)",
        out_dir=out_dir, name="landscape",
    )
    print(f"  [DW]  landscape -> results/plots/{folder}/landscape.pdf")

def plot_landscape_harm(folder, n_snapshots=5):
    """
    Harmonic potential V(q,t) = (1/2) kappa_0(t) q^2 at n_snapshots
    equally time-spaced points, with a kappa_0(t)/kappa_0_dot(t) inset.
    """
    d = load_run(folder)
    T = d["meta"]["T"]
    out_dir = _plots_dir(folder)

    def potential_harm(qv, kap):
        return 0.5 * kap * qv**2

    x_vals = np.linspace(sim.q_np.min(), sim.q_np.max(), 400) * 0.75
    t_vals = np.linspace(0, T, 200)
    t_snapshots = np.linspace(0, T, n_snapshots)
    kap_snapshots = sim.kappa0(t_snapshots, T)

    _landscape_plot(
        x_vals=x_vals, t_vals=t_vals, t_snapshots=t_snapshots,
        protocol_vals=sim.kappa0(t_vals, T), protocol_snapshots=kap_snapshots,
        dprotocol_vals=sim.kappa0_dot(t_vals, T),
        protocol_i=sim.kap_i, protocol_f=sim.kap_f,
        potential_fn=potential_harm,
        xlabel=r"$x$",
        ylabel=r"Potential profile, $V(x, \kappa_0(t))$",
        protocol_symbol=r"$\kappa_0$", dprotocol_symbol=r"$\dot{\kappa}_0$",
        protocol_label_fmt=lambda t_snap, p: rf"$t = {t_snap:.2f}\,\tau$ ($\kappa_0 = {p:.2f}$)",
        out_dir=out_dir, name="landscape",
    )
    print(f"  [Harm] landscape -> results/plots/{folder}/landscape.pdf")

# ═══════════════════════════════════════════════════════════════
# MAIN: auto-discover every run folder and plot it
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    dw_folders   = discover_folders("DW_T_*_dt_*")
    harm_folders = discover_folders("harmonic_T_*_dt_*")

    print(f"Found {len(dw_folders)} double-well run(s), {len(harm_folders)} harmonic run(s).")

    for folder in dw_folders:
        plot_dw_timeseries(folder)
        plot_landscape_dw(folder)

    for folder in harm_folders:
        plot_harm_timeseries(folder)
        plot_landscape_harm(folder)

    print("\nAll figures written under results/plots/<run_folder>/")
