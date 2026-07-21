"""
Spectral diagnostics -- LCD Magnus4 analysis, T=0.1, dt=1e-5.
================================================================
Figures (per potential, written to results/plots/<folder>/):
  1. spectral_gap.pdf     -- 1/|lambda_n(t)| for n=1..5 vs t/T
  2. r0_vs_pieq.pdf       -- r_0(t) vs pi_eq(t) at 5 snapshot times
  3-4. tvd/kl, wdiss      -- tvd/kl regenerated via plotter's standard
                             functions; wdiss uses a LOCAL Simpson-refined
                             heat integral (see wdiss_refined below) that
                             supersedes plotter's version for these two
                             folders only -- Lcd_full_magnus4.py itself is
                             untouched, so re-running plain plotter.py on
                             these folders later will revert to the
                             coarser (but still correct at t=T) version.
  5. rho_snapshots.pdf    -- rho(x,t) bare vs LCD vs pi_eq at 6 times
  6. spectral_coeffs.pdf  -- c_n bar chart + cumulative weight, evaluated
                             at the time of maximum total drive
                             sum_n|c_n(t)| (auto-selected)

Data (.npy) is written to results/data/<folder>/ alongside the standard
run_and_save_dw/harm output, under names not used by that pipeline.

Run:
    python spectral_diagnostics.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

import Lcd_full_magnus4 as sim
import plotter as plot_style  # PRL rcParams (applied on import) + I/O helpers

T_VAL  = 0.1
DT_VAL = 1e-5

DW_FOLDER   = sim.dw_folder_name(T_VAL, DT_VAL)
HARM_FOLDER = sim.harm_folder_name(T_VAL, DT_VAL)

RHO_SNAP_FRACS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
R0_SNAP_FRACS  = [0.0, 0.25, 0.5, 0.75, 1.0]

N_GAP_MODES  = 5
N_SCAN_TIMES = 300
CN_BAR_NMAX  = 15
CN_ACTIVE_THRESHOLD = 1e-4

C_BARE = plot_style.C_BARE
C_LCD  = plot_style.C_LCD
C_AN   = plot_style.C_AN
_save  = plot_style._save

POTENTIALS = {
    'dw': dict(
        folder=DW_FOLDER, N=sim.N_dw, grid=sim.x_np,
        protocol=sim.lam_dw, protocol_dot=sim.lam_dw_dot,
        build_L=sim.build_L_dw, get_dL=sim.get_dL_dw,
        rho_eq=sim.rho_eq_dw, make_gen=sim.make_gen_dw,
        potential_fn=lambda t: sim.potential_dw(sim.x_np, sim.lam_dw(t, T_VAL)),
        free_energy_fn=lambda t: sim.free_energy_dw(sim.lam_dw(t, T_VAL)),
        dV_dprotocol=sim.x_np,  # dV/dzeta = x  (V = x^4-2x^2+zeta*x)
        lcd_mode='lcd', xlabel=r"$x$",
        title="Double well",
    ),
    'harm': dict(
        folder=HARM_FOLDER, N=sim.N_h, grid=sim.q_np,
        protocol=sim.kappa0, protocol_dot=sim.kappa0_dot,
        build_L=sim.build_L_harm, get_dL=sim.get_dL_harm,
        rho_eq=sim.rho_eq_harm, make_gen=sim.make_gen_harm,
        potential_fn=lambda t: 0.5 * sim.kappa0(t, T_VAL) * sim.q_np**2,
        free_energy_fn=lambda t: sim.free_energy_harm(sim.kappa0(t, T_VAL)),
        dV_dprotocol=0.5 * sim.q_np**2,  # dV/dkappa0 = q^2/2
        lcd_mode='spectral', xlabel=r"$x$",
        title="Harmonic trap",
    ),
}

# ═══════════════════════════════════════════════════════════════
# 1. Main TVD/KL/Wdiss run (reuses the standard pipeline unmodified)
# ═══════════════════════════════════════════════════════════════

def ensure_main_run():
    """Writes results/data/{DW_FOLDER,HARM_FOLDER}/ if not already present."""
    dw_meta = os.path.join(sim.DATA_ROOT, DW_FOLDER, "meta.npy")
    if not os.path.exists(dw_meta):
        sim.run_and_save_dw(T_VAL, DT_VAL)
    else:
        print(f"[skip] {DW_FOLDER}/ already exists")

    harm_meta = os.path.join(sim.DATA_ROOT, HARM_FOLDER, "meta.npy")
    if not os.path.exists(harm_meta):
        sim.run_and_save_harm(T_VAL, DT_VAL)
    else:
        print(f"[skip] {HARM_FOLDER}/ already exists")

# ═══════════════════════════════════════════════════════════════
# 2. Spectral scan: eigenvalue trajectory (plot 1) + c_n(t) (plot 6)
# ═══════════════════════════════════════════════════════════════

def spectral_scan(key, n_time=N_SCAN_TIMES, n_gap_modes=N_GAP_MODES):
    p = POTENTIALS[key]
    N = p['N']
    ts = np.linspace(0, T_VAL, n_time)
    eigval_traj = np.zeros((n_time, n_gap_modes))
    cn_traj     = np.zeros((n_time, N - 1))

    for i, t in enumerate(ts):
        val  = p['protocol'](t, T_VAL)
        vdot = p['protocol_dot'](t, T_VAL)
        L, _ = p['build_L'](val)
        dL   = p['get_dL'](val, vdot)
        eigvals, R, l0, Linv = sim.biorthogonal_decomp(L, N)
        r0 = R[:, 0]
        eigval_traj[i] = eigvals[1:n_gap_modes + 1]
        for n in range(1, N):
            if abs(eigvals[n]) > 1e-10:
                cn_traj[i, n - 1] = (Linv[n, :] @ dL @ r0) / eigvals[n]

    return ts, eigval_traj, cn_traj

# ═══════════════════════════════════════════════════════════════
# 3. r0(t) vs pi_eq(t) at snapshot times
# ═══════════════════════════════════════════════════════════════

def r0_vs_pieq(key, fracs=R0_SNAP_FRACS):
    p = POTENTIALS[key]
    N = p['N']
    ts = np.array(fracs) * T_VAL
    r0_arr = np.zeros((len(fracs), N))
    pi_arr = np.zeros((len(fracs), N))
    for i, t in enumerate(ts):
        val = p['protocol'](t, T_VAL)
        L, pi = p['build_L'](val)
        eigvals, R, l0, Linv = sim.biorthogonal_decomp(L, N)
        r0_arr[i] = R[:, 0]
        pi_arr[i] = pi
    maxdiff = np.max(np.abs(r0_arr - pi_arr), axis=1)
    return ts, r0_arr, pi_arr, maxdiff

# ═══════════════════════════════════════════════════════════════
# 4. rho(x,t) trajectory snapshots: bare vs LCD vs pi_eq
# ═══════════════════════════════════════════════════════════════

def rho_snapshots(key, dt=None, fracs=RHO_SNAP_FRACS):
    # dt defaults to None (resolved to the CURRENT DT_VAL here, not at
    # def time) so overriding module-level DT_VAL for a different T (see
    # run_diagnostics_for_T.py) is actually honored -- `dt=DT_VAL` as a
    # literal default binds once at import time and silently goes stale.
    if dt is None:
        dt = DT_VAL
    p = POTENTIALS[key]
    ts_full = np.arange(0, T_VAL + dt, dt)
    snap_ts = np.array(fracs) * T_VAL
    snap_idx = {int(np.argmin(np.abs(ts_full - st))): j for j, st in enumerate(snap_ts)}

    pi_eq = np.array([p['rho_eq'](t, T_VAL) for t in snap_ts])

    out = {}
    for label, mode, step_fn in [
        ('bare', 'bare', sim.midpoint_step),
        ('lcd',  p['lcd_mode'], sim.magnus4_step),
    ]:
        gen  = p['make_gen'](T_VAL, mode)
        rho  = p['rho_eq'](0, T_VAL).copy()
        snaps = [None] * len(fracs)
        for i, t in enumerate(ts_full):
            if i in snap_idx:
                snaps[snap_idx[i]] = rho.copy()
            if t < T_VAL - dt / 2:
                rho = step_fn(rho, t, dt, gen)
        out[label] = np.array(snaps)
        print(f"    [{key}] {label} snapshots done")

    out['pi_eq']   = pi_eq
    out['snap_ts'] = snap_ts
    return out

# ═══════════════════════════════════════════════════════════════
# 4b. Simpson-refined heat integral -> Wdiss(t) (bare + LCD/spectral)
# ═══════════════════════════════════════════════════════════════
#
# Lcd_full_magnus4.run_dw/run_harm accumulate the heat integral as
# Q += U(t+dt/2) . (rho(t+dt)-rho(t)) per step -- a single-midpoint
# quadrature of the (smooth, closed-form) potential weighting the exact
# simulated jump in rho. That's O(dt^2) globally, same class of error
# run_harm_analytic's docstring already flags and fixes for the analytic
# CD path via fine substeps. It was never extended to bare/LCD. Confirmed
# empirically: DW T=0.1 LCD max|Wdiss(t)| = 7.84e-7 at dt=1e-4 vs 7.84e-9
# at dt=1e-5 -- exactly the 100x drop for a 10x finer dt that O(dt^2)
# predicts, settling to the correct ~1e-12 value only at t=T where the
# quadrature error happens to cancel.
#
# A first attempt at fixing this locally (Simpson-averaging the potential
# U(t) sampled at 3 points per step, keeping the same Q += U_avg . drho
# bookkeeping) was tried and REJECTED: it barely helped the double well
# (max|Wdiss| 7.84e-9 -> 5.93e-9) and made the harmonic trap WORSE (final
# 1.32e-9 -> 2.64e-9, monotonically drifting instead of returning near
# zero). Reweighting U(t) doesn't address the actual error source, which
# is not knowing how rho(t) itself (not U(t)) varies within a step.
#
# The fix that actually works (validated in Wdiss_highorder.py, provided
# separately): abandon the Delta-E-minus-heat split entirely and compute
# work directly as the standard stochastic-thermodynamics power integral,
#     W(t) = integral_0^t  protocol_dot(s) * <dV/dprotocol>_{rho(s)}  ds,
# using scipy's cumulative_simpson (composite 4th-order quadrature) over
# the already-available per-step power samples -- no extra eig() calls,
# same rho(t) trajectory as before. Validated at T=0.1, dt=1e-5: max
# |Wdiss| drops to 2.8e-12 (DW LCD), 5.2e-12 (harmonic LCD), 8.0e-14
# (harmonic analytic), growing smoothly/monotonically from 0 rather than
# spiking mid-protocol and snapping back at t=T.

from scipy.integrate import cumulative_simpson

def wdiss_refined(key, dt=None):
    if dt is None:  # see rho_snapshots' comment: a literal DT_VAL default goes stale
        dt = DT_VAL
    p = POTENTIALS[key]
    ts = np.arange(0, T_VAL + dt, dt)
    DF0 = p['free_energy_fn'](0.0)
    dV_dp = p['dV_dprotocol']

    out = {}
    for label, mode, step_fn in [
        ('bare', 'bare', sim.midpoint_step),
        ('lcd',  p['lcd_mode'], sim.magnus4_step),
    ]:
        gen = p['make_gen'](T_VAL, mode)
        rho = p['rho_eq'](0, T_VAL).copy()
        powers = np.zeros(len(ts))
        for i, t in enumerate(ts):
            pdot = p['protocol_dot'](t, T_VAL)
            powers[i] = pdot * np.dot(dV_dp, rho)
            if t < T_VAL - dt / 2:
                rho = step_fn(rho, t, dt, gen)
        W  = np.concatenate([[0.0], cumulative_simpson(powers, x=ts)])[:len(ts)]
        DF = np.array([p['free_energy_fn'](t) - DF0 for t in ts])
        Wd = W - DF
        out[label] = Wd
        print(f"    [{key}] {label}: refined max|Wdiss|={np.abs(Wd).max():.3e}, "
              f"final={Wd[-1]:.3e}")

    out['ts'] = ts
    return out

# ═══════════════════════════════════════════════════════════════
# 5. Driver: compute everything, save .npy under results/data/<folder>/
# ═══════════════════════════════════════════════════════════════

def compute_and_save(key):
    p = POTENTIALS[key]
    out_dir = os.path.join(sim.DATA_ROOT, p['folder'])
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{key}] spectral scan ({N_SCAN_TIMES} time points)...")
    ts_scan, eigval_traj, cn_traj = spectral_scan(key)
    total_drive = np.sum(np.abs(cn_traj), axis=1)
    i_star = int(np.argmax(total_drive))
    t_star = ts_scan[i_star]
    cn_star = cn_traj[i_star]
    cumulative_pct = np.cumsum(np.abs(cn_star)) / np.sum(np.abs(cn_star)) * 100.0

    np.save(os.path.join(out_dir, "spectral_gap.npy"),
            dict(ts=ts_scan, eigvals=eigval_traj, n_modes=N_GAP_MODES), allow_pickle=True)
    np.save(os.path.join(out_dir, "spectral_coeffs.npy"),
            dict(ts_scan=ts_scan, cn_traj=cn_traj, t_star=t_star, i_star=i_star,
                 cn_snapshot=cn_star, cumulative_pct=cumulative_pct,
                 threshold=CN_ACTIVE_THRESHOLD), allow_pickle=True)

    print(f"[{key}] r0(t) vs pi_eq(t) at fracs={R0_SNAP_FRACS}...")
    ts_r0, r0_arr, pi_arr, maxdiff = r0_vs_pieq(key)
    np.save(os.path.join(out_dir, "r0_check.npy"),
            dict(fracs=R0_SNAP_FRACS, ts=ts_r0, r0=r0_arr, pi=pi_arr, maxdiff=maxdiff),
            allow_pickle=True)
    print(f"    max|r0-pi_eq| per snapshot: {maxdiff}")

    print(f"[{key}] rho(x,t) snapshots (bare vs LCD) at dt={DT_VAL}...")
    snaps = rho_snapshots(key)
    np.save(os.path.join(out_dir, "rho_snapshots.npy"),
            dict(fracs=RHO_SNAP_FRACS, snap_ts=snaps['snap_ts'],
                 bare=snaps['bare'], lcd=snaps['lcd'], pi_eq=snaps['pi_eq']),
            allow_pickle=True)

    print(f"[{key}] Simpson-refined Wdiss(t) (bare + LCD/spectral)...")
    wd = wdiss_refined(key)
    np.save(os.path.join(out_dir, "Wd_b_refined.npy"), wd['bare'], allow_pickle=True)
    np.save(os.path.join(out_dir, "Wd_l_refined.npy"), wd['lcd'],  allow_pickle=True)

    print(f"[{key}] done -> results/data/{p['folder']}/")

# ═══════════════════════════════════════════════════════════════
# 6. Plots
# ═══════════════════════════════════════════════════════════════

_GAP_COLORS = ['mediumvioletred', 'indigo', 'lime', 'tomato', 'yellowgreen']

def plot_spectral_gap(key):
    p = POTENTIALS[key]
    out_dir = plot_style._plots_dir(p['folder'])
    d = np.load(os.path.join(sim.DATA_ROOT, p['folder'], "spectral_gap.npy"),
                allow_pickle=True).item()
    ts, eigvals = d['ts'], d['eigvals']

    fig, ax = plt.subplots()
    for n in range(eigvals.shape[1]):
        rate = 1.0 / np.abs(eigvals[:, n])
        ax.plot(ts / T_VAL, rate, color=_GAP_COLORS[n % len(_GAP_COLORS)],
                 lw=3.0 if n == 0 else 1.0, ls='-' if n == 0 else '--',
                 label=rf"$1/|\lambda_{{{n+1}}}(t)|$")
    ax.set_xlabel(r"$t/\tau$", fontsize=18)
    ax.set_ylabel(r"Inverse relaxation rate, $1/|\lambda_n(t)|$", fontsize=13)
    ax.legend(frameon=False, ncol=2, fontsize=10.0, loc="best")
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "spectral_gap")
    print(f"  [{key}] spectral_gap -> results/plots/{p['folder']}/")

def plot_r0_vs_pieq(key):
    p = POTENTIALS[key]
    out_dir = plot_style._plots_dir(p['folder'])
    d = np.load(os.path.join(sim.DATA_ROOT, p['folder'], "r0_check.npy"),
                allow_pickle=True).item()
    fracs, r0_arr, pi_arr, maxdiff = d['fracs'], d['r0'], d['pi'], d['maxdiff']
    grid = p['grid']

    n = len(fracs)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.6), sharey=True)
    for j, ax in enumerate(axes):
        ax.plot(grid, r0_arr[j], color=C_LCD, ls='--', lw=4.0, label=r"$r_0(t)$")
        ax.plot(grid, pi_arr[j], color='chartreuse',ls=':' ,lw=4.0, label=r"$\pi_{\rm eq}(t)$")
        ax.set_title(rf"$t={fracs[j]:.2f}\,\tau$" + "\n" + rf"max diff: {maxdiff[j]:.1e}",
                     fontsize=14)
        ax.set_xlabel(r"$x$", fontsize=18)
        ax.tick_params(direction="in")
        if j == 0:
            ax.set_ylabel("Probability density", fontsize=13)
            ax.legend(frameon=False, fontsize=12.0, loc="upper left")
    fig.tight_layout()
    _save(fig, out_dir, "r0_vs_pieq")
    print(f"  [{key}] r0_vs_pieq -> results/plots/{p['folder']}/  (max diffs: {maxdiff})")

def plot_rho_snapshots(key):
    p = POTENTIALS[key]
    out_dir = plot_style._plots_dir(p['folder'])
    d = np.load(os.path.join(sim.DATA_ROOT, p['folder'], "rho_snapshots.npy"),
                allow_pickle=True).item()
    fracs = d['fracs']
    grid = p['grid']
    n = len(fracs)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows + 0.6),
                              sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    for j in range(n):
        ax = axes[j]
        ax.plot(grid, d['bare'][j],  color=C_BARE, lw=4.0, label="Bare")
        ax.plot(grid, d['lcd'][j],   color=C_LCD,  lw=4.0, ls='--', label="LCD")
        ax.plot(grid, d['pi_eq'][j], color='chartreuse', lw=4.0, ls=':',  label=r"$\pi_{\rm eq}(t)$")
        ax.set_title(rf"$t={fracs[j]:.2f}\,\tau$", fontsize=15)
        ax.tick_params(direction="in")
    for j in range(n, len(axes)):
        axes[j].axis('off')

    # ── High-Precision Label & Legend Tucking ───────────────────────────────
    
    # Identify the central axis index of the bottom plotting row
    # (For 3 columns, index 1 is the bottom-middle plot, which perfectly centers the label)
    bottom_center_idx = ncols + 1 if nrows > 1 and n > ncols + 1 else 1
    if bottom_center_idx < n:
        # Bind the x-label directly to the bottom axis frame to eliminate gaps
        axes[bottom_center_idx].set_xlabel(r"$x$", fontsize=26, labelpad=4)

    # Set up global y-label relative to the figure boundary
    bottom_margin = 0.12  # Lowered from 0.18 to pull everything downward
    fig.supylabel(r"$\rho(x,t)$", fontsize=30, y=(1 + bottom_margin) / 2)
    
    legend_handles = [
        Line2D([0], [0], color=C_BARE, lw=7.0, label="Bare"),
        Line2D([0], [0], color=C_LCD,  lw=7.0, ls='--', label="LCD"),
        Line2D([0], [0], color='chartreuse', lw=7.0, ls=':', label=r"$\pi_{\rm eq}(t)$"),
    ]
    
    # Placed the legend bounding box immediately below the x-axis frame boundary
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
               frameon=False, fontsize=16.0, bbox_to_anchor=(0.5, 0.06))
    
    # Re-calibrated tight layout bounding box allocation
    fig.tight_layout(rect=[0.02, bottom_margin, 1, 1])
    
    _save(fig, out_dir, "rho_snapshots")
    print(f"  [{key}] rho_snapshots -> results/plots/{p['folder']}/")

# def plot_rho_snapshots(key):
#     p = POTENTIALS[key]
#     out_dir = plot_style._plots_dir(p['folder'])
#     d = np.load(os.path.join(sim.DATA_ROOT, p['folder'], "rho_snapshots.npy"),
#                 allow_pickle=True).item()
#     fracs = d['fracs']
#     grid = p['grid']
#     n = len(fracs)
#     ncols = 3
#     nrows = int(np.ceil(n / ncols))
#     fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows + 0.6),
#                               sharex=True, sharey=True)
#     axes = np.array(axes).reshape(-1)
#     for j in range(n):
#         ax = axes[j]
#         ax.plot(grid, d['bare'][j],  color=C_BARE, lw=4.0, label="Bare")
#         ax.plot(grid, d['lcd'][j],   color=C_LCD,  lw=4.0, ls='--', label="LCD")
#         ax.plot(grid, d['pi_eq'][j], color='chartreuse', lw=4.0, ls=':',  label=r"$\pi_{\rm eq}(t)$")
#         ax.set_title(rf"$t={fracs[j]:.2f}\,\tau$", fontsize=14)
#         ax.tick_params(direction="in")
#     for j in range(n, len(axes)):
#         axes[j].axis('off')

#     # Single shared x/y labels for the whole grid, plus one shared
#     # horizontal legend below the figure (thicker proxy handles so the
#     # legend bars read clearly, independent of the lw=4.0 plotted lines).
#     bottom_margin = 0.18
#     fig.supxlabel(r"$x$", fontsize=20, y=bottom_margin - 0.025)
#     fig.supylabel(r"$\rho(x,t)$", fontsize=30, y=(1 + bottom_margin) / 2)
#     legend_handles = [
#         Line2D([0], [0], color=C_BARE, lw=7.0, label="Bare"),
#         Line2D([0], [0], color=C_LCD,  lw=7.0, ls='--', label="LCD"),
#         Line2D([0], [0], color='chartreuse', lw=7.0, ls=':', label=r"$\pi_{\rm eq}(t)$"),
#     ]
#     fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
#                frameon=False, fontsize=14.0, bbox_to_anchor=(0.5, 0.0))
#     fig.tight_layout(rect=[0.02, bottom_margin, 1, 1])
#     _save(fig, out_dir, "rho_snapshots")
#     print(f"  [{key}] rho_snapshots -> results/plots/{p['folder']}/")

def plot_wdiss_refined(key):
    """
    Overwrites wdiss.png/.pdf for this run folder with the Simpson-refined
    Wdiss(t) (see wdiss_refined). Supersedes plotter.py's version, which
    uses the coarser quadrature baked into Lcd_full_magnus4.run_dw/run_harm
    -- re-running plain plotter.py on this folder later will revert it.
    """
    p = POTENTIALS[key]
    out_dir = os.path.join(sim.DATA_ROOT, p['folder'])
    plots_dir = plot_style._plots_dir(p['folder'])

    Wd_b = np.load(os.path.join(out_dir, "Wd_b_refined.npy"))
    Wd_l = np.load(os.path.join(out_dir, "Wd_l_refined.npy"))
    ts   = np.load(os.path.join(out_dir, "ts.npy"))

    fig, ax = plt.subplots()
    t_frac = ts / T_VAL
    ax.semilogy(t_frac, np.abs(Wd_b), color=C_BARE, lw=4.0, label="Bare")
    if key == 'harm':
        Wd_a = np.load(os.path.join(out_dir, "Wd_a.npy"))  # already fine-substep refined
        ax.semilogy(t_frac, np.abs(Wd_a), color=C_AN, ls="solid", lw=4.0, label="Analytic CD")
        ax.semilogy(t_frac, np.abs(Wd_l), color=C_LCD, ls="--", lw=4.0, label="LCD (spectral)")
    else:
        ax.semilogy(t_frac, np.abs(Wd_l), color=C_LCD, ls="--", lw=4.0,label="LCD")
    ax.set_xlabel(r"$t/\tau$", fontsize=18)
    ax.set_ylabel(r"$|W_{\rm diss}(t)|$", fontsize=14)
    
    # Shifted upward along the right boundary via bbox_to_anchor (y adjusted from 0.5 to 0.6)
    ax.legend(frameon=False, loc='center right', bbox_to_anchor=(1.0, 0.63), fontsize=14)
    
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()
    _save(fig, plots_dir, "wdiss")
    print(f"  [{key}] wdiss (Simpson-refined) -> results/plots/{p['folder']}/")

# def plot_wdiss_refined(key):
#     """
#     Overwrites wdiss.png/.pdf for this run folder with the Simpson-refined
#     Wdiss(t) (see wdiss_refined). Supersedes plotter.py's version, which
#     uses the coarser quadrature baked into Lcd_full_magnus4.run_dw/run_harm
#     -- re-running plain plotter.py on this folder later will revert it.
#     """
#     p = POTENTIALS[key]
#     out_dir = os.path.join(sim.DATA_ROOT, p['folder'])
#     plots_dir = plot_style._plots_dir(p['folder'])

#     Wd_b = np.load(os.path.join(out_dir, "Wd_b_refined.npy"))
#     Wd_l = np.load(os.path.join(out_dir, "Wd_l_refined.npy"))
#     ts   = np.load(os.path.join(out_dir, "ts.npy"))

#     fig, ax = plt.subplots()
#     t_frac = ts / T_VAL
#     ax.semilogy(t_frac, np.abs(Wd_b), color=C_BARE, lw=4.0, label="Bare")
#     if key == 'harm':
#         Wd_a = np.load(os.path.join(out_dir, "Wd_a.npy"))  # already fine-substep refined
#         ax.semilogy(t_frac, np.abs(Wd_a), color=C_AN, ls="solid", lw=4.0, label="Analytic CD")
#         ax.semilogy(t_frac, np.abs(Wd_l), color=C_LCD, ls="--", lw=4.0, label="LCD (spectral)")
#     else:
#         ax.semilogy(t_frac, np.abs(Wd_l), color=C_LCD, ls="--", lw=4.0,label="LCD")
#     ax.set_xlabel(r"$t/\tau$", fontsize=18)
#     ax.set_ylabel(r"$|W_{\rm diss}(t)|$", fontsize=14)
#     ax.legend(frameon=False, loc='center right', fontsize=11)
#     ax.grid(True, linestyle=":", alpha=0.3)
#     fig.tight_layout()
#     _save(fig, plots_dir, "wdiss")
#     print(f"  [{key}] wdiss (Simpson-refined) -> results/plots/{p['folder']}/")

def plot_spectral_coeffs(key):
    p = POTENTIALS[key]
    out_dir = plot_style._plots_dir(p['folder'])
    d = np.load(os.path.join(sim.DATA_ROOT, p['folder'], "spectral_coeffs.npy"),
                allow_pickle=True).item()
    cn = d['cn_snapshot']
    n_max = min(CN_BAR_NMAX, len(cn))
    n_vals = np.arange(1, n_max + 1)
    c_vals = cn[:n_max]
    cumulative = d['cumulative_pct'][:n_max]
    active = np.abs(c_vals) > CN_ACTIVE_THRESHOLD

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    ax.bar(n_vals[active], c_vals[active], color=C_LCD,
           label=rf"Active ($|c_n|>{CN_ACTIVE_THRESHOLD:g}$)")
    ax.bar(n_vals[~active], c_vals[~active], color="lightgray", label="Negligible")
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel("Liouvillian eigenmode index, $n$", fontsize=13)
    ax.set_ylabel(r"$c_n$", fontsize=16)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=10.0)
    ax.grid(True, linestyle=":", alpha=0.3)

    ax2 = axes[1]
    ax2.plot(n_vals, cumulative, color=C_LCD, marker='o', ms=3)
    ax2.axhline(95, color='gray', ls=':', lw=0.8)
    ax2.axhline(99, color='gray', ls=':', lw=0.8)
    ax2.set_xlabel("Liouvillian eigenmode index, $n$", fontsize=13)
    ax2.set_ylabel(r"Cumulative $\sum_{k\leq n}|c_k|$ (%)", fontsize=12)
    ax2.set_ylim(0, 105)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.grid(True, linestyle=":", alpha=0.3)

    for pct, va, dy in [(95, "bottom", 3), (99, "bottom", 3)]:
        idx = int(np.argmax(cumulative >= pct)) if np.any(cumulative >= pct) else None
        if idx is not None:
            n_hit = n_vals[idx]
            ax2.annotate(f"{pct}% at $n={n_hit}$",
                         xy=(n_hit, cumulative[idx]),
                         xytext=(n_hit + 0.4, cumulative[idx] - 14),
                         fontsize=6.5, color="gray",
                         arrowprops=dict(arrowstyle="->", color="gray", lw=0.6))

    fig.suptitle(rf"{p['title']}: $t^\star={d['t_star']:.4f}$ ($t^\star/\tau={d['t_star']/T_VAL:.2f}$)",
                 fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "spectral_coeffs")
    print(f"  [{key}] spectral_coeffs -> results/plots/{p['folder']}/  (t*={d['t_star']:.4f})")

# ═══════════════════════════════════════════════════════════════
# 7. Truncated biorthogonal LCD: TVD/KL/Wdiss time series vs n_trunc
# ═══════════════════════════════════════════════════════════════
#
# Lcd_full_magnus4.run_truncation_study already checks this at coarser
# dt=0.005 and only saves scalar summary rows (Max TVD etc.), not full
# t-series. Here we want the actual TVD(t)/KL(t)/Wdiss(t) CURVES at the
# same fine T=0.1, dt=1e-5 resolution as the rest of this script's
# figures, for a specific handful of truncation levels, to show curves
# collapsing onto the full solution by n~7-9. Wdiss uses the same
# validated Simpson power-integral as wdiss_refined (not the coarser
# Lcd_full_magnus4 heat integral), for consistency with the rest of this
# script's Wdiss numbers.
#
# Cost control: biorthogonal_decomp(L(t)) does NOT depend on n_trunc, so
# a decomp_cache dict shared across the whole n_trunc sweep (passed into
# make_gen_dw/harm -> build_ACD_from_L) reuses the same eig() calls for
# every n -- the whole sweep costs about the same as ONE full LCD run,
# not len(TRUNC_N_LIST)x (identical trick to run_truncation_study).

TRUNC_N_LIST = [5, 10, 15, 20, 25, 30, 35]

def run_lcd_series(key, n_list=TRUNC_N_LIST, dt=None):
    if dt is None:  # see rho_snapshots' comment: a literal DT_VAL default goes stale
        dt = DT_VAL
    p = POTENTIALS[key]
    ts = np.arange(0, T_VAL + dt, dt)
    DF0 = p['free_energy_fn'](0.0)
    DF = np.array([p['free_energy_fn'](t) - DF0 for t in ts])
    dV_dp = p['dV_dprotocol']

    decomp_cache = {}
    out = {}
    for n in n_list:
        gen = p['make_gen'](T_VAL, p['lcd_mode'], n_trunc=n, decomp_cache=decomp_cache)
        rho = p['rho_eq'](0, T_VAL).copy()
        tvds = np.zeros(len(ts)); kls = np.zeros(len(ts)); powers = np.zeros(len(ts))
        for i, t in enumerate(ts):
            req = p['rho_eq'](t, T_VAL)
            tvds[i]   = 0.5 * np.abs(rho - req).sum()
            kls[i]    = sim.kl_divergence(rho, req)
            powers[i] = p['protocol_dot'](t, T_VAL) * np.dot(dV_dp, rho)
            if t < T_VAL - dt / 2:
                rho = sim.magnus4_step(rho, t, dt, gen)
        W  = np.concatenate([[0.0], cumulative_simpson(powers, x=ts)])[:len(ts)]
        Wd = W - DF
        out[n] = dict(tvd=tvds, kl=kls, wdiss=Wd)
        print(f"    [{key}] n_trunc={n}: max TVD={tvds.max():.3e}, max KL={kls.max():.3e}, "
              f"max|Wdiss|={np.abs(Wd).max():.3e}")

    return ts, out

def compute_and_save_truncation(key, n_list=TRUNC_N_LIST):
    p = POTENTIALS[key]
    out_dir = os.path.join(sim.DATA_ROOT, p['folder'])
    print(f"[{key}] truncated-LCD time series, n_trunc={n_list} (T={T_VAL}, dt={DT_VAL})...")
    ts, series = run_lcd_series(key, n_list=n_list)

    # "Full" solution: reuse the already-saved main-run data instead of
    # recomputing (tvd_l/kl_l from run_and_save_dw/harm, Wd_l_refined
    # from wdiss_refined -- same Simpson-power-integral method used above).
    tvd_full = np.load(os.path.join(out_dir, "tvd_l.npy"))
    kl_full  = np.load(os.path.join(out_dir, "kl_l.npy"))
    wd_full  = np.load(os.path.join(out_dir, "Wd_l_refined.npy"))
    full = dict(tvd=tvd_full, kl=kl_full, wdiss=wd_full)

    save_dict = dict(ts=ts, n_list=np.array(n_list), full=full)
    for n in n_list:
        save_dict[f"n{n}"] = series[n]
    np.save(os.path.join(out_dir, "truncation_series.npy"), save_dict, allow_pickle=True)

    def row(label, tvd, kl, wd):
        return (f"| {label} | {tvd.max():.3e} | {kl.max():.3e} | "
                f"{wd[-1]:.3e} | {np.abs(wd).max():.3e} |")

    md = [
        f"# Truncated Biorthogonal LCD -- Time-Series Comparison ({p['title']})\n",
        f"Protocol: $\\tau={T_VAL}$, $dt={DT_VAL}$ (same fine resolution as the main "
        "spectral diagnostics figures). \"Full\" uses all $N-1$ non-trivial "
        "biorthogonal modes; truncated rows retain only the $n_{\\rm trunc}$ "
        "slowest-decaying modes in the counterdiabatic generator $A_{CD}$ "
        "(`build_ACD_from_L`). $W_{\\rm diss}$ computed via the Simpson "
        "power-integral (same method as `wdiss_refined`), not the coarser "
        "heat integral baked into `Lcd_full_magnus4.run_dw`/`run_harm`.\n",
        "| $n_{\\rm trunc}$ | Max TVD | Max $D_{KL}$ | $W_{\\rm diss}(\\tau)$ | Max $|W_{\\rm diss}(t)|$ |",
        "|---|---:|---:|---:|---:|",
    ]
    for n in n_list:
        md.append(row(str(n), series[n]['tvd'], series[n]['kl'], series[n]['wdiss']))
    md.append(row(f"Full ($n={p['N']-1}$)", tvd_full, kl_full, wd_full))
    md.append("")

    with open(os.path.join(out_dir, "truncation_series.md"), "w") as f:
        f.write("\n".join(md))
    print(f"[{key}] saved results/data/{p['folder']}/truncation_series.{{npy,md}}")

_TRUNC_COLORS = ['mediumvioletred', 'darkorange', 'gold', 'seagreen', 'dodgerblue', 'indigo', 'darkslateblue']

def plot_truncation_series(key):
    p = POTENTIALS[key]
    out_dir = os.path.join(sim.DATA_ROOT, p['folder'])
    plots_dir = plot_style._plots_dir(p['folder'])
    d = np.load(os.path.join(out_dir, "truncation_series.npy"), allow_pickle=True).item()
    ts, n_list, full = d['ts'], list(d['n_list']), d['full']
    t_frac = ts / T_VAL

    fig, axes = plt.subplots(1, 3, figsize=(6.2 * 3, 5.2))
    quantities = [
        ('tvd',   r"$\mathrm{TVD}(\rho,\pi_{\rm eq})$"),
        ('kl',    r"$D_{\rm KL}(\rho\|\pi_{\rm eq})$"),
        ('wdiss', r"$|W_{\rm diss}(t)|$"),
    ]
    handles, labels = [], []
    for ax, (qkey, ylabel) in zip(axes, quantities):
        for n, c in zip(n_list, _TRUNC_COLORS):
            y = np.maximum(np.abs(d[f"n{n}"][qkey]), 1e-16)
            line, = ax.semilogy(t_frac, y, color=c, lw=3.0, label=rf"$n_{{\rm trunc}}={n}$")
            if ax is axes[0]:
                handles.append(line); labels.append(rf"$n_{{\rm trunc}}={n}$")
        yfull = np.maximum(np.abs(full[qkey]), 1e-16)
        lfull, = ax.semilogy(t_frac, yfull, color='k', lw=3.5, ls='--', label="Full")
        if ax is axes[0]:
            handles.append(lfull); labels.append("Full")
        ax.set_xlabel(r"$t/\tau$")
        ax.set_ylabel(ylabel)
        ax.tick_params(direction="in")
        ax.grid(True, linestyle=":", alpha=0.3)

    # fig.suptitle(p['title'], fontsize=12)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, fontsize=12.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    _save(fig, plots_dir, "truncation_series")
    print(f"  [{key}] truncation_series -> results/plots/{p['folder']}/")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print(f"SPECTRAL DIAGNOSTICS: T={T_VAL}, dt={DT_VAL}")
    print("=" * 65)

    ensure_main_run()

    print("\nGenerating standard tvd/kl/wdiss/landscape figures (plotter.py)...")
    plot_style.plot_dw_timeseries(DW_FOLDER)
    plot_style.plot_landscape_dw(DW_FOLDER)
    plot_style.plot_harm_timeseries(HARM_FOLDER)
    plot_style.plot_landscape_harm(HARM_FOLDER)

    for key in ('dw', 'harm'):
        print(f"\n--- {POTENTIALS[key]['title']} ---")
        compute_and_save(key)
        # plot_spectral_gap(key)
        # plot_r0_vs_pieq(key)
        # plot_rho_snapshots(key)
        # plot_wdiss_refined(key)
        # plot_spectral_coeffs(key)
        compute_and_save_truncation(key)
        plot_truncation_series(key)

    print("\nDONE. All spectral diagnostics figures written under results/plots/<folder>/")
