"""
Harmonic-trap-only tau-sweep for dissipated work computation = W(tau) vs tau.
================================================================

The harmonic-trap analogue of a "work vs. protocol
duration" -- bare FP,
analytic CD, and LCD/spectral). Shows dissipated work growing sharply
for fast (small tau) protocols and W(tau) converging to the true
equilibrium Delta F as tau grows (quasistatic limit).

Work for bare and analytic CD is taken directly from
Lcd_full_magnus4.run_harm/run_harm_analytic (their W accuracy is fine at
these dissipation scales -- the coarse-Q-integral artifact identified in
spectral_diagnostics.py's wdiss_refined only matters when dissipation is
near machine precision). LCD uses the SAME validated Simpson power-
integral method as wdiss_refined (run_lcd_simpson below), since near-tau
LCD dissipation can be small enough for that artifact to matter.

Saves:
    results/data/harmonic_tau_sweep/tau_sweep.md    -- requested table
    results/data/harmonic_tau_sweep/tau_sweep.npy   -- full ts/tvd/kl/W
                                                        arrays per tau/mode,
                                                        for replotting later
    results/plots/harmonic_tau_sweep/DeltaF_vs_tau.{png,pdf}

Run:
    python harmonic_tau_sweep.py
"""

import os
import numpy as np
from scipy.integrate import cumulative_simpson
import matplotlib.pyplot as plt

import Lcd_full_magnus4 as sim
import plotter as plot_style  # PRL rcParams (applied on import) + colors/_save

OUT_DATA  = os.path.join(sim.DATA_ROOT, "harmonic_tau_sweep")
OUT_PLOTS = os.path.join(sim.PLOTS_ROOT, "harmonic_tau_sweep")
os.makedirs(OUT_DATA, exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)

TAU_LIST = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

# Explicit per-tau dt (finer for faster protocols), rather than a fixed
# step count -- user-specified schedule.
TAU_DT_MAP = {
    0.001: 1e-7, 0.003: 1e-7,
    0.01:  1e-6, 0.03:  1e-6,
    0.1:   1e-5, 0.3:   1e-5, 1.0: 1e-5,
    3.0:   1e-4, 10.0:  1e-4,
}

def dt_for_tau(tau):
    return TAU_DT_MAP[tau]

DELTA_F = sim.free_energy_harm(sim.kap_f) - sim.free_energy_harm(sim.kap_i)  # tau-independent

def run_lcd_simpson(tau, dt):
    """Simpson power-integral W(t) for the spectral LCD generator -- same
    validated method as spectral_diagnostics.wdiss_refined -- plus TVD/KL,
    computed in one pass."""
    ts  = np.arange(0, tau + dt, dt)
    gen = sim.make_gen_harm(tau, 'spectral')
    rho = sim.rho_eq_harm(0, tau).copy()
    powers = np.zeros(len(ts)); tvds = np.zeros(len(ts)); kls = np.zeros(len(ts))
    for i, t in enumerate(ts):
        req = sim.rho_eq_harm(t, tau)
        tvds[i]   = 0.5 * np.abs(rho - req).sum()
        kls[i]    = sim.kl_divergence(rho, req)
        powers[i] = sim.kappa0_dot(t, tau) * np.dot(0.5 * sim.q_np**2, rho)
        if t < tau - dt / 2:
            rho = sim.magnus4_step(rho, t, dt, gen)
    W = np.concatenate([[0.0], cumulative_simpson(powers, x=ts)])[:len(ts)]
    return ts, tvds, kls, W

def run_sweep():
    rows = []
    traj = {}
    for tau in TAU_LIST:
        dt = dt_for_tau(tau)
        n_steps = int(round(tau / dt))
        print(f"[tau={tau:g}, dt={dt:.3e}, steps={n_steps}]")

        ts_b, tvd_b, kl_b, W_b, DF_b = sim.run_harm(tau, dt=dt, mode='bare', integrator='midpoint')
        ts_a, tvd_a, kl_a, W_a, DF_a = sim.run_harm_analytic(tau, dt=dt)
        ts_l, tvd_l, kl_l, W_l       = run_lcd_simpson(tau, dt)

        row = dict(
            tau=tau,
            W_bare=W_b[-1], W_analytic=W_a[-1], W_lcd=W_l[-1],
            DF=DELTA_F,
            Wdiss_bare=W_b[-1] - DELTA_F, Wdiss_analytic=W_a[-1] - DELTA_F, Wdiss_lcd=W_l[-1] - DELTA_F,
            TVD_bare=tvd_b[-1], TVD_analytic=tvd_a[-1], TVD_lcd=tvd_l[-1],
        )
        rows.append(row)
        traj[tau] = dict(
            ts_b=ts_b, tvd_b=tvd_b, kl_b=kl_b, W_b=W_b,
            ts_a=ts_a, tvd_a=tvd_a, kl_a=kl_a, W_a=W_a,
            ts_l=ts_l, tvd_l=tvd_l, kl_l=kl_l, W_l=W_l,
        )
        print(f"  W_bare={row['W_bare']:.4f}  W_analytic={row['W_analytic']:.4f}  "
              f"W_lcd={row['W_lcd']:.4f}  (Delta F={DELTA_F:.4f})")
        print(f"  Wdiss_bare={row['Wdiss_bare']:.4f}  Wdiss_analytic={row['Wdiss_analytic']:.3e}  "
              f"Wdiss_lcd={row['Wdiss_lcd']:.3e}")

    return rows, traj

def save_table_md(rows):
    md = [
        "# Harmonic Trap -- Work vs Protocol Duration $\\tau$\n",
        f"$\\Delta F$ (exact, $\\tau$-independent) $= {DELTA_F:.6f}$. "
        "$\\W(\\tau)$ (final accumulated work); "
        "$W_{\\rm diss}(\\tau) = W(\\tau) - \\Delta F$. TVD is the FINAL-time value "
        "$\\mathrm{TVD}(\\rho(\\tau),\\pi_{\\rm eq}(\\tau))$, not the trajectory max. "
        "Bare/analytic $W$ from `Lcd_full_magnus4.run_harm`/`run_harm_analytic`; "
        "LCD $W$ from the Simpson power-integral (`run_lcd_simpson`, same method as "
        "`spectral_diagnostics.wdiss_refined`).\n",
        "| $\\tau$ | $W^{\\rm bare}(\\tau)$ | $W^{\\rm analytic}(\\tau)$ | $W^{\\rm LCD}(\\tau)$ | $\\Delta F$ "
        "| $W_{\\rm diss}^{\\rm bare}$ | $W_{\\rm diss}^{\\rm analytic}$ | $W_{\\rm diss}^{\\rm LCD}$ "
        "| $\\mathrm{TVD}^{\\rm bare}$ | $\\mathrm{TVD}^{\\rm analytic}$ | $\\mathrm{TVD}^{\\rm LCD}$ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['tau']:g} | {r['W_bare']:.4f} | {r['W_analytic']:.4f} | {r['W_lcd']:.4f} | {r['DF']:.4f} "
            f"| {r['Wdiss_bare']:.4f} | {r['Wdiss_analytic']:.3e} | {r['Wdiss_lcd']:.3e} "
            f"| {r['TVD_bare']:.3e} | {r['TVD_analytic']:.3e} | {r['TVD_lcd']:.3e} |"
        )
    md.append("")
    with open(os.path.join(OUT_DATA, "tau_sweep.md"), "w") as f:
        f.write("\n".join(md))
    print(f"\nSaved {OUT_DATA}/tau_sweep.md")

def plot_tau_sweep(rows):
    taus   = [r['tau'] for r in rows]
    W_bare = [r['W_bare'] for r in rows]
    W_an   = [r['W_analytic'] for r in rows]
    W_lcd  = [r['W_lcd'] for r in rows]

    fig, ax = plt.subplots()
    ax.plot(taus, W_bare, color=plot_style.C_BARE, marker='o', ms=6, lw=2.0, label="Bare FP")
    ax.plot(taus, W_an,   color=plot_style.C_AN,   marker='^', ms=6, lw=2.0, ls=':',  label="Analytic CD")
    ax.plot(taus, W_lcd,  color=plot_style.C_LCD,  marker='s', ms=6, lw=2.0, ls='--', label="LCD (spectral)")
    ax.axhline(DELTA_F, color='k', lw=1.5, label=r"$\Delta F$ (exact)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\Delta W(\tau)$")
    ax.legend(frameon=False)
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()
    plot_style._save(fig, OUT_PLOTS, "DeltaF_vs_tau")
    print(f"Saved {OUT_PLOTS}/W_vs_tau.{{png,pdf}}")

if __name__ == "__main__":
    print("=" * 65)
    print(f"HARMONIC TRAP: W(tau) vs tau, tau in {TAU_LIST}")
    print("=" * 65)
    rows, traj = run_sweep()

    np.save(os.path.join(OUT_DATA, "tau_sweep.npy"),
            dict(rows=rows, traj=traj, tau_list=TAU_LIST, delta_F=DELTA_F,
                 tau_dt_map=TAU_DT_MAP),
            allow_pickle=True)
    print(f"Saved {OUT_DATA}/tau_sweep.npy (full ts/tvd/kl/W arrays per tau/mode)")

    save_table_md(rows)
    plot_tau_sweep(rows)
    print("\nDONE.")
