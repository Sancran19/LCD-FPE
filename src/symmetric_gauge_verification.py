"""
Verification study: symmetric-gauge biorthogonal decomposition.
=================================================================

Validates Lcd_full_magnus4.biorthogonal_decomp_sym (the detailed-balance
symmetrization L_sym = D^{-1/2} L D^{1/2}, D=diag(pi_eq)) against the
dense nonsymmetric biorthogonal_decomp already used by the pipeline.

Three things are checked, at t/tau in {0, 0.25, 0.5, 0.75, 1.0} for
tau in {0.01, 0.1, 1.0}, for both potentials:
  1. Symmetry of L_sym to machine precision.
  2. Eigenvalue agreement: dense np.linalg.eig(L) vs symmetric
     eigh_tridiagonal(L_sym).
  3. Wall-clock speedup of eigh_tridiagonal(L_sym) over dense eig(L).

Plus a separate wall-clock-vs-N benchmark (N=50..120) showing the
asymptotic benefit of the symmetric/tridiagonal solve.

Run:
    python symmetric_gauge_verification.py
"""

import os
import time
import numpy as np
from scipy.linalg import eigh_tridiagonal

import Lcd_full_magnus4 as sim
import plotter as plot_style  # PRL rcParams (applied on import) + colors/_save

OUT_DATA  = os.path.join(sim.DATA_ROOT, "symmetric_gauge_verification")
OUT_PLOTS = os.path.join(sim.PLOTS_ROOT, "symmetric_gauge_verification")
os.makedirs(OUT_DATA, exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)

TAU_LIST   = [0.01, 0.1, 1.0]
FRACS      = [0.0, 0.25, 0.5, 0.75, 1.0]
N_TIMING_REP = 100
N_WARMUP     = 10

POTENTIALS = {
    'dw':   dict(build_L=sim.build_L_dw,   protocol=sim.lam_dw,   N=sim.N_dw, title="Double well"),
    'harm': dict(build_L=sim.build_L_harm, protocol=sim.kappa0,   N=sim.N_h,  title="Harmonic trap"),
}

# ═══════════════════════════════════════════════════════════════
# 1-3. Symmetry / eigenvalue agreement / speedup table
# ═══════════════════════════════════════════════════════════════

def time_call(fn, n_rep=N_TIMING_REP, n_warmup=N_WARMUP):
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_rep):
        fn()
    return (time.perf_counter() - t0) / n_rep

def analyze_point(key, tau, frac):
    p = POTENTIALS[key]
    t = frac * tau
    val = p['protocol'](t, tau)
    L, pi = p['build_L'](val)
    N = p['N']

    D_sqrt = np.sqrt(pi)
    L_sym  = (L * D_sqrt[None, :]) / D_sqrt[:, None]
    sym_violation = np.max(np.abs(L_sym - L_sym.T))

    ev_dense = np.sort(np.linalg.eigvals(L).real)
    ev_sym   = np.sort(np.linalg.eigvalsh(L_sym))
    eig_diff = np.max(np.abs(ev_dense - ev_sym))

    diag = np.diag(L_sym)
    off  = np.diag(L_sym, k=1)
    t_dense = time_call(lambda: np.linalg.eig(L))
    t_sym   = time_call(lambda: eigh_tridiagonal(diag, off, eigvals_only=True))

    return dict(sym_violation=sym_violation, eig_diff=eig_diff,
                t_dense_ms=t_dense * 1e3, t_sym_ms=t_sym * 1e3,
                speedup=t_dense / t_sym)

def run_table():
    rows = []
    for key in ('dw', 'harm'):
        for tau in TAU_LIST:
            for frac in FRACS:
                r = analyze_point(key, tau, frac)
                r.update(key=key, tau=tau, frac=frac)
                rows.append(r)
                print(f"  [{key}] tau={tau:<5} t/tau={frac:.2f}  "
                      f"sym_viol={r['sym_violation']:.2e}  eig_diff={r['eig_diff']:.2e}  "
                      f"speedup={r['speedup']:.1f}x")
    return rows

def save_table_md(rows):
    md = [
        "# Symmetric-Gauge Biorthogonal Decomposition -- Verification\n",
        "Checks $L_{\\rm sym} = D^{-1/2} L D^{1/2}$ ($D=\\mathrm{diag}(\\pi_{\\rm eq})$) against the dense "
        "nonsymmetric decomposition used by the main pipeline, at $N=80$ (current grid for both "
        "potentials), for $t/\\tau \\in \\{0, 0.25, 0.5, 0.75, 1.0\\}$ and "
        f"$\\tau \\in \\{{{', '.join(str(t) for t in TAU_LIST)}\\}}$. "
        f"Timings are means over {N_TIMING_REP} calls (after {N_WARMUP} warm-up calls).\n",
        "| Potential | $\\tau$ | $t/\\tau$ | $\\max\\|L_{\\rm sym}-L_{\\rm sym}^T\\|$ "
        "| $\\max\\|\\mathrm{eig}(L)-\\mathrm{eig}(L_{\\rm sym})\\|$ "
        "| dense eig (ms) | eigh\\_tridiagonal (ms) | Speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        title = POTENTIALS[r['key']]['title']
        md.append(f"| {title} | {r['tau']:g} | {r['frac']:.2f} | "
                   f"{r['sym_violation']:.2e} | {r['eig_diff']:.2e} | "
                   f"{r['t_dense_ms']:.4f} | {r['t_sym_ms']:.4f} | {r['speedup']:.1f}x |")
    md.append("")
    with open(os.path.join(OUT_DATA, "verification_table.md"), "w") as f:
        f.write("\n".join(md))
    print(f"\nSaved {OUT_DATA}/verification_table.md")

# ═══════════════════════════════════════════════════════════════
# Wall-clock vs N benchmark
# ═══════════════════════════════════════════════════════════════

N_SCAN = [50, 60, 70, 80, 100, 120]

def build_L_at_N(potential_fn, protocol_val, x_min, x_max, N, beta=1.0):
    """Mirrors build_L_dw/build_L_harm's exact recipe, parameterized by N
    (Lcd_full_magnus4.N_dw/N_h are fixed module-level constants, so this
    reimplements the same formula standalone for the N-scan only)."""
    x  = np.linspace(x_min, x_max, N)
    dx = x[1] - x[0]
    V  = potential_fn(x, protocol_val)
    pi = np.exp(-beta * V); pi /= pi.sum()
    Dc = 1. / dx**2; dV = np.diff(V)
    kf = Dc * np.exp(-beta * dV / 2.); kb = Dc * np.exp(beta * dV / 2.)
    L = np.zeros((N, N))
    L[np.arange(1, N), np.arange(0, N - 1)] = kf
    L[np.arange(0, N - 1), np.arange(1, N)] = kb
    for i in range(N):
        L[i, i] = -L[:, i].sum()
    return L, pi

def harmonic_potential(qv, kap):
    return 0.5 * kap * qv**2

N_SCAN_SPECS = {
    'dw':   dict(potential_fn=sim.potential_dw, val=0.3, x_min=-2.5, x_max=2.5, title="Double well"),
    'harm': dict(potential_fn=harmonic_potential, val=2.5, x_min=-4.0, x_max=4.0, title="Harmonic trap"),
}

def run_n_scan():
    results = {key: dict(N=[], t_dense=[], t_sym=[]) for key in N_SCAN_SPECS}
    for key, spec in N_SCAN_SPECS.items():
        for N in N_SCAN:
            L, pi = build_L_at_N(spec['potential_fn'], spec['val'], spec['x_min'], spec['x_max'], N)
            D_sqrt = np.sqrt(pi)
            L_sym = (L * D_sqrt[None, :]) / D_sqrt[:, None]
            diag = np.diag(L_sym); off = np.diag(L_sym, k=1)

            t_dense = time_call(lambda: np.linalg.eig(L))
            t_sym   = time_call(lambda: eigh_tridiagonal(diag, off, eigvals_only=True))

            results[key]['N'].append(N)
            results[key]['t_dense'].append(t_dense * 1e3)
            results[key]['t_sym'].append(t_sym * 1e3)
            print(f"  [{key}] N={N:<4} dense={t_dense*1e3:.4f} ms  "
                  f"sym={t_sym*1e3:.4f} ms  speedup={t_dense/t_sym:.1f}x")

    np.save(os.path.join(OUT_DATA, "n_scan.npy"), results, allow_pickle=True)

    md = ["# Wall-clock vs N -- dense eig(L) vs eigh_tridiagonal(L_sym)\n",
          f"Means over {N_TIMING_REP} calls (after {N_WARMUP} warm-up calls).\n",
          "| Potential | N | dense eig (ms) | eigh\\_tridiagonal (ms) | Speedup |",
          "|---|---:|---:|---:|---:|"]
    for key, spec in N_SCAN_SPECS.items():
        for i, N in enumerate(results[key]['N']):
            td, ts = results[key]['t_dense'][i], results[key]['t_sym'][i]
            md.append(f"| {spec['title']} | {N} | {td:.4f} | {ts:.4f} | {td/ts:.1f}x |")
    md.append("")
    with open(os.path.join(OUT_DATA, "n_scan.md"), "w") as f:
        f.write("\n".join(md))
    print(f"Saved {OUT_DATA}/n_scan.md")

    return results

def plot_n_scan(results):
    import matplotlib.pyplot as plt
    C_DENSE = plot_style.C_BARE
    C_SYM   = plot_style.C_LCD

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for key, spec in N_SCAN_SPECS.items():
        ls = '-' if key == 'dw' else '--'
        marker = 'o' if key == 'dw' else 's'
        ax.plot(results[key]['N'], results[key]['t_dense'], color=C_DENSE, ls=ls,
                 marker=marker, ms=6, lw=2.0, label=f"Dense eig(L) -- {spec['title']}")
        ax.plot(results[key]['N'], results[key]['t_sym'], color=C_SYM, ls=ls,
                 marker=marker, ms=6, lw=2.0, label=f"eigh_tridiagonal($L_{{\\rm sym}}$) -- {spec['title']}")
    ax.set_yscale('log')
    ax.set_xlabel("Spatial discretization, $N$")
    ax.set_ylabel("Wall-clock time per call (ms)")
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.legend(frameon=False, fontsize=9, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0.02, 0.20, 1, 1])
    plot_style._save(fig, OUT_PLOTS, "wallclock_vs_N")
    print(f"Saved {OUT_PLOTS}/wallclock_vs_N.{{png,pdf}}")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("Symmetric-gauge verification: symmetry, eigenvalues, speedup")
    print("=" * 65)
    rows = run_table()
    save_table_md(rows)

    print("\n" + "=" * 65)
    print(f"Wall-clock vs N benchmark: N in {N_SCAN}")
    print("=" * 65)
    results = run_n_scan()
    plot_n_scan(results)

    print("\nDONE.")
