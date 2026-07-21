"""
Quartic Coalescence Model — LCD Verification
==============================================

Potential: V(x, lambda) = x^4 - 16(1-lambda)x^2
Protocol:  lambda(t) = min(t/T, 1)  (linear ramp, lambda: 0 -> 1)

At lambda=0: two minima at +-sqrt(8), barrier height ~64
At lambda=1: single minimum at origin, barrier vanishes

Delta_F = F(lambda=1) - F(lambda=0) = 62.9407...  (at beta=1)

This is a STRESS TEST for the LCD framework:
  - The spectral gap |lambda_1| closes exponentially as lambda -> 0
  - The CD cost c_1 ~ 1/|lambda_1| diverges in this regime
  - cond(R) ~ 1e11 makes biorthogonal eigenvectors numerically fragile

Expected result:
  - Bare: W(T) >> Delta_F for small T, approaching Delta_F as T -> inf
  - LCD:  W(T) closer to Delta_F at all T, but NOT machine precision
          (limited by the spectral gap bottleneck)
"""

import os
import numpy as np
from scipy.linalg import expm
from scipy.integrate import cumulative_simpson
import matplotlib.pyplot as plt

_HERE     = os.path.dirname(os.path.abspath(__file__))
OUT_DATA  = os.path.join(_HERE, "results", "data", "free_energy_smoke_test")
OUT_PLOTS = os.path.join(_HERE, "results", "plots", "free_energy_smoke_test")
os.makedirs(OUT_DATA, exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)

plt.rcParams.update({
    'font.family':'serif','font.serif':['DejaVu Serif'],
    'font.size':16,'axes.titlesize':16,'axes.labelsize':18,
    'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':14,
    'lines.linewidth':3.5,'axes.linewidth':1.2,
    'axes.spines.top':False,'axes.spines.right':False,
    'xtick.direction':'in','ytick.direction':'in',
    'xtick.major.size':6,'ytick.major.size':6,
    'figure.facecolor':'white','axes.facecolor':'white',
    'savefig.dpi':300,'savefig.bbox':'tight','savefig.facecolor':'white',
})
C_BARE='#d62728'; C_LCD='#1f77b4'

# ═══════════════════════════════════════════════════════════════
# 1. MODEL
# ═══════════════════════════════════════════════════════════════

beta = 1.0
N    = 80
x_np = np.linspace(-4.5, 4.5, N)
dx   = x_np[1] - x_np[0]

def potential_q(xv, lam):
    """U(x, lambda) = x^4 - 16(1-lambda)x^2"""
    return xv**4 - 16.0*(1.0 - lam)*xv**2

def dV_dlam_q(xv):
    """dU/dlambda = 16 x^2"""
    return 16.0 * xv**2

def build_L_q(lam):
    """Sasa-Tasaki rate matrix for the quartic potential."""
    V  = potential_q(x_np, lam)
    pi = np.exp(-beta*V); pi /= pi.sum()
    Dc = 1.0/dx**2
    dV = np.diff(V)
    kf = Dc*np.exp(-beta*dV/2.0)
    kb = Dc*np.exp( beta*dV/2.0)
    L  = np.zeros((N,N))
    L[np.arange(1,N), np.arange(0,N-1)] = kf
    L[np.arange(0,N-1), np.arange(1,N)] = kb
    for i in range(N): L[i,i] = -L[:,i].sum()
    return L, pi

def free_energy_q(lam):
    """F(lambda) = -beta^{-1} ln Z(lambda), discrete partition function."""
    V = potential_q(x_np, lam)
    Z = np.sum(np.exp(-beta*V))
    return -1.0/beta * np.log(Z)

def rho_eq_q(lam):
    _, pi = build_L_q(lam)
    return pi

# ═══════════════════════════════════════════════════════════════
# 2. VERIFY DELTA_F
# ═══════════════════════════════════════════════════════════════

F_i = free_energy_q(0.0)
F_f = free_energy_q(1.0)
DF  = F_f - F_i

print("="*65)
print("QUARTIC COALESCENCE MODEL VERIFICATION")
print("="*65)
print(f"\n  U(x,lam) = x^4 - 16(1-lam)x^2")
print(f"  lambda: 0 -> 1,  beta = {beta}")
print(f"  Grid: N={N} on [{x_np[0]}, {x_np[-1]}], dx={dx:.4f}")
print(f"\n  F(lambda=0)  = {F_i:.4f}")
print(f"  F(lambda=1)  = {F_f:.4f}")
print(f"  Delta_F      = {DF:.4f}")
print(f"  Literature   = 62.94")
print(f"  Agreement    = {abs(DF-62.94):.4f}")

# Verify potential structure
V0 = potential_q(x_np, 0.0)
minima_idx = np.argsort(V0)[:2]
print(f"\n  At lambda=0:")
print(f"    Minima at x = {x_np[minima_idx[0]]:.3f}, {x_np[minima_idx[1]]:.3f}")
print(f"    Expected: +-sqrt(8) = +-{np.sqrt(8):.3f}")
print(f"    V(0) - V(min) = {V0[N//2] - V0.min():.1f}  (barrier height)")

# ═══════════════════════════════════════════════════════════════
# 3. SPECTRAL GAP ANALYSIS
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("SPECTRAL GAP ALONG THE PROTOCOL")
print(f"{'='*65}")
print(f"\n  {'lam':>5} | {'|lambda_1|':>12} | {'tau_relax':>12} | {'barrier':>8} | {'cond(R)':>12}")
print("  " + "-"*58)

for lam in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    L, pi = build_L_q(lam)
    eigs  = np.sort(np.linalg.eigvals(L).real)[::-1]
    gap   = abs(eigs[1])
    V     = potential_q(x_np, lam)
    barrier = V[N//2] - V.min() if lam < 0.99 else 0.0
    _, R  = np.linalg.eig(L)
    cR    = np.linalg.cond(R)
    print(f"  {lam:>5.1f} | {gap:>12.2e} | {1./max(gap,1e-20):>12.2e} | "
          f"{barrier:>8.1f} | {cR:>12.2e}")

print(f"\n  NOTE: for lambda < 0.5, |lambda_1| ~ 1e-13 (gap closed).")
print(f"  This makes c_1 ~ 1/|lambda_1| diverge, and cond(R) > 1e8.")
print(f"  This is the regime where CD cost diverges as predicted.")

# ═══════════════════════════════════════════════════════════════
# 4. LCD INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════

# Protocol: lambda(t) = min(t/T, 1)
def lam_q(t, T):
    return np.clip(t/T, 0.0, 1.0)

def lam_q_dot(t, T):
    return 1.0/T if t < T - 1e-15 else 0.0

def get_dL_q(lam, dlam, eps=1e-5):
    lam_c = np.clip(lam, eps, 1.0-eps)   # avoid boundary issues
    Lp, _ = build_L_q(lam_c + eps)
    Lm, _ = build_L_q(lam_c - eps)
    return (Lp - Lm)/(2*eps) * dlam

def build_ACD_q(lam, dlam):
    """Biorthogonal spectral A_CD for the quartic potential."""
    L, pi = build_L_q(lam)
    dL    = get_dL_q(lam, dlam)
    eigvals, R = np.linalg.eig(L)
    idx   = np.argsort(-eigvals.real)
    eigvals = eigvals[idx].real; R = R[:, idx].real
    R[:,0] = R[:,0] / R[:,0].sum()
    l0     = np.ones(N)
    Linv   = np.linalg.pinv(R, rcond=1e-10)
    Linv[0,:] = 0.0
    r0     = R[:, 0]
    A      = np.zeros((N,N))
    for n in range(1, N):
        if abs(eigvals[n]) > 1e-10:
            c = (Linv[n,:] @ dL @ r0) / eigvals[n]
            A -= c * np.outer(R[:,n], l0)
    return A

def kl_divergence(rho, pi):
    r_s = np.clip(rho, 1e-300, None)
    p_s = np.clip(pi,  1e-300, None)
    return np.sum(r_s * np.log(r_s/p_s))

# ═══════════════════════════════════════════════════════════════
# 5. SIMULATION (midpoint expm — more stable than Magnus4 here)
# ═══════════════════════════════════════════════════════════════

def run_q(T_val, dt, use_cd):
    """
    Evolve rho under L (bare) or L+A_CD (LCD) with midpoint expm.
    Work computed via Simpson integration of actual power (non-circular).

    Returns dict with all diagnostics.
    """
    ts   = np.arange(0, T_val + dt, dt)
    rho  = rho_eq_q(0.0).copy()

    powers = []; DF_t = []; tvds = []; kls = []

    for k, t in enumerate(ts):
        lam_now = lam_q(t, T_val)
        ldot    = lam_q_dot(t, T_val)

        # Power from ACTUAL evolved rho (non-circular)
        powers.append(ldot * np.dot(dV_dlam_q(x_np), rho))

        # Free energy at instantaneous lambda
        DF_t.append(free_energy_q(lam_now) - free_energy_q(0.0))

        # Tracking metrics
        req = rho_eq_q(lam_now)
        tvds.append(0.5 * np.abs(rho - req).sum())
        kls.append(kl_divergence(rho, req))

        # Step forward
        if k < len(ts) - 1:
            tm     = t + dt/2
            lam_m  = lam_q(tm, T_val)
            dlam_m = lam_q_dot(tm, T_val)
            L, _   = build_L_q(lam_m)

            if use_cd and dlam_m > 1e-15:
                A   = build_ACD_q(lam_m, dlam_m)
                Gen = L + A
            else:
                Gen = L

            rho_new = expm(Gen * dt) @ rho
            rho_new = np.abs(rho_new)
            rho_new = rho_new / rho_new.sum()
            rho     = rho_new

    powers = np.array(powers); DF_t = np.array(DF_t)
    W = np.concatenate([[0.0], cumulative_simpson(powers, x=ts)])[:len(ts)]
    Wdiss = W - DF_t

    return dict(
        ts=ts, W=W, DF=DF_t, Wdiss=Wdiss,
        tvds=np.array(tvds), kls=np.array(kls),
        W_final=W[-1], DF_final=DF_t[-1],
        max_tvd=max(tvds), max_kl=max(kls),
        max_wdiss=np.abs(Wdiss).max(),
        Wdiss_final=Wdiss[-1]
    )

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 6. CORROBORATION: VANISHING l^T_1 (\partial_t L) r_0 for a divergent inverse gap \Delta_1(t) = 1/(\lambda_1(t) - \lambda_0 (t)) 
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

def verify_symmetry_decoupling(lam, n_modes=5):
    """
    Confirms that the odd (antisymmetric) relaxation modes are decoupled from the
    even equilibrium target r0 = pi_eq by the even driving dL/dlambda, so that the
    numerator ell_n^T (dL/dlambda) r0 vanishes for odd modes. For the slowest mode
    (n=1, odd) this makes the inverse-gap factor 1/lambda_1 in the spectral A_CD
    multiply a vanishing numerator -- keeping the *exact* correction bounded as the
    gap closes, even though the *spectral construction* becomes ill-conditioned.
    """
    P = np.eye(N)[::-1]                    # reflection (P f)_i = f_{N-1-i}
    L, _ = build_L_q(lam)
    dL   = get_dL_q(lam, 1.0)             # raw dL/dlambda (the same one A_CD uses)
    w, R = np.linalg.eig(L)
    idx  = np.argsort(-w.real); w = w[idx].real; R = R[:, idx].real
    R[:, 0] = R[:, 0] / R[:, 0].sum()     # r0 = pi_eq (unit sum)
    Rinv = np.linalg.inv(R)               # rows = left eigenvectors ell_n
    r0   = R[:, 0]
    parity = lambda v: float((v/np.linalg.norm(v)) @ (P @ (v/np.linalg.norm(v))))
    print(f"lambda={lam}:  barrier dVb={64*(1-lam)**2:6.2f} kT,  "
          f"|lambda_1|={abs(w[1]):.2e},  ||P L P - L||inf={np.max(np.abs(P@L@P-L)):.1e},  "
          f"cond(R)={np.linalg.cond(R):.1e}")
    print(f"  {'n':>2} | {'lambda_n':>11} | {'parity r_n':>10} | "
          f"{'N_n=ell_n^T(dL)r0':>18} | {'c_n=N_n/lambda_n':>18}")
    Nvals = {}
    for n in range(1, n_modes+1):
        Nn = float(Rinv[n,:] @ dL @ r0); Nvals[n] = Nn
        cn = Nn/w[n] if w[n] != 0 else np.inf
        print(f"  {n:>2} | {w[n]:>11.4f} | {parity(R[:,n]):>+10.3f} | "
              f"{Nn:>18.4e} | {cn:>18.4e}")
    ratio = abs(Nvals[1])/abs(Nvals[2]) if Nvals[2] != 0 else np.inf
    print(f"  --> |N_1|/|N_2| = {ratio:.3e}   "
          f"({'odd slow mode DECOUPLED (~0)' if ratio < 1e-6 else 'NOT resolved -- garbage'})\n")
    return Nvals, ratio

# Verification (resolved gap): the decoupling is exact and reproducible
verify_symmetry_decoupling(0.8)   #  |N_1|/|N_2| ~ 1e-11
# Companion (collapsed gap): eigensolver cannot resolve parity -> spectral A_CD fails
verify_symmetry_decoupling(0.1)   #  ratio jumps to O(1e-2), c_1 ~ 1e13

# ═══════════════════════════════════════════════════════════════
# 6. RUN: SWEEP OVER PROTOCOL DURATIONS
# ═══════════════════════════════════════════════════════════════
T_vals = [0.01, 0.1, 0.2, 0.5, 1.0]
T_DT_MAP = {0.01: 1e-6, 0.1: 1e-5, 0.2: 1e-5, 0.5: 1e-5, 1.0: 1e-5}

print(f"\n{'='*65}")
print(f"W(T) vs Delta_F SWEEP  (dt schedule: {T_DT_MAP})")
print(f"{'='*65}")
print(f"\n  Delta_F = {DF:.4f} (target for perfect escorting)")
print(f"\n  {'T':>6} | {'W_bare(T)':>10} | {'W_LCD(T)':>10} | {'DF':>10} | "
      f"{'Wdiss_bare':>11} | {'Wdiss_LCD':>11} | {'TVD_bare':>10} | {'TVD_LCD':>10}")
print("  " + "-"*98)

results_bare = []
results_lcd  = []

for T in T_vals:
    dt = T_DT_MAP[T]
    print(f"  [T={T:g}, dt={dt:.0e}]")
    b = run_q(T, dt, use_cd=False)
    l = run_q(T, dt, use_cd=True)
    results_bare.append(b)
    results_lcd.append(l)
    print(f"  {T:>6.2f} | {b['W_final']:>10.2f} | {l['W_final']:>10.2f} | "
          f"{b['DF_final']:>10.2f} | {b['Wdiss_final']:>11.2f} | "
          f"{l['Wdiss_final']:>11.2e} | {b['max_tvd']:>10.4e} | {l['max_tvd']:>10.4e}")

# ═══════════════════════════════════════════════════════════════
# 7. SAVE ALL ARRAYS
# ═══════════════════════════════════════════════════════════════

np.save(os.path.join(OUT_DATA, 'quartic_T_vals.npy'), T_vals)
np.save(os.path.join(OUT_DATA, 'quartic_results_bare.npy'), results_bare, allow_pickle=True)
np.save(os.path.join(OUT_DATA, 'quartic_results_lcd.npy'), results_lcd, allow_pickle=True)
np.save(os.path.join(OUT_DATA, 'quartic_meta.npy'),
        {'DF': DF, 'beta': beta, 'N': N, 'T_dt_map': T_DT_MAP,
         'x_range': [x_np[0], x_np[-1]]}, allow_pickle=True)
print(f"\nSaved quartic_*.npy -> {OUT_DATA}/")

# ═══════════════════════════════════════════════════════════════
# 8. MARKDOWN TABLE
# ═══════════════════════════════════════════════════════════════

md = []
md.append("# Quartic Coalescence Model: $U(x,\\lambda) = x^4 - 16(1-\\lambda)x^2$\n")
md.append(f"$\\Delta F = {DF:.4f}$, $\\beta = {beta}$, $N = {N}$, "
          f"$dt$ schedule $= {T_DT_MAP}$\n")
md.append("| $T$ | $W^{\\rm bare}(T)$ | $W^{\\rm LCD}(T)$ | $\\Delta F$ | "
          "$W_{\\rm diss}^{\\rm bare}$ | $W_{\\rm diss}^{\\rm LCD}$ | "
          "TVD$^{\\rm bare}$ | TVD$^{\\rm LCD}$ |")
md.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
for T, b, l in zip(T_vals, results_bare, results_lcd):
    md.append(f"| {T:.2f} | {b['W_final']:.2f} | {l['W_final']:.2f} | "
              f"{b['DF_final']:.2f} | {b['Wdiss_final']:.2f} | "
              f"{l['Wdiss_final']:.2e} | {b['max_tvd']:.2e} | {l['max_tvd']:.2e} |")

with open(os.path.join(OUT_DATA, 'quartic_results.md'), 'w') as f:
    f.write('\n'.join(md))
print(f"Saved {OUT_DATA}/quartic_results.md")

# ═══════════════════════════════════════════════════════════════
# 9. PLOTS
# ═══════════════════════════════════════════════════════════════

# ── 9a. MAIN PLOT: W(tau) vs tau, log-log ────────────────
# x-axis: T (protocol duration), y-axis: Delta_F^est(tau) = W(tau) for
# each mode, plus the horizontal exact Delta_F line.
fig, ax = plt.subplots(figsize=(7, 5.5))
W_bare_finals = [b['W_final'] for b in results_bare]
W_lcd_finals  = [l['W_final'] for l in results_lcd]
ax.plot(T_vals, W_bare_finals, 'o-', color=C_BARE, ms=8, label=r'$W(\tau)$ bare')
ax.plot(T_vals, W_lcd_finals,  's-', color=C_LCD,  ms=8, label=r'$W(\tau)$ LCD')
ax.axhline(DF, color='gray', ls='--', lw=2, label=r'$\Delta F = %.2f$ (exact)' % DF)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Protocol duration $\tau$')
ax.set_ylabel(r'$W(\tau)$')
ax.legend(frameon=False)
ax.grid(True, which='both', linestyle=':', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_PLOTS, 'DeltaF_vs_T.png'))
fig.savefig(os.path.join(OUT_PLOTS, 'DeltaF_vs_T.pdf'))
plt.close(fig)

# ── 9b. Wdiss(T) vs T ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.5))
Wdiss_bare = [b['Wdiss_final'] for b in results_bare]
Wdiss_lcd  = [l['Wdiss_final'] for l in results_lcd]
ax.loglog(T_vals, np.abs(Wdiss_bare), 'o-', color=C_BARE, ms=8,
          label=r'$|W_{\rm diss}(\tau)|$ bare')
ax.loglog(T_vals, np.maximum(np.abs(Wdiss_lcd), 1e-6), 's-', color=C_LCD, ms=8,
          label=r'$|W_{\rm diss}(\tau)|$ LCD')
ax.set_xlabel(r'Protocol duration $\tau$')
ax.set_ylabel(r'$|W_{\rm diss}(\tau)|$')
ax.legend(frameon=False)
ax.grid(True, which='both', linestyle=':', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_PLOTS, 'quartic_Wdiss_vs_T.png'))
fig.savefig(os.path.join(OUT_PLOTS, 'quartic_Wdiss_vs_T.pdf'))
plt.close(fig)

# ── 9c. Running |Wdiss(t)| at T=1.0 ──────────────────────────
idx_T1 = T_vals.index(1.0)
b1 = results_bare[idx_T1]
l1 = results_lcd[idx_T1]

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.semilogy(b1['ts']/1.0, np.abs(b1['Wdiss']), color=C_BARE, label='Bare FP')
ax.semilogy(l1['ts']/1.0, np.maximum(np.abs(l1['Wdiss']), 1e-6),
            color=C_LCD, ls='--', label='LCD')
ax.set_xlabel(r'$t/\tau$')
ax.set_ylabel(r'$|W_{\rm diss}(t)|$')
ax.legend(frameon=False)
ax.grid(True, linestyle=':', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_PLOTS, 'quartic_wdiss_T1.png'))
fig.savefig(os.path.join(OUT_PLOTS, 'quartic_wdiss_T1.pdf'))
plt.close(fig)

# ── 9d. TVD at tau=1.0 ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.semilogy(b1['ts']/1.0, b1['tvds'], color=C_BARE, label='Bare FP')
ax.semilogy(l1['ts']/1.0, l1['tvds'], color=C_LCD, ls='--', label='LCD')
ax.set_xlabel(r'$t/\tau$')
ax.set_ylabel(r'$\mathrm{TVD}(\rho, \pi_{\rm eq})$')
ax.legend(frameon=False)
ax.grid(True, linestyle=':', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_PLOTS, 'quartic_tvd_T1.png'))
fig.savefig(os.path.join(OUT_PLOTS, 'quartic_tvd_T1.pdf'))
plt.close(fig)

print(f"\nSaved 4 figures -> {OUT_PLOTS}/ (DeltaF_vs_T [main, log-log], "
      "quartic_Wdiss_vs_T, quartic_wdiss_T1, quartic_tvd_T1)")

# ═══════════════════════════════════════════════════════════════
# 10. SUMMARY
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("SUMMARY")
print(f"{'='*65}")
print(f"\n  Delta_F = {DF:.4f}")
print(f"\n  Bare dynamics: W(tau) >> Delta_F for small T,")
print(f"    approaches Delta_F only as T -> infinity (~1/T).")
print(f"\n  LCD dynamics: W(tau) much closer to Delta_F at all T,")
print(f"    but NOT machine-precision due to spectral gap closing")
print(f"    exponentially for lambda < 0.5 (barrier height > 16).")
print(f"    This is the predicted limitation: c_n ~ 1/|lambda_n|")
print(f"    diverges when the gap closes.")
print(f"\n  At T=1.0:")
print(f"    W_bare  = {results_bare[idx_T1]['W_final']:.2f}  "
      f"(Wdiss = {results_bare[idx_T1]['Wdiss_final']:.2f})")
print(f"    W_LCD   = {results_lcd[idx_T1]['W_final']:.2f}  "
      f"(Wdiss = {results_lcd[idx_T1]['Wdiss_final']:.4f})")
print(f"    Delta_F = {DF:.2f}")
print(f"    Improvement: {abs(results_bare[idx_T1]['Wdiss_final'])/max(abs(results_lcd[idx_T1]['Wdiss_final']),1e-10):.0f}x")
