"""
Complete LCD Analysis — Magnus 4th Order
==========================================

Magnus4 integrator time-evolutor, for both potentials
(double well and harmonic trap):

  - Double well:    TVD, KL divergence, dissipated work (Magnus4 LCD vs bare)
  - Harmonic trap:  TVD, KL divergence, dissipated work
                    (bare vs analytic CD vs Magnus4 spectral LCD)
"""

import os
import numpy as np
from scipy.linalg import expm, eigh_tridiagonal
from scipy.integrate import solve_ivp

beta = 1.0

# Root directories (relative to this file, not the CWD it's launched from)
_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(_HERE, "results", "data")
PLOTS_ROOT = os.path.join(_HERE, "results", "plots")

# ═══════════════════════════════════════════════════════════════
# SHARED: biorthogonal decomposition, A_CD, Magnus4 integrator
# ═══════════════════════════════════════════════════════════════

def biorthogonal_decomp(L, N):
    """
    l0 = 1^T (analytical, from column-sum-zero of L).
    r0 = R[:,0] normalised to sum=1 -> r0 = pi_eq.
    Linv = pinv(R); row 0 zeroed (never used).
    Eigenpairs are sorted by DESCENDING eigenvalue (index 0 = 0, the
    stationary mode; index 1 = slowest-decaying mode; etc.), so that
    a truncation to the first n_trunc non-trivial modes keeps exactly
    the slowest, most dynamically relevant ones.
    """
    eigvals, R = np.linalg.eig(L)
    idx     = np.argsort(-eigvals.real)
    eigvals = eigvals[idx].real
    R       = R[:, idx].real
    R[:,0]  = R[:,0] / R[:,0].sum()
    l0      = np.ones(N)
    Linv    = np.linalg.pinv(R, rcond=1e-10)
    Linv[0,:] = 0.0
    return eigvals, R, l0, Linv

# ═══════════════════════════════════════════════════════════════
# FASTER NUMERICS (opt-in, not wired into the pipeline above):
# symmetric-gauge biorthogonal decomposition
# ═══════════════════════════════════════════════════════════════
#
# L satisfies detailed balance w.r.t. its stationary distribution pi --
# built in by construction via the symmetric half-step rate assignment
# kf=Dc*exp(-beta*dV/2), kb=Dc*exp(+beta*dV/2) used above (a
# Sasa-Tasaki-style discretization), i.e. L[j,i]*pi[i] = L[i,j]*pi[j] for
# every pair. That means L_sym = D^{-1/2} L D^{1/2} (D=diag(pi)) is REAL
# SYMMETRIC: (D^{-1/2} L D^{1/2})_{ij} = L_{ij} sqrt(pi_j/pi_i), and
# symmetry <=> L_{ij} pi_j = L_{ji} pi_i, exactly the detailed-balance
# condition. Since L is tridiagonal (nearest-neighbor birth-death chain)
# and a diagonal similarity transform introduces no fill-in, L_sym is
# ALSO tridiagonal -- so its eigendecomposition can be computed with
# scipy.linalg.eigh_tridiagonal instead of the dense nonsymmetric
# np.linalg.eig(L) that biorthogonal_decomp above uses. Verified (see
# symmetric_gauge_verification.py / results/data/symmetric_gauge_verification/):
# symmetric to ~1e-13, eigenvalues agree with the dense solve to
# ~1e-8-1e-10, and eigh_tridiagonal is ~12-14x faster per call at N=80.
# NOT used by build_ACD_from_L / the main pipeline below -- every
# existing result in this repo is unaffected. Provided as a validated,
# opt-in faster alternative (biorthogonal_decomp_sym is a drop-in
# replacement for biorthogonal_decomp, given pi).

def biorthogonal_decomp_sym(L, pi, N):
    """
    Symmetric-gauge equivalent of biorthogonal_decomp. Returns eigvals,
    R, l0, Linv in EXACTLY the same convention (descending eigenvalues,
    R[:,0]=pi normalised to sum=1, l0=ones(N), Linv=left eigenvectors as
    rows with Linv[0,:] zeroed) -- a drop-in replacement anywhere
    biorthogonal_decomp(L, N) is called, given pi (already returned by
    build_L_dw/build_L_harm, free to obtain).

    Derivation: writing L_sym's orthonormal eigenpairs as L_sym v_n =
    lam_n v_n (V=[v_0,...,v_{N-1}] orthogonal), substituting L_sym =
    D^{-1/2} L D^{1/2} gives L (D^{1/2} v_n) = lam_n (D^{1/2} v_n), i.e.
        right eigenvectors  r_n = D^{1/2} v_n      (columns of R)
    and L = D^{1/2} L_sym D^{-1/2} = (D^{1/2}V) Lambda (V^T D^{-1/2})
          = R Lambda R^{-1},  R^{-1} = V^T D^{-1/2}, i.e.
        left eigenvectors   l_n = D^{-1/2} v_n      (rows of Linv)
    using V^T V = V V^T = I (V square orthogonal). For n=0 (lam_0=0),
    L@pi=0 gives v_0 proportional to sqrt(pi) elementwise, so r_0 = D^{1/2}v_0 is proportional to pi
    exactly -- the R[:,0]/=R[:,0].sum() line below both normalises it to
    pi_eq and fixes eigh_tridiagonal's arbitrary overall eigenvector sign.
    """
    D_sqrt = np.sqrt(pi)
    L_sym  = (L * D_sqrt[None, :]) / D_sqrt[:, None]   # (D^-1/2 L D^1/2)_ij = L_ij*sqrt(pi_j/pi_i)
    diag   = np.diag(L_sym)
    off    = np.diag(L_sym, k=1)

    eigvals, V = eigh_tridiagonal(diag, off, eigvals_only=False)
    idx     = np.argsort(-eigvals)         # ascending -> descending, matches biorthogonal_decomp
    eigvals = eigvals[idx]
    V       = V[:, idx]

    R    = D_sqrt[:, None] * V             # r_n = D^{1/2} v_n, as columns
    Linv = V.T / D_sqrt[None, :]           # l_n = D^{-1/2} v_n, as rows

    R[:, 0]   = R[:, 0] / R[:, 0].sum()
    l0        = np.ones(N)
    Linv[0, :] = 0.0
    return eigvals, R, l0, Linv

def build_ACD_from_L(L, dL_dt, N, n_trunc=None, decomp_cache=None, cache_key=None):
    """
    A_CD = -sum_{n=1}^{n_max} c_n |r_n><l0|,  c_n = <l_n|dL/dt|r0>/lambda_n.

    n_trunc: number of non-trivial (n>=1) biorthogonal modes to retain.
    None (default) uses the full set (n_max = N-1), i.e. the exact LCD
    generator. n_trunc=k restricts the sum to the k slowest-decaying
    modes (n=1..k), giving the truncated biorthogonal-decomposition
    approximation to the counterdiabatic drive.
    """
    if decomp_cache is not None and cache_key in decomp_cache:
        eigvals, R, l0, Linv = decomp_cache[cache_key]
    else:
        eigvals, R, l0, Linv = biorthogonal_decomp(L, N)
        if decomp_cache is not None:
            decomp_cache[cache_key] = (eigvals, R, l0, Linv)
    r0 = R[:,0]
    n_max = (N - 1) if n_trunc is None else min(n_trunc, N - 1)
    A  = np.zeros((N, N))
    for n in range(1, n_max + 1):
        if abs(eigvals[n]) > 1e-10:
            mel = Linv[n,:] @ dL_dt @ r0
            c_n = mel / eigvals[n]
            A  -= c_n * np.outer(R[:,n], l0)
    return A

def magnus4_step(rho, t, dt, gen_fn):
    """
    4th-order Magnus step:
        t1,2 = t + (1/2 -+ sqrt(3)/6)*dt   (Gauss-Legendre nodes)
        Omega = (dt/2)(G1+G2) + (sqrt(3)dt^2/12)[G2,G1]
        rho_new = expm(Omega) @ rho
    """
    c  = np.sqrt(3) / 6
    t1 = t + (0.5 - c) * dt
    t2 = t + (0.5 + c) * dt
    G1 = gen_fn(t1)
    G2 = gen_fn(t2)
    comm  = G2 @ G1 - G1 @ G2
    Omega = (dt/2)*(G1+G2) + (dt**2 * np.sqrt(3)/12)*comm
    rho_new = expm(Omega) @ rho
    rho_new = np.abs(rho_new)
    rho_new = rho_new / rho_new.sum()
    return rho_new

def midpoint_step(rho, t, dt, gen_fn):
    """Midpoint step (used only for bare dynamics, sufficient there)."""
    Gen = gen_fn(t + dt/2)
    rho_new = expm(Gen * dt) @ rho
    rho_new = np.abs(rho_new)
    rho_new = rho_new / rho_new.sum()
    return rho_new

def kl_divergence(rho, pi):
    """D_KL(rho||pi) = sum rho_i log(rho_i/pi_i)."""
    r_s = np.clip(rho, 1e-300, None)
    p_s = np.clip(pi,  1e-300, None)
    return np.sum(r_s * np.log(r_s/p_s))

# ═══════════════════════════════════════════════════════════════
# DOUBLE WELL:  V(x,zeta) = x^4 - 2x^2 + zeta*x,  zeta: -1 -> +1
# (annealing parameter denoted zeta; kept as "lam" in code/variable
#  names for continuity, displayed as $\zeta$ in plots/labels)
# ═══════════════════════════════════════════════════════════════

N_dw  = 80
x_np  = np.linspace(-2.5, 2.5, N_dw)
dx    = x_np[1] - x_np[0]
lam_i_dw, lam_f_dw = -1.0, 1.0

def potential_dw(xv, lam):
    return xv**4 - 2.0*xv**2 + lam*xv

def build_L_dw(lam):
    V  = potential_dw(x_np, lam)
    pi = np.exp(-beta*V); pi /= pi.sum()
    Dc = 1./dx**2; dV = np.diff(V)
    kf = Dc*np.exp(-beta*dV/2.); kb = Dc*np.exp(beta*dV/2.)
    L  = np.zeros((N_dw, N_dw))
    L[np.arange(1,N_dw), np.arange(0,N_dw-1)] = kf
    L[np.arange(0,N_dw-1), np.arange(1,N_dw)] = kb
    for i in range(N_dw): L[i,i] = -L[:,i].sum()
    return L, pi

def free_energy_dw(lam):
    V = potential_dw(x_np, lam); Z = np.sum(np.exp(-beta*V))
    return -1./beta * np.log(Z)

def lam_dw(t, T):
    s=np.clip(t/T,0,1); sm=s**3*(6*s**2-15*s+10)
    return lam_i_dw + (lam_f_dw-lam_i_dw)*sm
def lam_dw_dot(t, T):
    s=np.clip(t/T,0,1); ds=30*s**2*(1-s)**2
    return (lam_f_dw-lam_i_dw)*ds/T

def get_dL_dw(lam, dlam, eps=1e-5):
    Lp,_=build_L_dw(lam+eps); Lm,_=build_L_dw(lam-eps)
    return (Lp-Lm)/(2*eps)*dlam

def make_gen_dw(T, mode, n_trunc=None, decomp_cache=None):
    def gen(t):
        lam  = lam_dw(t, T)
        dlam = lam_dw_dot(t, T)
        L, _ = build_L_dw(lam)
        if mode == 'bare':
            return L
        dL = get_dL_dw(lam, dlam)
        A  = build_ACD_from_L(L, dL, N_dw, n_trunc=n_trunc,
                               decomp_cache=decomp_cache, cache_key=lam)
        return L + A
    return gen

def rho_eq_dw(t, T):
    _, pi = build_L_dw(lam_dw(t, T)); return pi

def run_dw(T_val, dt=0.02, mode='bare', integrator='midpoint', n_trunc=None, decomp_cache=None):
    """
    Returns ts, tvd, kl, W, DF (running work and Delta_F(lam(t))).
    n_trunc: number of biorthogonal modes retained by the LCD generator
    (mode='lcd' only); None = full/exact.
    """
    ts   = np.arange(0, T_val+dt, dt)
    rho  = rho_eq_dw(0, T_val).copy()
    U_i  = potential_dw(x_np, lam_i_dw)
    Q    = 0.0
    tvds=[]; kls=[]; W_t=[]; DF_t=[]
    gen  = make_gen_dw(T_val, mode, n_trunc=n_trunc, decomp_cache=decomp_cache)
    step = magnus4_step if integrator=='magnus4' else midpoint_step

    for t in ts:
        req = rho_eq_dw(t, T_val)
        tvds.append(0.5*np.abs(rho-req).sum())
        kls.append(kl_divergence(rho, req))

        lam_now = lam_dw(t, T_val)
        U_now   = potential_dw(x_np, lam_now)
        DeltaE  = np.dot(U_now, rho) - np.dot(U_i, rho_eq_dw(0,T_val))
        W_t.append(DeltaE - Q)
        DF_t.append(free_energy_dw(lam_now) - free_energy_dw(lam_i_dw))

        if t < T_val-dt/2:
            t_mid = t + dt/2
            U_mid = potential_dw(x_np, lam_dw(t_mid, T_val))
            rho_new = step(rho, t, dt, gen)
            Q += np.dot(U_mid, rho_new-rho)
            rho = rho_new

    return ts, np.array(tvds), np.array(kls), np.array(W_t), np.array(DF_t)

# ═══════════════════════════════════════════════════════════════
# HARMONIC TRAP:  V(q,t) = (1/2) kappa0(t) q^2,  kappa: 1 -> 4
# ═══════════════════════════════════════════════════════════════

N_h     = 80
q_np    = np.linspace(-4.0, 4.0, N_h)
dq      = q_np[1] - q_np[0]
kap_i, kap_f = 1.0, 4.0
gamma_h = 1.0

def build_L_harm(kap):
    V  = 0.5*kap*q_np**2
    pi = np.exp(-beta*V); pi /= pi.sum()
    Dc = 1./dq**2; dV = np.diff(V)
    kf = Dc*np.exp(-beta*dV/2.); kb = Dc*np.exp(beta*dV/2.)
    L  = np.zeros((N_h, N_h))
    L[np.arange(1,N_h), np.arange(0,N_h-1)] = kf
    L[np.arange(0,N_h-1), np.arange(1,N_h)] = kb
    for i in range(N_h): L[i,i] = -L[:,i].sum()
    return L, pi

def free_energy_harm(kap):
    V = 0.5*kap*q_np**2; Z = np.sum(np.exp(-beta*V))
    return -1./beta * np.log(Z)

def kappa0(t, T):
    s=np.clip(t/T,0,1); sm=s**3*(6*s**2-15*s+10)
    return kap_i+(kap_f-kap_i)*sm
def kappa0_dot(t, T):
    s=np.clip(t/T,0,1); ds=30*s**2*(1-s)**2
    return (kap_f-kap_i)*ds/T
def kappa_CD(t, T):
    """Analytic CD (Jarzynski/Martinez Eq. 75): kappa0 + gamma*kappa0_dot/(2*kappa0)."""
    k0=kappa0(t,T); k0d=kappa0_dot(t,T)
    return k0+gamma_h*k0d/(2.*k0)

def get_dL_harm(kap, kdot, eps=1e-5):
    Lp,_=build_L_harm(kap+eps); Lm,_=build_L_harm(kap-eps)
    return (Lp-Lm)/(2*eps)*kdot

def make_gen_harm(T, mode, n_trunc=None, decomp_cache=None):
    """mode: 'bare', 'spectral' (Magnus4 LCD). Analytic CD handled separately
    via the scalar variance ODE (run_harm_analytic), not via the rate matrix."""
    def gen(t):
        k0  = kappa0(t, T)
        k0d = kappa0_dot(t, T)
        if mode == 'bare':
            L, _ = build_L_harm(k0)
            return L
        elif mode == 'spectral':
            L, _ = build_L_harm(k0)
            dL   = get_dL_harm(k0, k0d)
            A    = build_ACD_from_L(L, dL, N_h, n_trunc=n_trunc,
                                     decomp_cache=decomp_cache, cache_key=k0)
            return L + A
    return gen

def rho_eq_harm(t, T):
    _, pi = build_L_harm(kappa0(t, T)); return pi

# ─────────────────────────────────────────────────────────────
# 4b. SCALAR-ODE ANALYTIC CD (PDE-free, exact for harmonic trap)
# ─────────────────────────────────────────────────────────────
#
# For a harmonic potential, rho(q,t) stays Gaussian:
#     rho(q,t) = sqrt(sigma(t)/pi) * exp(-sigma(t) q^2),  sigma=1/(2 Var)
# and the FULL Fokker-Planck dynamics reduces to a SCALAR ODE:
#     dVar/dt = -2*kappa(t)*Var/friction + 2*D/friction   (friction=1, D=1/beta)
#
# Under the analytic CD protocol kappa_CD(t) = kappa0 + gamma*kappa0_dot/(2*kappa0),
# this ODE gives sigma_analytic(t) = sigma_target(t) = beta*kappa0(t)/2 to
# machine precision (verified ~1e-11), with NO spatial discretisation.
#
# We then evaluate TVD, KL, W, DF on the SAME spatial grid q_np by
# constructing the Gaussian rho(q,t)=sqrt(sigma(t)/pi)*exp(-sigma(t)q^2)*dq
# (probability MASS per grid cell, sum=1 convention) from sigma_analytic(t).

friction_h = 1.0
D_diff_h   = 1.0/(beta*friction_h)

def variance_ode_harm(t, Var, T_val, mode):
    """dVar/dt = -2*kappa(t)*Var/friction + 2*D/friction."""
    if mode == 'bare':
        kap = kappa0(t, T_val)
    elif mode == 'analytic':
        kap = kappa_CD(t, T_val)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return [-2.0*kap/friction_h*Var[0] + 2.0*D_diff_h/friction_h]

def gaussian_on_grid(sigma):
    """
    rho_i = sqrt(sigma/pi) * exp(-sigma*q_i^2) * dq,  normalised to sum=1.
    (discrete probability mass on the q_np grid; renormalised to absorb
    O(dq^2) quadrature mismatch so sum(rho)=1 exactly, consistent with
    rho_eq_harm's discrete convention.)
    """
    r = np.sqrt(sigma/np.pi) * np.exp(-sigma*q_np**2) * dq
    return r / r.sum()

def run_harm_analytic(T_val, dt=0.0001, n_sub=50):
    """
    Analytic CD via the scalar variance ODE (no Liouville matrix).

    Returns ts, tvd, kl, W, DF -- same signature/convention as run_harm,
    so Wd_a_h = W_a_h - DF_a_h is directly comparable.

    sigma(t) is obtained by solving dVar/dt = -2*kappa_CD(t)*Var + 2/beta
    with Var(0)=1/(beta*kappa0(0)), via dense_output (continuous in t).
    rho(q,t) is the Gaussian sqrt(sigma(t)/pi)*exp(-sigma(t)q^2) on q_np (sum=1).

    HEAT INTEGRAL Q(t):
        Computed with FINE substeps (n_sub per reporting interval) using
        the continuous dense_output solution for sigma(t), NOT the coarse
        dt midpoint rule. This reduces the O(dt^2) quadrature error in the
        heat integral to ~1e-7, so |Wdiss| reflects the analytic CD's true
        tracking accuracy rather than a shared discretisation artifact.
    """
    ts = np.arange(0, T_val+dt, dt)

    Var0 = 1.0/(beta*kappa0(0.0, T_val))
    sol = solve_ivp(
        variance_ode_harm, [0, T_val], [Var0],
        args=(T_val, 'analytic'),
        dense_output=True, rtol=1e-12, atol=1e-14, method='DOP853'
    )

    def sigma_of(t_):
        return 1.0/(2.0*sol.sol(t_)[0])

    U_i  = 0.5*kappa0(0,T_val)*q_np**2
    pi0  = gaussian_on_grid(sigma_of(0.0))
    Q    = 0.0
    tvds=[]; kls=[]; W_t=[]; DF_t=[]

    for k, t in enumerate(ts):
        req = rho_eq_harm(t, T_val)
        rho = gaussian_on_grid(sigma_of(t))

        tvds.append(0.5*np.abs(rho-req).sum())
        kls.append(kl_divergence(rho, req))

        k0_now = kappa0(t, T_val)
        U_now  = 0.5*k0_now*q_np**2
        DeltaE = np.dot(U_now, rho) - np.dot(U_i, pi0)
        W_t.append(DeltaE - Q)
        DF_t.append(free_energy_harm(k0_now) - free_energy_harm(kap_i))

        if k < len(ts)-1:
            # Fine-substep quadrature for the heat integral over [t, t+dt],
            # using the continuous sigma(t) from dense_output.
            t_fine     = np.linspace(t, t+dt, n_sub+1)
            sigma_fine = sigma_of(t_fine)
            rho_fine   = np.array([gaussian_on_grid(s) for s in sigma_fine])
            for j in range(n_sub):
                t_mid_fine = 0.5*(t_fine[j]+t_fine[j+1])
                U_mid_fine = 0.5*kappa0(t_mid_fine, T_val)*q_np**2
                Q += np.dot(U_mid_fine, rho_fine[j+1]-rho_fine[j])

    return ts, np.array(tvds), np.array(kls), np.array(W_t), np.array(DF_t)


def run_harm(T_val, dt=0.0001, mode='bare', integrator='midpoint', n_trunc=None, decomp_cache=None):
    """
    Returns ts, tvd, kl, W, DF.
    DF is ALWAYS Delta_F(kappa0(t)) -- the bare protocol's free energy --
    regardless of which generator (bare/analytic/spectral) evolves rho.
    U_mid is ALWAYS the bare potential (1/2)*kappa0(t)*q^2.
    n_trunc: number of biorthogonal modes retained by the LCD generator
    (mode='spectral' only); None = full/exact.
    """
    ts   = np.arange(0, T_val+dt, dt)
    rho  = rho_eq_harm(0, T_val).copy()
    U_i  = 0.5*kappa0(0,T_val)*q_np**2
    Q    = 0.0
    tvds=[]; kls=[]; W_t=[]; DF_t=[]
    gen  = make_gen_harm(T_val, mode, n_trunc=n_trunc, decomp_cache=decomp_cache)
    step = magnus4_step if integrator=='magnus4' else midpoint_step

    for t in ts:
        req = rho_eq_harm(t, T_val)
        tvds.append(0.5*np.abs(rho-req).sum())
        kls.append(kl_divergence(rho, req))

        k0_now = kappa0(t, T_val)
        U_now  = 0.5*k0_now*q_np**2
        DeltaE = np.dot(U_now, rho) - np.dot(U_i, rho_eq_harm(0,T_val))
        W_t.append(DeltaE - Q)
        DF_t.append(free_energy_harm(k0_now) - free_energy_harm(kap_i))

        if t < T_val-dt/2:
            t_mid  = t+dt/2
            U_mid  = 0.5*kappa0(t_mid,T_val)*q_np**2
            rho_new = step(rho, t, dt, gen)
            Q += np.dot(U_mid, rho_new-rho)
            rho = rho_new

    return ts, np.array(tvds), np.array(kls), np.array(W_t), np.array(DF_t)

# ═══════════════════════════════════════════════════════════════
# RESULT I/O: folder naming + save helpers (results/data/<folder>/)
# ═══════════════════════════════════════════════════════════════

def _fmt(x):
    return f"{x:g}"

def dw_folder_name(T, dt):
    return f"DW_T_{_fmt(T)}_dt_{_fmt(dt)}"

def harm_folder_name(T, dt):
    return f"harmonic_T_{_fmt(T)}_dt_{_fmt(dt)}"

def _save_arrays(root, folder, arrays):
    out_dir = os.path.join(root, folder)
    os.makedirs(out_dir, exist_ok=True)
    for name, arr in arrays.items():
        np.save(os.path.join(out_dir, f"{name}.npy"), arr, allow_pickle=True)
    return out_dir

# ═══════════════════════════════════════════════════════════════
# RUN + SAVE: double well
# ═══════════════════════════════════════════════════════════════

def run_and_save_dw(T_val, dt):
    folder = dw_folder_name(T_val, dt)
    print(f"[DW] T={T_val}, dt={dt}  ->  results/data/{folder}/")

    ts, tvd_b, kl_b, W_b, DF_b = run_dw(T_val, dt=dt, mode='bare', integrator='midpoint')
    ts, tvd_l, kl_l, W_l, DF_l = run_dw(T_val, dt=dt, mode='lcd',  integrator='magnus4')
    Wd_b = W_b - DF_b
    Wd_l = W_l - DF_l

    arrays = dict(
        ts=ts, tvd_b=tvd_b, tvd_l=tvd_l, kl_b=kl_b, kl_l=kl_l,
        W_b=W_b, W_l=W_l, DF_b=DF_b, DF_l=DF_l, Wd_b=Wd_b, Wd_l=Wd_l,
        meta=np.array({'T': T_val, 'dt': dt, 'N': N_dw,
                        'lam_i': lam_i_dw, 'lam_f': lam_f_dw}, dtype=object),
    )
    out_dir = _save_arrays(DATA_ROOT, folder, arrays)

    md = [
        "# LCD Results — Double Well (Magnus4 Integrator)\n",
        f"Protocol: $T={T_val}$, step size $dt={dt}$, $N={N_dw}$ grid points.\n",
        f"$V(x,\\zeta)=x^4-2x^2+\\zeta x$, $\\zeta: {lam_i_dw}\\to{lam_f_dw}$\n",
        "| Quantity | Bare FP | LCD (Magnus4) |",
        "|---|---:|---:|",
        f"| Max TVD | {tvd_b.max():.4f} | {tvd_l.max():.2e} |",
        f"| Max $D_{{KL}}$ | {kl_b.max():.4f} | {kl_l.max():.2e} |",
        f"| $\\Delta F$ | {DF_b[-1]:.4f} | {DF_l[-1]:.4f} |",
        f"| $W_{{\\rm diss}}(T)$ | {Wd_b[-1]:.4f} | {Wd_l[-1]:.2e} |",
        f"| Max $\\|W_{{\\rm diss}}(t)\\|$ | {np.abs(Wd_b).max():.4f} | {np.abs(Wd_l).max():.2e} |",
        "",
    ]
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write("\n".join(md))

    print(f"  Max TVD  bare:        {tvd_b.max():.4f}")
    print(f"  Max TVD  LCD Magnus4: {tvd_l.max():.2e}")
    print(f"  Final Wdiss bare:     {Wd_b[-1]:.4f}")
    print(f"  Final Wdiss LCD:      {Wd_l[-1]:.2e}")
    return folder

# ═══════════════════════════════════════════════════════════════
# RUN + SAVE: harmonic trap
# ═══════════════════════════════════════════════════════════════

def run_and_save_harm(T_val, dt):
    folder = harm_folder_name(T_val, dt)
    print(f"[Harmonic] T={T_val}, dt={dt}  ->  results/data/{folder}/")

    ts, tvd_b, kl_b, W_b, DF_b = run_harm(T_val, dt=dt, mode='bare',     integrator='midpoint')
    ts, tvd_a, kl_a, W_a, DF_a = run_harm_analytic(T_val, dt=dt)
    ts, tvd_l, kl_l, W_l, DF_l = run_harm(T_val, dt=dt, mode='spectral', integrator='magnus4')
    Wd_b = W_b - DF_b
    Wd_a = W_a - DF_a
    Wd_l = W_l - DF_l

    arrays = dict(
        ts=ts, tvd_b=tvd_b, tvd_a=tvd_a, tvd_l=tvd_l,
        kl_b=kl_b, kl_a=kl_a, kl_l=kl_l,
        W_b=W_b, W_a=W_a, W_l=W_l, DF_b=DF_b, DF_a=DF_a, DF_l=DF_l,
        Wd_b=Wd_b, Wd_a=Wd_a, Wd_l=Wd_l,
        meta=np.array({'T': T_val, 'dt': dt, 'N': N_h,
                        'kap_i': kap_i, 'kap_f': kap_f}, dtype=object),
    )
    out_dir = _save_arrays(DATA_ROOT, folder, arrays)

    md = [
        "# LCD Results — Harmonic Trap (Magnus4 Integrator)\n",
        f"Protocol: $T={T_val}$, step size $dt={dt}$, $N={N_h}$ grid points.\n",
        f"$V(q,t)=\\frac{{1}}{{2}}\\kappa_0(t)q^2$, $\\kappa_0: {kap_i}\\to{kap_f}$\n",
        "| Quantity | Bare FP | Analytic CD | LCD (Magnus4, spectral) |",
        "|---|---:|---:|---:|",
        f"| Max TVD | {tvd_b.max():.4f} | {tvd_a.max():.2e} | {tvd_l.max():.2e} |",
        f"| Max $D_{{KL}}$ | {kl_b.max():.4f} | {kl_a.max():.2e} | {kl_l.max():.2e} |",
        f"| $\\Delta F$ | {DF_b[-1]:.4f} | {DF_a[-1]:.4f} | {DF_l[-1]:.4f} |",
        f"| $W_{{\\rm diss}}(T)$ | {Wd_b[-1]:.4f} | {Wd_a[-1]:.2e} | {Wd_l[-1]:.2e} |",
        f"| Max $\\|W_{{\\rm diss}}(t)\\|$ | {np.abs(Wd_b).max():.4f} | {np.abs(Wd_a).max():.2e} | {np.abs(Wd_l).max():.2e} |",
        "",
    ]
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write("\n".join(md))

    print(f"  Max TVD  bare:        {tvd_b.max():.4f}")
    print(f"  Max TVD  analytic CD: {tvd_a.max():.2e}")
    print(f"  Max TVD  LCD Magnus4: {tvd_l.max():.2e}")
    print(f"  Final Wdiss bare:     {Wd_b[-1]:.4f}")
    print(f"  Final Wdiss analytic: {Wd_a[-1]:.2e}")
    print(f"  Final Wdiss LCD:      {Wd_l[-1]:.2e}")
    return folder

# ═══════════════════════════════════════════════════════════════
# TRUNCATED BIORTHOGONAL DECOMPOSITION: n=1..10 modes vs full
# ═══════════════════════════════════════════════════════════════

def run_truncation_study(T_val, dt, n_list=range(1, 11)):
    """
    Compares the LCD generator built from a TRUNCATED biorthogonal
    decomposition (only the n_trunc slowest-decaying non-trivial modes)
    against the FULL decomposition (all N-1 modes), for both potentials.
    Writes:

        results/data/truncation_T_<T>_dt_<dt>/truncation_comparison.md
        results/data/truncation_T_<T>_dt_<dt>/*.npy  (summary arrays)

    NOTE on dt: this study intentionally may use a COARSER dt than the
    main experiment. The biorthogonal decomposition of L(t) does not
    depend on n_trunc, so a single shared `decomp_cache` dict per
    potential lets every n in n_list (and the final "full" run) reuse
    the SAME eigendecomposition at each t -- an ~len(n_list)x speedup
    with zero effect on the numbers, since np.linalg.eig(L) on these
    stiff birth-death generators (~100-250 ms for N=50-80) dominates
    runtime far more than the O(dt) step count does. Even with caching,
    running this sweep at the main experiment's dt=0.0001 costs tens of
    minutes per potential; dt=0.005 (default from __main__) keeps it to
    a couple of minutes while still resolving the n-convergence trend.
    """
    folder = f"truncation_T_{_fmt(T_val)}_dt_{_fmt(dt)}"
    print(f"[Truncation study] T={T_val}, dt={dt}, n={list(n_list)} -> results/data/{folder}/")

    def sweep(run_fn, mode, n_dof):
        n_max_full = n_dof - 1
        decomp_cache = {}
        rows = []
        for n in n_list:
            _, tvd_n, kl_n, W_n, DF_n = run_fn(T_val, dt=dt, mode=mode,
                                                integrator='magnus4', n_trunc=n,
                                                decomp_cache=decomp_cache)
            Wd_n = W_n - DF_n
            rows.append((n, tvd_n.max(), kl_n.max(), Wd_n[-1], np.abs(Wd_n).max()))
        _, tvd_f, kl_f, W_f, DF_f = run_fn(T_val, dt=dt, mode=mode,
                                           integrator='magnus4', n_trunc=None,
                                           decomp_cache=decomp_cache)
        Wd_f = W_f - DF_f
        rows.append((n_max_full, tvd_f.max(), kl_f.max(), Wd_f[-1], np.abs(Wd_f).max()))
        return rows, n_max_full

    rows_dw, n_full_dw     = sweep(run_dw,   'lcd',      N_dw)
    rows_harm, n_full_harm = sweep(run_harm, 'spectral', N_h)

    def rows_to_arrays(rows):
        rows = np.array(rows, dtype=float)
        return dict(n_trunc=rows[:,0], tvd_max=rows[:,1], kl_max=rows[:,2],
                    wdiss_final=rows[:,3], wdiss_max=rows[:,4])

    arrays = {f"dw_{k}": v for k, v in rows_to_arrays(rows_dw).items()}
    arrays.update({f"harm_{k}": v for k, v in rows_to_arrays(rows_harm).items()})
    arrays["meta"] = np.array({'T': T_val, 'dt': dt, 'n_list': list(n_list),
                                'n_full_dw': n_full_dw, 'n_full_harm': n_full_harm},
                               dtype=object)
    out_dir = _save_arrays(DATA_ROOT, folder, arrays)

    def fmt_row(n, tvd, kl, wf, wm, is_full, n_full):
        label = f"Full ($n={n_full}$)" if is_full else f"{int(n)}"
        return f"| {label} | {tvd:.3e} | {kl:.3e} | {wf:.3e} | {wm:.3e} |"

    md = [
        "# Truncated vs Full Biorthogonal Decomposition\n",
        "Comparison of the LCD counterdiabatic generator built from a "
        "**truncated** biorthogonal decomposition (only the $n$ slowest-decaying "
        "non-trivial eigenmodes of $L$ retained) against the **full** "
        f"decomposition. Protocol: $T={T_val}$, $dt={dt}$ (same as the main "
        "experiment).\n",
        "## Double Well\n",
        f"$N={N_dw}$ grid points, full decomposition uses $n={n_full_dw}$ non-trivial modes.\n",
        "| $n$ retained | Max TVD | Max $D_{KL}$ | $W_{\\rm diss}(T)$ | Max $\\|W_{\\rm diss}(t)\\|$ |",
        "|---|---:|---:|---:|---:|",
    ]
    for i in range(len(n_list)):
        md.append(fmt_row(*rows_dw[i], is_full=False, n_full=n_full_dw))
    md.append(fmt_row(*rows_dw[-1], is_full=True, n_full=n_full_dw))
    md.append("")

    md += [
        "## Harmonic Trap\n",
        f"$N={N_h}$ grid points, full decomposition uses $n={n_full_harm}$ non-trivial modes.\n",
        "| $n$ retained | Max TVD | Max $D_{KL}$ | $W_{\\rm diss}(T)$ | Max $\\|W_{\\rm diss}(t)\\|$ |",
        "|---|---:|---:|---:|---:|",
    ]
    for i in range(len(n_list)):
        md.append(fmt_row(*rows_harm[i], is_full=False, n_full=n_full_harm))
    md.append(fmt_row(*rows_harm[-1], is_full=True, n_full=n_full_harm))
    md.append("")

    with open(os.path.join(out_dir, "truncation_comparison.md"), "w") as f:
        f.write("\n".join(md))

    print(f"  Saved results/data/{folder}/truncation_comparison.md")
    return folder

# ═══════════════════════════════════════════════════════════════
# T-SWEEP: per-T dt schedule + consolidated sweep-summary md
# ═══════════════════════════════════════════════════════════════

# Short protocols get the fine dt=0.0001 used for the main T=0.5 experiment;
# everything else falls back to a coarser dt=0.0005. Rationale (see the
# runtime notes on biorthogonal_decomp/eig cost): dt=0.0001 for T > ~2 makes
# the LCD/spectral runs impractically slow (step count, and hence eig() call
# count, scales as T/dt). Edit FINE_DT_TS / the fallback below to change the
# policy; run_sweep() always looks it up via dt_for_T(T), so this is the one
# place that needs editing.
FINE_DT_TS = {0.1, 0.2, 0.3, 0.5, 0.7, 1.0}

def dt_for_T(T, fine_dt=0.0001, coarse_dt=0.0005, fine_Ts=FINE_DT_TS):
    """dt schedule for the T-sweep: fine_dt for T in fine_Ts, coarse_dt otherwise."""
    return fine_dt if T in fine_Ts else coarse_dt

def _load_saved(folder):
    """Reload every .npy written by run_and_save_dw/run_and_save_harm for `folder`."""
    out_dir = os.path.join(DATA_ROOT, folder)
    data = {}
    for fn in os.listdir(out_dir):
        if fn.endswith(".npy"):
            data[fn[:-4]] = np.load(os.path.join(out_dir, fn), allow_pickle=True)
    if "meta" in data:
        data["meta"] = data["meta"].item()
    return data

def run_sweep(T_list, dt_fn=dt_for_T):
    """
    Runs the main (bare vs LCD, and bare vs analytic-CD vs spectral-LCD)
    experiment for every T in T_list, at a per-T step size chosen by dt_fn
    (see dt_for_T), and writes:

      - each T's own results/data/{DW,harmonic}_T_<T>_dt_<dt>/ (npy + results.md),
        exactly as run_and_save_dw/run_and_save_harm always do
      - a CONSOLIDATED sweep summary across all T's:
            results/data/sweep_summary.md
            results/data/sweep_summary_dw.npy
            results/data/sweep_summary_harm.npy

    The summary md is in "long format": one row per (T, mode), reporting
    Max TVD, Max D_KL, and excess dissipated work W_diss(T) & max|W_diss(t)|
    for every mode -- Bare FP and LCD (Magnus4) for the double well; Bare FP,
    Analytic CD (scalar variance ODE, NOT the Liouville/rate-matrix
    evolution), and LCD (Magnus4, spectral) for the harmonic trap.
    """
    dw_rows, harm_rows = [], []

    for T in T_list:
        dt = dt_fn(T)
        print(f"\n[Sweep] T={T}, dt={dt}")

        dw_folder = run_and_save_dw(T, dt)
        d = _load_saved(dw_folder)
        dw_rows.append((T, dt, "Bare FP",
                         d["tvd_b"].max(), d["kl_b"].max(), d["Wd_b"][-1], np.abs(d["Wd_b"]).max()))
        dw_rows.append((T, dt, "LCD (Magnus4)",
                         d["tvd_l"].max(), d["kl_l"].max(), d["Wd_l"][-1], np.abs(d["Wd_l"]).max()))

        harm_folder = run_and_save_harm(T, dt)
        h = _load_saved(harm_folder)
        harm_rows.append((T, dt, "Bare FP",
                           h["tvd_b"].max(), h["kl_b"].max(), h["Wd_b"][-1], np.abs(h["Wd_b"]).max()))
        harm_rows.append((T, dt, "Analytic CD (scalar)",
                           h["tvd_a"].max(), h["kl_a"].max(), h["Wd_a"][-1], np.abs(h["Wd_a"]).max()))
        harm_rows.append((T, dt, "LCD (Magnus4, spectral)",
                           h["tvd_l"].max(), h["kl_l"].max(), h["Wd_l"][-1], np.abs(h["Wd_l"]).max()))

    def rows_to_npy_dict(rows):
        return dict(
            T=np.array([r[0] for r in rows], dtype=float),
            dt=np.array([r[1] for r in rows], dtype=float),
            mode=np.array([r[2] for r in rows], dtype=object),
            tvd_max=np.array([r[3] for r in rows], dtype=float),
            kl_max=np.array([r[4] for r in rows], dtype=float),
            wdiss_final=np.array([r[5] for r in rows], dtype=float),
            wdiss_max=np.array([r[6] for r in rows], dtype=float),
        )

    np.save(os.path.join(DATA_ROOT, "sweep_summary_dw.npy"),
            rows_to_npy_dict(dw_rows), allow_pickle=True)
    np.save(os.path.join(DATA_ROOT, "sweep_summary_harm.npy"),
            rows_to_npy_dict(harm_rows), allow_pickle=True)

    def fmt_row(row):
        T, dt, mode, tvd, kl, wf, wm = row
        return f"| {T:g} | {dt:g} | {mode} | {tvd:.3e} | {kl:.3e} | {wf:.3e} | {wm:.3e} |"

    md = [
        "# T-Sweep Summary — LCD Magnus4 Analysis\n",
        "dt schedule: $dt=0.0001$ for $T \\in \\{0.1,0.2,0.3,0.5,0.7,1.0\\}$; "
        "$dt=0.0005$ for every other $T$ swept here (see `dt_for_T`). Chosen "
        "because `eig()`-dominated runtime scales with the step count "
        "$T/dt$, and $dt=0.0001$ becomes impractically slow beyond $T\\sim2$.\n",
        "## Double Well\n",
        "| $T$ | $dt$ | Mode | Max TVD | Max $D_{KL}$ | $W_{\\rm diss}(T)$ | Max $\\|W_{\\rm diss}(t)\\|$ |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    md += [fmt_row(r) for r in dw_rows]
    md.append("")

    md += [
        "## Harmonic Trap\n",
        "| $T$ | $dt$ | Mode | Max TVD | Max $D_{KL}$ | $W_{\\rm diss}(T)$ | Max $\\|W_{\\rm diss}(t)\\|$ |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    md += [fmt_row(r) for r in harm_rows]
    md.append("")

    with open(os.path.join(DATA_ROOT, "sweep_summary.md"), "w") as f:
        f.write("\n".join(md))

    print(f"\nSaved results/data/sweep_summary.md "
          f"(+ sweep_summary_dw.npy, sweep_summary_harm.npy)")
    return dw_rows, harm_rows

# ═══════════════════════════════════════════════════════════════
# MAIN: run everything for T=0.5, dt=0.0001 (both potentials)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if "--sweep" in sys.argv:
        # Default T grid: the fine-dt list plus a representative set of
        # longer protocols at the coarser dt. Edit freely -- this is a
        # starting grid, not a fixed requirement.
        T_SWEEP_LIST = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
        print("="*65)
        print(f"T-SWEEP: {T_SWEEP_LIST}")
        print("="*65)
        run_sweep(T_SWEEP_LIST)
        raise SystemExit(0)

    T_dw   = 0.5
    T_harm = 0.5
    dt     = 0.0001        # main experiment step size (both potentials)
    dt_trunc = 0.005       # coarser dt for the truncation study only (see
                           # run_truncation_study docstring for why: same
                           # T, cheaper because eig() dominates runtime)

    print("="*65)
    print("DOUBLE WELL  (Magnus4 LCD vs Bare)")
    print("="*65)
    dw_folder = run_and_save_dw(T_dw, dt)

    print("\n" + "="*65)
    print("HARMONIC TRAP  (Bare vs Analytic CD vs Magnus4 spectral LCD)")
    print("="*65)
    harm_folder = run_and_save_harm(T_harm, dt)

    print("\n" + "="*65)
    print("TRUNCATED VS FULL BIORTHOGONAL DECOMPOSITION")
    print("="*65)
    trunc_folder = run_truncation_study(0.5, dt_trunc, n_list=range(1, 11))

    print("\nDone. Data written to:")
    print(f"  results/data/{dw_folder}/")
    print(f"  results/data/{harm_folder}/")
    print(f"  results/data/{trunc_folder}/")
    print("Run plotter.py to generate the PRL-style figures from these files.")
