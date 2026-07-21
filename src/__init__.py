"""
biorthogonal_fpe
=================

Biorthogonal-decomposition local counterdiabatic (LCD) driving for
discrete Fokker-Planck / master-equation dynamics, integrated with a
4th-order Magnus scheme, benchmarked against bare (uncontrolled)
dynamics and (where available) an exact analytic counterdiabatic
solution.

Systems studied:
  - Double well:        V(x,zeta)   = x^4 - 2x^2 + zeta*x
  - Harmonic trap:       V(q,t)     = (1/2) kappa_0(t) q^2  (has an
                          exact analytic CD solution via a scalar
                          variance ODE, in addition to the spectral LCD)
  - Quartic coalescence: V(x,zeta)  = x^4 - 16(1-zeta)x^2  (a
                          deliberately adversarial stress test with an
                          exponentially closing spectral gap)

Module map (see README.md for the full guide):
  Lcd_full_magnus4.py           core physics: Magnus4/midpoint
                                 integrators, biorthogonal decomposition
                                 (dense + symmetric/tridiagonal), the
                                 counterdiabatic generator, TVD/KL
                                 metrics, save/load I/O
  plotter.py                    PRL-style plotting for the standard
                                 TVD/KL/Wdiss/landscape figures
  spectral_diagnostics.py       fine-resolution diagnostics: spectral
                                 gap, r0 vs pi_eq, rho(x,t) snapshots,
                                 Simpson-refined Wdiss, spectral
                                 coefficients, truncated-mode series
  plot_spectral_diagnostics.py  plot-only companion (no recompute)
  run_diagnostics_for_T.py      re-parameterize spectral_diagnostics.py
                                 for an arbitrary (T, dt)
  plot_truncation_series_all.py plot-only, auto-discovers every
                                 computed truncation series
  symmetric_gauge_verification.py
                                 validates the detailed-balance
                                 symmetrization L_sym = D^-1/2 L D^1/2
                                 and its eigh_tridiagonal speedup
  harmonic_tau_sweep.py         W(tau) vs tau (harmonic trap)
  free_energy_diff.py           quartic coalescence model: W(T) vs
                                tau (protocol duration) sweeps
  quartic_diagnostics.py        quartic model: potential-landscape
                                 snapshots + spectral gap

Every script here is import-safe (importing does not trigger
simulation) except free_energy_diff.py, which runs its full sweep at
import time by design -- see README.md.
"""

__version__ = "0.1.0"
