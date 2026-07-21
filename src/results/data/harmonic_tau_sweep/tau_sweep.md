# Harmonic Trap -- Work $\mathcal{W}(\tau)$ vs Protocol Duration $\tau$

$\Delta F$ (deterministically obtained via evolving the FP equation, and not via estimation against $\tau$) $= 0.693096$. $W(\tau)$ (final accumulated work); $W_{\rm diss}(\tau) = W(\tau) - \Delta F$. TVD is the FINAL-time value $\mathrm{TVD}(\rho(\tau),\pi_{\rm eq}(\tau))$, not the trajectory max. Bare/analytic $W$ from `Lcd_full_magnus4.run_harm`/`run_harm_analytic`; LCD $W$ from the Simpson power-integral (`run_lcd_simpson`, same method as `spectral_diagnostics.wdiss_refined`).

| $\tau$ | $W^{\rm bare}(\tau)$ | $W^{\rm analytic}(\tau)$ | $W^{\rm LCD}(\tau)$ | $\Delta F$ | $W_{\rm diss}^{\rm bare}$ | $W_{\rm diss}^{\rm analytic}$ | $W_{\rm diss}^{\rm LCD}$ | $\mathrm{TVD}^{\rm bare}$ | $\mathrm{TVD}^{\rm analytic}$ | $\mathrm{TVD}^{\rm LCD}$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 1.4977 | 0.6931 | 0.6931 | 0.6931 | 0.8046 | 4.005e-13 | -6.867e-12 | 3.220e-01 | 1.764e-12 | 2.249e-12 |
| 0.003 | 1.4958 | 0.6931 | 0.6931 | 0.6931 | 0.8027 | 2.265e-14 | -5.163e-12 | 3.207e-01 | 5.262e-14 | 2.139e-12 |
| 0.01 | 1.4890 | 0.6931 | 0.6931 | 0.6931 | 0.7959 | 5.818e-13 | -5.701e-12 | 3.160e-01 | 7.559e-14 | 2.197e-12 |
| 0.03 | 1.4702 | 0.6931 | 0.6931 | 0.6931 | 0.7771 | 2.898e-14 | -4.944e-12 | 3.030e-01 | 7.276e-14 | 1.952e-12 |
| 0.1 | 1.4097 | 0.6931 | 0.6931 | 0.6931 | 0.7166 | 5.933e-13 | -5.149e-12 | 2.605e-01 | 7.128e-14 | 1.843e-12 |
| 0.3 | 1.2727 | 0.6931 | 0.6931 | 0.6931 | 0.5796 | 3.186e-13 | -3.715e-12 | 1.581e-01 | 2.179e-14 | 1.025e-12 |
| 1 | 1.0215 | 0.6931 | 0.6931 | 0.6931 | 0.3284 | -5.635e-13 | -2.268e-12 | 2.001e-02 | 8.105e-15 | 9.336e-14 |
| 3 | 0.8274 | 0.6931 | 0.6931 | 0.6931 | 0.1343 | 5.285e-14 | -8.511e-13 | 6.648e-04 | 1.312e-15 | 7.332e-15 |
| 10 | 0.7339 | 0.6931 | 0.6931 | 0.6931 | 0.0408 | -2.345e-13 | -2.666e-13 | 2.019e-05 | 9.035e-17 | 3.719e-15 |
