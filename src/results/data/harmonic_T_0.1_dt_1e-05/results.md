# LCD Results — Harmonic Trap (Magnus4 Integrator)

Protocol: $T=0.1$, step size $dt=1e-05$, $N=80$ grid points.

$V(q,t)=\frac{1}{2}\kappa_0(t)q^2$, $\kappa_0: 1.0\to4.0$

| Quantity | Bare FP | Analytic CD | LCD (Magnus4, spectral) |
|---|---:|---:|---:|
| Max TVD | 0.2742 | 1.77e-12 | 1.95e-12 |
| Max $D_{KL}$ | 0.5226 | 3.79e-16 | 4.26e-16 |
| $\Delta F$ | 0.6931 | 0.6931 | 0.6931 |
| $W_{\rm diss}(T)$ | 0.7166 | 5.93e-13 | 1.32e-09 |
| Max $\|W_{\rm diss}(t)\|$ | 0.7166 | 7.59e-13 | 1.88e-09 |
