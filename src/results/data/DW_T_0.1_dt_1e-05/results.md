# LCD Results — Double Well (Magnus4 Integrator)

Protocol: $T=0.1$, step size $dt=1e-05$, $N=80$ grid points.

$V(x,\zeta)=x^4-2x^2+\zeta x$, $\zeta: -1.0\to1.0$

| Quantity | Bare FP | LCD (Magnus4) |
|---|---:|---:|
| Max TVD | 0.6644 | 2.14e-12 |
| Max $D_{KL}$ | 1.2655 | 3.88e-16 |
| $\Delta F$ | 0.0000 | 0.0000 |
| $W_{\rm diss}(T)$ | 1.3659 | 2.84e-12 |
| Max $\|W_{\rm diss}(t)\|$ | 1.3659 | 7.84e-09 |
