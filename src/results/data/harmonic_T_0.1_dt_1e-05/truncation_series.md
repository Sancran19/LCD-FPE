# Truncated Biorthogonal LCD -- Time-Series Comparison (Harmonic trap)

Protocol: $\tau=0.1$, $dt=1e-05$ (same fine resolution as the main spectral diagnostics figures). "Full" uses all $N-1$ non-trivial biorthogonal modes; truncated rows retain only the $n_{\rm trunc}$ slowest-decaying modes in the counterdiabatic generator $A_{CD}$ (`build_ACD_from_L`). $W_{\rm diss}$ computed via the Simpson power-integral (same method as `wdiss_refined`), not the coarser heat integral baked into `Lcd_full_magnus4.run_dw`/`run_harm`.

| $n_{\rm trunc}$ | Max TVD | Max $D_{KL}$ | $W_{\rm diss}(\tau)$ | Max $|W_{\rm diss}(t)|$ |
|---|---:|---:|---:|---:|
| 5 | 2.069e-04 | 1.172e-04 | 5.692e-05 | 5.692e-05 |
| 10 | 6.557e-05 | 9.006e-06 | 2.336e-05 | 2.336e-05 |
| 15 | 3.208e-05 | 2.504e-06 | 6.097e-06 | 6.097e-06 |
| 20 | 1.321e-05 | 2.635e-07 | 7.277e-07 | 7.277e-07 |
| 25 | 7.864e-06 | 5.040e-08 | 1.614e-07 | 1.614e-07 |
| 30 | 3.969e-06 | 3.744e-09 | 1.410e-08 | 1.410e-08 |
| 35 | 2.644e-06 | 5.936e-10 | 3.705e-09 | 3.705e-09 |
| Full ($n=79$) | 1.952e-12 | 4.257e-16 | -5.149e-12 | 5.151e-12 |
