# Truncated Biorthogonal LCD -- Time-Series Comparison (Double well)

Protocol: $\tau=0.1$, $dt=1e-05$ (same fine resolution as the main spectral diagnostics figures). "Full" uses all $N-1$ non-trivial biorthogonal modes; truncated rows retain only the $n_{\rm trunc}$ slowest-decaying modes in the counterdiabatic generator $A_{CD}$ (`build_ACD_from_L`). $W_{\rm diss}$ computed via the Simpson power-integral (same method as `wdiss_refined`), not the coarser heat integral baked into `Lcd_full_magnus4.run_dw`/`run_harm`.

| $n_{\rm trunc}$ | Max TVD | Max $D_{KL}$ | $W_{\rm diss}(\tau)$ | Max $|W_{\rm diss}(t)|$ |
|---|---:|---:|---:|---:|
| 5 | 2.662e-03 | 3.325e-05 | 1.342e-04 | 1.342e-04 |
| 10 | 7.435e-05 | 2.705e-08 | 1.971e-07 | 1.971e-07 |
| 15 | 1.348e-06 | 1.100e-11 | 1.958e-10 | 1.958e-10 |
| 20 | 1.385e-07 | 1.156e-13 | 5.415e-12 | 5.420e-12 |
| 25 | 7.774e-09 | 8.539e-16 | 2.854e-12 | 2.859e-12 |
| 30 | 1.348e-09 | 4.531e-16 | 2.840e-12 | 2.845e-12 |
| 35 | 1.525e-10 | 4.536e-16 | 2.839e-12 | 2.844e-12 |
| Full ($n=79$) | 2.137e-12 | 3.882e-16 | 2.843e-12 | 2.848e-12 |
