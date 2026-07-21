# Symmetric-Gauge Biorthogonal Decomposition -- Verification

Checks $L_{\rm sym} = D^{-1/2} L D^{1/2}$ ($D=\mathrm{diag}(\pi_{\rm eq})$) against the dense nonsymmetric decomposition used by the main pipeline, at $N=80$ (current grid for both potentials), for $t/\tau \in \{0, 0.25, 0.5, 0.75, 1.0\}$ and $\tau \in \{0.01, 0.1, 1.0\}$. Timings are means over 100 calls (after 10 warm-up calls).

| Potential | $\tau$ | $t/\tau$ | $\max\|L_{\rm sym}-L_{\rm sym}^T\|$ | $\max\|\mathrm{eig}(L)-\mathrm{eig}(L_{\rm sym})\|$ | dense eig (ms) | eigh\_tridiagonal (ms) | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| Double well | 0.01 | 0.00 | 1.42e-13 | 5.60e-08 | 2.0164 | 0.1662 | 12.1x |
| Double well | 0.01 | 0.25 | 1.42e-13 | 4.25e-08 | 1.9061 | 0.1665 | 11.4x |
| Double well | 0.01 | 0.50 | 1.42e-13 | 1.55e-08 | 2.0144 | 0.1652 | 12.2x |
| Double well | 0.01 | 0.75 | 1.42e-13 | 7.02e-09 | 2.0725 | 0.1671 | 12.4x |
| Double well | 0.01 | 1.00 | 1.14e-13 | 5.74e-09 | 1.9527 | 0.1674 | 11.7x |
| Double well | 0.1 | 0.00 | 1.42e-13 | 5.60e-08 | 2.0050 | 0.1660 | 12.1x |
| Double well | 0.1 | 0.25 | 1.42e-13 | 4.25e-08 | 1.9039 | 0.1662 | 11.5x |
| Double well | 0.1 | 0.50 | 1.42e-13 | 1.55e-08 | 1.9909 | 0.1656 | 12.0x |
| Double well | 0.1 | 0.75 | 1.14e-13 | 1.20e-08 | 2.0562 | 0.1667 | 12.3x |
| Double well | 0.1 | 1.00 | 1.14e-13 | 5.74e-09 | 1.9405 | 0.1677 | 11.6x |
| Double well | 1 | 0.00 | 1.42e-13 | 5.60e-08 | 2.0048 | 0.1658 | 12.1x |
| Double well | 1 | 0.25 | 1.42e-13 | 4.25e-08 | 1.8937 | 0.1660 | 11.4x |
| Double well | 1 | 0.50 | 1.42e-13 | 1.55e-08 | 1.9906 | 0.1652 | 12.0x |
| Double well | 1 | 0.75 | 1.42e-13 | 7.02e-09 | 2.0585 | 0.1668 | 12.3x |
| Double well | 1 | 1.00 | 1.14e-13 | 5.74e-09 | 1.9509 | 0.1674 | 11.7x |
| Harmonic trap | 0.01 | 0.00 | 4.26e-14 | 2.56e-12 | 2.4443 | 0.1629 | 15.0x |
| Harmonic trap | 0.01 | 0.25 | 4.26e-14 | 2.27e-12 | 2.3915 | 0.1650 | 14.5x |
| Harmonic trap | 0.01 | 0.50 | 4.26e-14 | 1.81e-10 | 2.2751 | 0.1659 | 13.7x |
| Harmonic trap | 0.01 | 0.75 | 4.26e-14 | 2.56e-08 | 2.6821 | 0.1692 | 15.8x |
| Harmonic trap | 0.01 | 1.00 | 5.68e-14 | 6.56e-08 | 2.5677 | 0.1700 | 15.1x |
| Harmonic trap | 0.1 | 0.00 | 4.26e-14 | 2.56e-12 | 2.4503 | 0.1626 | 15.1x |
| Harmonic trap | 0.1 | 0.25 | 4.26e-14 | 2.27e-12 | 2.3946 | 0.1651 | 14.5x |
| Harmonic trap | 0.1 | 0.50 | 4.26e-14 | 1.81e-10 | 2.2790 | 0.1658 | 13.7x |
| Harmonic trap | 0.1 | 0.75 | 2.84e-14 | 1.71e-08 | 2.6490 | 0.1689 | 15.7x |
| Harmonic trap | 0.1 | 1.00 | 5.68e-14 | 6.56e-08 | 2.5706 | 0.1700 | 15.1x |
| Harmonic trap | 1 | 0.00 | 4.26e-14 | 2.56e-12 | 2.4508 | 0.1630 | 15.0x |
| Harmonic trap | 1 | 0.25 | 4.26e-14 | 2.27e-12 | 2.3900 | 0.1651 | 14.5x |
| Harmonic trap | 1 | 0.50 | 4.26e-14 | 1.81e-10 | 2.2667 | 0.1658 | 13.7x |
| Harmonic trap | 1 | 0.75 | 4.26e-14 | 2.56e-08 | 2.6696 | 0.1692 | 15.8x |
| Harmonic trap | 1 | 1.00 | 5.68e-14 | 6.56e-08 | 2.5630 | 0.1700 | 15.1x |
