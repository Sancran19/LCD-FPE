# Wall-clock vs N -- dense eig(L) vs eigh_tridiagonal(L_sym)

Means over 100 calls (after 10 warm-up calls).

| Potential | N | dense eig (ms) | eigh\_tridiagonal (ms) | Speedup |
|---|---:|---:|---:|---:|
| Double well | 50 | 0.4636 | 0.0902 | 5.1x |
| Double well | 60 | 0.7733 | 0.1142 | 6.8x |
| Double well | 70 | 1.1990 | 0.1392 | 8.6x |
| Double well | 80 | 1.9824 | 0.1669 | 11.9x |
| Double well | 100 | 3.2033 | 0.2297 | 13.9x |
| Double well | 120 | 6.1974 | 0.3223 | 19.2x |
| Harmonic trap | 50 | 0.5276 | 0.0936 | 5.6x |
| Harmonic trap | 60 | 0.8017 | 0.1205 | 6.7x |
| Harmonic trap | 70 | 1.2588 | 0.1388 | 9.1x |
| Harmonic trap | 80 | 2.2701 | 0.1657 | 13.7x |
| Harmonic trap | 100 | 3.9710 | 0.2289 | 17.3x |
| Harmonic trap | 120 | 6.4160 | 0.3197 | 20.1x |
