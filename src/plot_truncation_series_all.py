"""Plotter function for truncated series."""

import os
import glob
import numpy as np

import spectral_diagnostics as sd

def discover_truncation_runs():
    """Yields (key, T, dt, folder) for every run folder with a saved
    truncation_series.npy, key in {'dw','harm'}."""
    for key, pattern in [('dw', "DW_T_*_dt_*"), ('harm', "harmonic_T_*_dt_*")]:
        for path in sorted(glob.glob(os.path.join(sd.sim.DATA_ROOT, pattern))):
            folder = os.path.basename(path)
            if not os.path.exists(os.path.join(path, "truncation_series.npy")):
                continue
            meta = np.load(os.path.join(path, "meta.npy"), allow_pickle=True).item()
            yield key, meta['T'], meta['dt'], folder

if __name__ == "__main__":
    runs = list(discover_truncation_runs())
    print(f"Found {len(runs)} run(s) with truncation_series.npy:")
    for key, T, dt, folder in runs:
        print(f"  [{key}] T={T}, dt={dt}  ({folder})")

    for key, T, dt, folder in runs:
        sd.T_VAL  = T
        sd.DT_VAL = dt
        sd.POTENTIALS[key]['folder'] = folder
        sd.plot_truncation_series(key)

    print("\nDONE (plot-only, no recompute).")
