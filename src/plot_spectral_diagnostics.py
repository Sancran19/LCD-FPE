"""
Plot-only companion to spectral_diagnostics.py.
"""

import os

import spectral_diagnostics as sd
import plotter as plot_style

REQUIRED_FILES = [
    "spectral_gap.npy", "r0_check.npy", "rho_snapshots.npy",
    "Wd_b_refined.npy", "Wd_l_refined.npy", "spectral_coeffs.npy",
    "truncation_series.npy",
]

def _check_data(folder):
    out_dir = os.path.join(sd.sim.DATA_ROOT, folder)
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(out_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"results/data/{folder}/ is missing {missing} -- "
            f"run `python spectral_diagnostics.py` once first to compute them."
        )

if __name__ == "__main__":
    _check_data(sd.DW_FOLDER)
    _check_data(sd.HARM_FOLDER)

    print("Plotting standard tvd/kl/wdiss/landscape figures (plotter.py)...")
    plot_style.plot_dw_timeseries(sd.DW_FOLDER)
    plot_style.plot_landscape_dw(sd.DW_FOLDER)
    plot_style.plot_harm_timeseries(sd.HARM_FOLDER)
    plot_style.plot_landscape_harm(sd.HARM_FOLDER)

    for key in ('dw', 'harm'):
        print(f"\n--- {sd.POTENTIALS[key]['title']} ---")
        sd.plot_spectral_gap(key)
        sd.plot_r0_vs_pieq(key)
        sd.plot_rho_snapshots(key)
        sd.plot_wdiss_refined(key)
        sd.plot_spectral_coeffs(key)
        sd.plot_truncation_series(key)

    print("\nDONE (plot-only, no recompute). Figures written under results/plots/<folder>/")
