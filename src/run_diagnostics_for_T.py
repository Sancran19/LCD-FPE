"""
Run the full spectral_diagnostics.py compute+plot pipeline for an
arbitrary (T, dt), reusing every function in spectral_diagnostics.py
unmodified.

Usage:
    python run_diagnostics_for_T.py <T> [dt]
    (dt defaults to 1e-5, matching the rest of this diagnostic series)
"""

import sys
import spectral_diagnostics as sd

def run_for(T, dt=1e-5):
    sd.T_VAL  = T
    sd.DT_VAL = dt
    sd.DW_FOLDER   = sd.sim.dw_folder_name(T, dt)
    sd.HARM_FOLDER = sd.sim.harm_folder_name(T, dt)
    sd.POTENTIALS['dw']['folder']   = sd.DW_FOLDER
    sd.POTENTIALS['harm']['folder'] = sd.HARM_FOLDER

    print("=" * 65)
    print(f"SPECTRAL DIAGNOSTICS: T={T}, dt={dt}")
    print("=" * 65)

    sd.ensure_main_run()

    print("\nGenerating standard tvd/kl/wdiss/landscape figures (plotter.py)...")
    sd.plot_style.plot_dw_timeseries(sd.DW_FOLDER)
    sd.plot_style.plot_landscape_dw(sd.DW_FOLDER)
    sd.plot_style.plot_harm_timeseries(sd.HARM_FOLDER)
    sd.plot_style.plot_landscape_harm(sd.HARM_FOLDER)

    for key in ('dw', 'harm'):
        print(f"\n--- {sd.POTENTIALS[key]['title']} (T={T}) ---")
        sd.compute_and_save(key)
        sd.plot_spectral_gap(key)
        sd.plot_r0_vs_pieq(key)
        sd.plot_rho_snapshots(key)
        sd.plot_wdiss_refined(key)
        sd.plot_spectral_coeffs(key)
        sd.compute_and_save_truncation(key)
        sd.plot_truncation_series(key)

    print(f"\nDONE_T_{T}. Figures written under results/plots/<folder>/")

if __name__ == "__main__":
    T_arg  = float(sys.argv[1])
    dt_arg = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-5
    run_for(T_arg, dt_arg)
