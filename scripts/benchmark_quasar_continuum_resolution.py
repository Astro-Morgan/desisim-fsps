"""
Benchmark demiurge.quasar_continuum.QuasarContinuum generation time as a
function of `n_energy_bins`, tracking energy conservation (total integrated
luminosity vs. the model's own analytic mdot*L_Edd, per KD18's definitions
-- the same independent check quasar_continuum/geometry.py's module
docstring and the sandbox validation this channel was built against both
use) at each resolution, so a future change to the default `n_energy_bins`
(600, chosen ad hoc when the channel was first built -- not itself
benchmarked at the time) has real data behind it instead of a guess.

`icor`/`iout` (radial-grid resolution) were checked for sensitivity during
the channel's original sandbox validation and found already converged at
their current defaults (icor=20, iout=400 -- doubling either did not move
the energy-conservation ratio at all) -- not re-swept here; this script is
specifically the n_energy_bins axis, which was NOT checked at the time.

Fiducial parameters: M=1e8 Msun, mdot=0.10, cosi=0.5 (the anisotropy-
neutral case, cosi/0.5=1, so the energy-conservation ratio is directly
comparable to mdot without an extra cosi-scaling factor -- see
quasar_continuum/continuum.py's own `cosi_scale` comment), a spread of
kTe_hot/kTe_warm/Gamma_warm/hard_xray_luminosity_fraction within their
Tier-2 registry ranges (not the QSOSED-typical defaults exclusively, so the
benchmark isn't accidentally tuned to one easy case).

Usage:
    python scripts/benchmark_quasar_continuum_resolution.py
    python scripts/benchmark_quasar_continuum_resolution.py --repeats 5 --plot out.png

Findings (confirmed empirically, 2026-09-09, `--repeats 5`, this
hardware -- see this script's own printed table for exact numbers):
- Generation time grows *sub-linearly* with n_energy_bins: ~54ms at 100
  bins, ~67-74ms at the current default (600), ~168-180ms at 3600 (a 36x
  bin-count increase costs only ~3x the runtime). Consistent with the
  dominant cost being the ~icor+1 `donthcomp` calls (each with its own
  internal, `ear`-independent Kompaneets-solve grid of size <900), not the
  final energy-grid binning -- `n_energy_bins` mostly affects the *output*
  spectral resolution, not the physics solve's own cost.
- **Energy conservation (integrated L / (mdot*L_Edd)) is completely FLAT
  across the whole n_energy_bins range tested (100-3600), for both a
  QSOSED-typical case (ratio ~1.088-1.091) and a harder/cooler-warm,
  high-hard-Xray-fraction case (ratio ~1.132-1.134, pre-fix numbers -- see
  below).** This was a real, useful negative result at the time: the then-
  documented ~9-13% energy-conservation gap was NOT a discretization
  artifact of the energy grid (nor, per the channel's original sandbox
  validation, of the icor/iout radial grid either, which was checked
  separately and also found already converged) -- ruling out discretization
  is exactly what motivated chasing the gap down to its real cause. That
  cause has since been root-caused and fixed (2026-09-09, same day): the
  disc/warm zones' own spectral synthesis was never subtracting the
  fraction of each annulus's photons intercepted by the corona as seed
  photons, double-counting that energy against the corona's own `L_hot`
  output -- see quasar_continuum/geometry.py's module docstring for the
  full diagnosis and fix. Post-fix, the ratios above land close to 1.0
  instead (~0.95-1.10 depending on reprocess/hard_xray_luminosity_fraction)
  -- this script's own resolution sweep was not rerun after the fix, so the
  ratio values quoted above are historical, pre-fix numbers, kept for
  record of how the diagnosis was reached. The n_energy_bins/speed
  finding itself (sub-linear growth, ~2x win available toward 200-300
  bins) is unaffected by the fix and still holds.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from demiurge.quasar_continuum import QuasarContinuum

FIDUCIAL_CASES = [
    dict(m_msun=1.0e8, astar=0.0, mdot_edd=0.10, cosi=0.5,
         hard_xray_luminosity_fraction=0.02, kte_hot_kev=100.0, kte_warm_kev=0.2, gamma_warm=2.5,
         label="typical (QSOSED-like)"),
    dict(m_msun=1.0e8, astar=0.0, mdot_edd=0.10, cosi=0.5,
         hard_xray_luminosity_fraction=0.04, kte_hot_kev=45.0, kte_warm_kev=0.8, gamma_warm=3.0,
         label="harder/cooler warm, high hard-Xray fraction"),
]

N_ENERGY_BINS_GRID = [100, 150, 200, 300, 450, 600, 900, 1200, 1800, 2400, 3600]

L_SUN_ERG_S = 3.828e33


def energy_conservation_ratio(qc: QuasarContinuum, geom) -> float:
    """Integrate the actual output L_nu [Lsun/Hz] over frequency and
    compare to mdot*L_Edd (erg/s) -- valid at cosi=0.5 specifically, where
    the disk/warm cosi/0.5 anisotropy factor is exactly 1."""
    freq_hz = 2.99792458e18 / qc.wave  # wave is Angstrom, ascending -> freq descending
    order = np.argsort(freq_hz)
    integrated_lsun = np.trapezoid(qc.flux[order], freq_hz[order])
    integrated_erg_s = integrated_lsun * L_SUN_ERG_S
    target_erg_s = geom.mdot_edd * geom.ledd
    return integrated_erg_s / target_erg_s


def main(repeats: int, plot_path: str | None) -> None:
    results = []  # (case_label, n_energy_bins, median_time_s, energy_ratio)

    for case in FIDUCIAL_CASES:
        label = case["label"]
        kwargs = {k: v for k, v in case.items() if k != "label"}
        for n_bins in N_ENERGY_BINS_GRID:
            times = []
            qc = None
            for _ in range(repeats):
                t0 = time.perf_counter()
                qc = QuasarContinuum.from_parameters(**kwargs, n_energy_bins=n_bins)
                times.append(time.perf_counter() - t0)
            ratio = energy_conservation_ratio(qc, qc.meta["geometry"])
            median_t = float(np.median(times))
            results.append((label, n_bins, median_t, ratio))
            print(f"[{label:38s}] n_energy_bins={n_bins:5d}  "
                  f"median_time={median_t*1000:7.2f} ms  energy_ratio={ratio:.4f}")

    if plot_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(11, 4.5))
        for label in {r[0] for r in results}:
            rows = [r for r in results if r[0] == label]
            rows.sort(key=lambda r: r[1])
            n_bins = [r[1] for r in rows]
            times_ms = [r[2] * 1000 for r in rows]
            ratios = [r[3] for r in rows]
            ax_t.plot(n_bins, times_ms, marker="o", label=label)
            ax_e.plot(n_bins, ratios, marker="o", label=label)

        ax_t.set_xlabel("n_energy_bins")
        ax_t.set_ylabel("median generation time [ms]")
        ax_t.set_xscale("log")
        ax_t.set_yscale("log")
        ax_t.grid(alpha=0.3, which="both")
        ax_t.legend(fontsize=8)
        ax_t.set_title("QuasarContinuum.from_parameters generation time")

        ax_e.axhline(1.0, color="black", lw=1, ls="--", alpha=0.5)
        ax_e.set_xlabel("n_energy_bins")
        ax_e.set_ylabel("integrated L / (mdot * L_Edd)")
        ax_e.set_xscale("log")
        ax_e.grid(alpha=0.3, which="both")
        ax_e.legend(fontsize=8)
        ax_e.set_title("Energy conservation vs. resolution")

        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        print(f"\nsaved {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5, help="timing repeats per (case, n_energy_bins) point")
    parser.add_argument("--plot", type=str, default=None, help="path to save a benchmark PNG (optional)")
    args = parser.parse_args()
    main(args.repeats, args.plot)
