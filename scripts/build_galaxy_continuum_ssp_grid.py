"""
Precompute and cache FSPS's full native age-grid response at a log-spaced
grid of metallicity (Z) points, for both IMF-treatment modes GalaxyContinuum
supports ("shared": fixed canonical Kroupa 2001 IMF at every Z; "dynamic":
Z-dependent IMF via demiurge.galaxy_continuum.imf.metallicity_dependent_slopes).

This is the one-time "pretabulation" build step for
demiurge.galaxy_continuum.pretabulated's fast numpy/torch interpolation-
based synthesis path -- run this ONCE (per Z-grid design decision) to
produce the shipped package data (src/demiurge/galaxy_continuum/data/
ssp_grid_{shared,dynamic}.npy + wave_grid.npy + metadata.json), not at
install time or runtime.

Requires python-fsps (see BUILD.md) -- NOT required by anyone just using the
resulting shipped data files at runtime. That asymmetry is the whole point:
the fsps_direct GalaxyContinuum backend needs python-fsps (and therefore a
Fortran compiler; WSL2 on Windows, see BUILD.md), the pretabulated backend
does not.

Cost: 20 Z points x ~25s/point x 2 IMF modes =~ 1000s (~17 min) for the
one-time per-Z isochrone setup, on this hardware/library (C3K_HR)
combination, confirmed empirically 2026-09-04 -- PLUS the per-age Monte
Carlo sampling described below, which adds roughly another ~15 min (K x
N_AGE extra `get_spectrum` calls per Z point at ~0.008s each once the
isochrone is cached; confirmed empirically 2026-09-14), for a total of
roughly ~30 min per full two-IMF-mode build.

Usage:
    python scripts/build_galaxy_continuum_ssp_grid.py
    python scripts/build_galaxy_continuum_ssp_grid.py --imf-modes dynamic  # rebuild just one

Why each age is Monte Carlo-averaged, not sampled once (found + approved
2026-09-14): querying FSPS at a single exact age near certain evolutionary
transitions (e.g. ~60-71 Myr at Z~0.006) can jump by >10 orders of magnitude
in the EUV/He+-ionizing tail (~100-300A), because that flux is dominated by
a vanishingly small number of extremely short-lived hot post-main-sequence
stars -- whether the isochrone's finite mass grid happens to sample one at a
given exact age is essentially a coin flip, confirmed to be a property of
FSPS/MIST's own isochrone response (reproduced by querying the official
python-fsps directly at densely-spaced ages), not an artifact of this
script or its interpolation. That jump is real, physical, and not the bug
-- suppressing it would misrepresent the isochrone. The actual bug is that
a single raw sample bakes in one arbitrary, non-reproducible realization of
that stochastic response into the shipped grid forever. Averaging many
finely-jittered sub-age samples around each target age instead estimates
the *expectation* of that stochastic process -- a smooth, reproducible
function of age -- which is what a fixed, shipped lookup table should
represent. See `_JITTER_HALF_WIDTH_DEX`/`N_MC_SAMPLES` below for the
exact scheme.

Design rationale for the Z-grid bounds/resolution (2026-09-04, full
derivation in project history -- summarized here so this script is
self-contained):
- Z_FLOOR=1e-4, Z_MASTER_MAX=0.25: covers the full range the registered
  closed-box metallicity priors (galaxy_continuum.metallicity.yield x
  star_formation_efficiency) can in principle produce -- Z_max =
  yield * ln(1/(1-efficiency)) ~ 0.05 * ln(1/0.01) ~= 0.23, with margin --
  not just the range any one example draw happened to show. A Z draw that
  still exceeds this (a genuine tail case) is clipped at runtime by
  pretabulated.py, which flags when it does so.
- N_Z=20, log-spaced: the largest point count that keeps each resulting
  grid file under GitHub's 100MB per-file hard limit (measured ~4.71
  MB/point; 20 points ~= 94MB). Log-spacing (not linear) empirically gives
  substantially better fidelity per point than linear spacing over the same
  range (RMS error ~4.4x lower at matched resolution, confirmed via direct
  A/B benchmark) -- Z(t) draws from the SFH generator spend most of their
  dynamic range compressed at low Z, so log-spacing concentrates resolution
  where the spectral shape actually varies fastest.
- Two separate grids (not one): "shared" IMF mode uses the fixed canonical
  Kroupa (2001) IMF at every Z; "dynamic" mode uses this project's own
  simplified Z-dependent IMF (see demiurge.galaxy_continuum.imf's module
  docstring for the full citation, derivation, and documented limitations
  of that simplification).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "src" / "demiurge" / "galaxy_continuum" / "data"

Z_FLOOR = 1.0e-4
Z_MASTER_MAX = 0.25
N_Z = 20

AGE_LOGYR_MIN, AGE_LOGYR_MAX, N_AGE = 5.0, 10.3, 107  # FSPS's own native ssp_ages grid, verified 2026-09-04

# Monte Carlo per-age averaging (2026-09-14, see module docstring for the full why):
N_MC_SAMPLES = 25  # within the approved 20-30 range
# Half-width of each age's jitter window, in dex (log10 yr) -- set to exactly half the
# native grid spacing so each age's window is its own non-overlapping cell (the range
# that would map to this grid point under the runtime bilinear interpolation), not an
# arbitrary number: neither so narrow it fails to average over a several-Myr-scale
# transition, nor so wide it blurs into a neighboring age's own territory.
_JITTER_HALF_WIDTH_DEX = 0.5 * (AGE_LOGYR_MAX - AGE_LOGYR_MIN) / (N_AGE - 1)
_JITTER_SEED = 20260914  # fixed so the shipped grid is exactly reproducible from source


def build_grid(imf_mode: str, out_path: Path):
    import fsps  # deferred: only this build script needs python-fsps, not the runtime package

    from demiurge.galaxy_continuum import imf as imf_module

    if imf_mode not in ("shared", "dynamic"):
        raise ValueError(f"imf_mode must be 'shared' or 'dynamic', got {imf_mode!r}")

    sp = fsps.StellarPopulation(zcontinuous=1, sfh=0, imf_type=2, dust1=0.0, dust2=0.0)
    z_grid = np.logspace(np.log10(Z_FLOOR), np.log10(Z_MASTER_MAX), N_Z)
    age_logyr_grid = np.linspace(AGE_LOGYR_MIN, AGE_LOGYR_MAX, N_AGE)
    rng = np.random.default_rng(_JITTER_SEED)

    wave = None
    grid = None
    t0 = time.time()
    for i, z_val in enumerate(z_grid):
        slopes = (
            imf_module.canonical_slopes()
            if imf_mode == "shared"
            else imf_module.metallicity_dependent_slopes(float(z_val))
        )
        sp.params["logzsol"] = float(np.log10(z_val / imf_module.Z_SUN))
        sp.params["imf1"] = slopes.imf1
        sp.params["imf2"] = slopes.imf2
        sp.params["imf3"] = slopes.imf3

        for j, target_logyr in enumerate(age_logyr_grid):
            jittered_logyr = rng.uniform(
                target_logyr - _JITTER_HALF_WIDTH_DEX, target_logyr + _JITTER_HALF_WIDTH_DEX, size=N_MC_SAMPLES
            )
            summed = None
            for logyr_sample in jittered_logyr:
                tage_gyr = float(10.0**logyr_sample / 1.0e9)
                w, f = sp.get_spectrum(tage=tage_gyr, peraa=False)
                if wave is None:
                    wave = w
                if summed is None:
                    summed = np.zeros_like(f)
                summed += f
            averaged = summed / N_MC_SAMPLES

            if grid is None:
                grid = np.lib.format.open_memmap(
                    str(out_path), mode="w+", dtype=np.float32, shape=(N_Z, N_AGE, averaged.shape[0])
                )
            grid[i, j] = averaged.astype(np.float32)

        print(
            f"  [{imf_mode}] Z[{i}]={z_val:.6f} (logzsol={sp.params['logzsol']:.3f}) done, "
            f"elapsed={time.time()-t0:.1f}s"
        )
    grid.flush()
    elapsed = time.time() - t0
    file_size_mb = out_path.stat().st_size / 1.0e6
    print(f"[{imf_mode}] grid precompute done in {elapsed:.1f}s, file size={file_size_mb:.1f}MB")

    info = dict(
        imf_mode=imf_mode,
        z_grid=z_grid.tolist(),
        n_z=N_Z,
        z_floor=Z_FLOOR,
        z_master_max=Z_MASTER_MAX,
        n_age=N_AGE,
        age_logyr_min=AGE_LOGYR_MIN,
        age_logyr_max=AGE_LOGYR_MAX,
        n_mc_samples=N_MC_SAMPLES,
        mc_jitter_half_width_dex=_JITTER_HALF_WIDTH_DEX,
        mc_jitter_seed=_JITTER_SEED,
        precompute_time_s=elapsed,
        file_size_mb=file_size_mb,
    )
    return wave, info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--imf-modes", nargs="+", default=["shared", "dynamic"], choices=["shared", "dynamic"])
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = DATA_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    wave = None
    for imf_mode in args.imf_modes:
        out_path = DATA_DIR / f"ssp_grid_{imf_mode}.npy"
        wave_this, info = build_grid(imf_mode, out_path)
        metadata[imf_mode] = info
        if wave is None:
            wave = wave_this

    wave_path = DATA_DIR / "wave_grid.npy"
    if not wave_path.exists():
        np.save(wave_path, wave)
        print(f"Wrote {wave_path}")

    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
