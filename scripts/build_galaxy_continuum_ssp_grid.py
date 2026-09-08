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

Cost: 20 Z points x ~25s/point x 2 IMF modes =~ 1000s (~17 min) on this
hardware/library (C3K_HR) combination, confirmed empirically 2026-09-04.

Usage:
    python scripts/build_galaxy_continuum_ssp_grid.py
    python scripts/build_galaxy_continuum_ssp_grid.py --imf-modes dynamic  # rebuild just one

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


def build_grid(imf_mode: str, out_path: Path):
    import fsps  # deferred: only this build script needs python-fsps, not the runtime package

    from demiurge.galaxy_continuum import imf as imf_module

    if imf_mode not in ("shared", "dynamic"):
        raise ValueError(f"imf_mode must be 'shared' or 'dynamic', got {imf_mode!r}")

    sp = fsps.StellarPopulation(zcontinuous=1, sfh=0, imf_type=2, dust1=0.0, dust2=0.0)
    z_grid = np.logspace(np.log10(Z_FLOOR), np.log10(Z_MASTER_MAX), N_Z)

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
        w, f_grid = sp.get_spectrum(tage=0, peraa=False)
        if wave is None:
            wave = w
            grid = np.lib.format.open_memmap(
                str(out_path), mode="w+", dtype=np.float32, shape=(N_Z, f_grid.shape[0], f_grid.shape[1])
            )
        grid[i] = f_grid.astype(np.float32)
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
