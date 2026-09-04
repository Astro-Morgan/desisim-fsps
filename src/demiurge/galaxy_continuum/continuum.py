"""
GalaxyContinuum -- pure stellar-continuum spectrum for the galaxy mock path
(the `continuum` bucket of the 3-bucket decomposition; dust attenuation is
explicitly NOT part of this channel and is handled separately downstream,
matching the pre-refactor convention of forcing FSPS's own dust2=0 here).

Three construction paths:

- `from_single_population`: wraps FSPS's native `sfh=0` (simple stellar
  population) mode directly -- one age, one metallicity, one IMF, no
  binning, no GP, no closed-box enrichment. This is deliberately almost
  nothing beyond python-fsps itself, specifically so it can be checked
  against a raw `fsps.StellarPopulation` call as a regression baseline
  (project direction, 2026-09-04: "we should be able to recover the fsps
  baseline").
- `from_arrays`: fully user-supplied SFR(t)/Z(t) history. Z(t) monotonicity
  is validated (raises by default; see `metallicity.validate_monotonic_z`).
  A length-1 history is degenerate and delegates to
  `from_single_population`, so the baseline-recovery guarantee holds here
  too, not just via the explicit constructor.
- `from_dense_basis`: the Tier-2 stochastic default (HANDOFF3 Sec. 6.1/6.4)
  -- draws SFH(t) (`sfh.draw_sfh`) and Z(t) (`metallicity.draw_metallicity`)
  from their registered priors, then synthesizes exactly like `from_arrays`.

`imf_mode` ("shared" or "dynamic") applies to `from_arrays` and
`from_dense_basis` alike: "shared" uses the fixed canonical Kroupa (2001)
IMF for the whole history; "dynamic" applies
`imf.metallicity_dependent_slopes` per time-bin (see `imf.py`'s module
docstring for exactly what this simplified treatment does and doesn't
capture). Metallicity itself is always binned (one representative Z per
FSPS call) regardless of `imf_mode` -- `python-fsps` 0.5.0's
multi-metallicity tabulated-SFH mode (`zcontinuous=3`) is unusable (see
`imf.py`'s module docstring for the confirmed upstream bug), so "shared
IMF" does not mean "single Z for the whole history," only "single IMF."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import imf as imf_module
from . import metallicity as metallicity_module
from . import sfh as sfh_module
from ..parameters.samplers import ParameterSampler, PriorSampler

try:
    import fsps as _fsps
except ImportError:  # pragma: no cover -- exercised only in environments without python-fsps
    _fsps = None


def _require_fsps():
    if _fsps is None:
        raise ImportError(
            "python-fsps is required for GalaxyContinuum but is not installed -- see BUILD.md "
            "(requires a Fortran compiler; WSL2 on Windows, native on Linux/macOS)."
        )
    return _fsps


@dataclass(frozen=True)
class GalaxyContinuum:
    wave: np.ndarray
    flux: np.ndarray
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_single_population(
        cls,
        age_gyr: float,
        z_absolute: float,
        *,
        imf_slopes: Optional[imf_module.IMFSlopes] = None,
        peraa: bool = False,
    ) -> "GalaxyContinuum":
        """Direct FSPS SSP (`sfh=0`) call -- the exact-baseline-recovery path."""
        if age_gyr <= 0.0:
            raise ValueError(f"age_gyr must be > 0, got {age_gyr}")
        if z_absolute <= 0.0:
            raise ValueError(f"z_absolute must be > 0, got {z_absolute}")
        fsps = _require_fsps()
        if imf_slopes is None:
            imf_slopes = imf_module.canonical_slopes()

        sp = fsps.StellarPopulation(zcontinuous=1, sfh=0, imf_type=2)
        sp.params["logzsol"] = float(np.log10(z_absolute / imf_module.Z_SUN))
        sp.params["imf1"] = imf_slopes.imf1
        sp.params["imf2"] = imf_slopes.imf2
        sp.params["imf3"] = imf_slopes.imf3
        wave, flux = sp.get_spectrum(tage=float(age_gyr), peraa=peraa)

        return cls(
            wave=wave,
            flux=flux,
            meta=dict(age_gyr=age_gyr, z_absolute=z_absolute, imf_slopes=imf_slopes),
        )

    @classmethod
    def from_arrays(
        cls,
        t_grid_gyr,
        sfr_msun_per_yr,
        z_grid,
        *,
        imf_mode: str = "shared",
        n_bins: int = 8,
        z_monotonic_enforce: bool = False,
        peraa: bool = False,
    ) -> "GalaxyContinuum":
        t_grid_gyr = np.atleast_1d(np.asarray(t_grid_gyr, dtype=float))
        sfr_msun_per_yr = np.atleast_1d(np.asarray(sfr_msun_per_yr, dtype=float))
        z_grid = np.atleast_1d(np.asarray(z_grid, dtype=float))
        if imf_mode not in ("shared", "dynamic"):
            raise ValueError(f"imf_mode must be 'shared' or 'dynamic', got {imf_mode!r}")

        if t_grid_gyr.size == 1:
            imf_slopes = (
                imf_module.canonical_slopes()
                if imf_mode == "shared"
                else imf_module.metallicity_dependent_slopes(float(z_grid[-1]))
            )
            return cls.from_single_population(
                float(t_grid_gyr[-1]), float(z_grid[-1]), imf_slopes=imf_slopes, peraa=peraa
            )

        z_grid = metallicity_module.validate_monotonic_z(z_grid, enforce=z_monotonic_enforce)
        return cls._synthesize(t_grid_gyr, sfr_msun_per_yr, z_grid, imf_mode=imf_mode, n_bins=n_bins, peraa=peraa)

    @classmethod
    def from_dense_basis(
        cls,
        rng: np.random.Generator,
        t_obs_gyr: float,
        *,
        sampler: Optional[ParameterSampler] = None,
        imf_mode: str = "dynamic",
        n_bins: int = 8,
        n_grid: int = 200,
        peraa: bool = False,
    ) -> "GalaxyContinuum":
        if sampler is None:
            sampler = PriorSampler()

        sfh_result = sfh_module.draw_sfh(rng, t_obs_gyr, sampler=sampler, n_grid=n_grid)
        z_result = metallicity_module.draw_metallicity(rng, sfh_result.cumulative_mass_fraction, sampler=sampler)

        result = cls._synthesize(
            sfh_result.t_grid_gyr,
            sfh_result.sfr_msun_per_yr,
            z_result.z_grid,
            imf_mode=imf_mode,
            n_bins=n_bins,
            peraa=peraa,
        )
        result.meta["sfh"] = sfh_result
        result.meta["metallicity"] = z_result
        return result

    @classmethod
    def _synthesize(cls, t_grid_gyr, sfr_msun_per_yr, z_grid, *, imf_mode, n_bins, peraa) -> "GalaxyContinuum":
        fsps = _require_fsps()
        sp = fsps.StellarPopulation(zcontinuous=1, sfh=3, imf_type=2)
        t_obs = float(t_grid_gyr[-1])

        bins = _bin_sfh(t_grid_gyr, sfr_msun_per_yr, z_grid, n_bins)
        if not bins:
            raise ValueError("SFH has zero total star formation -- nothing to synthesize.")

        wave = None
        total_flux = None
        bin_meta = []
        for segment_sfr, mean_z in bins:
            imf_slopes = (
                imf_module.canonical_slopes() if imf_mode == "shared" else imf_module.metallicity_dependent_slopes(mean_z)
            )
            sp.params["logzsol"] = float(np.log10(mean_z / imf_module.Z_SUN))
            sp.params["imf1"] = imf_slopes.imf1
            sp.params["imf2"] = imf_slopes.imf2
            sp.params["imf3"] = imf_slopes.imf3
            sp.set_tabular_sfh(t_grid_gyr, segment_sfr)
            w, f = sp.get_spectrum(tage=t_obs, peraa=peraa)

            if wave is None:
                wave = w
                total_flux = np.zeros_like(f)
            total_flux = total_flux + f
            bin_meta.append(dict(mean_z=mean_z, imf_slopes=imf_slopes, mass_msun=float(segment_sfr.sum())))

        return cls(wave=wave, flux=total_flux, meta=dict(imf_mode=imf_mode, n_bins=n_bins, bins=bin_meta))


def _bin_sfh(t_grid: np.ndarray, sfr: np.ndarray, z_grid: np.ndarray, n_bins: int):
    """Split (t, SFR, Z) into up to `n_bins` contiguous time segments, each
    with its own SFR-weighted mean Z. Segments with zero total SFR are
    dropped. Returns a list of (segment_sfr, mean_z) -- `segment_sfr` is
    zeroed outside the segment but full-length, so each drives an
    independent tabulated-SFH FSPS call whose sum over segments reconstructs
    the full history (verified additive to floating-point precision for a
    shared Z/IMF; see galaxy_continuum test suite).
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    edges = np.linspace(0, len(t_grid), n_bins + 1).astype(int)

    segments = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        segment_total = sfr[lo:hi].sum()
        if segment_total <= 0.0:
            continue
        mean_z = float(np.average(z_grid[lo:hi], weights=sfr[lo:hi]))
        segment_sfr = np.zeros_like(sfr)
        segment_sfr[lo:hi] = sfr[lo:hi]
        segments.append((segment_sfr, mean_z))
    return segments
