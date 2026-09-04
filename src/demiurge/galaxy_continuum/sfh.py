"""
Continuous star-formation-history (SFH) generation for the galaxy continuum
channel.

The Tier-2/3-default stochastic path implemented here is a Dense-Basis-style
non-parametric reconstruction (Iyer & Gawiser 2017, ApJ 838, 127; Iyer et al.
2019, ApJ 879, 116, "Non-parametric Star Formation History Reconstruction
with Gaussian Processes I"): a small number of stellar-mass-formation-time
quantiles are drawn, then a Gaussian Process with a Matern-3/2 kernel is
conditioned to pass exactly through those quantile constraints in
cumulative-mass-fraction-vs-cosmic-time space (Iyer et al. 2019 Sec. 2.1).
The continuous star-formation rate SFR(t) is the time-derivative of that
reconstructed cumulative curve.

This is one of two ways `GalaxyContinuum` can be given a star-formation
history -- the other is a fully user-supplied SFR(t) array, which does not
touch this module at all.

Implementation notes, departures from the cited method (documented, not
hidden -- see `demiurge.parameters.registry`'s entries for this channel for
which specific numbers here are cited vs. MAGIC):

- We take the GP posterior MEAN through the quantile constraints rather than
  drawing a random posterior sample. Stochasticity instead comes from where
  the quantile-time constraints themselves land, drawn from a Dirichlet
  prior over the time-gaps between them -- this matches Iyer et al. 2019's
  own framing of the length-scale hyperparameter as controlling "the tension
  in a string that passes through all the constraints," and avoids needing a
  posterior-covariance Cholesky/sampling step.
- SFR(t) = M_total * dF/dt is obtained by numerically differentiating the
  reconstructed cumulative-mass-fraction curve on a fine grid, not via the
  analytic derivative of the GP kernel.
- Negative SFR is clipped to zero and the curve is rescaled to conserve
  M_total exactly -- a practical safety net for the same failure mode Iyer
  et al. 2019 designed their length-scale calibration to avoid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..parameters.samplers import ParameterSampler, PriorSampler

# Stellar-mass-formation quantiles used as GP constraints, e.g. (0.25, 0.5, 0.75)
# means "25%/50%/75% of total stellar mass formed by these times". Fixed at
# module level (not a runtime argument) because the registered
# `galaxy_continuum.sfh.mass_quantile_gap_fractions` Dirichlet prior has a
# fixed dimensionality (len(QUANTILE_FRACTIONS) + 1 gaps) tied to this choice.
QUANTILE_FRACTIONS: tuple = (0.25, 0.5, 0.75)

_SQRT3 = np.sqrt(3.0)


@dataclass(frozen=True)
class SFHResult:
    """A realized, continuous star-formation history on a fixed time grid."""

    t_grid_gyr: np.ndarray
    sfr_msun_per_yr: np.ndarray
    cumulative_mass_fraction: np.ndarray
    total_stellar_mass_msun: float
    quantile_times_gyr: np.ndarray
    gp_length_scale_gyr: float


def _matern32_kernel(t1: np.ndarray, t2: np.ndarray, length_scale: float) -> np.ndarray:
    d = np.abs(t1[:, None] - t2[None, :]) / length_scale
    return (1.0 + _SQRT3 * d) * np.exp(-_SQRT3 * d)


def reconstruct_cumulative_sfh(
    t_grid_gyr: np.ndarray,
    quantile_times_gyr: np.ndarray,
    t_obs_gyr: float,
    length_scale_gyr: float,
    *,
    nugget: float = 1e-10,
) -> np.ndarray:
    """GP (Matern-3/2) posterior-mean reconstruction of the cumulative
    stellar-mass-fraction curve F(t) through the exact constraints
    (0, 0), (quantile_times_gyr, QUANTILE_FRACTIONS), (t_obs_gyr, 1),
    clipped to be non-decreasing and within [0, 1].
    """
    t_train = np.concatenate(([0.0], quantile_times_gyr, [t_obs_gyr]))
    f_train = np.concatenate(([0.0], QUANTILE_FRACTIONS, [1.0]))

    k_train = _matern32_kernel(t_train, t_train, length_scale_gyr) + nugget * np.eye(len(t_train))
    k_query = _matern32_kernel(t_grid_gyr, t_train, length_scale_gyr)
    weights = np.linalg.solve(k_train, f_train)
    f_grid = k_query @ weights

    f_grid = np.clip(np.maximum.accumulate(f_grid), 0.0, 1.0)
    return f_grid


def draw_sfh(
    rng: np.random.Generator,
    t_obs_gyr: float,
    *,
    sampler: Optional[ParameterSampler] = None,
    n_grid: int = 200,
) -> SFHResult:
    """Draw a full stochastic SFH realization: quantile times, GP length
    scale, and total stellar mass come from `sampler` (a `PriorSampler` by
    default -- swappable for an `NPESampler` later without this function
    changing, per HANDOFF3 Sec. 6.1); the continuous SFR(t) curve is
    reconstructed deterministically from those draws.

    `t_obs_gyr` is the galaxy's age at the observation epoch (e.g. the age
    of the universe at its redshift) -- not itself drawn here, matching the
    "caller supplies redshift-derived quantities explicitly" convention
    already established for `zqso` in the pre-refactor orchestrator.
    """
    if sampler is None:
        sampler = PriorSampler()
    if t_obs_gyr <= 0.0:
        raise ValueError(f"t_obs_gyr must be > 0, got {t_obs_gyr}")

    draws = sampler.sample(
        [
            "galaxy_continuum.sfh.total_stellar_mass",
            "galaxy_continuum.sfh.mass_quantile_gap_fractions",
            "galaxy_continuum.sfh.gp_length_scale_fraction",
        ],
        rng=rng,
    )
    total_mass = float(draws["galaxy_continuum.sfh.total_stellar_mass"])
    gap_fractions = np.asarray(draws["galaxy_continuum.sfh.mass_quantile_gap_fractions"])
    length_scale_gyr = float(draws["galaxy_continuum.sfh.gp_length_scale_fraction"]) * t_obs_gyr

    quantile_times_gyr = np.cumsum(gap_fractions * t_obs_gyr)[:-1]

    t_grid_gyr = np.linspace(0.0, t_obs_gyr, n_grid)
    f_grid = reconstruct_cumulative_sfh(t_grid_gyr, quantile_times_gyr, t_obs_gyr, length_scale_gyr)

    sfr_grid = total_mass * np.gradient(f_grid, t_grid_gyr) / 1.0e9  # Msun/Gyr -> Msun/yr
    sfr_grid = np.clip(sfr_grid, 0.0, None)

    formed_mass = np.trapezoid(sfr_grid * 1.0e9, t_grid_gyr)  # back to Msun/Gyr for the integral
    if formed_mass > 0.0:
        sfr_grid *= total_mass / formed_mass

    return SFHResult(
        t_grid_gyr=t_grid_gyr,
        sfr_msun_per_yr=sfr_grid,
        cumulative_mass_fraction=f_grid,
        total_stellar_mass_msun=total_mass,
        quantile_times_gyr=quantile_times_gyr,
        gp_length_scale_gyr=length_scale_gyr,
    )
