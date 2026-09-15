"""
Inverting galaxy_continuum's own SFH+metallicity generating parameters
(`galaxy_continuum.sfh.mass_quantile_gap_fractions`,
`galaxy_continuum.sfh.gp_length_scale_fraction`,
`galaxy_continuum.metallicity.yield`,
`galaxy_continuum.metallicity.star_formation_efficiency`) against real
literature (age, metallicity) targets for disc/halo/elliptical zones (see
project memory: `project-dust-emission-planning-handoff`'s disc/halo/
elliptical literature pull) -- needed because none of those real observed
[Fe/H]/[Z/H]/age numbers map directly onto this project's `yield`/
`star_formation_efficiency` closed-box parameterization; this module finds
which region of the EXISTING generating-parameter space actually reproduces
a given target, rather than hand-deriving a new formula.

Andy's confirmed design (2026-09-15): a hybrid dense-grid + Monte-Carlo
approach -- a moderate REGULAR grid over the generating-parameter space
(explicit boundary coverage, including corners the registered priors rarely
visit) UNIONED with Monte Carlo draws from the CURRENT registered priors
(resolution where probability mass actually concentrates). The combined
point cloud is queried with a density-aware nearest-neighbor lookup, not
strict grid interpolation (the union isn't a regular lattice).

Dimensionality (confirmed 2026-09-15, not assumed): 6 free generating
dimensions feed the (age, Z) summary below -- the SFH quantile
gap-fraction Dirichlet(4 components) has 3 free coordinates (parameterized
here via stick-breaking over an ordinary [0,1]^3 cube, which grids far more
simply than a raw 4-simplex), plus `gp_length_scale_fraction`, `yield`,
`star_formation_efficiency`. `total_stellar_mass` is deliberately EXCLUDED
-- it cancels out of both summary statistics below by construction (mass
FRACTION, not absolute mass, is all either one depends on; verified in
tests, not just asserted).

Two closed-form identities used below, both verified against trivial
analytic cases in tests, not just derived on paper:

- Mass-weighted age: age = integral_0^t_obs F(t) dt (via integration by
  parts of the usual "age = integral (t_obs - t) dF(t)" definition,
  using F(t_obs)=1 and F(0)=0) -- avoids ever differentiating F(t) for
  SFR(t), which `sfh.draw_sfh` needs but this module does not.
- Mass-weighted metallicity: Z_bar = integral Z(t) dF(t), computed directly
  as `np.trapezoid(z_grid, f_grid)` (integrating Z as a function of F,
  i.e. literally weighting by mass fraction formed) -- the metallicity of
  the stars actually present, not `Z(t_obs)` (which is only the
  metallicity of the LAST gas enriched, and -- a real closed-box property,
  not a limitation of this module -- depends only on `yield`/`efficiency`
  regardless of SFH shape at all, since `F(t_obs)=1` always).

`t_obs_gyr` is fixed for the whole grid (default 13.8 Gyr, a representative
z~0 age of the Universe) since the literature targets this module targets
(Bensby/An/Xue/Gallazzi/Thomas, [[project-dust-emission-planning-handoff]])
are themselves local-Universe (z~0) population properties -- a deliberate
scope limit for this first pass, not a hidden assumption; a future need to
target non-local zones would need `t_obs_gyr` as its own grid axis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .metallicity import evaluate_closed_box
from .sfh import QUANTILE_FRACTIONS, reconstruct_cumulative_sfh
from ..parameters.registry import get_parameter
from ..parameters.samplers import ParameterSampler, PriorSampler

_GAP_FRACTIONS_NAME = "galaxy_continuum.sfh.mass_quantile_gap_fractions"
_LENGTH_SCALE_NAME = "galaxy_continuum.sfh.gp_length_scale_fraction"
_YIELD_NAME = "galaxy_continuum.metallicity.yield"
_EFFICIENCY_NAME = "galaxy_continuum.metallicity.star_formation_efficiency"

DEFAULT_T_OBS_GYR = 13.8


@dataclass(frozen=True)
class PopulationSummary:
    mass_weighted_age_gyr: float
    mass_weighted_z: float


def stick_breaking_gap_fractions(v: np.ndarray) -> np.ndarray:
    """Maps 3 independent coordinates in [0,1] to a 4-component vector
    summing to 1 -- a simplex parameterization chosen specifically because
    an ordinary cube [0,1]^3 is trivial to lay a regular grid over
    (including its corners), unlike the raw 4-simplex
    `mass_quantile_gap_fractions` is actually drawn from
    (`Dirichlet((2,2,2,2))`). Used only for building this module's grid
    axis, not as a replacement for the registered Dirichlet prior itself
    (Monte Carlo draws below use the real Dirichlet directly).
    """
    v = np.asarray(v, dtype=float)
    if v.shape != (3,):
        raise ValueError(f"stick_breaking_gap_fractions expects 3 coordinates, got shape {v.shape}")
    g1 = v[0]
    g2 = (1.0 - v[0]) * v[1]
    g3 = (1.0 - v[0]) * (1.0 - v[1]) * v[2]
    g4 = (1.0 - v[0]) * (1.0 - v[1]) * (1.0 - v[2])
    return np.array([g1, g2, g3, g4])


def evaluate_population(
    mass_quantile_gap_fractions: np.ndarray,
    gp_length_scale_fraction: float,
    yield_: float,
    star_formation_efficiency: float,
    *,
    t_obs_gyr: float = DEFAULT_T_OBS_GYR,
    n_grid: int = 200,
) -> PopulationSummary:
    """Pure Tier-1 forward map (no registry/sampler involvement) -- the
    deterministic (age, Z) a given set of generating parameters produces.
    `mass_quantile_gap_fractions` must be a length-4 array summing to 1
    (the same convention `sfh.draw_sfh` consumes internally, exposed here
    for direct grid-node evaluation rather than a random draw).
    """
    gap_fractions = np.asarray(mass_quantile_gap_fractions, dtype=float)
    if gap_fractions.shape != (len(QUANTILE_FRACTIONS) + 1,):
        raise ValueError(
            f"mass_quantile_gap_fractions must have {len(QUANTILE_FRACTIONS) + 1} components, "
            f"got shape {gap_fractions.shape}"
        )

    length_scale_gyr = gp_length_scale_fraction * t_obs_gyr
    quantile_times_gyr = np.cumsum(gap_fractions * t_obs_gyr)[:-1]

    t_grid_gyr = np.linspace(0.0, t_obs_gyr, n_grid)
    f_grid = reconstruct_cumulative_sfh(t_grid_gyr, quantile_times_gyr, t_obs_gyr, length_scale_gyr)
    z_grid = evaluate_closed_box(f_grid, yield_, star_formation_efficiency)

    mass_weighted_age_gyr = float(np.trapezoid(f_grid, t_grid_gyr))
    mass_weighted_z = float(np.trapezoid(z_grid, f_grid))

    return PopulationSummary(mass_weighted_age_gyr=mass_weighted_age_gyr, mass_weighted_z=mass_weighted_z)


@dataclass(frozen=True)
class PopulationTargetGrid:
    """The combined dense-grid + Monte-Carlo point cloud: each row is one
    (gap_fractions, gp_length_scale_fraction, yield, efficiency) generating-
    parameter combination, alongside its resulting (age, Z) summary."""

    gap_fractions: np.ndarray  # shape (n, 4)
    gp_length_scale_fraction: np.ndarray  # shape (n,)
    yield_: np.ndarray  # shape (n,)
    star_formation_efficiency: np.ndarray  # shape (n,)
    mass_weighted_age_gyr: np.ndarray  # shape (n,)
    mass_weighted_z: np.ndarray  # shape (n,)
    t_obs_gyr: float

    def __len__(self) -> int:
        return self.mass_weighted_age_gyr.shape[0]

    def nearest(self, target_age_gyr: float, target_z: float, k: int = 50) -> np.ndarray:
        """Indices of the `k` grid/MC points closest to (target_age_gyr,
        target_z) in NORMALIZED (age, Z) space -- each axis scaled by this
        grid's own observed spread, so the two summary statistics (very
        different natural units/ranges) contribute comparably to distance,
        rather than one dominating purely from unit choice."""
        age_scale = float(np.std(self.mass_weighted_age_gyr)) or 1.0
        z_scale = float(np.std(self.mass_weighted_z)) or 1.0
        d2 = (
            ((self.mass_weighted_age_gyr - target_age_gyr) / age_scale) ** 2
            + ((self.mass_weighted_z - target_z) / z_scale) ** 2
        )
        k = min(k, d2.shape[0])
        return np.argpartition(d2, k - 1)[:k]


def build_population_target_grid(
    *,
    rng: np.random.Generator,
    t_obs_gyr: float = DEFAULT_T_OBS_GYR,
    n_grid_points_per_axis: int = 6,
    n_monte_carlo: int = 2000,
    sampler: Optional[ParameterSampler] = None,
) -> PopulationTargetGrid:
    """Builds the hybrid grid described in this module's docstring: a
    regular grid over (3 stick-breaking coordinates, gp_length_scale_fraction,
    yield, star_formation_efficiency) -- `n_grid_points_per_axis` values per
    axis, INCLUDING each registered parameter's own min/max boundary --
    unioned with `n_monte_carlo` draws from the real registered priors.
    """
    if sampler is None:
        sampler = PriorSampler()

    length_scale_low, length_scale_high = get_parameter(_LENGTH_SCALE_NAME).distribution.support
    yield_low, yield_high = get_parameter(_YIELD_NAME).distribution.support
    efficiency_low, efficiency_high = get_parameter(_EFFICIENCY_NAME).distribution.support

    # ---- regular grid, explicit boundary inclusion on every axis ----
    v_axis = np.linspace(0.0, 1.0, n_grid_points_per_axis)
    length_scale_axis = np.linspace(length_scale_low, length_scale_high, n_grid_points_per_axis)
    yield_axis = np.geomspace(yield_low, yield_high, n_grid_points_per_axis)  # LogUniform prior -> log-spaced grid
    efficiency_axis = np.linspace(efficiency_low, efficiency_high, n_grid_points_per_axis)

    grid_gap_fractions, grid_length_scale, grid_yield, grid_efficiency = [], [], [], []
    grid_age, grid_z = [], []
    for v0 in v_axis:
        for v1 in v_axis:
            for v2 in v_axis:
                gap_fractions = stick_breaking_gap_fractions(np.array([v0, v1, v2]))
                for length_scale_fraction in length_scale_axis:
                    for yield_ in yield_axis:
                        for efficiency in efficiency_axis:
                            summary = evaluate_population(
                                gap_fractions, length_scale_fraction, yield_, efficiency, t_obs_gyr=t_obs_gyr
                            )
                            grid_gap_fractions.append(gap_fractions)
                            grid_length_scale.append(length_scale_fraction)
                            grid_yield.append(yield_)
                            grid_efficiency.append(efficiency)
                            grid_age.append(summary.mass_weighted_age_gyr)
                            grid_z.append(summary.mass_weighted_z)

    # ---- Monte Carlo draws from the real registered priors ----
    mc_draws = sampler.sample(
        [_GAP_FRACTIONS_NAME, _LENGTH_SCALE_NAME, _YIELD_NAME, _EFFICIENCY_NAME],
        rng=rng,
        size=n_monte_carlo,
    )
    mc_gap_fractions = np.asarray(mc_draws[_GAP_FRACTIONS_NAME])
    mc_length_scale = np.asarray(mc_draws[_LENGTH_SCALE_NAME])
    mc_yield = np.asarray(mc_draws[_YIELD_NAME])
    mc_efficiency = np.asarray(mc_draws[_EFFICIENCY_NAME])
    for i in range(n_monte_carlo):
        summary = evaluate_population(
            mc_gap_fractions[i], float(mc_length_scale[i]), float(mc_yield[i]), float(mc_efficiency[i]),
            t_obs_gyr=t_obs_gyr,
        )
        grid_gap_fractions.append(mc_gap_fractions[i])
        grid_length_scale.append(mc_length_scale[i])
        grid_yield.append(mc_yield[i])
        grid_efficiency.append(mc_efficiency[i])
        grid_age.append(summary.mass_weighted_age_gyr)
        grid_z.append(summary.mass_weighted_z)

    return PopulationTargetGrid(
        gap_fractions=np.asarray(grid_gap_fractions),
        gp_length_scale_fraction=np.asarray(grid_length_scale),
        yield_=np.asarray(grid_yield),
        star_formation_efficiency=np.asarray(grid_efficiency),
        mass_weighted_age_gyr=np.asarray(grid_age),
        mass_weighted_z=np.asarray(grid_z),
        t_obs_gyr=t_obs_gyr,
    )
