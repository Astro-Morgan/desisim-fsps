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

Extended 2026-09-16 with TARGETED sampling (`build_targeted_samples`,
wired into `build_population_target_grid` via `target_z_values`): verified
numerically that plain grid+Monte-Carlo induces a Z-distribution that is
genuinely sparser near low-Z (halo-like) targets than near moderate/high-Z
(disc/elliptical-like) ones (~2.6x lower density at the same tolerance,
out of 2e6 draws from the real registered priors) -- a real, structural
property of `Z = yield*ln(1/(1-efficiency))` being a MULTIPLICATIVE
function of two independently-uniform-ish factors (the level set near
Z->0 occupies a genuinely smaller corner of the (yield,efficiency)
rectangle than the level set near Z's upper reach), not an artifact of
this module's own sampling choices. `sample_yield_efficiency_for_target_z`
exploits `mass_weighted_z_closed_form` (below) to draw (yield,efficiency)
pairs landing EXACTLY on a given Z target, directly filling in whichever
region turns out sparse rather than hoping more blind random draws
eventually land nearby -- though the SFH-shape draws paired with those
points still come from the unconditional shape prior, so AGE-matching
within the targeted set is still down to luck, not similarly targeted yet
(a real, honestly-flagged limitation, not fixed in this pass).

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
  the stars actually present, NOT `Z(t_obs)` (the metallicity of only the
  LAST gas enriched -- these are genuinely different numbers, e.g. ~2x
  apart at low efficiency; conflating them was a real bug this module
  once shipped with, see `mass_weighted_z_closed_form`'s own docstring).
  Substituting u=F(t) in the integral shows Z_bar is ALSO independent of
  SFH shape, just via a different closed form than Z(t_obs)'s -- both
  facts are real closed-box properties, not limitations of this module.

`t_obs_gyr` is fixed for the whole grid (default 13.8 Gyr, a representative
z~0 age of the Universe) since the literature targets this module targets
(Bensby/An/Xue/Gallazzi/Thomas, [[project-dust-emission-planning-handoff]])
are themselves local-Universe (z~0) population properties -- a deliberate
scope limit for this first pass, not a hidden assumption; a future need to
target non-local zones would need `t_obs_gyr` as its own grid axis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

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


def mass_weighted_z_closed_form(yield_: float, star_formation_efficiency: float) -> float:
    """Exact, SFH-shape-independent closed form for `mass_weighted_z`,
    derived (and numerically verified against `evaluate_population` to
    ~1e-4 relative precision -- the residual is exactly GP-reconstruction
    discretization, not a modeling error) after a real bug this module
    shipped with once: `mass_weighted_z = integral Z(t) dF(t)` was
    (wrongly) assumed to be well-approximated by `Z(t_obs) = yield *
    ln(1/(1-efficiency))` when building targeted samples -- Z(t_obs) is
    the metallicity of the LAST gas enriched, not the mass-weighted
    average, and the two differ by a real, efficiency-dependent factor
    (e.g. ~2x at low efficiency), not just discretization noise.

    The truth: since `Z(t) = yield * ln(1/(1 - efficiency*F(t)))` is a
    function of `F(t)` ALONE (at fixed yield/efficiency), substituting
    `u = F(t)` in `integral Z(t) dF(t)` gives `integral_0^1 Z(u) du` --
    a definite integral over u, with NO reference to the original t-shape
    of F(t) left at all. Evaluated analytically:

        mass_weighted_z = yield * (1 + ((1-efficiency)/efficiency) * ln(1-efficiency))

    (`efficiency`'s registered support, `(0.01, 0.99)`, safely avoids the
    removable singularity at efficiency=0, where the bracket -> 0.)
    """
    eff = star_formation_efficiency
    return yield_ * (1.0 + ((1.0 - eff) / eff) * np.log(1.0 - eff))


def sample_yield_efficiency_for_target_z(
    target_z: float,
    rng: np.random.Generator,
    *,
    n_points: int = 50,
    sampler: Optional[ParameterSampler] = None,
    max_attempts: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Rejection-samples (yield, efficiency) pairs that land EXACTLY on the
    `target_z` contour of `mass_weighted_z_closed_form` (exact and SFH-
    shape-independent -- see that function's docstring). Draws
    `efficiency` from its REAL registered prior (not an arbitrary grid over
    the contour) and solves for the `yield` that hits `target_z` exactly
    given each draw, keeping only draws whose implied yield falls within
    yield's own registered support -- a legitimate importance/rejection
    sample conditioned on hitting the target exactly, not an ad hoc curve.

    Motivation (Andy, 2026-09-16): plain grid+Monte-Carlo sampling of the
    generating-parameter space induces a Z-distribution that is genuinely
    sparser near low-Z (halo-like) targets than near moderate/high-Z
    (disc/elliptical-like) targets -- verified numerically (only ~1.1% of
    uniform (yield,efficiency) draws land within 10% of a real halo target
    Z, vs. ~2.9% for a real elliptical target, out of 2e6 draws). This
    function supplies exactly the extra density a sparse target needs,
    directly, rather than hoping more blind random draws eventually land
    nearby.
    """
    yield_low, yield_high = get_parameter(_YIELD_NAME).distribution.support
    efficiency_dist = get_parameter(_EFFICIENCY_NAME).distribution

    yields: list = []
    efficiencies: list = []
    attempts = 0
    while len(yields) < n_points and attempts < max_attempts:
        batch = min(max(n_points * 4, 100), max_attempts - attempts)
        attempts += batch
        efficiency_candidates = efficiency_dist.draw(rng, size=batch)
        shape_factor = 1.0 + ((1.0 - efficiency_candidates) / efficiency_candidates) * np.log(1.0 - efficiency_candidates)
        yield_candidates = target_z / shape_factor
        valid = (yield_candidates >= yield_low) & (yield_candidates <= yield_high)
        yields.extend(yield_candidates[valid].tolist())
        efficiencies.extend(efficiency_candidates[valid].tolist())

    if len(yields) < n_points:
        raise ValueError(
            f"Could not find {n_points} (yield, efficiency) pairs hitting target_z={target_z:.4e} "
            f"within {max_attempts} draws -- target_z may be outside the achievable range given the "
            f"registered yield support ({yield_low:.4e}, {yield_high:.4e})."
        )
    return np.asarray(yields[:n_points]), np.asarray(efficiencies[:n_points])


def build_targeted_samples(
    target_z: float,
    *,
    rng: np.random.Generator,
    t_obs_gyr: float = DEFAULT_T_OBS_GYR,
    n_yield_efficiency_points: int = 50,
    n_shape_draws_per_point: int = 20,
    sampler: Optional[ParameterSampler] = None,
) -> dict[str, np.ndarray]:
    """For `target_z`: draws `n_yield_efficiency_points` (yield, efficiency)
    pairs exactly on its contour (`sample_yield_efficiency_for_target_z`),
    then for EACH pair draws `n_shape_draws_per_point` real SFH-shape
    combinations (gap fractions + GP length scale, from their own
    registered priors) to spread the resulting age while Z stays
    (essentially) fixed at the target. Returns a dict with the same keys
    `PopulationTargetGrid`'s fields use, ready to concatenate in.
    """
    if sampler is None:
        sampler = PriorSampler()

    yields, efficiencies = sample_yield_efficiency_for_target_z(
        target_z, rng, n_points=n_yield_efficiency_points, sampler=sampler
    )

    n_total = n_yield_efficiency_points * n_shape_draws_per_point
    shape_draws = sampler.sample([_GAP_FRACTIONS_NAME, _LENGTH_SCALE_NAME], rng=rng, size=n_total)
    gap_fractions_all = np.asarray(shape_draws[_GAP_FRACTIONS_NAME])
    length_scale_all = np.asarray(shape_draws[_LENGTH_SCALE_NAME])

    out_gap_fractions, out_length_scale, out_yield, out_efficiency = [], [], [], []
    out_age, out_z = [], []
    shape_idx = 0
    for yield_, efficiency in zip(yields, efficiencies):
        for _ in range(n_shape_draws_per_point):
            gap_fractions = gap_fractions_all[shape_idx]
            length_scale_fraction = float(length_scale_all[shape_idx])
            shape_idx += 1
            summary = evaluate_population(gap_fractions, length_scale_fraction, yield_, efficiency, t_obs_gyr=t_obs_gyr)
            out_gap_fractions.append(gap_fractions)
            out_length_scale.append(length_scale_fraction)
            out_yield.append(yield_)
            out_efficiency.append(efficiency)
            out_age.append(summary.mass_weighted_age_gyr)
            out_z.append(summary.mass_weighted_z)

    return dict(
        gap_fractions=np.asarray(out_gap_fractions),
        gp_length_scale_fraction=np.asarray(out_length_scale),
        yield_=np.asarray(out_yield),
        star_formation_efficiency=np.asarray(out_efficiency),
        mass_weighted_age_gyr=np.asarray(out_age),
        mass_weighted_z=np.asarray(out_z),
    )


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

    def concatenate(self, extra: dict[str, np.ndarray]) -> "PopulationTargetGrid":
        """Returns a NEW grid with `extra` (the dict shape
        `build_targeted_samples` returns) appended -- used to supplement
        the base dense-grid+Monte-Carlo cloud with targeted samples around
        specific sparse targets (Andy, 2026-09-16)."""
        return PopulationTargetGrid(
            gap_fractions=np.concatenate([self.gap_fractions, extra["gap_fractions"]]),
            gp_length_scale_fraction=np.concatenate(
                [self.gp_length_scale_fraction, extra["gp_length_scale_fraction"]]
            ),
            yield_=np.concatenate([self.yield_, extra["yield_"]]),
            star_formation_efficiency=np.concatenate(
                [self.star_formation_efficiency, extra["star_formation_efficiency"]]
            ),
            mass_weighted_age_gyr=np.concatenate([self.mass_weighted_age_gyr, extra["mass_weighted_age_gyr"]]),
            mass_weighted_z=np.concatenate([self.mass_weighted_z, extra["mass_weighted_z"]]),
            t_obs_gyr=self.t_obs_gyr,
        )


def build_population_target_grid(
    *,
    rng: np.random.Generator,
    t_obs_gyr: float = DEFAULT_T_OBS_GYR,
    n_grid_points_per_axis: int = 6,
    n_monte_carlo: int = 2000,
    target_z_values: Optional[Sequence[float]] = None,
    n_yield_efficiency_points_per_target: int = 50,
    n_shape_draws_per_target_point: int = 20,
    sampler: Optional[ParameterSampler] = None,
) -> PopulationTargetGrid:
    """Builds the hybrid grid described in this module's docstring: a
    regular grid over (3 stick-breaking coordinates, gp_length_scale_fraction,
    yield, star_formation_efficiency) -- `n_grid_points_per_axis` values per
    axis, INCLUDING each registered parameter's own min/max boundary --
    unioned with `n_monte_carlo` draws from the real registered priors.

    `target_z_values`, when given, supplements the grid with targeted
    samples around each listed Z value (`build_targeted_samples`) -- use
    this whenever a known target is expected to fall in a sparse region
    (verified 2026-09-16: low-Z/halo-like targets are genuinely under-
    sampled by plain grid+Monte-Carlo alone, ~2.6x sparser than a
    moderate/high-Z/elliptical-like target at the same tolerance) rather
    than relying on `n_monte_carlo`/`n_grid_points_per_axis` alone to
    eventually land nearby.
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

    result = PopulationTargetGrid(
        gap_fractions=np.asarray(grid_gap_fractions),
        gp_length_scale_fraction=np.asarray(grid_length_scale),
        yield_=np.asarray(grid_yield),
        star_formation_efficiency=np.asarray(grid_efficiency),
        mass_weighted_age_gyr=np.asarray(grid_age),
        mass_weighted_z=np.asarray(grid_z),
        t_obs_gyr=t_obs_gyr,
    )

    for target_z in target_z_values or []:
        extra = build_targeted_samples(
            target_z,
            rng=rng,
            t_obs_gyr=t_obs_gyr,
            n_yield_efficiency_points=n_yield_efficiency_points_per_target,
            n_shape_draws_per_point=n_shape_draws_per_target_point,
            sampler=sampler,
        )
        result = result.concatenate(extra)

    return result
