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

=== Self-consistent extension (2026-09-17), per Andy's stated preference ===

`evaluate_population_self_consistent`/`build_self_consistent_targeted_samples`/
`build_self_consistent_population_target_grid` are the SAME idea, using
`metal_yield.py`'s real metallicity-dependent yield(Z) instead of the
constant-`yield`-per-mock closed-box relation above -- what Andy asked for
after `metal_yield.py` was built ("my preference is for the SFH/Z inversion
to use the more accurate metallicity-dependent IMF and yield model").

This is NOT a drop-in swap, for a real reason: `mass_weighted_z_closed_form`
above only exists BECAUSE `Z(t_obs)` and `mass_weighted_z` have closed forms
independent of SFH shape under the OLD constant-yield relation. Once yield
depends on the evolving Z(t) itself, that independence is gone -- there is
no equivalent closed form, and `sample_yield_efficiency_for_target_z`'s
whole trick (solve algebra for the exact (yield,efficiency) pair) has
nothing to solve algebraically anymore.

The fix, benchmarked for real (sandboxed) before writing any of this:
NUMERICAL bisection over `yield_scatter` (the one knob that shifts the
whole yield(Z) curve up or down) replaces algebra -- monotonic in its
effect on the final Z, so bisection converges reliably. Two real costs
this surfaced, both addressed below:

1. **The self-consistent forward evaluation is ~300x slower per call**
   than the old closed form (measured directly: ~53ms vs ~0.17ms) --
   entirely the cost of RK4-stepping through the enrichment history
   instead of one formula evaluation. Addressed two ways: (a) a coarser
   step grid, `DEFAULT_N_GRID_SELF_CONSISTENT=30` instead of 200 -- checked
   directly against n_grid=200 "ground truth" and found <5% deviation,
   negligible next to the 35-70% astrophysical scatter `metal_yield.py`
   already models; (b) `metal_yield.integrate_self_consistent_z_batch`
   (vectorized across many points' bisection trials at once, instead of
   one Python-level call per point) -- combined, roughly 100x faster than
   the naive per-point approach, confirmed by direct benchmark.
2. **Blind rejection sampling for AGE (accept within a fixed window of
   the target) is biased** -- if the underlying shape prior isn't locally
   symmetric around the target age, the accepted sample inherits that
   asymmetry. Fixed by switching to "draw a large, cheap pool of candidate
   shapes (age alone is cheap to check -- no ODE needed) and keep whichever
   are CLOSEST to the target," rather than "accept the first ones inside a
   fixed radius." For common ages this converges with a modest pool; for
   genuinely rare ages (e.g. a 12 Gyr halo-like target, where old shapes
   are intrinsically uncommon under the current symmetric Dirichlet(2,2,2,2)
   prior), a larger pool is needed to close the bias fully -- checked
   directly (pool size 4,000 -> 60,000 shrank the halo bias from -0.47 Gyr
   to -0.01 Gyr) rather than assumed adequate at a fixed size.
3. Reachability is checked BEFORE nearest-in-age selection (not after) --
   an earlier sandboxed pass picked nearest-in-age first and only then
   checked which of those happened to also reach the target Z, which
   under-fills the requested batch size unpredictably; filtering for
   reachability first guarantees the requested count whenever the pool is
   large enough to contain that many reachable candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .metal_yield import integrate_self_consistent_z_batch
from .metallicity import evaluate_closed_box
from .sfh import QUANTILE_FRACTIONS, reconstruct_cumulative_sfh
from ..parameters.registry import get_parameter
from ..parameters.samplers import ParameterSampler, PriorSampler

_GAP_FRACTIONS_NAME = "galaxy_continuum.sfh.mass_quantile_gap_fractions"
_LENGTH_SCALE_NAME = "galaxy_continuum.sfh.gp_length_scale_fraction"
_YIELD_NAME = "galaxy_continuum.metallicity.yield"
_EFFICIENCY_NAME = "galaxy_continuum.metallicity.star_formation_efficiency"
_YIELD_SCATTER_NAME = "galaxy_continuum.metallicity.yield_scatter"

DEFAULT_T_OBS_GYR = 13.8
DEFAULT_N_GRID_SELF_CONSISTENT = 30  # vs. 200 for the closed-box path -- checked directly
# (see module docstring): <5% deviation from n_grid=200, negligible next to the 35-70%
# astrophysical scatter metal_yield.py already models. ~6x fewer RK4 steps per evaluation.


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


def _reconstruct_age_f_and_mu_grid(
    mass_quantile_gap_fractions: np.ndarray,
    gp_length_scale_fraction: float,
    star_formation_efficiency: float,
    *,
    t_obs_gyr: float = DEFAULT_T_OBS_GYR,
    n_grid: int = DEFAULT_N_GRID_SELF_CONSISTENT,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Shared by `evaluate_population_self_consistent` and the targeted-
    batch builder below: age, F(t), and mu(t) depend only on SFH shape +
    efficiency, never on yield_scatter -- computing them once and reusing
    across many yield_scatter trials (the whole point of the bisection
    search) avoids redundant work. Deliberately cheap: no ODE integration
    happens in this function. Returns `f_grid` (needed afterward for the
    mass-weighted Z integral, `trapz(z_grid, f_grid)`) alongside `mu_grid`
    (needed to DRIVE that integral) -- returning only one and re-deriving
    the other would be more error-prone than just returning both.
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
    age_gyr = float(np.trapezoid(f_grid, t_grid_gyr))
    mu_grid = 1.0 - star_formation_efficiency * f_grid
    return age_gyr, f_grid, mu_grid


def evaluate_population_self_consistent(
    mass_quantile_gap_fractions: np.ndarray,
    gp_length_scale_fraction: float,
    star_formation_efficiency: float,
    yield_scatter: float,
    *,
    t_obs_gyr: float = DEFAULT_T_OBS_GYR,
    n_grid: int = DEFAULT_N_GRID_SELF_CONSISTENT,
) -> PopulationSummary:
    """Self-consistent sibling of `evaluate_population`: uses
    `metal_yield.py`'s real metallicity-dependent yield(Z) (via the RK4
    ODE) instead of the constant-`yield`-per-mock closed-box relation --
    `yield_scatter` replaces `yield_` as the free per-mock knob (see module
    docstring for why there is no drop-in closed form once yield depends
    on the evolving Z(t) itself). `mass_weighted_z` is
    `trapz(z_grid, f_grid)`, the SAME mass-weighted definition
    `evaluate_population` uses -- NOT `Z(t_obs)` (the endpoint), which is a
    genuinely different, and here wrong, quantity (see module docstring).
    """
    age_gyr, f_grid, mu_grid = _reconstruct_age_f_and_mu_grid(
        mass_quantile_gap_fractions, gp_length_scale_fraction, star_formation_efficiency,
        t_obs_gyr=t_obs_gyr, n_grid=n_grid,
    )
    z_grid = integrate_self_consistent_z_batch(mu_grid[np.newaxis, :], np.array([yield_scatter]))[0]
    mass_weighted_z = float(np.trapezoid(z_grid, f_grid))
    return PopulationSummary(mass_weighted_age_gyr=age_gyr, mass_weighted_z=mass_weighted_z)


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


def build_self_consistent_targeted_samples(
    target_age_gyr: float,
    target_z: float,
    n_points: int,
    rng: np.random.Generator,
    *,
    sampler: Optional[ParameterSampler] = None,
    t_obs_gyr: float = DEFAULT_T_OBS_GYR,
    n_grid: int = DEFAULT_N_GRID_SELF_CONSISTENT,
    n_candidate_shapes: int = 4_000,
    n_bisect_iter: int = 25,
    yield_scatter_bounds: tuple[float, float] = (-8.0, 8.0),
    max_candidate_shapes: int = 500_000,
) -> dict[str, np.ndarray]:
    """The self-consistent analogue of `build_targeted_samples` -- built
    via numerical bisection over `yield_scatter` instead of algebra (no
    closed form exists once yield depends on the evolving Z(t) itself; see
    module docstring), and TARGETING AGE TOO (not just Z, an improvement
    over the old function): draws a large, cheap pool of candidate SFH
    shapes (checking a shape's AGE needs no ODE integration at all -- only
    Z-matching does), keeps the `n_points` CLOSEST in age among those that
    can ALSO reach `target_z` (reachability checked FIRST, then
    nearest-in-age -- checking the other order first under-fills the
    requested count unpredictably, found and fixed 2026-09-17), then
    batch-bisects `yield_scatter` across exactly those `n_points` at once.

    The candidate pool grows automatically (doubling) if too few reachable
    points are found -- genuinely rare target ages (e.g. an old, halo-like
    12 Gyr target under the current symmetric shape prior) need a much
    bigger pool than common ones to close the age gap fully; checked
    directly, not assumed adequate at a fixed size (2026-09-16 sandbox:
    4,000 candidates left the halo case biased by -0.47 Gyr, 60,000 closed
    it to -0.01 Gyr). Raises if `max_candidate_shapes` is exhausted first.
    """
    if sampler is None:
        sampler = PriorSampler()

    n_candidates = n_candidate_shapes
    while True:
        draws = sampler.sample(
            [_GAP_FRACTIONS_NAME, _LENGTH_SCALE_NAME, _EFFICIENCY_NAME], rng=rng, size=n_candidates,
        )
        gap_all = np.asarray(draws[_GAP_FRACTIONS_NAME])
        ls_all = np.asarray(draws[_LENGTH_SCALE_NAME])
        eff_all = np.asarray(draws[_EFFICIENCY_NAME])

        ages = np.empty(n_candidates)
        f_grids = np.empty((n_candidates, n_grid))
        mu_grids = np.empty((n_candidates, n_grid))
        for i in range(n_candidates):
            ages[i], f_grids[i], mu_grids[i] = _reconstruct_age_f_and_mu_grid(
                gap_all[i], ls_all[i], eff_all[i], t_obs_gyr=t_obs_gyr, n_grid=n_grid,
            )

        lo_bound, hi_bound = yield_scatter_bounds
        z_lo = np.trapezoid(
            integrate_self_consistent_z_batch(mu_grids, np.full(n_candidates, lo_bound)), f_grids, axis=1
        )
        z_hi = np.trapezoid(
            integrate_self_consistent_z_batch(mu_grids, np.full(n_candidates, hi_bound)), f_grids, axis=1
        )
        reachable = (z_lo <= target_z) & (target_z <= z_hi)

        if reachable.sum() >= n_points or n_candidates >= max_candidate_shapes:
            break
        n_candidates = min(n_candidates * 4, max_candidate_shapes)

    if reachable.sum() < n_points:
        raise ValueError(
            f"Only found {int(reachable.sum())} reachable candidates for target_z={target_z:.4e} out of "
            f"{n_candidates} tried (max_candidate_shapes={max_candidate_shapes}) -- target_z may be outside "
            f"this model's achievable range."
        )

    reachable_idx = np.flatnonzero(reachable)
    nearest = reachable_idx[np.argsort(np.abs(ages[reachable_idx] - target_age_gyr))[:n_points]]

    gap_batch = gap_all[nearest]
    ls_batch = ls_all[nearest]
    eff_batch = eff_all[nearest]
    mu_batch = mu_grids[nearest]
    f_batch = f_grids[nearest]

    lo = np.full(n_points, lo_bound)
    hi = np.full(n_points, hi_bound)
    for _ in range(n_bisect_iter):
        mid = 0.5 * (lo + hi)
        z_mid = np.trapezoid(integrate_self_consistent_z_batch(mu_batch, mid), f_batch, axis=1)
        go_up = z_mid < target_z
        lo = np.where(go_up, mid, lo)
        hi = np.where(go_up, hi, mid)
    yield_scatter_batch = 0.5 * (lo + hi)
    z_final = np.trapezoid(integrate_self_consistent_z_batch(mu_batch, yield_scatter_batch), f_batch, axis=1)

    return dict(
        gap_fractions=gap_batch,
        gp_length_scale_fraction=ls_batch,
        star_formation_efficiency=eff_batch,
        yield_scatter=yield_scatter_batch,
        mass_weighted_age_gyr=ages[nearest],
        mass_weighted_z=z_final,
    )


@dataclass(frozen=True)
class SelfConsistentPopulationTargetGrid:
    """Self-consistent analogue of `PopulationTargetGrid` -- `yield_scatter`
    replaces `yield_` as the free per-mock knob (see module docstring)."""

    gap_fractions: np.ndarray
    gp_length_scale_fraction: np.ndarray
    star_formation_efficiency: np.ndarray
    yield_scatter: np.ndarray
    mass_weighted_age_gyr: np.ndarray
    mass_weighted_z: np.ndarray
    t_obs_gyr: float

    def __len__(self) -> int:
        return self.mass_weighted_age_gyr.shape[0]

    def nearest(self, target_age_gyr: float, target_z: float, k: int = 50) -> np.ndarray:
        """Same normalized-distance k-NN as `PopulationTargetGrid.nearest`
        -- still useful for querying the blind grid+MC portion of a
        combined grid, even though the targeted portion (built via
        `build_self_consistent_targeted_samples`) doesn't need it."""
        age_scale = float(np.std(self.mass_weighted_age_gyr)) or 1.0
        z_scale = float(np.std(self.mass_weighted_z)) or 1.0
        d2 = (
            ((self.mass_weighted_age_gyr - target_age_gyr) / age_scale) ** 2
            + ((self.mass_weighted_z - target_z) / z_scale) ** 2
        )
        k = min(k, d2.shape[0])
        return np.argpartition(d2, k - 1)[:k]

    def concatenate(self, extra: dict[str, np.ndarray]) -> "SelfConsistentPopulationTargetGrid":
        return SelfConsistentPopulationTargetGrid(
            gap_fractions=np.concatenate([self.gap_fractions, extra["gap_fractions"]]),
            gp_length_scale_fraction=np.concatenate(
                [self.gp_length_scale_fraction, extra["gp_length_scale_fraction"]]
            ),
            star_formation_efficiency=np.concatenate(
                [self.star_formation_efficiency, extra["star_formation_efficiency"]]
            ),
            yield_scatter=np.concatenate([self.yield_scatter, extra["yield_scatter"]]),
            mass_weighted_age_gyr=np.concatenate([self.mass_weighted_age_gyr, extra["mass_weighted_age_gyr"]]),
            mass_weighted_z=np.concatenate([self.mass_weighted_z, extra["mass_weighted_z"]]),
            t_obs_gyr=self.t_obs_gyr,
        )


def build_self_consistent_population_target_grid(
    *,
    rng: np.random.Generator,
    t_obs_gyr: float = DEFAULT_T_OBS_GYR,
    n_grid: int = DEFAULT_N_GRID_SELF_CONSISTENT,
    n_grid_points_per_axis: int = 4,
    n_monte_carlo: int = 500,
    targets: Optional[Sequence[tuple[float, float]]] = None,
    n_points_per_target: int = 30,
    sampler: Optional[ParameterSampler] = None,
) -> SelfConsistentPopulationTargetGrid:
    """Self-consistent analogue of `build_population_target_grid`: a modest
    blind dense-grid + Monte-Carlo cloud (smaller defaults than the old
    closed-box version's -- each point now costs real ODE integration, not
    free algebra, so blind coverage is for general/boundary sanity, not
    for carrying sparse regions -- that's `targets`' job) UNIONED with
    targeted batches (`build_self_consistent_targeted_samples`) for each
    `(target_age_gyr, target_z)` pair in `targets`.
    """
    if sampler is None:
        sampler = PriorSampler()

    length_scale_low, length_scale_high = get_parameter(_LENGTH_SCALE_NAME).distribution.support
    efficiency_low, efficiency_high = get_parameter(_EFFICIENCY_NAME).distribution.support

    v_axis = np.linspace(0.0, 1.0, n_grid_points_per_axis)
    length_scale_axis = np.linspace(length_scale_low, length_scale_high, n_grid_points_per_axis)
    efficiency_axis = np.linspace(efficiency_low, efficiency_high, n_grid_points_per_axis)

    grid_gap, grid_ls, grid_eff = [], [], []
    for v0 in v_axis:
        for v1 in v_axis:
            for v2 in v_axis:
                gap = stick_breaking_gap_fractions(np.array([v0, v1, v2]))
                for ls in length_scale_axis:
                    for eff in efficiency_axis:
                        grid_gap.append(gap)
                        grid_ls.append(ls)
                        grid_eff.append(eff)

    mc_draws = sampler.sample(
        [_GAP_FRACTIONS_NAME, _LENGTH_SCALE_NAME, _EFFICIENCY_NAME], rng=rng, size=n_monte_carlo,
    )
    all_gap = np.concatenate([np.asarray(grid_gap), np.asarray(mc_draws[_GAP_FRACTIONS_NAME])])
    all_ls = np.concatenate([np.asarray(grid_ls), np.asarray(mc_draws[_LENGTH_SCALE_NAME])])
    all_eff = np.concatenate([np.asarray(grid_eff), np.asarray(mc_draws[_EFFICIENCY_NAME])])
    n_blind = all_gap.shape[0]

    blind_ages = np.empty(n_blind)
    blind_f = np.empty((n_blind, n_grid))
    blind_mu = np.empty((n_blind, n_grid))
    for i in range(n_blind):
        blind_ages[i], blind_f[i], blind_mu[i] = _reconstruct_age_f_and_mu_grid(
            all_gap[i], all_ls[i], all_eff[i], t_obs_gyr=t_obs_gyr, n_grid=n_grid,
        )
    blind_yield_scatter = sampler.sample([_YIELD_SCATTER_NAME], rng=rng, size=n_blind)[_YIELD_SCATTER_NAME]
    blind_z_grid = integrate_self_consistent_z_batch(blind_mu, np.asarray(blind_yield_scatter))
    blind_z = np.trapezoid(blind_z_grid, blind_f, axis=1)

    result = SelfConsistentPopulationTargetGrid(
        gap_fractions=all_gap,
        gp_length_scale_fraction=all_ls,
        star_formation_efficiency=all_eff,
        yield_scatter=np.asarray(blind_yield_scatter),
        mass_weighted_age_gyr=blind_ages,
        mass_weighted_z=blind_z,
        t_obs_gyr=t_obs_gyr,
    )

    for target_age_gyr, target_z in targets or []:
        extra = build_self_consistent_targeted_samples(
            target_age_gyr, target_z, n_points_per_target, rng,
            sampler=sampler, t_obs_gyr=t_obs_gyr, n_grid=n_grid,
        )
        result = result.concatenate(extra)

    return result


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
