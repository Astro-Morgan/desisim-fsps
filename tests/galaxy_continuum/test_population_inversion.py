import numpy as np
import pytest

from demiurge.galaxy_continuum.metallicity import evaluate_closed_box
from demiurge.galaxy_continuum.population_inversion import (
    PopulationTargetGrid,
    build_population_target_grid,
    build_targeted_samples,
    evaluate_population,
    mass_weighted_z_closed_form,
    sample_yield_efficiency_for_target_z,
    stick_breaking_gap_fractions,
)
from demiurge.galaxy_continuum.sfh import reconstruct_cumulative_sfh

T_OBS = 12.0


def _gap_fractions(early=False, late=False):
    """A large FIRST gap fraction means a long time elapses before the
    first mass quantile is reached -- i.e. delayed/LATE star formation (a
    YOUNG mass-weighted population). A large LAST gap fraction means most
    mass already formed early and little happens afterward -- an OLD
    population. (Named for the resulting population age, not the raw
    gap-fraction pattern -- easy to get backwards, as an earlier version
    of this fixture did, caught by the monotonicity test below.)"""
    if early:
        return np.array([0.05, 0.05, 0.05, 0.85])
    if late:
        return np.array([0.85, 0.05, 0.05, 0.05])
    return np.array([0.25, 0.25, 0.25, 0.25])


# ---- stick_breaking_gap_fractions ----


def test_stick_breaking_sums_to_one():
    for v in ([0.1, 0.5, 0.9], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.3, 0.7, 0.2]):
        g = stick_breaking_gap_fractions(np.array(v))
        assert g.sum() == pytest.approx(1.0)


def test_stick_breaking_corner_cases():
    np.testing.assert_allclose(stick_breaking_gap_fractions(np.array([0.0, 0.0, 0.0])), [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(stick_breaking_gap_fractions(np.array([1.0, 1.0, 1.0])), [1.0, 0.0, 0.0, 0.0])


def test_stick_breaking_rejects_wrong_shape():
    with pytest.raises(ValueError):
        stick_breaking_gap_fractions(np.array([0.1, 0.2]))


# ---- evaluate_population: internal-consistency (identity) checks ----


def test_age_matches_trapz_of_the_actual_reconstructed_f():
    gap_fractions = _gap_fractions()
    length_scale_fraction = 0.2
    quantile_times = np.cumsum(gap_fractions * T_OBS)[:-1]
    t_grid = np.linspace(0.0, T_OBS, 200)
    f_grid = reconstruct_cumulative_sfh(t_grid, quantile_times, T_OBS, length_scale_fraction * T_OBS)
    expected_age = float(np.trapezoid(f_grid, t_grid))

    summary = evaluate_population(gap_fractions, length_scale_fraction, yield_=0.01, star_formation_efficiency=0.5, t_obs_gyr=T_OBS)
    assert summary.mass_weighted_age_gyr == pytest.approx(expected_age, rel=1e-9)


def test_z_matches_trapz_of_z_over_f():
    gap_fractions = _gap_fractions()
    length_scale_fraction = 0.2
    yield_, efficiency = 0.02, 0.6
    quantile_times = np.cumsum(gap_fractions * T_OBS)[:-1]
    t_grid = np.linspace(0.0, T_OBS, 200)
    f_grid = reconstruct_cumulative_sfh(t_grid, quantile_times, T_OBS, length_scale_fraction * T_OBS)
    z_grid = evaluate_closed_box(f_grid, yield_, efficiency)
    expected_z = float(np.trapezoid(z_grid, f_grid))

    summary = evaluate_population(gap_fractions, length_scale_fraction, yield_, efficiency, t_obs_gyr=T_OBS)
    assert summary.mass_weighted_z == pytest.approx(expected_z, rel=1e-9)


def test_z_at_t_obs_depends_only_on_yield_and_efficiency():
    """A real closed-box property this module's docstring relies on: since
    F(t_obs)=1 always, Z(t_obs) has a closed form independent of SFH
    shape -- confirmed directly via evaluate_closed_box, the same function
    evaluate_population uses internally."""
    yield_, efficiency = 0.015, 0.4
    expected = yield_ * np.log(1.0 / (1.0 - efficiency))
    z_at_f_equals_one = evaluate_closed_box(np.array([1.0]), yield_, efficiency)[0]
    assert z_at_f_equals_one == pytest.approx(expected)


def test_evaluate_population_rejects_wrong_gap_fraction_shape():
    with pytest.raises(ValueError):
        evaluate_population(np.array([0.5, 0.5]), 0.2, 0.01, 0.5, t_obs_gyr=T_OBS)


# ---- evaluate_population: physical sanity/monotonicity ----


def test_mass_weighted_age_bounded_by_t_obs():
    for gap_fractions in (_gap_fractions(early=True), _gap_fractions(), _gap_fractions(late=True)):
        summary = evaluate_population(gap_fractions, 0.2, 0.01, 0.5, t_obs_gyr=T_OBS)
        assert 0.0 <= summary.mass_weighted_age_gyr <= T_OBS


def test_early_sfh_gives_older_mass_weighted_age_than_late_sfh():
    early = evaluate_population(_gap_fractions(early=True), 0.2, 0.01, 0.5, t_obs_gyr=T_OBS)
    late = evaluate_population(_gap_fractions(late=True), 0.2, 0.01, 0.5, t_obs_gyr=T_OBS)
    assert early.mass_weighted_age_gyr > late.mass_weighted_age_gyr


def test_mass_weighted_z_increases_with_efficiency():
    lo = evaluate_population(_gap_fractions(), 0.2, yield_=0.01, star_formation_efficiency=0.1, t_obs_gyr=T_OBS)
    hi = evaluate_population(_gap_fractions(), 0.2, yield_=0.01, star_formation_efficiency=0.9, t_obs_gyr=T_OBS)
    assert hi.mass_weighted_z > lo.mass_weighted_z


def test_mass_weighted_z_increases_with_yield():
    lo = evaluate_population(_gap_fractions(), 0.2, yield_=0.001, star_formation_efficiency=0.5, t_obs_gyr=T_OBS)
    hi = evaluate_population(_gap_fractions(), 0.2, yield_=0.04, star_formation_efficiency=0.5, t_obs_gyr=T_OBS)
    assert hi.mass_weighted_z > lo.mass_weighted_z


# ---- build_population_target_grid / PopulationTargetGrid ----


def test_build_population_target_grid_shapes_and_gap_fractions_sum_to_one():
    grid = build_population_target_grid(
        rng=np.random.default_rng(0), t_obs_gyr=T_OBS, n_grid_points_per_axis=2, n_monte_carlo=20
    )
    expected_len = 2 ** 6 + 20
    assert len(grid) == expected_len
    assert grid.gap_fractions.shape == (expected_len, 4)
    np.testing.assert_allclose(grid.gap_fractions.sum(axis=1), 1.0, atol=1e-9)
    assert grid.mass_weighted_age_gyr.shape == (expected_len,)
    assert np.all(np.isfinite(grid.mass_weighted_age_gyr))
    assert np.all(np.isfinite(grid.mass_weighted_z))


def test_build_population_target_grid_includes_registered_boundaries():
    from demiurge.parameters.registry import get_parameter

    grid = build_population_target_grid(
        rng=np.random.default_rng(1), t_obs_gyr=T_OBS, n_grid_points_per_axis=3, n_monte_carlo=10
    )
    yield_low, yield_high = get_parameter("galaxy_continuum.metallicity.yield").distribution.support
    efficiency_low, efficiency_high = get_parameter(
        "galaxy_continuum.metallicity.star_formation_efficiency"
    ).distribution.support

    assert grid.yield_.min() == pytest.approx(yield_low)
    assert grid.yield_.max() == pytest.approx(yield_high)
    assert grid.star_formation_efficiency.min() == pytest.approx(efficiency_low)
    assert grid.star_formation_efficiency.max() == pytest.approx(efficiency_high)


def test_nearest_returns_the_actually_closest_points():
    grid = PopulationTargetGrid(
        gap_fractions=np.tile(np.array([0.25, 0.25, 0.25, 0.25]), (5, 1)),
        gp_length_scale_fraction=np.full(5, 0.2),
        yield_=np.full(5, 0.01),
        star_formation_efficiency=np.full(5, 0.5),
        mass_weighted_age_gyr=np.array([0.0, 1.0, 5.0, 9.0, 10.0]),
        mass_weighted_z=np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
        t_obs_gyr=T_OBS,
    )
    idx = grid.nearest(target_age_gyr=9.5, target_z=0.35, k=2)
    assert set(idx.tolist()) == {3, 4}


def test_nearest_k_larger_than_grid_returns_everything():
    grid = PopulationTargetGrid(
        gap_fractions=np.tile(np.array([0.25, 0.25, 0.25, 0.25]), (3, 1)),
        gp_length_scale_fraction=np.full(3, 0.2),
        yield_=np.full(3, 0.01),
        star_formation_efficiency=np.full(3, 0.5),
        mass_weighted_age_gyr=np.array([1.0, 2.0, 3.0]),
        mass_weighted_z=np.array([0.1, 0.2, 0.3]),
        t_obs_gyr=T_OBS,
    )
    idx = grid.nearest(target_age_gyr=2.0, target_z=0.2, k=100)
    assert set(idx.tolist()) == {0, 1, 2}


def test_build_population_target_grid_reproducible_given_same_seed():
    a = build_population_target_grid(rng=np.random.default_rng(42), t_obs_gyr=T_OBS, n_grid_points_per_axis=2, n_monte_carlo=10)
    b = build_population_target_grid(rng=np.random.default_rng(42), t_obs_gyr=T_OBS, n_grid_points_per_axis=2, n_monte_carlo=10)
    np.testing.assert_array_equal(a.mass_weighted_age_gyr, b.mass_weighted_age_gyr)
    np.testing.assert_array_equal(a.mass_weighted_z, b.mass_weighted_z)


# ---- mass_weighted_z_closed_form ----


def test_mass_weighted_z_closed_form_matches_numeric_evaluate_population():
    """The identity this module's targeted-sampling relies on: substituting
    u=F(t) in integral Z(t) dF(t) makes mass_weighted_z a definite integral
    in u alone, hence SFH-shape-independent -- verified numerically against
    the real (shape-dependent machinery) evaluate_population, not just
    derived on paper. A real bug here (using Z(t_obs) instead) was caught
    by this exact kind of check landing off by ~2x, not within noise."""
    rng = np.random.default_rng(11)
    for _ in range(30):
        yield_ = 10.0 ** rng.uniform(-3, -1.3)
        efficiency = rng.uniform(0.01, 0.99)
        gap_fractions = rng.dirichlet([2.0, 2.0, 2.0, 2.0])
        length_scale_fraction = rng.uniform(0.05, 0.5)
        summary = evaluate_population(gap_fractions, length_scale_fraction, yield_, efficiency, t_obs_gyr=T_OBS)
        closed_form = mass_weighted_z_closed_form(yield_, efficiency)
        assert summary.mass_weighted_z == pytest.approx(closed_form, rel=1e-3)


# ---- targeted sampling for sparse Z regions ----


def test_sample_yield_efficiency_hits_target_z_exactly():
    target_z = 2.5e-4
    yields, efficiencies = sample_yield_efficiency_for_target_z(target_z, np.random.default_rng(0), n_points=30)
    implied_z = mass_weighted_z_closed_form(yields, efficiencies)
    np.testing.assert_allclose(implied_z, target_z, rtol=1e-9)


def test_sample_yield_efficiency_respects_registered_yield_support():
    from demiurge.parameters.registry import get_parameter

    yield_low, yield_high = get_parameter("galaxy_continuum.metallicity.yield").distribution.support
    yields, _ = sample_yield_efficiency_for_target_z(2.5e-4, np.random.default_rng(1), n_points=30)
    assert np.all((yields >= yield_low) & (yields <= yield_high))


def test_sample_yield_efficiency_raises_for_unachievable_target():
    with pytest.raises(ValueError):
        sample_yield_efficiency_for_target_z(1.0, np.random.default_rng(2), n_points=10, max_attempts=1000)


def test_build_targeted_samples_lands_close_to_target_z():
    target_z = 2.5e-4
    extra = build_targeted_samples(
        target_z, rng=np.random.default_rng(3), t_obs_gyr=T_OBS, n_yield_efficiency_points=10, n_shape_draws_per_point=5
    )
    n = 10 * 5
    for key in ("gap_fractions", "gp_length_scale_fraction", "yield_", "star_formation_efficiency", "mass_weighted_age_gyr", "mass_weighted_z"):
        assert key in extra
    assert extra["mass_weighted_z"].shape == (n,)
    # evaluate_population's own numeric mass_weighted_z should match the exact closed-form target tightly --
    # the only residual is GP-reconstruction discretization noise (~1e-3 level), not shape-dependence.
    np.testing.assert_allclose(extra["mass_weighted_z"], target_z, rtol=1e-2)


def test_targeted_samples_measurably_improve_density_near_a_sparse_target():
    """The actual point of this feature: supplementing a sparse-target
    query with build_targeted_samples should shrink the achieved Z error
    dramatically compared to the base grid+MC cloud alone."""
    target_age, target_z = 12.0, 2.5252e-4  # halo-like

    base = build_population_target_grid(rng=np.random.default_rng(4), t_obs_gyr=T_OBS, n_grid_points_per_axis=3, n_monte_carlo=500)
    idx_base = base.nearest(target_age, target_z, k=30)
    base_z_error = np.abs(base.mass_weighted_z[idx_base] - target_z).mean()

    extra = build_targeted_samples(
        target_z, rng=np.random.default_rng(5), t_obs_gyr=T_OBS, n_yield_efficiency_points=10, n_shape_draws_per_point=10
    )
    supplemented = base.concatenate(extra)
    idx_supp = supplemented.nearest(target_age, target_z, k=30)
    supplemented_z_error = np.abs(supplemented.mass_weighted_z[idx_supp] - target_z).mean()

    assert supplemented_z_error < base_z_error


def test_build_population_target_grid_with_target_z_values_concatenates_extra_points():
    grid_without = build_population_target_grid(
        rng=np.random.default_rng(6), t_obs_gyr=T_OBS, n_grid_points_per_axis=2, n_monte_carlo=10
    )
    grid_with = build_population_target_grid(
        rng=np.random.default_rng(6), t_obs_gyr=T_OBS, n_grid_points_per_axis=2, n_monte_carlo=10,
        target_z_values=[2.5e-4], n_yield_efficiency_points_per_target=5, n_shape_draws_per_target_point=4,
    )
    assert len(grid_with) == len(grid_without) + 5 * 4
