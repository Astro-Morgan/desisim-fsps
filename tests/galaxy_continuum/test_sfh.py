import numpy as np
import pytest

from demiurge.galaxy_continuum.sfh import QUANTILE_FRACTIONS, draw_sfh, reconstruct_cumulative_sfh


def test_draw_sfh_conserves_total_mass():
    rng = np.random.default_rng(1)
    sfh = draw_sfh(rng, t_obs_gyr=10.0)
    integrated_mass = np.trapezoid(sfh.sfr_msun_per_yr * 1.0e9, sfh.t_grid_gyr)
    assert integrated_mass == pytest.approx(sfh.total_stellar_mass_msun, rel=1e-6)


def test_draw_sfh_cumulative_fraction_boundary_conditions():
    rng = np.random.default_rng(2)
    sfh = draw_sfh(rng, t_obs_gyr=8.0)
    assert sfh.cumulative_mass_fraction[0] == pytest.approx(0.0, abs=1e-6)
    assert sfh.cumulative_mass_fraction[-1] == pytest.approx(1.0, abs=1e-6)


def test_draw_sfh_cumulative_fraction_is_monotonic():
    rng = np.random.default_rng(3)
    sfh = draw_sfh(rng, t_obs_gyr=12.0)
    assert np.all(np.diff(sfh.cumulative_mass_fraction) >= -1e-12)


def test_draw_sfh_sfr_is_nonnegative():
    rng = np.random.default_rng(4)
    sfh = draw_sfh(rng, t_obs_gyr=6.0)
    assert np.all(sfh.sfr_msun_per_yr >= 0.0)


def test_draw_sfh_quantile_times_are_ordered_and_within_bounds():
    rng = np.random.default_rng(5)
    t_obs = 9.0
    sfh = draw_sfh(rng, t_obs_gyr=t_obs)
    assert len(sfh.quantile_times_gyr) == len(QUANTILE_FRACTIONS)
    assert np.all(np.diff(sfh.quantile_times_gyr) > 0)
    assert sfh.quantile_times_gyr[0] > 0.0
    assert sfh.quantile_times_gyr[-1] < t_obs


def test_draw_sfh_reproducible_given_same_seed():
    a = draw_sfh(np.random.default_rng(123), t_obs_gyr=10.0)
    b = draw_sfh(np.random.default_rng(123), t_obs_gyr=10.0)
    np.testing.assert_array_equal(a.sfr_msun_per_yr, b.sfr_msun_per_yr)
    np.testing.assert_array_equal(a.quantile_times_gyr, b.quantile_times_gyr)


def test_draw_sfh_rejects_nonpositive_t_obs():
    with pytest.raises(ValueError):
        draw_sfh(np.random.default_rng(0), t_obs_gyr=0.0)
    with pytest.raises(ValueError):
        draw_sfh(np.random.default_rng(0), t_obs_gyr=-1.0)


def test_draw_sfh_robust_across_many_seeds_and_ages():
    for seed in range(50):
        rng = np.random.default_rng(seed)
        t_obs = np.random.default_rng(seed + 10_000).uniform(0.5, 13.5)
        sfh = draw_sfh(rng, t_obs_gyr=t_obs)
        assert not np.any(np.isnan(sfh.sfr_msun_per_yr))
        assert np.all(sfh.sfr_msun_per_yr >= 0.0)
        assert sfh.cumulative_mass_fraction[-1] == pytest.approx(1.0, abs=1e-5)


def test_reconstruct_cumulative_sfh_exact_at_constraint_points():
    """The GP is conditioned to pass exactly through its constraints (before
    the monotonic clip) -- verify the reconstruction actually hits them,
    which is the whole point of using exact-interpolation GP regression
    rather than a smoothed fit."""
    t_obs = 10.0
    quantile_times = np.array([2.0, 5.0, 8.0])
    t_grid = np.sort(np.concatenate([np.linspace(0, t_obs, 50), [0.0], quantile_times, [t_obs]]))
    f_grid = reconstruct_cumulative_sfh(t_grid, quantile_times, t_obs, length_scale_gyr=2.0)
    for t_q, f_q in zip(quantile_times, QUANTILE_FRACTIONS):
        idx = np.argmin(np.abs(t_grid - t_q))
        assert f_grid[idx] == pytest.approx(f_q, abs=1e-3)


def test_reconstruct_cumulative_sfh_shorter_length_scale_allows_more_structure():
    """Sanity check on the smoothness hyperparameter's documented meaning:
    a much shorter length scale should not produce a smoother/flatter curve
    than a long one -- check the reconstructed curve's second-derivative
    roughness is not smaller for the short length scale."""
    t_obs = 10.0
    quantile_times = np.array([1.0, 5.0, 9.0])
    t_grid = np.linspace(0, t_obs, 300)

    f_short = reconstruct_cumulative_sfh(t_grid, quantile_times, t_obs, length_scale_gyr=0.5)
    f_long = reconstruct_cumulative_sfh(t_grid, quantile_times, t_obs, length_scale_gyr=5.0)

    roughness_short = np.sum(np.diff(f_short, n=2) ** 2)
    roughness_long = np.sum(np.diff(f_long, n=2) ** 2)
    assert roughness_short >= roughness_long
