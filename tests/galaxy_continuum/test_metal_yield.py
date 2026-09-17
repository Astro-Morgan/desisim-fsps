import numpy as np
import pytest

from demiurge.galaxy_continuum import imf as imf_module
from demiurge.galaxy_continuum.metal_yield import (
    _ANCHOR_FEH,
    _ANCHOR_YIELD_AT_FEH,
    _METAL_EJECTA_MSUN,
    SelfConsistentMetallicityResult,
    draw_self_consistent_metallicity,
    effective_yield,
    effective_yield_at_z,
    integrate_self_consistent_z,
    integrate_self_consistent_z_batch,
    yield_mean,
    yield_sigma_relative,
)


class _FixedSampler:
    def __init__(self, values: dict):
        self._values = values

    def sample(self, names, *, rng, condition=None, size=None):
        return {name: self._values[name] for name in names}


def _fixed_sampler(yield_scatter):
    return _FixedSampler({"galaxy_continuum.metallicity.yield_scatter": yield_scatter})


# ---- effective_yield (the IMF integral) -- regression against the real sandbox numbers ----


@pytest.mark.parametrize(
    "feh,expected",
    [(0, 0.00980), (-1, 0.01115), (-2, 0.01213), (-3, 0.01485)],
)
def test_effective_yield_matches_the_validated_sandbox_numbers(feh, expected):
    z_abs = imf_module.Z_SUN * 10.0 ** feh
    slopes = imf_module.metallicity_dependent_slopes(z_abs)
    y = effective_yield(slopes.imf1, slopes.imf2, slopes.imf3, _METAL_EJECTA_MSUN[feh])
    assert y == pytest.approx(expected, rel=2e-3)


def test_canonical_yield_matches_recchi_kroupa_2015_within_2_percent():
    """The actual validation criterion: at the untouched canonical Kroupa
    IMF, using the solar table, this must land close to Recchi & Kroupa
    (2015)'s own stated 'typical' value of 0.01."""
    y = effective_yield(1.3, 2.3, 2.3, _METAL_EJECTA_MSUN[0])
    assert y == pytest.approx(0.01, rel=0.02)


def test_yield_increases_as_alpha3_decreases_top_heavier():
    y_canonical = effective_yield(1.3, 2.3, 2.3, _METAL_EJECTA_MSUN[0])
    y_top_heavy = effective_yield(1.3, 2.3, 1.8, _METAL_EJECTA_MSUN[0])
    assert y_top_heavy > y_canonical


# ---- anchor points ----


def test_anchor_points_are_monotonically_increasing_toward_low_feh():
    order = np.argsort(_ANCHOR_FEH)
    sorted_yields = _ANCHOR_YIELD_AT_FEH[order]
    assert np.all(np.diff(sorted_yields) <= 0), "yield should decrease as [Fe/H] increases from -3 to 0"


def test_solar_anchor_uses_canonical_imf():
    idx = list(_ANCHOR_FEH).index(0.0)
    assert _ANCHOR_YIELD_AT_FEH[idx] == pytest.approx(0.00980, rel=2e-3)


# ---- yield_mean (the PCHIP spline) ----


def test_yield_mean_passes_through_every_real_anchor_exactly():
    for feh, y in zip(_ANCHOR_FEH, _ANCHOR_YIELD_AT_FEH):
        z_abs = imf_module.Z_SUN * 10.0 ** feh
        assert yield_mean(z_abs) == pytest.approx(y, rel=1e-6)


def test_yield_mean_is_monotonic_between_anchors():
    fehs = np.linspace(-3.0, 0.0, 200)
    z_values = imf_module.Z_SUN * 10.0 ** fehs
    yields = np.array([yield_mean(z) for z in z_values])
    assert np.all(np.diff(yields) <= 1e-12), "spline must stay monotone (PCHIP), not overshoot between real anchors"


def test_yield_mean_held_flat_outside_real_coverage():
    z_below = imf_module.Z_SUN * 10.0 ** (-5.0)
    z_at_floor = imf_module.Z_SUN * 10.0 ** (-3.0)
    assert yield_mean(z_below) == pytest.approx(yield_mean(z_at_floor), rel=1e-9)

    z_above = imf_module.Z_SUN * 10.0 ** (0.5)
    z_at_ceiling = imf_module.Z_SUN * 10.0 ** (0.0)
    assert yield_mean(z_above) == pytest.approx(yield_mean(z_at_ceiling), rel=1e-9)


# ---- yield_sigma_relative ----


def test_sigma_smaller_inside_real_coverage_than_outside():
    inside = yield_sigma_relative(imf_module.Z_SUN * 10.0 ** (-1.5))
    outside_low = yield_sigma_relative(imf_module.Z_SUN * 10.0 ** (-4.0))
    outside_high = yield_sigma_relative(imf_module.Z_SUN * 10.0 ** (0.5))
    assert inside == pytest.approx(0.35)
    assert outside_low == pytest.approx(0.70)
    assert outside_high == pytest.approx(0.70)
    assert inside < outside_low
    assert inside < outside_high


# ---- effective_yield_at_z ----


def test_zero_scatter_gives_exactly_the_mean_curve():
    z_abs = imf_module.Z_SUN * 10.0 ** (-1.0)
    assert effective_yield_at_z(z_abs, 0.0) == pytest.approx(yield_mean(z_abs))


def test_scatter_is_floored_well_above_zero():
    z_abs = imf_module.Z_SUN * 10.0 ** (-1.0)
    y = effective_yield_at_z(z_abs, yield_scatter=-100.0)  # an absurdly large negative draw
    assert y > 0.0
    assert y == pytest.approx(0.05 * yield_mean(z_abs))


def test_positive_scatter_increases_yield_negative_decreases_it():
    z_abs = imf_module.Z_SUN * 10.0 ** (-1.0)
    mean = yield_mean(z_abs)
    assert effective_yield_at_z(z_abs, 1.0) > mean
    assert effective_yield_at_z(z_abs, -1.0) < mean


# ---- integrate_self_consistent_z: the ODE, validated against the OLD closed form ----


def test_rk4_integrator_reproduces_the_old_closed_form_for_a_truly_constant_yield():
    """The key correctness check for the RK4 mechanism itself, isolated
    from the real (Z-varying) yield_mean curve via an injected synthetic
    constant-yield RHS: dZ/dmu=-y0/mu, Z(1)=0 has the exact analytic
    solution Z(mu)=y0*ln(1/mu) -- the OLD closed-box relation this project
    used before this module existed. If the numerics reproduce this
    exactly, the RK4 stepping is correct regardless of what curve is fed
    into it."""
    mu_grid = np.linspace(1.0, 0.3, 500)
    y0 = 0.015

    def constant_yield(z_val, yield_scatter):
        return y0

    z_numeric = integrate_self_consistent_z(mu_grid, yield_scatter=0.0, yield_fn=constant_yield)
    z_closed_form = y0 * np.log(1.0 / mu_grid)
    np.testing.assert_allclose(z_numeric, z_closed_form, rtol=1e-6, atol=1e-10)


def test_ode_with_the_real_yield_curve_gives_a_smaller_final_z_than_the_naive_constant_estimate():
    """Sanity check that the real (Z-varying) feedback is doing something
    physically sensible: as Z rises during the integration, yield_mean(Z)
    DECREASES (moving away from the low-Z anchor toward the solar one), so
    the real ODE should predict LESS total enrichment than naively assuming
    the initial (low-Z, higher-yield) rate held constant throughout."""
    mu_grid = np.linspace(1.0, 0.3, 500)
    y0_at_start = yield_mean(0.0)  # yield_mean(Z=0) resolves into the low-Z flat region
    z_naive_constant = y0_at_start * np.log(1.0 / mu_grid)
    z_real = integrate_self_consistent_z(mu_grid, yield_scatter=0.0)
    assert z_real[-1] < z_naive_constant[-1]


def test_ode_z_starts_at_zero_and_is_nondecreasing():
    mu_grid = np.linspace(1.0, 0.4, 300)
    z = integrate_self_consistent_z(mu_grid, yield_scatter=0.5)
    assert z[0] == pytest.approx(0.0)
    assert np.all(np.diff(z) >= -1e-12)


def test_ode_higher_yield_scatter_gives_higher_final_z():
    mu_grid = np.linspace(1.0, 0.3, 300)
    z_lo = integrate_self_consistent_z(mu_grid, yield_scatter=-1.0)
    z_hi = integrate_self_consistent_z(mu_grid, yield_scatter=1.0)
    assert z_hi[-1] > z_lo[-1]


# ---- draw_self_consistent_metallicity ----


def test_draw_matches_formula_with_fixed_sampler():
    mu_grid = np.linspace(1.0, 0.5, 100)
    result = draw_self_consistent_metallicity(np.random.default_rng(0), mu_grid, sampler=_fixed_sampler(0.5))
    expected = integrate_self_consistent_z(mu_grid, 0.5)
    np.testing.assert_array_equal(result.z_grid, expected)
    assert result.yield_scatter == 0.5


def test_default_sampler_draws_the_registered_parameter():
    mu_grid = np.linspace(1.0, 0.6, 50)
    result = draw_self_consistent_metallicity(np.random.default_rng(3), mu_grid)
    assert isinstance(result, SelfConsistentMetallicityResult)
    assert np.isfinite(result.yield_scatter)
    assert np.all(np.isfinite(result.z_grid))


def test_reproducible_given_same_seed():
    mu_grid = np.linspace(1.0, 0.5, 60)
    a = draw_self_consistent_metallicity(np.random.default_rng(42), mu_grid)
    b = draw_self_consistent_metallicity(np.random.default_rng(42), mu_grid)
    assert a.yield_scatter == b.yield_scatter
    np.testing.assert_array_equal(a.z_grid, b.z_grid)


# ---- array-capable yield_mean/yield_sigma_relative/effective_yield_at_z (2026-09-17) ----


def test_yield_mean_array_input_matches_looping_the_scalar_path():
    fehs = np.linspace(-3.5, 0.5, 25)
    z_values = imf_module.Z_SUN * 10.0 ** fehs
    batched = yield_mean(z_values)
    looped = np.array([yield_mean(z) for z in z_values])
    assert batched.shape == z_values.shape
    np.testing.assert_array_equal(batched, looped)


def test_yield_sigma_relative_array_input_matches_looping_the_scalar_path():
    fehs = np.linspace(-4.0, 1.0, 25)
    z_values = imf_module.Z_SUN * 10.0 ** fehs
    batched = yield_sigma_relative(z_values)
    looped = np.array([yield_sigma_relative(z) for z in z_values])
    np.testing.assert_array_equal(batched, looped)


def test_effective_yield_at_z_array_input_matches_looping_the_scalar_path():
    rng = np.random.default_rng(21)
    z_values = imf_module.Z_SUN * 10.0 ** rng.uniform(-4.0, 1.0, size=25)
    scatters = rng.normal(size=25)
    batched = effective_yield_at_z(z_values, scatters)
    looped = np.array([effective_yield_at_z(z, s) for z, s in zip(z_values, scatters)])
    np.testing.assert_array_equal(batched, looped)


def test_scalar_inputs_still_return_plain_python_floats():
    """Regression guard: the array-support generalization must not turn
    scalar calls into length-1 arrays -- existing callers (and tests
    throughout this file) rely on a plain float coming back."""
    assert isinstance(yield_mean(0.01), float)
    assert isinstance(yield_sigma_relative(0.01), float)
    assert isinstance(effective_yield_at_z(0.01, 0.5), float)


# ---- integrate_self_consistent_z_batch (2026-09-17) ----


def test_batch_integrator_matches_looping_the_scalar_integrator():
    """Full trajectory, not just the endpoint -- population_inversion.py
    needs every grid point to compute the mass-weighted integral."""
    rng = np.random.default_rng(23)
    n_points, n_grid = 15, 40
    mu_batch = np.sort(rng.uniform(0.2, 1.0, size=(n_points, n_grid)), axis=1)[:, ::-1]
    mu_batch[:, 0] = 1.0
    scatters = rng.normal(size=n_points)

    batched = integrate_self_consistent_z_batch(mu_batch, scatters)
    assert batched.shape == (n_points, n_grid)
    looped = np.array([integrate_self_consistent_z(mu_batch[i], scatters[i]) for i in range(n_points)])
    np.testing.assert_array_equal(batched, looped)


def test_batch_integrator_starts_at_zero_and_is_nondecreasing_per_row():
    rng = np.random.default_rng(24)
    mu_batch = np.tile(np.linspace(1.0, 0.4, 30), (5, 1))
    scatters = rng.normal(size=5)
    z = integrate_self_consistent_z_batch(mu_batch, scatters)
    assert np.all(z[:, 0] == 0.0)
    assert np.all(np.diff(z, axis=1) >= -1e-12)


def test_registry_entry_matches_decided_distribution():
    from demiurge.parameters.distributions import Normal
    from demiurge.parameters.registry import get_parameter

    param = get_parameter("galaxy_continuum.metallicity.yield_scatter")
    assert param.tier == 3
    assert param.distribution == Normal(0.0, 1.0)
