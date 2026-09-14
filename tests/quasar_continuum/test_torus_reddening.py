import numpy as np
import pytest

from demiurge.quasar_continuum.torus_reddening import TorusReddeningResult, draw_torus_reddening

WAVE = np.linspace(1000.0, 10000.0, 200)


class _FixedSampler:
    """Minimal ParameterSampler stub -- returns fixed values regardless of rng,
    for deterministic torus_reddening tests (mirrors the _FixedMassSampler
    pattern in tests/quasar_continuum/test_quasar_continuum.py)."""

    def __init__(self, values: dict):
        self._values = values

    def sample(self, names, *, rng, condition=None, size=None):
        return {name: self._values[name] for name in names}


def _fixed_sampler(covering_angle_cosine, theta0_amplitude=1.0, theta1_slope=0.3, euv_curvature=0.0):
    return _FixedSampler(
        {
            "quasar_continuum.torus_reddening.covering_angle_cosine": covering_angle_cosine,
            "quasar_continuum.torus_reddening.theta0_amplitude": theta0_amplitude,
            "quasar_continuum.torus_reddening.theta1_slope": theta1_slope,
            "quasar_continuum.torus_reddening.euv_curvature": euv_curvature,
        }
    )


def test_low_cosi_below_covering_angle_intercepts():
    result = draw_torus_reddening(
        np.random.default_rng(0), cosi=0.1, sampler=_fixed_sampler(covering_angle_cosine=0.3)
    )
    assert result.intercepted is True


def test_high_cosi_above_covering_angle_does_not_intercept():
    result = draw_torus_reddening(
        np.random.default_rng(0), cosi=0.9, sampler=_fixed_sampler(covering_angle_cosine=0.3)
    )
    assert result.intercepted is False


def test_not_intercepted_gives_unit_transmission_everywhere():
    result = draw_torus_reddening(
        np.random.default_rng(0), cosi=0.9, sampler=_fixed_sampler(covering_angle_cosine=0.3)
    )
    t = result.transmission(WAVE)
    np.testing.assert_array_equal(t, np.ones_like(WAVE))


def test_intercepted_gives_transmission_below_one_and_finite():
    result = draw_torus_reddening(
        np.random.default_rng(0), cosi=0.1, sampler=_fixed_sampler(covering_angle_cosine=0.3, theta0_amplitude=1.0)
    )
    t = result.transmission(WAVE)
    assert np.all(np.isfinite(t))
    assert np.all((t > 0.0) & (t < 1.0))


def test_zero_amplitude_gives_unit_transmission_even_when_intercepted():
    result = draw_torus_reddening(
        np.random.default_rng(0), cosi=0.1, sampler=_fixed_sampler(covering_angle_cosine=0.3, theta0_amplitude=0.0)
    )
    t = result.transmission(WAVE)
    np.testing.assert_allclose(t, np.ones_like(WAVE))


def test_draws_are_unconditional_regardless_of_interception():
    """RNG-hygiene: the amplitude/slope draw happens even when not
    intercepted -- both branches should consume the sampler identically."""
    intercepted = draw_torus_reddening(
        np.random.default_rng(0), cosi=0.1, sampler=_fixed_sampler(covering_angle_cosine=0.3, theta0_amplitude=2.5)
    )
    not_intercepted = draw_torus_reddening(
        np.random.default_rng(0), cosi=0.9, sampler=_fixed_sampler(covering_angle_cosine=0.3, theta0_amplitude=2.5)
    )
    assert intercepted.theta0_amplitude == not_intercepted.theta0_amplitude == 2.5


def test_default_sampler_draws_every_registered_parameter():
    result = draw_torus_reddening(np.random.default_rng(5), cosi=0.5)
    assert isinstance(result, TorusReddeningResult)
    assert 0.0 <= result.covering_angle_cosine or result.covering_angle_cosine is not None
    assert np.isfinite(result.theta0_amplitude)
    assert np.isfinite(result.theta1_slope)
    assert np.isfinite(result.euv_curvature)


def test_transmission_continuous_across_the_validity_floor_for_any_curvature():
    from demiurge.dust.curve import VALIDITY_FLOOR_AA

    for curvature in (-2.0, -1.0, 0.0, 0.1):
        result = draw_torus_reddening(
            np.random.default_rng(0),
            cosi=0.1,
            sampler=_fixed_sampler(covering_angle_cosine=0.3, theta0_amplitude=1.0, theta1_slope=0.5, euv_curvature=curvature),
        )
        just_below = result.transmission(np.array([VALIDITY_FLOOR_AA - 1e-6]))
        just_above = result.transmission(np.array([VALIDITY_FLOOR_AA + 1e-6]))
        np.testing.assert_allclose(just_below, just_above, rtol=1e-6)


def test_zero_curvature_reproduces_the_plain_power_law_below_the_floor():
    from demiurge.dust.curve import VALIDITY_FLOOR_AA, k_lambda

    result = draw_torus_reddening(
        np.random.default_rng(0),
        cosi=0.1,
        sampler=_fixed_sampler(covering_angle_cosine=0.3, theta0_amplitude=1.0, theta1_slope=0.5, euv_curvature=0.0),
    )
    wave = np.array([500.0, 100.0])
    t = result.transmission(wave)
    expected_k = k_lambda(wave, 1.0, 0.5)
    np.testing.assert_allclose(t, 10.0 ** (-0.4 * expected_k), rtol=1e-9)
    assert wave.min() < VALIDITY_FLOOR_AA  # sanity: this is actually exercising the below-floor branch


def test_negative_curvature_turns_the_curve_back_toward_transparency():
    result = draw_torus_reddening(
        np.random.default_rng(0),
        cosi=0.1,
        sampler=_fixed_sampler(covering_angle_cosine=0.3, theta0_amplitude=1.0, theta1_slope=0.5, euv_curvature=-2.0),
    )
    t = result.transmission(np.array([900.0, 300.0, 50.0, 1.0]))
    assert t[-1] > t[1]  # far below the floor, transmission rises back toward 1


def test_reproducible_given_same_rng_seed():
    a = draw_torus_reddening(np.random.default_rng(42), cosi=0.4)
    b = draw_torus_reddening(np.random.default_rng(42), cosi=0.4)
    assert a == b
