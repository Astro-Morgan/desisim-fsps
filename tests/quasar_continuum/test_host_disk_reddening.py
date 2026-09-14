import numpy as np
import pytest

from demiurge.quasar_continuum.host_disk_reddening import (
    PATH_LENGTH_CAP,
    HostDiskReddeningResult,
    draw_host_disk_reddening,
)

WAVE = np.linspace(1000.0, 10000.0, 200)


class _FixedSampler:
    """Minimal ParameterSampler stub -- returns fixed values regardless of rng."""

    def __init__(self, values: dict):
        self._values = values

    def sample(self, names, *, rng, condition=None, size=None):
        return {name: self._values[name] for name in names}


def _fixed_sampler(cosi_disk, av_faceon, theta1_slope=0.5, euv_curvature=0.0):
    return _FixedSampler(
        {
            "quasar_continuum.host_disk_reddening.cosi_disk": cosi_disk,
            "quasar_continuum.host_disk_reddening.av_faceon": av_faceon,
            "quasar_continuum.host_disk_reddening.theta1_slope": theta1_slope,
            "quasar_continuum.host_disk_reddening.euv_curvature": euv_curvature,
        }
    )


def test_zero_av_faceon_gives_zero_av_and_unit_transmission_regardless_of_inclination():
    for cosi_disk in (0.0, 0.3, 1.0):
        result = draw_host_disk_reddening(
            np.random.default_rng(0), sampler=_fixed_sampler(cosi_disk, av_faceon=0.0)
        )
        assert result.av == 0.0
        np.testing.assert_array_equal(result.transmission(WAVE), np.ones_like(WAVE))


def test_face_on_dusty_host_gives_minimum_not_exactly_zero_av():
    """A genuinely dusty host (av_faceon > 0) still shows a nonzero minimum
    reddening at face-on (cosi_disk=1) -- exact zero only comes from a
    dust-free draw, per the module docstring."""
    result = draw_host_disk_reddening(
        np.random.default_rng(0), sampler=_fixed_sampler(cosi_disk=1.0, av_faceon=0.5)
    )
    assert result.av == pytest.approx(0.5)
    assert result.av > 0.0


def test_edge_on_amplifies_relative_to_face_on():
    face_on = draw_host_disk_reddening(
        np.random.default_rng(0), sampler=_fixed_sampler(cosi_disk=1.0, av_faceon=0.5)
    )
    edge_on = draw_host_disk_reddening(
        np.random.default_rng(0), sampler=_fixed_sampler(cosi_disk=0.05, av_faceon=0.5)
    )
    assert edge_on.av > face_on.av


def test_path_length_factor_is_capped_near_edge_on():
    edge_on = draw_host_disk_reddening(
        np.random.default_rng(0), sampler=_fixed_sampler(cosi_disk=0.0, av_faceon=1.0)
    )
    assert edge_on.av == pytest.approx(PATH_LENGTH_CAP)


def test_transmission_finite_and_bounded_for_dusty_host():
    result = draw_host_disk_reddening(
        np.random.default_rng(0), sampler=_fixed_sampler(cosi_disk=0.2, av_faceon=1.0)
    )
    t = result.transmission(WAVE)
    assert np.all(np.isfinite(t))
    assert np.all((t > 0.0) & (t < 1.0))


def test_default_sampler_returns_valid_result():
    result = draw_host_disk_reddening(np.random.default_rng(5))
    assert isinstance(result, HostDiskReddeningResult)
    assert 0.0 <= result.cosi_disk <= 1.0
    assert result.av_faceon >= 0.0
    assert np.isfinite(result.theta1_slope)
    assert np.isfinite(result.euv_curvature)


def test_transmission_continuous_across_the_validity_floor_for_any_curvature():
    from demiurge.dust.curve import VALIDITY_FLOOR_AA

    for curvature in (-2.0, -1.0, 0.0, 0.1):
        result = draw_host_disk_reddening(
            np.random.default_rng(0),
            sampler=_fixed_sampler(cosi_disk=0.5, av_faceon=1.0, theta1_slope=0.5, euv_curvature=curvature),
        )
        just_below = result.transmission(np.array([VALIDITY_FLOOR_AA - 1e-6]))
        just_above = result.transmission(np.array([VALIDITY_FLOOR_AA + 1e-6]))
        np.testing.assert_allclose(just_below, just_above, rtol=1e-6)


def test_reproducible_given_same_rng_seed():
    a = draw_host_disk_reddening(np.random.default_rng(42))
    b = draw_host_disk_reddening(np.random.default_rng(42))
    assert a == b


def test_zero_inflated_default_prior_produces_some_exact_zeros():
    draws = [draw_host_disk_reddening(np.random.default_rng(i)) for i in range(500)]
    zero_fraction = np.mean([d.av_faceon == 0.0 for d in draws])
    assert zero_fraction > 0.0
