import numpy as np
import pytest

from demiurge.galaxy_continuum.metallicity import (
    draw_metallicity,
    evaluate_closed_box,
    validate_monotonic_z,
)


def test_evaluate_closed_box_starts_near_zero():
    f = np.linspace(0.0, 1.0, 100)
    z = evaluate_closed_box(f, yield_=0.02, efficiency=0.5)
    assert z[0] == pytest.approx(0.0, abs=1e-9)


def test_evaluate_closed_box_is_monotonic():
    f = np.linspace(0.0, 1.0, 100)
    z = evaluate_closed_box(f, yield_=0.02, efficiency=0.5)
    assert np.all(np.diff(z) >= 0.0)


def test_evaluate_closed_box_scales_with_yield():
    f = np.linspace(0.0, 1.0, 100)
    z_low = evaluate_closed_box(f, yield_=0.01, efficiency=0.5)
    z_high = evaluate_closed_box(f, yield_=0.02, efficiency=0.5)
    np.testing.assert_allclose(z_high, 2.0 * z_low)


def test_evaluate_closed_box_raises_when_reservoir_exhausted():
    f = np.array([0.0, 0.5, 1.0])
    with pytest.raises(ValueError):
        evaluate_closed_box(f, yield_=0.02, efficiency=1.0)  # mu(1.0) = 0 exactly


def test_draw_metallicity_reproducible_given_same_seed():
    f = np.linspace(0.0, 1.0, 50)
    a = draw_metallicity(np.random.default_rng(7), f)
    b = draw_metallicity(np.random.default_rng(7), f)
    np.testing.assert_array_equal(a.z_grid, b.z_grid)


def test_draw_metallicity_robust_across_many_seeds():
    f = np.linspace(0.0, 1.0, 50)
    for seed in range(50):
        result = draw_metallicity(np.random.default_rng(seed), f)
        assert not np.any(np.isnan(result.z_grid))
        assert np.all(np.diff(result.z_grid) >= -1e-12)


def test_validate_monotonic_z_passes_valid_array():
    z = np.array([0.0, 0.01, 0.02, 0.02, 0.03])
    out = validate_monotonic_z(z)
    np.testing.assert_array_equal(out, z)


def test_validate_monotonic_z_raises_by_default_on_violation():
    z = np.array([0.0, 0.02, 0.01, 0.03])
    with pytest.raises(ValueError):
        validate_monotonic_z(z)


def test_validate_monotonic_z_enforce_coerces_via_running_max():
    z = np.array([0.0, 0.02, 0.01, 0.03])
    out = validate_monotonic_z(z, enforce=True)
    assert np.all(np.diff(out) >= 0.0)
    np.testing.assert_array_equal(out, np.array([0.0, 0.02, 0.02, 0.03]))
