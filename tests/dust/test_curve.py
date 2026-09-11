import numpy as np
import pytest

from demiurge.dust.curve import LAMBDA_V, drude, k_lambda, transmission

WAVE = np.linspace(1000.0, 10000.0, 500)


def test_drude_peaks_at_lambda0_with_value_one():
    peak = drude(np.array([2175.0]), lambda0=2175.0, gamma=350.0)
    assert peak[0] == pytest.approx(1.0)


def test_drude_falls_off_away_from_lambda0():
    # Symmetric in wave**2 - lambda0**2 (the standard Fitzpatrick & Massa
    # 1986 convention), NOT in linear (wave - lambda0) offset -- so this
    # checks monotonic falloff on both sides, not linear symmetry.
    peak = drude(np.array([2175.0]), lambda0=2175.0, gamma=350.0)[0]
    left = drude(np.array([2175.0 - 100.0]), lambda0=2175.0, gamma=350.0)[0]
    right = drude(np.array([2175.0 + 100.0]), lambda0=2175.0, gamma=350.0)[0]
    assert left < peak
    assert right < peak


def test_k_lambda_zero_theta_gives_zero_everywhere():
    k = k_lambda(WAVE, theta0=0.0, theta1=1.0, theta2=0.0, theta3=0.0)
    np.testing.assert_array_equal(k, np.zeros_like(WAVE))


def test_k_lambda_at_lambda_v_equals_theta0_plus_theta3_plus_bump_offset():
    # At wave == lambda_v, the power-law term reduces to exactly theta0.
    k = k_lambda(np.array([LAMBDA_V]), theta0=0.5, theta1=1.0, theta2=0.0, theta3=0.1)
    assert k[0] == pytest.approx(0.5 + 0.1)


def test_k_lambda_grey_floor_only_is_flat():
    k = k_lambda(WAVE, theta0=0.0, theta1=0.0, theta2=0.0, theta3=0.2)
    np.testing.assert_allclose(k, 0.2)


def test_k_lambda_power_law_decreases_with_wavelength_for_positive_theta1():
    k = k_lambda(WAVE, theta0=1.0, theta1=1.0, theta2=0.0, theta3=0.0)
    assert np.all(np.diff(k) < 0.0)


def test_transmission_is_one_when_k_is_zero():
    t = transmission(np.zeros(10))
    np.testing.assert_array_equal(t, np.ones(10))


def test_transmission_decreases_with_increasing_k():
    k = np.array([0.0, 0.5, 1.0, 2.0])
    t = transmission(k)
    assert np.all(np.diff(t) < 0.0)
    assert np.all((t > 0.0) & (t <= 1.0))


def test_bump_adds_local_excess_at_bump_center():
    k_no_bump = k_lambda(np.array([2175.0]), theta0=0.3, theta1=0.5, theta2=0.0, theta3=0.0)
    k_with_bump = k_lambda(np.array([2175.0]), theta0=0.3, theta1=0.5, theta2=0.5, theta3=0.0)
    assert k_with_bump[0] > k_no_bump[0]
    assert k_with_bump[0] == pytest.approx(k_no_bump[0] + 0.5)
