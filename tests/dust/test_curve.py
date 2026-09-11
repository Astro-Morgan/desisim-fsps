import numpy as np
import pytest

from demiurge.dust.curve import (
    LAMBDA_V,
    VALIDITY_FLOOR_AA,
    drude,
    k_lambda,
    k_lambda_with_floor,
    transmission,
    transmission_with_floor,
)

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


def test_k_lambda_with_floor_matches_raw_k_above_the_floor():
    wave = np.array([VALIDITY_FLOOR_AA, 5500.0, 10000.0])
    np.testing.assert_array_equal(
        k_lambda_with_floor(wave, 1.0, 1.0), k_lambda(wave, 1.0, 1.0)
    )


def test_k_lambda_with_floor_holds_the_boundary_value_below_the_floor():
    # Below the floor, k is held FLAT at its own real value AT the floor --
    # not reset to zero (which would mean "no reddening," an equally
    # arbitrary claim) and not left to keep diverging.
    boundary_k = k_lambda(np.array([VALIDITY_FLOOR_AA]), 1.0, 1.3)[0]
    wave = np.array([0.06, 1.0, 100.0, 500.0, 911.9])
    k = k_lambda_with_floor(wave, 1.0, 1.3)
    np.testing.assert_allclose(k, boundary_k)


def test_transmission_with_floor_is_perfectly_continuous_at_the_boundary():
    # The whole point of this fix: no discontinuity anywhere, for any
    # theta -- not "a smaller jump," an ACTUAL zero jump, verified
    # numerically rather than assumed.
    just_below = transmission_with_floor(np.array([VALIDITY_FLOOR_AA - 1e-6]), 1.0, 1.3)
    just_above = transmission_with_floor(np.array([VALIDITY_FLOOR_AA + 1e-6]), 1.0, 1.3)
    np.testing.assert_allclose(just_below, just_above, rtol=1e-6)


def test_transmission_with_floor_is_flat_below_the_floor_for_a_steep_slope():
    # Reproduces the exact case that originally surfaced this whole chain
    # of fixes (theta1=1.897, once a real registered draw): whatever the
    # slope, transmission below the floor must be perfectly flat, not
    # diverging (the original bug) and not a step to 1.0 (the first wrong
    # fix).
    wave = np.linspace(0.06, VALIDITY_FLOOR_AA, 2000)
    t = transmission_with_floor(wave, 0.166, 1.897)
    assert np.all(np.isfinite(t))
    np.testing.assert_allclose(t, t[0], rtol=1e-9)


def test_transmission_with_floor_unclamped_behavior_continues_above_the_floor():
    wave = np.linspace(VALIDITY_FLOOR_AA, 10000.0, 500)
    t = transmission_with_floor(wave, 1.0, 1.0)
    np.testing.assert_allclose(t, transmission(k_lambda(wave, 1.0, 1.0)))
    assert np.all(np.diff(t) > 0.0)  # transmission rises with wavelength (less reddening redward)


def test_realistic_theta1_ceiling_gives_a_modest_floor_value_not_near_total_extinction():
    # host_disk_reddening's registered theta1_slope ceiling is now 1.3
    # (Prevot et al. 1984's real SMC-bar measurement, n~1.2, plus a small
    # margin) -- confirms the fix actually addresses the root cause found
    # via visual verification (2026-09-11): a real, in-range theta0/theta1
    # combination should no longer produce anything close to the ~99% jump
    # the old, unchecked Uniform(0,2) range allowed.
    theta0, theta1 = 0.166, 1.3  # theta0 matches the real draw that surfaced the original bug
    t_at_floor = transmission_with_floor(np.array([VALIDITY_FLOOR_AA]), theta0, theta1)[0]
    assert t_at_floor > 0.1  # a modest dimming, nowhere near the old ~0.01
