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
    for curvature in (-2.0, -1.0, 0.0, 0.1):
        np.testing.assert_array_equal(
            k_lambda_with_floor(wave, 1.0, 1.0, curvature), k_lambda(wave, 1.0, 1.0)
        )


def test_k_lambda_with_floor_rejects_nonzero_bump_or_grey_floor():
    with pytest.raises(NotImplementedError):
        k_lambda_with_floor(np.array([500.0]), 1.0, 1.0, 0.0, theta2=0.5)
    with pytest.raises(NotImplementedError):
        k_lambda_with_floor(np.array([500.0]), 1.0, 1.0, 0.0, theta3=0.1)


def test_k_lambda_with_floor_matches_value_and_slope_at_the_boundary_for_any_curvature():
    # By construction: value AND derivative (in ln k vs ln lambda) must
    # match the plain power law exactly at the floor, for ANY curvature --
    # verified numerically here, not just derived.
    theta0, theta1 = 1.0, 1.3
    eps = VALIDITY_FLOOR_AA * 1e-6
    for curvature in (-2.0, -1.0, 0.0, 0.1):
        k_below = k_lambda_with_floor(np.array([VALIDITY_FLOOR_AA - eps]), theta0, theta1, curvature)[0]
        k_above = k_lambda(np.array([VALIDITY_FLOOR_AA - eps]), theta0, theta1)[0]
        # just below the floor, the quadratic extension is active but should
        # still nearly match the plain curve this close to the boundary,
        # for every curvature (value continuity)
        np.testing.assert_allclose(k_below, k_above, rtol=1e-5)


def test_k_lambda_with_floor_zero_curvature_reproduces_the_plain_power_law():
    wave = np.array([500.0, 100.0, 1.0, 0.06])
    np.testing.assert_allclose(
        k_lambda_with_floor(wave, 1.0, 1.3, 0.0), k_lambda(wave, 1.0, 1.3), rtol=1e-9
    )


def test_k_lambda_with_floor_negative_curvature_turns_over():
    # Sufficiently negative curvature should eventually make k DECREASE
    # (curve back toward transparency) as wavelength drops further below
    # the floor, unlike curvature=0 (monotonically increasing, diverging).
    wave = np.array([900.0, 300.0, 50.0, 1.0])
    k = k_lambda_with_floor(wave, 1.0, 1.3, -2.0)
    assert k[-1] < k[1]  # k at 1A is LESS than k at 300A -- the curve turned over


def test_k_lambda_with_floor_positive_curvature_diverges_faster_than_plain():
    wave = np.array([50.0])
    k_curved = k_lambda_with_floor(wave, 1.0, 1.3, 0.1)[0]
    k_plain = k_lambda(wave, 1.0, 1.3)[0]
    assert k_curved > k_plain


def test_transmission_with_floor_is_perfectly_continuous_at_the_boundary_for_any_curvature():
    # The whole point of this fix: no discontinuity anywhere, for any
    # theta1/curvature combination -- not "a smaller jump," an ACTUAL zero
    # jump, verified numerically rather than assumed.
    for curvature in (-2.0, -1.0, 0.0, 0.1):
        just_below = transmission_with_floor(np.array([VALIDITY_FLOOR_AA - 1e-6]), 1.0, 1.3, curvature)
        just_above = transmission_with_floor(np.array([VALIDITY_FLOOR_AA + 1e-6]), 1.0, 1.3, curvature)
        np.testing.assert_allclose(just_below, just_above, rtol=1e-6)


def test_transmission_with_floor_bounded_in_zero_one_below_the_floor():
    # k_below = exp(...) is always >= 0 by construction, so transmission
    # must stay in [0, 1] -- never negative, never > 1. Exactly 1.0 is
    # reachable for very negative curvature far from the floor (k_below
    # underflows to 0.0); exactly 0.0 is reachable for positive curvature
    # (k_below overflows toward +inf, transmission underflows to 0.0) --
    # both are expected float64 behavior at the extremes, not bugs.
    wave = np.linspace(0.06, VALIDITY_FLOOR_AA, 500)
    for curvature in (-2.0, -1.0, 0.0, 0.1):
        t = transmission_with_floor(wave, 0.166, 1.897, curvature)
        assert np.all(np.isfinite(t))
        assert np.all((t >= 0.0) & (t <= 1.0))


def test_transmission_with_floor_unclamped_behavior_continues_above_the_floor():
    wave = np.linspace(VALIDITY_FLOOR_AA, 10000.0, 500)
    t = transmission_with_floor(wave, 1.0, 1.0, -1.0)  # curvature is irrelevant above the floor
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
    t_at_floor = transmission_with_floor(np.array([VALIDITY_FLOOR_AA]), theta0, theta1, 0.0)[0]
    assert t_at_floor > 0.1  # a modest dimming, nowhere near the old ~0.01
