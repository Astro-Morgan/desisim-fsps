import pytest

from demiurge.galaxy_continuum.imf import (
    ALPHA3_CANONICAL,
    ALPHA3_R14_SLOPE,
    Z_SUN,
    canonical_slopes,
    metallicity_dependent_slopes,
)


def test_canonical_slopes_are_fixed_kroupa_2001_values():
    slopes = canonical_slopes()
    assert slopes.imf1 == pytest.approx(1.3)
    assert slopes.imf2 == pytest.approx(2.3)
    assert slopes.imf3 == pytest.approx(2.3)


def test_metallicity_dependent_slopes_match_canonical_at_solar_z():
    slopes = metallicity_dependent_slopes(Z_SUN)
    assert slopes.imf1 == pytest.approx(canonical_slopes().imf1, abs=1e-9)
    assert slopes.imf2 == pytest.approx(canonical_slopes().imf2, abs=1e-9)
    assert slopes.imf3 == pytest.approx(canonical_slopes().imf3, abs=1e-9)


def test_alpha3_is_now_metallicity_dependent_not_fixed():
    """The real fix (2026-09-16): alpha3 used to be hardcoded at
    ALPHA3_CANONICAL regardless of Z -- now it follows Recchi et al.
    (2014) eq. 6."""
    low_z = metallicity_dependent_slopes(Z_SUN * 1e-2)
    high_z = metallicity_dependent_slopes(Z_SUN * 10.0)
    assert low_z.imf3 != pytest.approx(ALPHA3_CANONICAL)
    assert high_z.imf3 != pytest.approx(ALPHA3_CANONICAL)
    assert low_z.imf3 < ALPHA3_CANONICAL < high_z.imf3, "lower Z must give a smaller (more top-heavy) alpha3"


def test_alpha3_matches_r14_formula_exactly():
    import numpy as np

    z = Z_SUN * 1e-2
    m_over_h = np.log10(z / Z_SUN)
    expected = ALPHA3_CANONICAL + ALPHA3_R14_SLOPE * m_over_h
    assert metallicity_dependent_slopes(z).imf3 == pytest.approx(expected)


def test_alpha3_shift_is_much_smaller_than_alpha1_alpha2_shift():
    """R14's alpha3 slope (0.0572) is deliberately much shallower than
    alpha1/alpha2's (0.5) -- the high-mass IMF is far less sensitive to
    metallicity than the low/intermediate-mass end, per the cited papers."""
    z = Z_SUN * 1e-2  # 2 dex below solar
    slopes = metallicity_dependent_slopes(z)
    shift1 = abs(slopes.imf1 - canonical_slopes().imf1)
    shift3 = abs(slopes.imf3 - canonical_slopes().imf3)
    assert shift3 < shift1


def test_metallicity_dependent_slopes_rejects_nonpositive_z():
    with pytest.raises(ValueError):
        metallicity_dependent_slopes(0.0)
    with pytest.raises(ValueError):
        metallicity_dependent_slopes(-0.01)
