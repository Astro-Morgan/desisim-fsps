import pytest

from demiurge.dust.depletion import (
    GCB16_TABLE1_AT_SOLAR,
    XI_D_SOLAR,
    depletion_factor,
    gas_phase_fraction,
)


def test_depletion_is_zero_at_xi_d_zero():
    for element in GCB16_TABLE1_AT_SOLAR:
        assert depletion_factor(element, 0.0) == pytest.approx(0.0)


def test_depletion_is_one_at_xi_d_one_for_refractory_elements():
    """Nitrogen is the one non-refractory element this project currently
    registers (GCB16's own Table 1 footnote) -- it does NOT follow the
    refractory (0,0)-(1,1) interpolation at all, see the dedicated
    nitrogen test below."""
    for element, f_solar in GCB16_TABLE1_AT_SOLAR.items():
        if f_solar == 0.0:
            continue
        assert depletion_factor(element, 1.0) == pytest.approx(1.0)


def test_depletion_matches_table1_at_solar_xi_d():
    for element, expected in GCB16_TABLE1_AT_SOLAR.items():
        assert depletion_factor(element, XI_D_SOLAR) == pytest.approx(expected)


def test_depletion_continuous_across_the_solar_anchor():
    """The two line segments (below/above XI_D_SOLAR) must agree exactly at
    the anchor for every element, not just by coincidence for one -- this is
    a real property of the construction, verified numerically here."""
    for element in GCB16_TABLE1_AT_SOLAR:
        just_below = depletion_factor(element, XI_D_SOLAR - 1e-9)
        just_above = depletion_factor(element, XI_D_SOLAR + 1e-9)
        assert just_below == pytest.approx(just_above, abs=1e-6)


def test_depletion_is_monotonically_nondecreasing_in_xi_d():
    for element in GCB16_TABLE1_AT_SOLAR:
        values = [depletion_factor(element, xi_d / 100.0) for xi_d in range(0, 101)]
        assert all(a <= b + 1e-12 for a, b in zip(values, values[1:]))


def test_nitrogen_is_undepleted_at_every_xi_d():
    """N's own GCB16 table value is 0 (not refractory in this model) -- so
    depletion should stay exactly 0 across the WHOLE xi_d range, not just at
    the solar anchor."""
    for xi_d in (0.0, 0.1, 0.36, 0.7, 1.0):
        assert depletion_factor("N", xi_d) == pytest.approx(0.0)


def test_gas_phase_fraction_is_one_minus_depletion():
    for element in GCB16_TABLE1_AT_SOLAR:
        for xi_d in (0.0, 0.2, XI_D_SOLAR, 0.6, 1.0):
            assert gas_phase_fraction(element, xi_d) == pytest.approx(1.0 - depletion_factor(element, xi_d))


def test_depletion_rejects_xi_d_out_of_range():
    with pytest.raises(ValueError):
        depletion_factor("O", -0.1)
    with pytest.raises(ValueError):
        depletion_factor("O", 1.1)


def test_depletion_rejects_unknown_element():
    with pytest.raises(KeyError):
        depletion_factor("Xe", 0.3)


def test_registry_entries_match_decided_distributions():
    from demiurge.parameters.distributions import TruncatedNormal, Uniform
    from demiurge.parameters.registry import get_parameter

    xi_d = get_parameter("dust.depletion.xi_d")
    assert xi_d.tier == 2
    assert xi_d.citation is not None
    assert xi_d.distribution == Uniform(0.0, 1.0)

    xi_d_nlr = get_parameter("dust.depletion.xi_d_nlr")
    assert xi_d_nlr.tier == 2
    assert xi_d_nlr.citation is not None
    assert xi_d_nlr.distribution == TruncatedNormal(mean=0.3, sigma=0.1, low=0.0, high=1.0)
