import numpy as np

from demiurge.quasar_continuum import novikov_thorne as nt


def test_isco_matches_known_schwarzschild_value():
    """astar=0 -> r_ms = 6 R_g, the textbook Schwarzschild ISCO."""
    assert nt.isco(0.0) == 6.0


def test_efficiency_matches_known_schwarzschild_value():
    """astar=0 -> eta = 1 - sqrt(8/9) ~ 0.0572, the standard Schwarzschild
    accretion efficiency (KD18 Sec. 2's own stated value, "0.057 for a=0")."""
    eff = nt.efficiency(0.0)
    assert abs(eff - 0.0571909584) < 1e-8


def test_isco_decreases_with_prograde_spin():
    assert nt.isco(0.998) < nt.isco(0.5) < nt.isco(0.0) < nt.isco(-0.5) < nt.isco(-0.998)


def test_efficiency_increases_with_prograde_spin():
    assert nt.efficiency(0.998) > nt.efficiency(0.0) > nt.efficiency(-0.998)


def test_nt_temperature4_positive_and_decreasing_outward():
    rms = nt.isco(0.0)
    mdot_gs = nt.mass_accretion_rate_gs(1e8, 0.1, nt.efficiency(0.0))
    r = np.array([10.0, 100.0, 1000.0])
    t4 = nt.nt_temperature4(1e8, 0.0, mdot_gs, rms, r)
    assert np.all(t4 > 0)
    assert np.all(np.diff(t4) < 0), "disc temperature must decrease outward"


def test_self_gravity_radius_matches_compiled_reference():
    """Cross-checked directly against the compiled agnsed.f/qsosed.f
    (gfortran -std=legacy, unmodified) for M=1e8 Msun, mdot=0.10:
    rout(rsg) = 772.66993950239748."""
    r_out = nt.self_gravity_radius(1.0e8, 0.10)
    assert abs(r_out - 772.66993950239748) / 772.66993950239748 < 1e-6


def test_eddington_luminosity_scales_linearly_with_mass():
    assert nt.eddington_luminosity(2e8) == 2 * nt.eddington_luminosity(1e8)
