"""
Regression tests against the real, compiled HEASARC `agnsed.f`/`qsosed.f`
(gfortran -std=legacy, unmodified upstream source -- see
`quasar_continuum/geometry.py`'s module docstring). Expected values below
are the Fortran's own printed output for these exact (M, mdot,
hard_xray_luminosity_fraction=0.02, r_warm_over_r_hot=2.0) inputs, not
independently re-derived -- this is a byte-for-byte-algorithm parity check,
not a physics correctness claim on its own (that's KD18 Table 2 and Figure 9,
checked in test_continuum.py).

T(R_hot)/T(R_warm) are deliberately NOT checked here: the Fortran's own
diagnostic print statements for those two quantities carry a confirmed,
precisely-diagnosed bug (a variable-reuse mistake -- see geometry.py's
module docstring) that this module deliberately does not replicate, so
those two outputs are expected to disagree with the naive printed reference.
"""
import pytest

from demiurge.quasar_continuum.geometry import solve_geometry

# (M_msun, mdot_edd) -> Fortran-printed (Gamma_hot, r_hot, r_warm, r_out,
# Ldiss_hot/Ledd, Lhot/Ledd), all at hard_xray_luminosity_fraction=0.02,
# r_warm_over_r_hot=2.0, astar=0.0, reprocess=True.
CASES = [
    (5.5e7, 0.027, dict(gamma_hot=1.6790, r_hot=67.440, r_warm=134.876, r_out=493.130,
                         ldiss_frac=0.020010, lhot_frac=0.020755)),
    (1.0e8, 0.10, dict(gamma_hot=1.9943, r_hot=14.264, r_warm=28.527, r_out=772.670,
                        ldiss_frac=0.020028, lhot_frac=0.024194)),
    (1.0e8, 0.40, dict(gamma_hot=2.2412, r_hot=9.284, r_warm=18.569, r_out=1430.790,
                        ldiss_frac=0.020262, lhot_frac=0.033804)),
    (1.0e8, 0.20, dict(gamma_hot=2.1201, r_hot=10.939, r_warm=21.878, r_out=1051.440,
                        ldiss_frac=0.020003, lhot_frac=0.027674)),
]


@pytest.mark.parametrize("m_msun,mdot_edd,expected", CASES)
def test_solve_geometry_matches_compiled_fortran(m_msun, mdot_edd, expected):
    g = solve_geometry(m_msun, astar=0.0, mdot_edd=mdot_edd, hard_xray_luminosity_fraction=0.02)

    assert g.gamma_hot == pytest.approx(expected["gamma_hot"], rel=1e-3)
    assert g.r_hot == pytest.approx(expected["r_hot"], rel=1e-3)
    assert g.r_warm == pytest.approx(expected["r_warm"], rel=1e-3)
    assert g.r_out == pytest.approx(expected["r_out"], rel=1e-3)
    assert g.displ_erg_s / g.ledd == pytest.approx(expected["ldiss_frac"], rel=1e-3)
    assert g.lumipl_erg_s / g.ledd == pytest.approx(expected["lhot_frac"], rel=1e-3)


def test_r_hot_increases_with_hard_xray_luminosity_fraction():
    """Physical sanity check on the Tier-2 generalization QSOSED itself
    never exercises (it hardcodes the target at 0.02): a colder/larger
    disc area is needed to sweep up more dissipated luminosity, so a
    higher target fraction must push r_hot further out."""
    fractions = [0.01, 0.02, 0.03, 0.05]
    r_hots = [solve_geometry(1e8, 0.0, 0.10, f).r_hot for f in fractions]
    assert r_hots == sorted(r_hots)


def test_r_hot_realizes_the_requested_fraction():
    for frac in [0.01, 0.02, 0.03, 0.05]:
        g = solve_geometry(1e8, 0.0, 0.10, frac)
        assert g.displ_erg_s / g.ledd == pytest.approx(frac, rel=0.01)


def test_r_warm_respects_the_requested_ratio():
    g = solve_geometry(1e8, 0.0, 0.10, 0.02, r_warm_over_r_hot=3.0)
    assert g.r_warm == pytest.approx(3.0 * g.r_hot, rel=1e-6)
