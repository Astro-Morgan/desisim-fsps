import numpy as np
import pytest

from demiurge.quasar_continuum import QuasarContinuum
from demiurge.quasar_continuum.continuum import _C_ANGSTROM_PER_S


def _fiducial(**overrides):
    kwargs = dict(
        m_msun=1.0e8, astar=0.0, mdot_edd=0.10, cosi=0.7071067811865476,
        hard_xray_luminosity_fraction=0.02, kte_hot_kev=100.0, kte_warm_kev=0.2, gamma_warm=2.5,
    )
    kwargs.update(overrides)
    return QuasarContinuum.from_parameters(**kwargs)


def test_from_parameters_produces_finite_nonnegative_flux():
    qc = _fiducial()
    assert np.all(np.isfinite(qc.flux))
    assert np.all(qc.flux >= 0.0)
    assert np.all(np.isfinite(qc.wave)) and np.all(qc.wave > 0)


def test_wave_is_ascending():
    qc = _fiducial()
    assert np.all(np.diff(qc.wave) > 0)


def test_peraa_conversion_matches_galaxy_continuum_convention():
    """peraa=True must relate to the default (L_nu [Lsun/Hz]) output via
    exactly the same f_lambda = f_nu * c / lambda**2 formula
    galaxy_continuum.continuum._fnu_to_flambda uses -- same physics, same
    constant, so the two channels' outputs are directly combinable without
    a unit-conversion step at blend time."""
    qc_nu = _fiducial()
    qc_lambda = _fiducial(peraa=True)
    np.testing.assert_allclose(qc_nu.wave, qc_lambda.wave)
    expected = qc_nu.flux * _C_ANGSTROM_PER_S / qc_nu.wave ** 2
    np.testing.assert_allclose(qc_lambda.flux, expected, rtol=1e-10)


def test_r_warm_over_r_hot_is_threaded_through():
    qc = _fiducial(r_warm_over_r_hot=3.0)
    assert qc.meta["geometry"].r_warm == pytest.approx(3.0 * qc.meta["geometry"].r_hot, rel=1e-6)


def test_higher_black_hole_mass_gives_higher_bolometric_output():
    """Sanity check independent of any specific normalization convention:
    a 10x more massive black hole at the same Eddington ratio must be more
    luminous, not just differently shaped."""
    qc_lo = _fiducial(m_msun=1e7)
    qc_hi = _fiducial(m_msun=1e8)
    assert np.trapezoid(qc_hi.flux, qc_hi.wave) > np.trapezoid(qc_lo.flux, qc_lo.wave)


def test_from_agnsed_produces_finite_nonnegative_flux():
    rng = np.random.default_rng(11)
    qc = QuasarContinuum.from_agnsed(rng)
    assert np.all(np.isfinite(qc.flux))
    assert np.all(qc.flux >= 0.0)
    assert "drawn_parameters" in qc.meta


def test_from_agnsed_reproducible_given_same_seed():
    a = QuasarContinuum.from_agnsed(np.random.default_rng(99))
    b = QuasarContinuum.from_agnsed(np.random.default_rng(99))
    np.testing.assert_array_equal(a.flux, b.flux)


def test_from_agnsed_draws_every_registered_parameter():
    rng = np.random.default_rng(3)
    qc = QuasarContinuum.from_agnsed(rng)
    drawn = qc.meta["drawn_parameters"]
    for name in [
        "quasar_continuum.agnsed.black_hole_mass",
        "quasar_continuum.agnsed.eddington_ratio",
        "quasar_continuum.agnsed.spin",
        "quasar_continuum.agnsed.cosi",
        "quasar_continuum.agnsed.hard_xray_luminosity_fraction",
        "quasar_continuum.agnsed.kte_hot",
        "quasar_continuum.agnsed.kte_warm",
        "quasar_continuum.agnsed.gamma_warm",
        "quasar_continuum.agnsed.r_warm_over_r_hot",
    ]:
        assert name in drawn


class _FixedMassSampler:
    """Minimal ParameterSampler stub -- fixes black_hole_mass, draws
    everything else from the real registry priors. Exercises from_agnsed's
    sampler-swap path without depending on PriorSampler's own condition
    mechanics (already covered in tests/parameters/test_samplers.py)."""

    def sample(self, names, *, rng, condition=None, size=None):
        from demiurge.parameters.samplers import PriorSampler

        base = PriorSampler().sample(names, rng=rng, condition=condition, size=size)
        if "quasar_continuum.agnsed.black_hole_mass" in names:
            base["quasar_continuum.agnsed.black_hole_mass"] = 5e8
        return base


def test_from_agnsed_accepts_a_custom_sampler():
    qc = QuasarContinuum.from_agnsed(np.random.default_rng(1), sampler=_FixedMassSampler())
    assert qc.meta["drawn_parameters"]["quasar_continuum.agnsed.black_hole_mass"] == 5e8


# --- Kubota & Done (2018) Figure 9: M=1e8 Msun, mdot=0.05 and 0.5,
# kTe_warm=0.2 keV, Gamma_warm=2.5, kTe_hot=100 keV,
# hard_xray_luminosity_fraction=0.02, reprocessing on, i=45deg (their own
# stated standard illustrative inclination, Sec. 3), evaluated as a flux
# at their own stated fiducial distance of 100 Mpc (Figure 3's caption,
# which Figure 9 builds on) in their own E^2 N(E) [keV^2 Photons/cm^2/s/keV]
# convention. Peak values read directly off their published figure: ~0.13
# (mdot=0.05) and ~2.0 (mdot=0.5) -- independent confirmation of BOTH the
# spectral shape and the absolute normalization together, not just the
# geometry (test_geometry.py) or a self-consistency check on this module's
# own output. ---
FIGURE_9_DISTANCE_CM = 100.0 * 1e6 * 3.0856775814913673e18  # 100 Mpc


@pytest.mark.parametrize("mdot_edd,expected_peak", [(0.05, 0.13), (0.5, 2.0)])
def test_reproduces_kubota_done_figure_9_peak_flux(mdot_edd, expected_peak):
    from demiurge.quasar_continuum.continuum import KEVHZ, _synthesize_photon_rates
    from demiurge.quasar_continuum.geometry import solve_geometry

    ear_kev = np.geomspace(1e-4, 200.0, 601)
    geom = solve_geometry(1.0e8, astar=0.0, mdot_edd=mdot_edd, hard_xray_luminosity_fraction=0.02)
    disk_rate, warm_rate, hot_rate, en_mid = _synthesize_photon_rates(
        geom, ear_kev, kte_hot_kev=100.0, kte_warm_kev=0.2, gamma_warm=2.5, icor=20, iout=400
    )
    cosi_45 = np.cos(np.deg2rad(45.0))
    total_rate = (disk_rate + warm_rate) * (cosi_45 / 0.5) + hot_rate

    bin_width_kev = np.diff(ear_kev)
    e2n_flux = en_mid ** 2 * (total_rate / bin_width_kev) / (4 * np.pi * FIGURE_9_DISTANCE_CM ** 2)

    assert e2n_flux.max() == pytest.approx(expected_peak, rel=0.15)


# --- Energy-conservation mechanism, pinned down 2026-09-09 (see
# geometry.py's module docstring for the full explanation): with
# reprocessing on, the total synthesized luminosity exceeds mdot*L_Edd by
# ~9-13%, and this is NOT a bug -- L_hot already includes seed photons
# intercepted from the disc/warm zones, and reprocessing then illuminates
# that same L_hot back onto the disc/warm zones, double-counting that
# energy. Confirmed by disabling reprocessing, where the excess disappears
# almost entirely (residual is discretization error, not this effect). ---
def _total_synthesized_over_edd(**overrides):
    from demiurge.quasar_continuum.continuum import KEV_TO_ERG, _synthesize_photon_rates
    from demiurge.quasar_continuum.geometry import solve_geometry

    kwargs = dict(m_msun=1.0e8, astar=0.0, mdot_edd=0.10, hard_xray_luminosity_fraction=0.02,
                   kte_hot_kev=100.0, kte_warm_kev=0.2, gamma_warm=2.5, reprocess=True)
    kwargs.update(overrides)
    reprocess = kwargs.pop("reprocess")

    ear_kev = np.geomspace(1e-4, 200.0, 601)
    geom = solve_geometry(kwargs.pop("m_msun"), kwargs.pop("astar"), kwargs.pop("mdot_edd"),
                           kwargs.pop("hard_xray_luminosity_fraction"), reprocess=reprocess)
    disk_rate, warm_rate, hot_rate, en_mid = _synthesize_photon_rates(
        geom, ear_kev, kwargs["kte_hot_kev"], kwargs["kte_warm_kev"], kwargs["gamma_warm"],
        icor=20, iout=400, reprocess=reprocess,
    )
    total_erg_s = np.sum((disk_rate + warm_rate + hot_rate) * en_mid * KEV_TO_ERG)
    return total_erg_s / (geom.mdot_edd * geom.ledd)


def test_reprocessing_produces_the_expected_luminosity_excess():
    ratio = _total_synthesized_over_edd(reprocess=True)
    assert 1.05 < ratio < 1.20, (
        f"expected the documented ~9-13% reprocessing-driven excess, got ratio={ratio:.4f} -- "
        f"if this moves outside that band, something about the reprocessing mechanism itself "
        f"changed, not just numerical noise (see geometry.py's module docstring)."
    )


def test_disabling_reprocessing_leaves_only_the_seed_photon_baseline():
    """reprocess=False turns off the Frep illumination boost (mechanism 2
    in geometry.py's module docstring) but NOT the corona's always-on
    interception of disc/warm seed photons (mechanism 1) -- so this is
    expected to land near ~1.03-1.04, not exactly 1.0. Bounded well below
    the full-reprocessing ratio (~1.09-1.13, see the test above) to keep
    this a meaningful regression guard on mechanism 2 specifically."""
    ratio = _total_synthesized_over_edd(reprocess=False)
    assert 1.0 < ratio < 1.06


def test_from_parameters_reprocess_flag_actually_changes_the_spectrum():
    """Regression guard for the 2026-09-09 bug where _synthesize_photon_rates
    hardcoded rep_flag=1.0 regardless of the reprocess argument -- from_parameters
    accepted `reprocess=False` but it silently had no effect on the synthesized
    spectrum (only on the geometry solve)."""
    qc_rep = _fiducial(reprocess=True)
    qc_norep = _fiducial(reprocess=False)
    assert not np.allclose(qc_rep.flux, qc_norep.flux)
