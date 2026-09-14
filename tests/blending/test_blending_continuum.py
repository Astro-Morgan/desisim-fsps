import numpy as np
import pytest

from demiurge.blending.continuum import (
    CompositeContinuum,
    _bolometric_luminosity,
    blend_continua,
)
from demiurge.galaxy_continuum.continuum import GalaxyContinuum
from demiurge.quasar_continuum.host_disk_reddening import HostDiskReddeningResult
from demiurge.quasar_continuum.torus_reddening import TorusReddeningResult


def _flat_channel(wave, level):
    """A trivial constant-L_nu channel -- enough to exercise blend_continua's
    own rescale math without depending on real FSPS/AGNSED synthesis."""
    return GalaxyContinuum(wave=np.asarray(wave, dtype=float), flux=np.full(len(wave), level), meta={})


WAVE_GAL = np.linspace(1000.0, 20000.0, 500)
WAVE_QSO = np.linspace(500.0, 25000.0, 500)


def _galaxy(level=1.0):
    return _flat_channel(WAVE_GAL, level)


def _quasar(level=1.0):
    # QuasarContinuum has the identical (wave, flux, meta) shape as
    # GalaxyContinuum -- reuse the same trivial constructor via duck typing
    # by importing the real class for an accurate isinstance/type story.
    from demiurge.quasar_continuum.continuum import QuasarContinuum

    return QuasarContinuum(wave=np.asarray(WAVE_QSO, dtype=float), flux=np.full(len(WAVE_QSO), level), meta={})


def test_bolometric_luminosity_of_flat_spectrum_is_positive_and_finite():
    l_bol = _bolometric_luminosity(WAVE_GAL, np.full(len(WAVE_GAL), 1.0))
    assert np.isfinite(l_bol)
    assert l_bol > 0.0


def test_quasar_frac_zero_gives_zero_quasar_contribution():
    result = blend_continua(_galaxy(2.0), _quasar(5.0), quasar_frac=0.0)
    assert result.meta["scale_quasar"] == 0.0
    assert result.meta["scale_galaxy"] > 0.0


def test_quasar_frac_one_gives_zero_galaxy_contribution():
    result = blend_continua(_galaxy(2.0), _quasar(5.0), quasar_frac=1.0)
    assert result.meta["scale_galaxy"] == 0.0
    assert result.meta["scale_quasar"] > 0.0


@pytest.mark.parametrize("quasar_frac", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
def test_achieved_bolometric_fraction_matches_quasar_frac_exactly(quasar_frac):
    galaxy = _galaxy(2.0)
    quasar = _quasar(5.0)
    result = blend_continua(galaxy, quasar, quasar_frac=quasar_frac)
    l_gal_final = result.meta["scale_galaxy"] * result.meta["l_bol_galaxy"]
    l_qso_final = result.meta["scale_quasar"] * result.meta["l_bol_quasar"]
    achieved = l_qso_final / (l_gal_final + l_qso_final)
    assert achieved == pytest.approx(quasar_frac, abs=1e-10)


def test_total_bolometric_luminosity_preserved_across_quasar_frac():
    galaxy = _galaxy(2.0)
    quasar = _quasar(5.0)
    totals = []
    for f in (0.0, 0.4, 1.0):
        result = blend_continua(galaxy, quasar, quasar_frac=f)
        totals.append(
            result.meta["scale_galaxy"] * result.meta["l_bol_galaxy"]
            + result.meta["scale_quasar"] * result.meta["l_bol_quasar"]
        )
    assert totals[0] == pytest.approx(totals[1]) == pytest.approx(totals[2])


def test_rejects_out_of_range_quasar_frac():
    with pytest.raises(ValueError):
        blend_continua(_galaxy(), _quasar(), quasar_frac=1.5)
    with pytest.raises(ValueError):
        blend_continua(_galaxy(), _quasar(), quasar_frac=-0.1)


def test_output_wave_grid_clipped_to_overlap_of_both_channels():
    result = blend_continua(_galaxy(), _quasar(), quasar_frac=0.5)
    assert result.wave.min() >= max(WAVE_GAL.min(), WAVE_QSO.min())
    assert result.wave.max() <= min(WAVE_GAL.max(), WAVE_QSO.max())


def test_flux_finite_and_nonnegative_for_positive_inputs():
    result = blend_continua(_galaxy(2.0), _quasar(5.0), quasar_frac=0.5)
    assert np.all(np.isfinite(result.flux))
    assert np.all(result.flux >= 0.0)


def test_torus_reddening_reduces_quasar_contribution_only():
    unreddened = blend_continua(_galaxy(1.0), _quasar(1.0), quasar_frac=0.7)
    torus = TorusReddeningResult(
        intercepted=True, covering_angle_cosine=0.5, theta0_amplitude=1.0, theta1_slope=0.3, euv_curvature=0.0, cosi=0.1
    )
    reddened = blend_continua(_galaxy(1.0), _quasar(1.0), quasar_frac=0.7, torus=torus)
    # Reddening reduces the OBSERVED quasar light reaching the composite,
    # without changing quasar_frac's own bolometric-luminosity accounting
    # (computed on the pre-reddening flux) -- so the composite's actual
    # flux should decrease overall, but scale_quasar (the accounting) is unchanged.
    assert reddened.meta["scale_quasar"] == pytest.approx(unreddened.meta["scale_quasar"])
    assert np.all(reddened.flux <= unreddened.flux + 1e-12)
    assert np.any(reddened.flux < unreddened.flux)


def test_host_disk_reddening_with_zero_av_faceon_has_no_effect():
    dust_free = HostDiskReddeningResult(cosi_disk=0.2, av_faceon=0.0, theta1_slope=0.5, euv_curvature=0.0)
    with_dust = blend_continua(_galaxy(1.0), _quasar(1.0), quasar_frac=0.5, host_disk=dust_free)
    without = blend_continua(_galaxy(1.0), _quasar(1.0), quasar_frac=0.5)
    np.testing.assert_allclose(with_dust.flux, without.flux)


def test_no_reddening_arguments_gives_unit_transmission():
    with_none = blend_continua(_galaxy(1.0), _quasar(1.0), quasar_frac=0.5)
    torus_transparent = TorusReddeningResult(
        intercepted=False, covering_angle_cosine=0.9, theta0_amplitude=1.0, theta1_slope=0.3, euv_curvature=0.0, cosi=0.95
    )
    with_transparent_torus = blend_continua(
        _galaxy(1.0), _quasar(1.0), quasar_frac=0.5, torus=torus_transparent
    )
    np.testing.assert_allclose(with_none.flux, with_transparent_torus.flux)


def test_rejects_zero_bolometric_luminosity_channel():
    zero_galaxy = _flat_channel(WAVE_GAL, 0.0)
    with pytest.raises(ValueError):
        blend_continua(zero_galaxy, _quasar(1.0), quasar_frac=0.5)


def test_provenance_meta_preserved():
    galaxy = _galaxy(2.0)
    galaxy = GalaxyContinuum(wave=galaxy.wave, flux=galaxy.flux, meta={"drawn_parameters": {"x": 1}})
    result = blend_continua(galaxy, _quasar(5.0), quasar_frac=0.5)
    assert result.meta["galaxy_meta"] == {"drawn_parameters": {"x": 1}}
    assert isinstance(result, CompositeContinuum)
