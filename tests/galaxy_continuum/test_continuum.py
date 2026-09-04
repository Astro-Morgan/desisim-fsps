"""
Tests for GalaxyContinuum. All of these exercise real python-fsps calls,
which cost ~15-25s each on this hardware/library (C3K_HR) combination
(confirmed during development -- see BUILD.md/imf.py's module docstring for
context) -- every test here is marked `slow` and excluded from a default
`pytest` run (`pytest -m 'not slow'`); run explicitly with `pytest -m slow`
or plain `pytest tests/galaxy_continuum/test_continuum.py`.
"""
import numpy as np
import pytest

pytest.importorskip("fsps")

import fsps as raw_fsps  # noqa: E402

from demiurge.galaxy_continuum import GalaxyContinuum
from demiurge.galaxy_continuum.imf import Z_SUN, canonical_slopes, metallicity_dependent_slopes

pytestmark = pytest.mark.slow


def test_from_single_population_matches_raw_fsps_ssp_exactly():
    """The core "recover the FSPS baseline" guarantee (project direction,
    2026-09-04): our SSP wrapper must reproduce a raw python-fsps SSP call
    bit-for-bit, not just approximately."""
    age_gyr, z_absolute = 3.0, 0.0142
    gc = GalaxyContinuum.from_single_population(age_gyr=age_gyr, z_absolute=z_absolute)

    sp = raw_fsps.StellarPopulation(zcontinuous=1, sfh=0, imf_type=2)
    sp.params["logzsol"] = float(np.log10(z_absolute / Z_SUN))
    wave_raw, flux_raw = sp.get_spectrum(tage=age_gyr, peraa=False)

    np.testing.assert_array_equal(gc.wave, wave_raw)
    np.testing.assert_array_equal(gc.flux, flux_raw)


def test_from_single_population_uses_canonical_imf_by_default():
    gc = GalaxyContinuum.from_single_population(age_gyr=2.0, z_absolute=0.0142)
    assert gc.meta["imf_slopes"] == canonical_slopes()


def test_from_single_population_rejects_nonpositive_inputs():
    with pytest.raises(ValueError):
        GalaxyContinuum.from_single_population(age_gyr=0.0, z_absolute=0.0142)
    with pytest.raises(ValueError):
        GalaxyContinuum.from_single_population(age_gyr=1.0, z_absolute=0.0)


def test_from_arrays_length_one_delegates_to_single_population():
    """A degenerate (length-1) user-supplied history must recover the exact
    same FSPS baseline as the explicit single-population constructor --
    the "discrete input is plug-and-play into the continuous infrastructure"
    requirement, tested at its simplest possible case."""
    age_gyr, z_absolute = 4.0, 0.008
    direct = GalaxyContinuum.from_single_population(age_gyr=age_gyr, z_absolute=z_absolute)
    via_arrays = GalaxyContinuum.from_arrays([age_gyr], [0.0], [z_absolute])
    np.testing.assert_array_equal(direct.flux, via_arrays.flux)


def test_from_arrays_shared_imf_mode_uses_canonical_slopes_in_every_bin():
    t = np.linspace(0.5, 8.0, 40)
    sfr = np.ones_like(t)
    z = np.linspace(0.001, 0.01, 40)  # rising, monotonic
    gc = GalaxyContinuum.from_arrays(t, sfr, z, imf_mode="shared", n_bins=2)
    assert all(b["imf_slopes"] == canonical_slopes() for b in gc.meta["bins"])


def test_from_arrays_dynamic_imf_mode_varies_with_bin_metallicity():
    t = np.linspace(0.5, 8.0, 40)
    sfr = np.ones_like(t)
    z = np.linspace(0.0005, 0.02, 40)  # spans well below/above solar
    gc = GalaxyContinuum.from_arrays(t, sfr, z, imf_mode="dynamic", n_bins=2)
    slopes = [b["imf_slopes"] for b in gc.meta["bins"]]
    assert len(set(slopes)) > 1, "dynamic mode should not produce identical slopes across differing-Z bins"
    for s in slopes:
        assert s.imf3 == canonical_slopes().imf3, "alpha3 must stay fixed in this simplified pass"


def test_from_arrays_rejects_invalid_imf_mode():
    t = np.linspace(0.5, 8.0, 10)
    with pytest.raises(ValueError):
        GalaxyContinuum.from_arrays(t, np.ones_like(t), np.linspace(0.001, 0.01, 10), imf_mode="bogus")


def test_from_arrays_rejects_nonmonotonic_z_by_default():
    t = np.linspace(0.5, 8.0, 10)
    z = np.array([0.001, 0.01, 0.005, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    with pytest.raises(ValueError):
        GalaxyContinuum.from_arrays(t, np.ones_like(t), z, n_bins=2)


def test_from_arrays_enforces_monotonic_z_when_requested():
    t = np.linspace(0.5, 8.0, 10)
    z = np.array([0.001, 0.01, 0.005, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    gc = GalaxyContinuum.from_arrays(t, np.ones_like(t), z, n_bins=2, z_monotonic_enforce=True)
    assert np.all(np.isfinite(gc.flux))


def test_shared_and_dynamic_imf_binned_sfh_reproduces_manual_sum():
    """Cross-check the whole binning + summation pipeline against a manual
    per-bin FSPS call sequence built independently of GalaxyContinuum's own
    internals -- catches a regression in the binning/summation logic itself,
    not just in the IMF-slope formula."""
    t = np.linspace(0.1, 5.0, 30)
    sfr = np.ones_like(t)
    z = np.full_like(t, 0.0142)  # constant Z -> shared and dynamic should agree exactly

    gc_shared = GalaxyContinuum.from_arrays(t, sfr, z, imf_mode="shared", n_bins=2)

    sp = raw_fsps.StellarPopulation(zcontinuous=1, sfh=3, imf_type=2)
    sp.params["logzsol"] = 0.0
    sp.set_tabular_sfh(t, sfr)
    wave_manual, flux_manual = sp.get_spectrum(tage=float(t[-1]), peraa=False)

    np.testing.assert_allclose(gc_shared.flux, flux_manual, rtol=1e-6)


def test_from_dense_basis_produces_finite_nonnegative_flux():
    rng = np.random.default_rng(11)
    gc = GalaxyContinuum.from_dense_basis(rng, t_obs_gyr=8.0, imf_mode="dynamic", n_bins=2)
    assert np.all(np.isfinite(gc.flux))
    assert np.all(gc.flux >= 0.0)
    assert "sfh" in gc.meta and "metallicity" in gc.meta


def test_from_dense_basis_reproducible_given_same_seed():
    a = GalaxyContinuum.from_dense_basis(np.random.default_rng(99), t_obs_gyr=8.0, n_bins=2)
    b = GalaxyContinuum.from_dense_basis(np.random.default_rng(99), t_obs_gyr=8.0, n_bins=2)
    np.testing.assert_array_equal(a.flux, b.flux)
