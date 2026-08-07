import unittest
import numpy as np

from desisim.agn_continuum import AGNPowerLawContinuum, DEFAULT_SLOPE_PRIORS, DEFAULT_BREAK_POINTS


class TestAGNPowerLawContinuum(unittest.TestCase):
    '''Tests for the simqso-wrapping AGN accretion-disk power-law
    continuum channel. See agn_continuum.py's module docstring for why
    this wraps simqso.sqgrids.BrokenPowerLawContinuumVar rather than
    reimplementing an independent power law, and for the documented
    upstream simqso seeding quirk this module works around.
    '''

    def setUp(self):
        self.wave = np.arange(1000.0, 12000.0, 5.0)
        self.agn = AGNPowerLawContinuum()

    def test_output_finite_and_positive(self):
        flux, wave, table = self.agn.spectrum(self.wave, seed=1)
        self.assertTrue(np.all(np.isfinite(flux)))
        self.assertTrue(np.all(flux > 0))
        self.assertEqual(flux.shape, self.wave.shape)
        np.testing.assert_array_equal(wave, self.wave)

    def test_slope_table_has_one_row_per_segment(self):
        _, _, table = self.agn.spectrum(self.wave, seed=1)
        self.assertEqual(len(table), len(DEFAULT_BREAK_POINTS) + 1)
        self.assertEqual(set(table.colnames), {'segment', 'alpha_nu'})

    def test_seed_reproducible_different_seed_not(self):
        '''This is the key regression test for the documented upstream
        simqso quirk: simqso's own seed= kwarg on
        BrokenPowerLawContinuumVar is a silent no-op (verified by direct
        inspection of its MRO), so reproducibility here depends entirely
        on this module's workaround of seeding the global numpy RNG
        itself before drawing slopes.'''
        f1, _, t1 = self.agn.spectrum(self.wave, seed=7)
        f2, _, t2 = self.agn.spectrum(self.wave, seed=7)
        f3, _, t3 = self.agn.spectrum(self.wave, seed=8)
        np.testing.assert_array_equal(f1, f2)
        np.testing.assert_array_equal(t1['alpha_nu'].data, t2['alpha_nu'].data)
        self.assertFalse(np.array_equal(f1, f3))

    def test_explicit_slopes_override_draw_and_are_echoed_in_table(self):
        explicit = np.array([[-1.5, -0.5, -0.37, -1.7, -1.03]])
        flux, _, table = self.agn.spectrum(self.wave, slopes=explicit)
        np.testing.assert_allclose(table['alpha_nu'].data, explicit[0])
        # explicit slopes must be reproducible with no seed at all
        flux2, _, table2 = self.agn.spectrum(self.wave, slopes=explicit)
        np.testing.assert_allclose(flux, flux2)

    def test_default_priors_match_forks_existing_simqso_model(self):
        '''DEFAULT_SLOPE_PRIORS/DEFAULT_BREAK_POINTS must match
        simqso.sqmodels.BossDr9_fiducial_continuum exactly -- the whole
        point of wrapping (rather than reimplementing) is numerical
        consistency with the model already driving this fork's existing
        SIMQSO.make_templates() QSO path.'''
        self.assertEqual(DEFAULT_BREAK_POINTS, [1100.0, 5700.0, 9730.0, 22300.0])
        expected_means = [-1.50, -0.50, -0.37, -1.70, -1.03]
        self.assertEqual([p['mean'] for p in DEFAULT_SLOPE_PRIORS], expected_means)
        self.assertTrue(all(p['sigma'] == 0.3 for p in DEFAULT_SLOPE_PRIORS))

    def test_custom_slope_priors_and_break_points(self):
        agn = AGNPowerLawContinuum(slope_priors=[dict(mean=-1.0, sigma=0.01),
                                                  dict(mean=-0.2, sigma=0.01)],
                                    break_points=[5000.0])
        _, _, table = agn.spectrum(self.wave, seed=1)
        self.assertEqual(len(table), 2)

    def test_mismatched_priors_and_break_points_raises(self):
        with self.assertRaises(ValueError):
            AGNPowerLawContinuum(slope_priors=[dict(mean=0.0, sigma=1.0)],
                                  break_points=[1000.0, 2000.0])

    def test_flux_norm_produces_finite_normalized_spectrum(self):
        from astropy.cosmology import Planck18
        flux_norm = dict(wavelength=1450.0, M_AB=-26.0,
                          DM=lambda z: Planck18.distmod(z).value)
        flux, _, _ = self.agn.spectrum(self.wave, z=1.0, flux_norm=flux_norm, seed=1)
        self.assertTrue(np.all(np.isfinite(flux)))
        self.assertTrue(np.all(flux > 0))


if __name__ == '__main__':
    unittest.main()
