import unittest
import numpy as np

from desisim.feii_continuum import FeIIPseudoContinuum, _bracket, _load_grid


class TestBracketHelper(unittest.TestCase):
    '''Unit tests for the trilinear-interpolation boundary-clipping helper
    (_bracket), independent of the grid data itself.'''

    def setUp(self):
        self.axis = np.array([10.0, 20.0, 30.0, 40.0])

    def test_interior_value_brackets_correctly(self):
        i0, i1, w0, w1 = _bracket(self.axis, 25.0)
        self.assertEqual((i0, i1), (1, 2))
        self.assertAlmostEqual(w0, 0.5)
        self.assertAlmostEqual(w1, 0.5)
        self.assertAlmostEqual(w0 * self.axis[i0] + w1 * self.axis[i1], 25.0)

    def test_exact_grid_point_gives_zero_weight_to_neighbor(self):
        i0, i1, w0, w1 = _bracket(self.axis, 20.0)
        self.assertAlmostEqual(w0 * self.axis[i0] + w1 * self.axis[i1], 20.0)

    def test_below_range_clips_to_lower_boundary(self):
        i0, i1, w0, w1 = _bracket(self.axis, -100.0)
        self.assertAlmostEqual(w0 * self.axis[i0] + w1 * self.axis[i1], 10.0)

    def test_above_range_clips_to_upper_boundary(self):
        i0, i1, w0, w1 = _bracket(self.axis, 1e6)
        self.assertAlmostEqual(w0 * self.axis[i0] + w1 * self.axis[i1], 40.0)


class TestGridLoads(unittest.TestCase):
    '''Sanity checks on the packaged grid data itself.'''

    def test_grid_loads_and_matches_documented_ranges(self):
        g = _load_grid()
        self.assertAlmostEqual(g['phi_grid'].min(), FeIIPseudoContinuum.PHI_RANGE[0])
        self.assertAlmostEqual(g['phi_grid'].max(), FeIIPseudoContinuum.PHI_RANGE[1])
        self.assertAlmostEqual(g['nH_grid'].min(), FeIIPseudoContinuum.LOGNH_RANGE[0])
        self.assertAlmostEqual(g['nH_grid'].max(), FeIIPseudoContinuum.LOGNH_RANGE[1])
        np.testing.assert_array_equal(g['turb_grid'], np.array(FeIIPseudoContinuum.TURB_VALUES))
        self.assertEqual(set(g['sed_labels']), set(FeIIPseudoContinuum.SED_SHAPES))

    def test_most_grid_cells_are_genuinely_measured(self):
        '''Regression check on the backfill: the vast majority of cells
        must be real CLOUDY runs, with only the documented handful of
        no-Fe-II-emission corners backfilled.'''
        g = _load_grid()
        frac_measured = g['measured'].mean()
        self.assertGreater(frac_measured, 0.9)

    def test_all_flux_values_finite_and_nonnegative(self):
        g = _load_grid()
        self.assertTrue(np.all(np.isfinite(g['flux'])))
        self.assertTrue(np.all(g['flux'] >= 0.0))


class TestFeIIPseudoContinuum(unittest.TestCase):
    '''Tests for the additive, independently-UV/optical-drawn Fe II
    pseudo-continuum channel. See feii_continuum.py's module docstring
    for the full physical/design rationale.
    '''

    def setUp(self):
        self.fe = FeIIPseudoContinuum(minwave=1200.0, maxwave=9000.0, cdelt_kms=40.0)

    def test_output_finite_and_correct_shape(self):
        flux, wave, params = self.fe.spectrum(seed=1)
        self.assertEqual(flux.shape, wave.shape)
        self.assertTrue(np.all(np.isfinite(flux)))
        self.assertTrue(np.all(flux >= -1e-25))  # additive emission, never negative

    def test_seed_reproducible_different_seed_not(self):
        f1, _, p1 = self.fe.spectrum(seed=5)
        f2, _, p2 = self.fe.spectrum(seed=5)
        f3, _, p3 = self.fe.spectrum(seed=6)
        np.testing.assert_array_equal(f1, f2)
        self.assertEqual(p1, p2)
        self.assertFalse(np.array_equal(f1, f3))

    def test_uv_and_optical_params_drawn_independently(self):
        '''Across many seeds, the UV band's and optical band's resolved
        physical parameters should not be forced equal to each other --
        confirms the decoupling is real, not accidentally correlated.'''
        mismatches = 0
        for seed in range(15):
            _, _, params = self.fe.spectrum(seed=seed)
            if (params['uv']['log_phi'], params['uv']['log_nH'], params['uv']['turb'], params['uv']['sed']) != \
               (params['optical']['log_phi'], params['optical']['log_nH'], params['optical']['turb'], params['optical']['sed']):
                mismatches += 1
        self.assertGreater(mismatches, 0)

    def test_explicit_band_params_always_win(self):
        uv_params = dict(log_phi=19.0, log_nH=11.0, turb=20.0, sed='AGN_SED')
        opt_params = dict(log_phi=20.0, log_nH=12.0, turb=50.0, sed='Intermediate_SED')
        _, _, params = self.fe.spectrum(uv_params=uv_params, optical_params=opt_params, seed=1)
        for key in ('log_phi', 'log_nH', 'turb', 'sed'):
            self.assertEqual(params['uv'][key], uv_params[key])
            self.assertEqual(params['optical'][key], opt_params[key])

    def test_out_of_range_params_clip_not_extrapolate(self):
        uv_params = dict(log_phi=1000.0, log_nH=-1000.0, turb=0.0, sed='AGN_SED')
        flux, _, params = self.fe.spectrum(uv_params=uv_params, uv_norm=1.0, optical_norm=0.0, seed=1)
        self.assertTrue(np.all(np.isfinite(flux)))
        # Clipped values recorded verbatim (the request), but the
        # interpolation itself must have pinned to the grid boundary --
        # verified indirectly via measured=False (that corner is the
        # known excluded/backfilled one).
        self.assertEqual(params['uv']['log_phi'], 1000.0)
        self.assertFalse(params['uv']['measured'])

    def test_unknown_sed_raises(self):
        with self.assertRaises(ValueError):
            self.fe.spectrum(uv_params=dict(sed='NotARealSED'), seed=1)

    def test_zero_norm_band_contributes_nothing(self):
        '''Setting one band's norm to exactly 0 must remove its
        contribution entirely (additive independence between bands).'''
        common = dict(uv_params=dict(log_phi=19.0, log_nH=11.0, turb=20.0, sed='AGN_SED'),
                      optical_params=dict(log_phi=19.0, log_nH=11.0, turb=20.0, sed='AGN_SED'),
                      sigma_kms=500.0, velshift_kms=0.0, seed=1)
        flux_both, wave, _ = self.fe.spectrum(uv_norm=1.0, optical_norm=1.0, **common)
        flux_uv_only, _, _ = self.fe.spectrum(uv_norm=1.0, optical_norm=0.0, **common)
        flux_opt_only, _, _ = self.fe.spectrum(uv_norm=0.0, optical_norm=1.0, **common)
        np.testing.assert_allclose(flux_both, flux_uv_only + flux_opt_only, atol=1e-25)

    def test_explicit_norm_scales_peak_linearly(self):
        common = dict(uv_params=dict(log_phi=19.0, log_nH=11.0, turb=0.0, sed='AGN_SED'),
                      optical_norm=0.0, sigma_kms=500.0, velshift_kms=0.0, seed=1)
        flux_a, _, _ = self.fe.spectrum(uv_norm=1.0, **common)
        flux_b, _, _ = self.fe.spectrum(uv_norm=3.0, **common)
        np.testing.assert_allclose(flux_b, flux_a * 3.0, atol=1e-25)

    def test_explicit_sigma_and_velshift_always_win(self):
        _, _, params = self.fe.spectrum(sigma_kms=777.0, velshift_kms=-123.0, seed=1)
        self.assertEqual(params['sigma_kms'], 777.0)
        self.assertEqual(params['velshift_kms'], -123.0)

    def test_sigma_kms_and_velshift_draw_within_range_and_reproducible(self):
        _, _, pA = self.fe.spectrum(seed=10)
        _, _, pA2 = self.fe.spectrum(seed=10)
        self.assertAlmostEqual(pA['sigma_kms'], pA2['sigma_kms'], places=8)
        self.assertGreaterEqual(pA['sigma_kms'], FeIIPseudoContinuum.SIGMA_KMS_RANGE[0])
        self.assertLessEqual(pA['sigma_kms'], FeIIPseudoContinuum.SIGMA_KMS_RANGE[1])
        self.assertGreaterEqual(pA['velshift_kms'], FeIIPseudoContinuum.VELSHIFT_KMS_RANGE[0])
        self.assertLessEqual(pA['velshift_kms'], FeIIPseudoContinuum.VELSHIFT_KMS_RANGE[1])

    def test_larger_sigma_reduces_peak_amplitude(self):
        '''Broadening at fixed integrated-ish scale must smear the peak
        down -- a basic physical sanity check on the convolution step.'''
        common = dict(uv_params=dict(log_phi=19.0, log_nH=11.0, turb=0.0, sed='AGN_SED'),
                      optical_norm=0.0, uv_norm=1.0, velshift_kms=0.0, seed=1)
        flux_narrow, _, _ = self.fe.spectrum(sigma_kms=FeIIPseudoContinuum.SIGMA_KMS_RANGE[0], **common)
        flux_wide, _, _ = self.fe.spectrum(sigma_kms=FeIIPseudoContinuum.SIGMA_KMS_RANGE[1], **common)
        self.assertGreater(flux_narrow.max(), flux_wide.max())

    def test_velshift_moves_flux_redward_or_blueward(self):
        common = dict(uv_params=dict(log_phi=19.0, log_nH=11.0, turb=0.0, sed='AGN_SED'),
                      optical_norm=0.0, uv_norm=1.0, sigma_kms=FeIIPseudoContinuum.SIGMA_KMS_RANGE[0], seed=1)
        flux0, wave, _ = self.fe.spectrum(velshift_kms=0.0, **common)
        flux_pos, _, _ = self.fe.spectrum(velshift_kms=400.0, **common)
        # Centroid (flux-weighted mean wavelength) must shift redward.
        centroid0 = np.sum(wave * flux0) / np.sum(flux0)
        centroid_pos = np.sum(wave * flux_pos) / np.sum(flux_pos)
        self.assertGreater(centroid_pos, centroid0)

    def test_explicit_log10wave_grid_is_respected(self):
        custom_grid = np.linspace(np.log10(2000.0), np.log10(6000.0), 3000)
        fe = FeIIPseudoContinuum(log10wave=custom_grid)
        flux, wave, _ = fe.spectrum(seed=1)
        np.testing.assert_array_equal(wave, 10 ** custom_grid)

    def test_zshift_moves_output_wavelength_grid_content(self):
        common = dict(uv_params=dict(log_phi=19.0, log_nH=11.0, turb=0.0, sed='AGN_SED'),
                      optical_norm=0.0, uv_norm=1.0, sigma_kms=FeIIPseudoContinuum.SIGMA_KMS_RANGE[0],
                      velshift_kms=0.0, seed=1)
        flux0, wave, _ = self.fe.spectrum(zshift=0.0, **common)
        flux_z, _, _ = self.fe.spectrum(zshift=0.05, **common)
        centroid0 = np.sum(wave * flux0) / np.sum(flux0)
        centroid_z = np.sum(wave * flux_z) / np.sum(flux_z)
        self.assertGreater(centroid_z, centroid0)


if __name__ == '__main__':
    unittest.main()
