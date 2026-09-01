import unittest
import numpy as np

from desisim.balmer_continuum import BalmerContinuum, _bracket, _bilinear_ratio, _planck_lambda, _load_grid


class TestBracketHelper(unittest.TestCase):
    '''Same boundary-clipping contract as feii_continuum._bracket -- see
    that module's tests for the rationale; duplicated here since this
    module intentionally does not import feii_continuum's copy (module
    self-containment, matching the project's existing per-module style).
    '''

    def setUp(self):
        self.axis = np.array([5.0, 8.0, 11.0, 14.0])

    def test_interior_value_brackets_correctly(self):
        i0, i1, w0, w1 = _bracket(self.axis, 9.5)
        self.assertEqual((i0, i1), (1, 2))
        self.assertAlmostEqual(w0 * self.axis[i0] + w1 * self.axis[i1], 9.5)

    def test_below_range_clips_to_lower_boundary(self):
        i0, i1, w0, w1 = _bracket(self.axis, -100.0)
        self.assertAlmostEqual(w0 * self.axis[i0] + w1 * self.axis[i1], 5.0)

    def test_above_range_clips_to_upper_boundary(self):
        i0, i1, w0, w1 = _bracket(self.axis, 1e6)
        self.assertAlmostEqual(w0 * self.axis[i0] + w1 * self.axis[i1], 14.0)


class TestGridLoads(unittest.TestCase):

    def test_grid_loads_and_matches_documented_ranges(self):
        g = _load_grid()
        self.assertAlmostEqual(g['dens_grid'].min(), 2.0)
        self.assertAlmostEqual(g['dens_grid'].max(), 14.0)
        self.assertIn(15000.0, g['temp_grid'])
        self.assertEqual(g['ratio'].shape[0], len(g['names']))
        self.assertEqual(g['ratio'].shape[1], g['dens_grid'].size)
        self.assertEqual(g['ratio'].shape[2], g['temp_grid'].size)

    def test_49_lines_lyalpha_plus_balmer_3_through_50(self):
        g = _load_grid()
        self.assertEqual(len(g['names']), 49)
        self.assertIn('Lyalpha', g['names'])
        self.assertIn('Balmer_n3', g['names'])
        self.assertIn('Balmer_n50', g['names'])

    def test_hbeta_ratio_identically_one_everywhere(self):
        g = _load_grid()
        hbeta_idx = g['names'].index('Balmer_n4')
        np.testing.assert_allclose(g['ratio'][hbeta_idx], 1.0)

    def test_all_ratios_finite_and_nonnegative(self):
        g = _load_grid()
        self.assertTrue(np.all(np.isfinite(g['ratio'])))
        self.assertTrue(np.all(g['ratio'] >= 0.0))

    def test_wavelengths_increasing_and_capped_at_series_limit(self):
        '''Every line wavelength must be >= 3646A (the series limit) and
        < Lyalpha's, with Balmer wavelengths approaching (but never
        reaching) 3646A as n grows.'''
        g = _load_grid()
        balmer_waves = np.array([g['wave'][g['names'].index('Balmer_n%d' % n)] for n in range(3, 51)])
        self.assertTrue(np.all(np.diff(balmer_waves) < 0))  # strictly decreasing with n
        self.assertTrue(np.all(balmer_waves > 3646.0))
        self.assertLess(balmer_waves[-1], 3660.0)  # n=50 should be very close to the limit


class TestBilinearRatio(unittest.TestCase):

    def test_matches_grid_at_exact_grid_point(self):
        g = _load_grid()
        out = _bilinear_ratio(g['ratio'], g['dens_grid'], g['temp_grid'], g['dens_grid'][3], g['temp_grid'][5])
        np.testing.assert_allclose(out, g['ratio'][:, 3, 5])

    def test_clips_rather_than_extrapolates(self):
        g = _load_grid()
        out_far = _bilinear_ratio(g['ratio'], g['dens_grid'], g['temp_grid'], -1000.0, -1000.0)
        out_boundary = _bilinear_ratio(g['ratio'], g['dens_grid'], g['temp_grid'],
                                        g['dens_grid'][0], g['temp_grid'][0])
        np.testing.assert_allclose(out_far, out_boundary)


class TestPlanckLambda(unittest.TestCase):

    def test_positive_and_finite_across_optical_range(self):
        wave = np.linspace(1000.0, 5000.0, 200)
        b = _planck_lambda(wave, 15000.0)
        self.assertTrue(np.all(np.isfinite(b)))
        self.assertTrue(np.all(b > 0))

    def test_peak_shifts_blueward_for_hotter_temperature(self):
        wave = np.linspace(500.0, 20000.0, 5000)
        b_cool = _planck_lambda(wave, 8000.0)
        b_hot = _planck_lambda(wave, 25000.0)
        self.assertLess(wave[np.argmax(b_hot)], wave[np.argmax(b_cool)])


class TestBalmerContinuum(unittest.TestCase):

    def setUp(self):
        self.bc = BalmerContinuum(minwave=1200.0, maxwave=9000.0, cdelt_kms=40.0)

    def test_output_finite_shape_and_nonnegative(self):
        flux, wave, params = self.bc.spectrum(seed=1, backend='numpy')
        self.assertEqual(flux.shape, wave.shape)
        self.assertTrue(np.all(np.isfinite(flux)))
        self.assertTrue(np.all(flux >= -1e-25))

    def test_seed_reproducible_different_seed_not(self):
        f1, _, p1 = self.bc.spectrum(seed=5, backend='numpy')
        f2, _, p2 = self.bc.spectrum(seed=5, backend='numpy')
        f3, _, p3 = self.bc.spectrum(seed=6, backend='numpy')
        np.testing.assert_array_equal(f1, f2)
        self.assertEqual(p1, p2)
        self.assertFalse(np.array_equal(f1, f3))

    def test_explicit_params_always_win(self):
        _, _, p = self.bc.spectrum(T_e=17000.0, log_ne=9.0, tau_BE=0.5, sigma_kms=900.0,
                                    velshift_kms=-50.0, seed=1, backend='numpy')
        self.assertEqual(p['T_e'], 17000.0)
        self.assertEqual(p['log_ne'], 9.0)
        self.assertEqual(p['tau_BE'], 0.5)
        self.assertEqual(p['sigma_kms'], 900.0)
        self.assertEqual(p['velshift_kms'], -50.0)

    def test_draws_within_documented_ranges_and_reproducible(self):
        _, _, p = self.bc.spectrum(seed=10, backend='numpy')
        _, _, p2 = self.bc.spectrum(seed=10, backend='numpy')
        self.assertEqual(p, p2)
        self.assertGreaterEqual(p['T_e'], BalmerContinuum.TE_RANGE[0])
        self.assertLessEqual(p['T_e'], BalmerContinuum.TE_RANGE[1])
        self.assertGreaterEqual(p['log_ne'], BalmerContinuum.LOGNE_RANGE[0])
        self.assertLessEqual(p['log_ne'], BalmerContinuum.LOGNE_RANGE[1])
        self.assertGreaterEqual(p['tau_BE'], BalmerContinuum.TAU_BE_RANGE[0])
        self.assertLessEqual(p['tau_BE'], BalmerContinuum.TAU_BE_RANGE[1])
        self.assertGreaterEqual(p['sigma_kms'], BalmerContinuum.SIGMA_KMS_RANGE[0])
        self.assertLessEqual(p['sigma_kms'], BalmerContinuum.SIGMA_KMS_RANGE[1])
        self.assertGreaterEqual(p['velshift_kms'], BalmerContinuum.VELSHIFT_KMS_RANGE[0])
        self.assertLessEqual(p['velshift_kms'], BalmerContinuum.VELSHIFT_KMS_RANGE[1])

    def test_zero_edge_norm_removes_edge_piece_only(self):
        '''With edge_norm=0, flux blueward of 3646A (only reachable by the
        free-bound edge, modulo broadening leakage) should vanish for a
        narrow enough sigma; use the narrowest allowed width to minimize
        leakage across the edge.'''
        common = dict(T_e=15000.0, log_ne=10.0, tau_BE=1.0, line_norm=1.0,
                      sigma_kms=BalmerContinuum.SIGMA_KMS_RANGE[0], velshift_kms=0.0, seed=1, backend='numpy')
        flux_with_edge, wave, _ = self.bc.spectrum(edge_norm=1.0, **common)
        flux_no_edge, _, _ = self.bc.spectrum(edge_norm=0.0, **common)
        # A window strictly between Lyalpha (~1216A, part of the *line*
        # series) and the edge (3646A) -- reachable only by piece (A).
        deep_blue = (wave > 3000.0) & (wave < 3550.0)
        np.testing.assert_allclose(flux_no_edge[deep_blue], 0.0, atol=1e-25)
        self.assertTrue(np.any(flux_with_edge[deep_blue] > 0))

    def test_zero_line_norm_removes_line_series_only(self):
        common = dict(T_e=15000.0, log_ne=10.0, tau_BE=1.0, edge_norm=0.0,
                      sigma_kms=BalmerContinuum.SIGMA_KMS_RANGE[0], velshift_kms=0.0, seed=1, backend='numpy')
        flux_with_lines, wave, _ = self.bc.spectrum(line_norm=1.0, **common)
        flux_no_lines, _, _ = self.bc.spectrum(line_norm=0.0, **common)
        far_red = wave > 5000.0  # only Lyalpha/Balmer lines reach here, not the edge
        np.testing.assert_allclose(flux_no_lines[far_red], 0.0, atol=1e-25)
        self.assertTrue(np.any(flux_with_lines[far_red] > 0))

    def test_line_norm_scales_linearly(self):
        common = dict(T_e=15000.0, log_ne=10.0, tau_BE=1.0, edge_norm=0.0,
                      sigma_kms=500.0, velshift_kms=0.0, seed=1, backend='numpy')
        f_a, _, _ = self.bc.spectrum(line_norm=1.0, **common)
        f_b, _, _ = self.bc.spectrum(line_norm=4.0, **common)
        np.testing.assert_allclose(f_b, f_a * 4.0, atol=1e-25)

    def test_edge_norm_scales_linearly(self):
        common = dict(T_e=15000.0, log_ne=10.0, tau_BE=1.0, line_norm=0.0,
                      sigma_kms=500.0, velshift_kms=0.0, seed=1, backend='numpy')
        f_a, _, _ = self.bc.spectrum(edge_norm=1.0, **common)
        f_b, _, _ = self.bc.spectrum(edge_norm=3.0, **common)
        np.testing.assert_allclose(f_b, f_a * 3.0, atol=1e-25)

    def test_larger_sigma_reduces_line_peak(self):
        common = dict(T_e=15000.0, log_ne=10.0, tau_BE=1.0, edge_norm=0.0, line_norm=1.0,
                      velshift_kms=0.0, seed=1, backend='numpy')
        f_narrow, _, _ = self.bc.spectrum(sigma_kms=BalmerContinuum.SIGMA_KMS_RANGE[0], **common)
        f_wide, _, _ = self.bc.spectrum(sigma_kms=BalmerContinuum.SIGMA_KMS_RANGE[1], **common)
        self.assertGreater(f_narrow.max(), f_wide.max())

    def test_velshift_moves_flux_redward(self):
        common = dict(T_e=15000.0, log_ne=10.0, tau_BE=1.0, edge_norm=0.0, line_norm=1.0,
                      sigma_kms=BalmerContinuum.SIGMA_KMS_RANGE[0], seed=1, backend='numpy')
        flux0, wave, _ = self.bc.spectrum(velshift_kms=0.0, **common)
        flux_pos, _, _ = self.bc.spectrum(velshift_kms=150.0, **common)
        centroid0 = np.sum(wave * flux0) / np.sum(flux0)
        centroid_pos = np.sum(wave * flux_pos) / np.sum(flux_pos)
        self.assertGreater(centroid_pos, centroid0)

    def test_different_log_ne_changes_line_ratios(self):
        '''Sanity check that (T_e, log_ne) actually drives the line
        shape, not just a no-op passthrough.'''
        common = dict(T_e=15000.0, tau_BE=1.0, edge_norm=0.0, line_norm=1.0,
                      sigma_kms=500.0, velshift_kms=0.0, seed=1, backend='numpy')
        f_lowne, _, _ = self.bc.spectrum(log_ne=5.0, **common)
        f_highne, _, _ = self.bc.spectrum(log_ne=14.0, **common)
        self.assertFalse(np.allclose(f_lowne, f_highne))

    def test_out_of_range_explicit_params_clip_not_crash(self):
        flux, _, p = self.bc.spectrum(T_e=1e9, log_ne=-1e9, tau_BE=1.0, seed=1, backend='numpy')
        self.assertTrue(np.all(np.isfinite(flux)))
        # clipped values recorded verbatim in params (the explicit request)
        self.assertEqual(p['T_e'], 1e9)
        self.assertEqual(p['log_ne'], -1e9)

    def test_numpy_and_torch_backends_agree(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest('torch not installed')
        common = dict(T_e=15000.0, log_ne=10.0, tau_BE=1.0, edge_norm=1.0, line_norm=1.0,
                      sigma_kms=800.0, velshift_kms=20.0, seed=1)
        f_numpy, _, _ = self.bc.spectrum(backend='numpy', **common)
        f_torch, _, _ = self.bc.spectrum(backend='torch', **common)
        np.testing.assert_allclose(f_numpy, f_torch, atol=1e-20, rtol=1e-6)


    def test_explicit_edge_norm_still_overrides_default(self):
        '''Task #41 backward compatibility: an explicit edge_norm must
        win over the new continuity-condition default exactly as it
        already won over the old implicit peak=1.0 default.'''
        flux_a, _, params_a = self.bc.spectrum(edge_norm=3.0, line_norm=1.0, T_e=15000.0,
                                                log_ne=10.0, tau_BE=1.0, seed=1, backend='numpy')
        flux_b, _, params_b = self.bc.spectrum(edge_norm=3.0, line_norm=1.0, T_e=15000.0,
                                                log_ne=10.0, tau_BE=1.0, seed=1, backend='numpy')
        np.testing.assert_array_equal(flux_a, flux_b)
        self.assertAlmostEqual(params_a['edge_norm'], 3.0, places=10)

    def test_default_line_norm_is_broad_narrow_hbeta_ratio(self):
        '''Task #41: with line_norm=None, the resolved value recorded in
        params must equal the new literature-anchored default, not the
        old placeholder of 1.0.'''
        _, _, params = self.bc.spectrum(edge_norm=0.0, T_e=15000.0, log_ne=10.0, tau_BE=1.0,
                                         seed=1, backend='numpy')
        self.assertAlmostEqual(params['line_norm'], BalmerContinuum.BROAD_NARROW_HBETA_RATIO, places=10)
        self.assertNotEqual(params['line_norm'], 1.0)

    def test_default_edge_norm_is_continuous_with_line_series_at_the_edge(self):
        '''Task #41: with edge_norm=None, the free-bound edge piece and
        the high-order line series must be continuous at lambda=3646A
        (Kovacevic, Popovic & Kollatschny 2015 continuity condition) --
        checked directly on the two pieces' own analytic/interpolated
        values at that wavelength, at zero velocity shift so the "shifted"
        and "rest-frame" edge wavelengths coincide exactly.'''
        from desisim.balmer_continuum import _planck_lambda, _bilinear_ratio
        T_e, log_ne, tau_BE = 15000.0, 10.0, 1.0
        flux, wave, params = self.bc.spectrum(T_e=T_e, log_ne=log_ne, tau_BE=tau_BE,
                                               line_norm=2.5, velshift_kms=0.0, zshift=0.0,
                                               seed=1, backend='numpy')
        edge_scale = params['edge_norm']
        edge_peak = np.where(wave <= BalmerContinuum.LAMBDA_BE,
                              _planck_lambda(wave, T_e) * (1.0 - np.exp(-tau_BE * (wave / BalmerContinuum.LAMBDA_BE) ** 3)),
                              0.0).max()
        edge_raw_at_edge = _planck_lambda(BalmerContinuum.LAMBDA_BE, T_e) * (1.0 - np.exp(-tau_BE))
        edge_value_at_edge = edge_scale * edge_raw_at_edge / edge_peak

        line_only, _, _ = self.bc.spectrum(T_e=T_e, log_ne=log_ne, tau_BE=tau_BE, edge_norm=0.0,
                                            line_norm=2.5, velshift_kms=0.0, zshift=0.0,
                                            seed=1, backend='numpy')
        line_value_at_edge = np.interp(BalmerContinuum.LAMBDA_BE, wave, line_only)

        self.assertAlmostEqual(edge_value_at_edge, line_value_at_edge, delta=1e-6)

    def test_default_edge_norm_tracks_explicit_line_norm(self):
        '''The continuity default must respond to whatever line_norm is
        actually in effect, not just its own literature default -- i.e.
        scaling line_norm up should scale the derived edge_norm up by the
        same factor.'''
        _, _, params_a = self.bc.spectrum(T_e=15000.0, log_ne=10.0, tau_BE=1.0, line_norm=1.0,
                                           seed=1, backend='numpy')
        _, _, params_b = self.bc.spectrum(T_e=15000.0, log_ne=10.0, tau_BE=1.0, line_norm=4.0,
                                           seed=1, backend='numpy')
        self.assertAlmostEqual(params_b['edge_norm'] / params_a['edge_norm'], 4.0, places=6)


if __name__ == '__main__':
    unittest.main()
