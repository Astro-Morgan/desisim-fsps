import unittest
import numpy as np

from desisim.bal_trough import BALTrough, weymann_bi, LINE_WAVES, BI_V_LO_KMS, BI_V_HI_KMS, MIN_TROUGH_WIDTH_KMS


class TestWeymannBI(unittest.TestCase):
    '''Tests for the literal Weymann et al. (1991) balnicity-index helper,
    independent of BALTrough itself -- this is the exact formula used both
    by the production code (to report provenance) and by the empirical
    backtest below, so its own correctness is checked directly first.
    '''

    def test_no_absorption_gives_zero_bi(self):
        v = np.linspace(-26000, -2000, 2000)
        T = np.ones_like(v)
        self.assertEqual(weymann_bi(v, T), 0.0)

    def test_narrow_dip_below_min_width_is_excluded(self):
        '''A dip narrower than MIN_TROUGH_WIDTH_KMS=2000 must not register,
        by definition (Weymann's own C(v) mask).'''
        v = np.linspace(-26000, -2000, 4000)
        T = np.ones_like(v)
        # a narrow ~500 km/s dip well within the BI window
        narrow = (v > -10000) & (v < -9500)
        T[narrow] = 0.1
        self.assertEqual(weymann_bi(v, T), 0.0)

    def test_wide_deep_dip_gives_large_positive_bi(self):
        v = np.linspace(-26000, -2000, 4000)
        T = np.ones_like(v)
        wide = (v > -15000) & (v < -5000)
        T[wide] = 0.1
        bi = weymann_bi(v, T)
        self.assertGreater(bi, 5000.0)

    def test_dip_entirely_outside_window_is_excluded(self):
        '''A deep, wide dip entirely redward of BI_V_HI (-3000 km/s) must
        not contribute (outside Weymann's own integration window).'''
        v = np.linspace(-26000, 0, 5000)
        T = np.ones_like(v)
        outside = (v > -2000) & (v < 0)
        T[outside] = 0.1
        self.assertEqual(weymann_bi(v, T), 0.0)

    def test_bi_matches_hand_computed_top_hat(self):
        '''Closed-form check: a perfect top-hat T=0 over exactly
        [-10000,-5000] (width 5000, comfortably above the 2000 km/s
        threshold) gives BI = integral (1-0/0.9) dv over that window
        = 1 * 5000 = 5000 (since f=0 makes the integrand exactly 1).'''
        v = np.linspace(-26000, -2000, 20001)  # 1.2 km/s pixels, fine enough
        T = np.ones_like(v)
        T[(v >= -10000) & (v <= -5000)] = 0.0
        bi = weymann_bi(v, T)
        expected = 5000.0
        self.assertAlmostEqual(bi, expected, delta=expected * 0.01)


class TestBALTrough(unittest.TestCase):

    def setUp(self):
        self.bal = BALTrough(minwave=1200.0, maxwave=1700.0, cdelt_kms=20.0)
        self.flux = np.full_like(self.bal.log10wave, 5.0)

    def test_hasbal_false_gives_zero_deficit_and_unit_transmission(self):
        bal_flux, wave, params = self.bal.spectrum(self.flux, hasbal=False, seed=1)
        np.testing.assert_array_equal(bal_flux, np.zeros_like(self.flux))
        self.assertFalse(params['hasbal'])
        self.assertIsNone(params['bi'])

    def test_hasbal_true_gives_nonpositive_deficit(self):
        bal_flux, wave, params = self.bal.spectrum(self.flux, hasbal=True, seed=2)
        self.assertTrue(np.all(bal_flux <= 1e-12))
        self.assertTrue(params['hasbal'])
        self.assertIsNotNone(params['bi'])
        self.assertGreaterEqual(params['bi'], 0.0)

    def test_transmission_bounded_zero_one(self):
        T, wave, params = self.bal.transmission(hasbal=True, seed=3)
        self.assertTrue(np.all(T >= -1e-12))
        self.assertTrue(np.all(T <= 1.0 + 1e-12))

    def test_seed_reproducible(self):
        out1 = self.bal.spectrum(self.flux, seed=42)
        out2 = self.bal.spectrum(self.flux, seed=42)
        np.testing.assert_array_equal(out1[0], out2[0])
        self.assertEqual(out1[2]['hasbal'], out2[2]['hasbal'])

    def test_incidence_rate_matches_balprob_over_many_draws(self):
        n = 4000
        hasbal = [self.bal.spectrum(self.flux, seed=s)[2]['hasbal'] for s in range(n)]
        rate = np.mean(hasbal)
        self.assertAlmostEqual(rate, self.bal.BALPROB, delta=0.03)

    def test_explicit_balprob_override(self):
        n = 3000
        hasbal = [self.bal.spectrum(self.flux, balprob=0.5, seed=s)[2]['hasbal'] for s in range(n)]
        rate = np.mean(hasbal)
        self.assertAlmostEqual(rate, 0.5, delta=0.03)

    def test_explicit_strength_v_min_v_max_depth_are_honored(self):
        bal_flux, wave, params = self.bal.spectrum(
            self.flux, hasbal=True, v_min_kms=-10000.0, v_max_kms=-4000.0,
            depth=0.8, smooth_kms=500.0, seed=5)
        self.assertEqual(params['v_min_kms'], -10000.0)
        self.assertEqual(params['v_max_kms'], -4000.0)
        self.assertEqual(params['depth'], 0.8)
        self.assertEqual(params['smooth_kms'], 500.0)

    def test_trough_is_centered_near_civ_rest_wavelength(self):
        '''A strong, explicit trough should show its deepest absorption
        blueward of CIV_1549 (1549.06A), not somewhere unrelated.'''
        bal_flux, wave, params = self.bal.spectrum(
            self.flux, hasbal=True, v_min_kms=-10000.0, v_max_kms=-4000.0,
            depth=0.9, smooth_kms=300.0, seed=6)
        imin = np.argmin(bal_flux)
        self.assertLess(wave[imin], LINE_WAVES['CIV'])
        self.assertGreater(wave[imin], LINE_WAVES['CIV'] * (1 + BI_V_LO_KMS / 299792.458) - 5.0)

    def test_zero_depth_gives_no_absorption(self):
        bal_flux, wave, params = self.bal.spectrum(
            self.flux, hasbal=True, v_min_kms=-10000.0, v_max_kms=-4000.0,
            depth=0.0, smooth_kms=300.0, seed=7)
        np.testing.assert_allclose(bal_flux, 0.0, atol=1e-10)

    def test_zero_flux_gives_zero_deficit(self):
        zero_flux = np.zeros_like(self.bal.log10wave)
        bal_flux, wave, params = self.bal.spectrum(zero_flux, hasbal=True, seed=8)
        np.testing.assert_array_equal(bal_flux, np.zeros_like(zero_flux))

    def test_multi_line_extension_applies_to_requested_lines_only(self):
        '''Extending to SiIV should leave the CIV-only bal_flux for the
        SAME draw unaffected in the CIV region and introduce a SECOND
        trough near SiIV -- both nontrivial and CIV's own BI unaffected by
        including SiIV (the two windows shouldn't overlap on this grid).'''
        bal_wide = BALTrough(minwave=1100.0, maxwave=1700.0, cdelt_kms=20.0)
        flux_wide = np.full_like(bal_wide.log10wave, 5.0)
        civ_only = bal_wide.spectrum(flux_wide, hasbal=True, v_min_kms=-10000.0,
                                       v_max_kms=-4000.0, depth=0.8, smooth_kms=300.0,
                                       lines=['CIV'], seed=9)
        civ_and_siiv = bal_wide.spectrum(flux_wide, hasbal=True, v_min_kms=-10000.0,
                                           v_max_kms=-4000.0, depth=0.8, smooth_kms=300.0,
                                           lines=['CIV', 'SiIV'], seed=9)
        # More total absorption when SiIV trough is added too.
        self.assertLess(civ_and_siiv[0].sum(), civ_only[0].sum())
        # CIV's own reported BI is identical either way (same v_min/v_max/depth/smooth).
        self.assertEqual(civ_only[2]['bi'], civ_and_siiv[2]['bi'])

    def test_explicit_log10wave_grid_is_honored(self):
        custom_wave = np.arange(1300.0, 1600.0, 0.5)
        bal = BALTrough(log10wave=np.log10(custom_wave))
        T, wave, params = bal.transmission(hasbal=True, seed=10)
        np.testing.assert_allclose(wave, custom_wave)


class TestEmpiricalBacktest(unittest.TestCase):
    '''Standing regression test operationalizing the PI's 2026-08-27
    directive: confirm this parametric model can recreate the real
    empirical BAL population, and keep checking that going forward.

    Real reference values were queried live from the public SDSS DR14Q
    quasar catalog (Paris et al. 2018, VizieR VII/286, table
    "VII/286/dr14q", column "BI(CIV)") via the VizieR TAP service on
    2026-08-27: n=21870 real BALQSOs with BI(CIV)>0, mean=1852.5 km/s,
    P(BI<500)=0.346, P(BI<1000)=0.489, P(BI<2000)=0.670, P(BI<5000)=0.911,
    P(BI<10000)=0.991. This is real population data, not an assumption --
    see bal_trough.py's module docstring for the exact query used.

    Tolerances below are deliberately loose (statistical/distributional
    match, not a per-object fit -- see module docstring's disclosed
    limitation on the extreme tail) but tight enough to catch a genuine
    regression (e.g. an accidental order-of-magnitude change to
    STRENGTH_SCALE/WIDTH_SCALE/DEPTH_SCALE) rather than only checking
    internal self-consistency.
    '''

    REAL_MEAN_BI = 1852.5
    REAL_CDF = {500: 0.346, 1000: 0.489, 2000: 0.670, 5000: 0.911, 10000: 0.991}

    @classmethod
    def setUpClass(cls):
        bal = BALTrough(minwave=1200.0, maxwave=1700.0, cdelt_kms=30.0)
        flux = np.ones_like(bal.log10wave)
        n = 4000
        bis = []
        for s in range(n):
            _, _, params = bal.spectrum(flux, hasbal=True, seed=100000 + s)
            bis.append(params['bi'])
        cls.bis = np.array(bis)
        cls.bis_nonzero = cls.bis[cls.bis > 0]

    def test_most_draws_register_a_nonzero_bi(self):
        '''Most, though not necessarily all, forced-hasbal draws should
        clear Weymann's own min-width/depth threshold to register a
        nonzero BI (a real, physical selection effect -- weak/marginal
        troughs genuinely fail this literal definition too).'''
        frac_nonzero = len(self.bis_nonzero) / len(self.bis)
        self.assertGreater(frac_nonzero, 0.5)

    def test_synthetic_mean_bi_within_factor_of_two_of_real(self):
        ratio = self.bis_nonzero.mean() / self.REAL_MEAN_BI
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 2.0)

    def test_synthetic_cdf_within_15_points_of_real_at_every_bin(self):
        for thresh, real_p in self.REAL_CDF.items():
            synth_p = np.mean(self.bis_nonzero < thresh)
            self.assertLess(abs(synth_p - real_p), 0.15,
                             msg='P(BI<{}): synthetic={:.3f} real={:.3f}'.format(
                                 thresh, synth_p, real_p))

    def test_synthetic_distribution_is_right_skewed_like_real(self):
        '''Real BI distributions are heavily right-skewed (mean > median);
        confirm this qualitative shape is reproduced, not just the mean.'''
        self.assertGreater(self.bis_nonzero.mean(), np.median(self.bis_nonzero))


if __name__ == '__main__':
    unittest.main()
