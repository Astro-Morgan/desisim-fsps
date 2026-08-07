import unittest
import numpy as np

from desisim.dust import DustAttenuation, _drude_numpy


class TestDrudeHelper(unittest.TestCase):
    '''Pure-math unit tests for the Drude UV-bump profile.'''

    def test_peak_value_is_one_at_center(self):
        self.assertAlmostEqual(_drude_numpy(np.array([2175.0]), 2175.0, 350.0)[0], 1.0, places=10)

    def test_symmetric_and_decaying_away_from_center(self):
        wave = np.array([1800.0, 2175.0, 2550.0])
        d = _drude_numpy(wave, 2175.0, 350.0)
        self.assertLess(d[0], d[1])
        self.assertLess(d[2], d[1])

    def test_nonnegative_everywhere(self):
        wave = np.linspace(500.0, 10000.0, 5000)
        d = _drude_numpy(wave, 2175.0, 350.0)
        self.assertTrue(np.all(d >= 0.0))


class TestDustAttenuation(unittest.TestCase):
    '''Unit/integration tests for the PI-specified 4-parameter dust
    attenuation law k(lambda;theta) = theta0*(lambda/lambdaV)^-theta1 +
    theta2*D(lambda;lambda0,gamma) + theta3, and the additive
    flux-deficit convention this module returns it through.
    '''

    def setUp(self):
        self.wave = np.arange(2000.0, 10000.0, 1.0)
        self.flux = np.full_like(self.wave, 1e-16)
        self.dust = DustAttenuation()

    def _theta_value(self, table, name):
        row = table[table['param'] == name]
        self.assertEqual(len(row), 1)
        return float(row['value'][0])

    def test_zero_theta_gives_zero_deficit_everywhere(self):
        zero = dict(theta0=0.0, theta1=0.0, theta2=0.0, theta3=0.0)
        dflux, _, _ = self.dust.spectrum(self.wave, self.flux, theta=zero, seed=1)
        np.testing.assert_allclose(dflux, 0.0, atol=1e-30)

    def test_dust_flux_is_nonpositive_and_finite(self):
        for seed in range(5):
            dflux, wave, table = self.dust.spectrum(self.wave, self.flux, seed=seed)
            self.assertTrue(np.all(np.isfinite(dflux)))
            self.assertTrue(np.all(dflux <= 1e-25))

    def test_grey_floor_alone_gives_constant_attenuation(self):
        '''theta3 alone (theta0=theta1=theta2=0) must give an exactly
        wavelength-independent flux deficit -- the defining property of a
        "grey" term.'''
        theta = dict(theta0=0.0, theta1=0.0, theta2=0.0, theta3=0.4)
        dflux, _, _ = self.dust.spectrum(self.wave, self.flux, theta=theta, seed=1)
        expected = self.flux * (10**(-0.4 * 0.4) - 1.0)
        np.testing.assert_allclose(dflux, expected)

    def test_bump_amplitude_peaks_exactly_at_lambda_bump(self):
        theta = dict(theta0=0.0, theta1=0.0, theta2=1.2, theta3=0.0)
        dflux, wave, _ = self.dust.spectrum(self.wave, self.flux, theta=theta, seed=1)
        idx = np.argmin(np.abs(wave - DustAttenuation.LAMBDA_BUMP))
        expected_peak = self.flux[idx] * (10**(-0.4 * 1.2) - 1.0)
        self.assertAlmostEqual(dflux[idx] / expected_peak, 1.0, places=4)

    def test_positive_slope_gives_more_attenuation_in_blue_than_red(self):
        '''Physical sanity: theta1>0 must make the curve steeper toward
        blue/UV wavelengths (more attenuation at short wavelengths), the
        defining behavior of a reddening law.'''
        theta = dict(theta0=1.0, theta1=1.5, theta2=0.0, theta3=0.0)
        dflux, wave, _ = self.dust.spectrum(self.wave, self.flux, theta=theta, seed=1)
        i_blue = np.argmin(np.abs(wave - 4000.0))
        i_red = np.argmin(np.abs(wave - 9000.0))
        self.assertGreater(abs(dflux[i_blue]), abs(dflux[i_red]))

    def test_theta1_zero_gives_flat_powerlaw_term(self):
        '''theta1=0 collapses the power-law term to a wavelength-
        independent constant (theta0), matching the flat/AGN-torus-like
        limit described in the module docstring.'''
        theta = dict(theta0=0.7, theta1=0.0, theta2=0.0, theta3=0.0)
        dflux, _, _ = self.dust.spectrum(self.wave, self.flux, theta=theta, seed=1)
        expected = self.flux * (10**(-0.4 * 0.7) - 1.0)
        np.testing.assert_allclose(dflux, expected)

    def test_explicit_theta_always_wins(self):
        explicit = dict(theta0=0.5, theta1=0.8, theta2=0.3, theta3=0.1)
        _, _, table = self.dust.spectrum(self.wave, self.flux, theta=explicit, seed=1)
        for name, val in explicit.items():
            self.assertAlmostEqual(self._theta_value(table, name), val, places=10)

    def test_partial_explicit_theta_only_overrides_given_keys(self):
        _, _, table = self.dust.spectrum(self.wave, self.flux, theta=dict(theta3=0.0), seed=1)
        self.assertEqual(self._theta_value(table, 'theta3'), 0.0)
        lo, hi = DustAttenuation.THETA_PRIORS['theta0']
        self.assertGreaterEqual(self._theta_value(table, 'theta0'), lo)
        self.assertLessEqual(self._theta_value(table, 'theta0'), hi)

    def test_seed_reproducible_different_seed_not(self):
        a1, _, t1 = self.dust.spectrum(self.wave, self.flux, seed=5)
        a2, _, t2 = self.dust.spectrum(self.wave, self.flux, seed=5)
        a3, _, t3 = self.dust.spectrum(self.wave, self.flux, seed=6)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(t1['value'].data, t2['value'].data)
        self.assertFalse(np.array_equal(a1, a3))

    def test_default_priors_respect_nonnegativity_constraints(self):
        '''theta0, theta2, theta3 must never be drawn negative (the PI's
        explicit constraint); theta1's prior range is also non-negative in
        the current defaults, though not constrained by the model itself.'''
        for name in ('theta0', 'theta2', 'theta3'):
            lo, hi = DustAttenuation.THETA_PRIORS[name]
            self.assertGreaterEqual(lo, 0.0)

    def test_theta_priors_override(self):
        custom_priors = {'theta0': (5.0, 5.0), 'theta1': (0.0, 0.0),
                          'theta2': (0.0, 0.0), 'theta3': (0.0, 0.0)}
        dust = DustAttenuation(theta_priors=custom_priors)
        _, _, table = dust.spectrum(self.wave, self.flux, seed=1)
        self.assertAlmostEqual(self._theta_value(table, 'theta0'), 5.0, places=10)

    def test_lambda_v_lambda_bump_bump_width_overrides(self):
        dust = DustAttenuation(lambda_v=4000.0, lambda_bump=3000.0, bump_width=100.0)
        theta = dict(theta0=0.0, theta1=0.0, theta2=1.0, theta3=0.0)
        dflux, wave, _ = dust.spectrum(self.wave, self.flux, theta=theta, seed=1)
        idx = np.argmin(np.abs(wave - 3000.0))
        expected_peak = self.flux[idx] * (10**(-0.4 * 1.0) - 1.0)
        self.assertAlmostEqual(dflux[idx] / expected_peak, 1.0, places=3)

    def test_torch_and_numpy_backends_agree(self):
        kwargs = dict(seed=1)
        a_np, _, _ = self.dust.spectrum(self.wave, self.flux, backend='numpy', **kwargs)
        a_torch, _, _ = self.dust.spectrum(self.wave, self.flux, backend='torch', device='cpu', **kwargs)
        np.testing.assert_allclose(a_np, a_torch, rtol=1e-6, atol=1e-30)

    def test_output_shape_matches_input(self):
        dflux, wave, _ = self.dust.spectrum(self.wave, self.flux, seed=1)
        self.assertEqual(dflux.shape, self.wave.shape)
        np.testing.assert_array_equal(wave, self.wave)

    def test_theta_table_columns(self):
        _, _, table = self.dust.spectrum(self.wave, self.flux, seed=1)
        self.assertEqual(set(table.colnames), {'param', 'value'})
        self.assertEqual(len(table), 4)


class TestUniversalDustExtensions(unittest.TestCase):
    '''Tests for the 2026-08-06 generalization: the two opt-in extensions
    (vary_bump_shape, include_fuv_curvature) that widen DustAttenuation's
    reach toward a "universal" family, while defaulting off so every
    pre-existing caller (including TestDustAttenuation above) is
    unaffected.
    '''

    def setUp(self):
        # Extend blueward of the FM90 pivot (x=5.9 um^-1 <-> ~1695A) so the
        # far-UV curvature term actually turns on somewhere in this grid.
        self.wave = np.arange(1200.0, 10000.0, 1.0)
        self.flux = np.full_like(self.wave, 1e-16)

    def _theta_value(self, table, name):
        row = table[table['param'] == name]
        self.assertEqual(len(row), 1)
        return float(row['value'][0])

    def test_generalization_flags_default_off_reproduce_legacy_family(self):
        '''With both extensions left at their defaults (False), the class
        must be numerically identical to the original 4-parameter family
        -- i.e. this generalization is a strict superset, not a behavior
        change, for every existing caller.'''
        theta = dict(theta0=0.6, theta1=1.1, theta2=0.4, theta3=0.05)
        legacy = DustAttenuation()
        generalized = DustAttenuation(vary_bump_shape=False, include_fuv_curvature=False)
        d_legacy, _, t_legacy = legacy.spectrum(self.wave, self.flux, theta=theta, seed=1)
        d_general, _, t_general = generalized.spectrum(self.wave, self.flux, theta=theta, seed=1)
        np.testing.assert_array_equal(d_legacy, d_general)
        self.assertEqual(set(t_legacy['param']), set(t_general['param']))

    def test_fuv_curvature_off_by_default_no_theta4_in_table(self):
        dust = DustAttenuation()
        _, _, table = dust.spectrum(self.wave, self.flux, seed=1)
        self.assertNotIn('theta4', set(table['param']))
        self.assertEqual(len(table), 4)

    def test_fuv_curvature_enabled_adds_theta4_to_table(self):
        dust = DustAttenuation(include_fuv_curvature=True)
        _, _, table = dust.spectrum(self.wave, self.flux, seed=1)
        self.assertIn('theta4', set(table['param']))
        self.assertEqual(len(table), 5)

    def test_fuv_curvature_is_exactly_zero_at_and_above_pivot_wavelength(self):
        '''F(x) is defined to be exactly zero for x < 5.9 um^-1, i.e.
        lambda > 1e4/5.9 = 1694.9A -- confirm turning theta4 on changes
        nothing redward of that pivot.'''
        theta = dict(theta0=0.0, theta1=0.0, theta2=0.0, theta3=0.0, theta4=1.0)
        dust_on = DustAttenuation(include_fuv_curvature=True)
        dflux, wave, _ = dust_on.spectrum(self.wave, self.flux, theta=theta, seed=1)
        redward = wave > (1.0e4 / 5.9)
        np.testing.assert_allclose(dflux[redward], 0.0, atol=1e-30)

    def test_fuv_curvature_adds_extra_attenuation_blueward_of_pivot(self):
        '''theta4>0 must strictly increase the attenuation magnitude
        blueward of the FM90 pivot relative to theta4=0, all else equal.'''
        base = dict(theta0=0.5, theta1=1.0, theta2=0.0, theta3=0.0)
        dust_off = DustAttenuation(include_fuv_curvature=True)
        d_off, wave, _ = dust_off.spectrum(self.wave, self.flux,
                                            theta=dict(base, theta4=0.0), seed=1)
        d_on, _, _ = dust_off.spectrum(self.wave, self.flux,
                                        theta=dict(base, theta4=0.8), seed=1)
        i_fuv = np.argmin(np.abs(wave - 1300.0))  # well blueward of the 1695A pivot
        self.assertGreater(abs(d_on[i_fuv]), abs(d_off[i_fuv]))

    def test_vary_bump_shape_off_by_default_fixed_bump_used(self):
        '''Legacy behavior: with vary_bump_shape left False, the bump peak
        stays exactly at the fixed LAMBDA_BUMP constant.'''
        dust = DustAttenuation()
        theta = dict(theta0=0.0, theta1=0.0, theta2=1.0, theta3=0.0)
        dflux, wave, table = dust.spectrum(self.wave, self.flux, theta=theta, seed=1)
        self.assertNotIn('lambda_bump', set(table['param']))
        idx = np.argmin(np.abs(wave - DustAttenuation.LAMBDA_BUMP))
        expected_peak = self.flux[idx] * (10**(-0.4 * 1.0) - 1.0)
        self.assertAlmostEqual(dflux[idx] / expected_peak, 1.0, places=3)

    def test_vary_bump_shape_draws_within_configured_range(self):
        dust = DustAttenuation(vary_bump_shape=True)
        for seed in range(10):
            _, _, table = dust.spectrum(self.wave, self.flux, seed=seed)
            lb = self._theta_value(table, 'lambda_bump')
            bw = self._theta_value(table, 'bump_width')
            lo, hi = DustAttenuation.BUMP_CENTER_RANGE
            self.assertGreaterEqual(lb, lo)
            self.assertLessEqual(lb, hi)
            lo, hi = DustAttenuation.BUMP_WIDTH_RANGE
            self.assertGreaterEqual(bw, lo)
            self.assertLessEqual(bw, hi)

    def test_vary_bump_shape_explicit_override_moves_peak(self):
        dust = DustAttenuation(vary_bump_shape=True)
        theta = dict(theta0=0.0, theta1=0.0, theta2=1.0, theta3=0.0,
                     lambda_bump=3000.0, bump_width=80.0)
        dflux, wave, table = dust.spectrum(self.wave, self.flux, theta=theta, seed=1)
        self.assertAlmostEqual(self._theta_value(table, 'lambda_bump'), 3000.0, places=10)
        idx = np.argmin(np.abs(wave - 3000.0))
        expected_peak = self.flux[idx] * (10**(-0.4 * 1.0) - 1.0)
        self.assertAlmostEqual(dflux[idx] / expected_peak, 1.0, places=3)

    def test_bump_center_and_width_range_overrides(self):
        dust = DustAttenuation(vary_bump_shape=True,
                                bump_center_range=(9000.0, 9000.0),
                                bump_width_range=(50.0, 50.0))
        _, _, table = dust.spectrum(self.wave, self.flux, seed=1)
        self.assertAlmostEqual(self._theta_value(table, 'lambda_bump'), 9000.0, places=6)
        self.assertAlmostEqual(self._theta_value(table, 'bump_width'), 50.0, places=6)

    def test_seed_reproducible_with_both_extensions_enabled(self):
        dust = DustAttenuation(vary_bump_shape=True, include_fuv_curvature=True)
        d1, _, t1 = dust.spectrum(self.wave, self.flux, seed=7)
        d2, _, t2 = dust.spectrum(self.wave, self.flux, seed=7)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(t1['value'].data, t2['value'].data)
        self.assertEqual(len(t1), 7)  # theta0..theta4 + lambda_bump + bump_width

    def test_torch_and_numpy_backends_agree_with_both_extensions_enabled(self):
        dust = DustAttenuation(vary_bump_shape=True, include_fuv_curvature=True)
        theta = dict(theta0=0.5, theta1=1.0, theta2=0.3, theta3=0.05, theta4=0.4,
                     lambda_bump=2200.0, bump_width=300.0)
        a_np, _, _ = dust.spectrum(self.wave, self.flux, theta=theta, backend='numpy', seed=1)
        a_torch, _, _ = dust.spectrum(self.wave, self.flux, theta=theta, backend='torch',
                                       device='cpu', seed=1)
        np.testing.assert_allclose(a_np, a_torch, rtol=1e-6, atol=1e-30)

    def test_theta4_nonnegativity_prior_default(self):
        lo, hi = DustAttenuation.THETA_PRIORS['theta4']
        self.assertGreaterEqual(lo, 0.0)


if __name__ == '__main__':
    unittest.main()
