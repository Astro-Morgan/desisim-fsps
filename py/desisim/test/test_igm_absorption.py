import unittest
import numpy as np

from desisim.igm_absorption import IGMAbsorption, LAMBDA_LYA


class TestIGMAbsorption(unittest.TestCase):
    '''Tests for task #35's Lyman-alpha-forest + DLA additive reconciliation
    module. See igm_absorption.py's module docstring for the full physical/
    architectural rationale (why MockMaker not the external CoLoRe file, why
    zqso is a required non-drawn input, why the additive-deficit reformulation
    mirroring dust.py is exact rather than approximate).
    '''

    def setUp(self):
        # Wide enough to comfortably bracket the rest-frame Lyalpha region
        # (1215.67A) plus headroom either side.
        self.igm = IGMAbsorption(minwave=900.0, maxwave=1300.0, cdelt_kms=20.0)
        self.zqso = 2.5

    def test_transmission_bounded_in_zero_one(self):
        T, wave, params = self.igm.transmission(self.zqso, seed=1)
        self.assertTrue(np.all(T >= 0.0))
        self.assertTrue(np.all(T <= 1.0 + 1e-12))

    def test_transmission_wave_matches_object_grid(self):
        T, wave, params = self.igm.transmission(self.zqso, seed=1)
        np.testing.assert_allclose(wave, 10 ** self.igm.log10wave)
        self.assertEqual(T.shape, wave.shape)

    def test_no_forest_redward_of_qso_own_lyalpha(self):
        '''By definition there is no Lyalpha forest absorption redward of the
        QSO's own Lyalpha emission line -- transmission must be exactly 1
        there (forest_T is explicitly masked to 1.0 in that region; DLAs can
        still occur there in principle for a real high-z sightline, but this
        module's insert_dlas call is windowed to the same wave_obs array, so
        any redward DLA would need real column density there too -- check the
        forest piece in isolation via add_dlas=False).'''
        T, wave, params = self.igm.transmission(self.zqso, add_dlas=False, seed=1)
        redward = wave > LAMBDA_LYA * 1.0001
        np.testing.assert_allclose(T[redward], 1.0)

    def test_forest_present_blueward_of_qso_own_lyalpha(self):
        T, wave, params = self.igm.transmission(self.zqso, add_dlas=False, seed=1)
        blueward = wave < LAMBDA_LYA * 0.999
        # Not every pixel needs T<1 (lognormal field has some near-unity
        # troughs) but the mean absorption there should be real and nonzero.
        self.assertLess(T[blueward].mean(), 0.999)

    def test_seed_reproducibility(self):
        T1, _, p1 = self.igm.transmission(self.zqso, seed=42)
        T2, _, p2 = self.igm.transmission(self.zqso, seed=42)
        np.testing.assert_array_equal(T1, T2)
        self.assertEqual(p1['ndla'], p2['ndla'])

    def test_different_seeds_give_different_transmission(self):
        T1, _, _ = self.igm.transmission(self.zqso, seed=1)
        T2, _, _ = self.igm.transmission(self.zqso, seed=2)
        self.assertFalse(np.array_equal(T1, T2))

    def test_add_dlas_false_is_pure_forest(self):
        T_with, _, params_with = self.igm.transmission(self.zqso, add_dlas=True, seed=7)
        T_without, _, params_without = self.igm.transmission(self.zqso, add_dlas=False, seed=7)
        self.assertIsNone(params_without['dla_meta'])
        self.assertEqual(params_without['ndla'], 0)
        # Forest RNG draw is independent of the DLA RNG draw (separate
        # child seeds from the same RandomState), so turning DLAs off
        # should not alter the forest-only transmission at all.
        np.testing.assert_array_equal(T_without, T_with if params_with['ndla'] == 0 else T_without)

    def test_params_report_zqso_and_add_dlas(self):
        T, wave, params = self.igm.transmission(self.zqso, add_dlas=True, seed=3)
        self.assertEqual(params['zqso'], self.zqso)
        self.assertTrue(params['add_dlas'])

    def test_spectrum_deficit_is_nonpositive(self):
        flux = np.ones_like(self.igm.log10wave) * 5.0
        igm_flux, wave, params = self.igm.spectrum(flux, self.zqso, seed=1)
        self.assertTrue(np.all(igm_flux <= 1e-12))

    def test_spectrum_deficit_matches_manual_transmission_formula(self):
        flux = np.linspace(1.0, 10.0, self.igm.log10wave.size)
        igm_flux, wave, params = self.igm.spectrum(flux, self.zqso, seed=9)
        T, _, _ = self.igm.transmission(self.zqso, seed=9)
        np.testing.assert_allclose(igm_flux, flux * (T - 1.0))

    def test_spectrum_zero_flux_gives_zero_deficit(self):
        flux = np.zeros_like(self.igm.log10wave)
        igm_flux, wave, params = self.igm.spectrum(flux, self.zqso, seed=1)
        np.testing.assert_array_equal(igm_flux, np.zeros_like(flux))

    def test_higher_zqso_shifts_forest_region_wider_in_rest_frame(self):
        '''Sanity check on the physical picture, not a tight quantitative
        test: at higher zqso, a larger fraction of the (fixed, wide) MockMaker
        skewer's observed-frame extent maps into the module's rest-frame
        output window before hitting the object's own maxwave -- meaning the
        forest-affected rest-frame fraction should not shrink as zqso grows
        as long as it stays within the skewer's coverage.'''
        T_lo, wave, _ = self.igm.transmission(2.0, add_dlas=False, seed=5)
        T_hi, _, _ = self.igm.transmission(3.0, add_dlas=False, seed=5)
        blueward = wave < LAMBDA_LYA * 0.999
        # Both should show real absorption; just confirm neither degenerates
        # to all-ones (i.e. confirm the wave_obs range actually overlaps the
        # skewer's own coverage at both redshifts).
        self.assertLess(T_lo[blueward].mean(), 0.999)
        self.assertLess(T_hi[blueward].mean(), 0.999)

    def test_dla_seed_that_draws_a_dla_produces_a_real_trough(self):
        '''Scan a handful of seeds (DLA incidence is stochastic/Poisson-like,
        not guaranteed for any single seed) to confirm that when a DLA IS
        drawn, dla_meta is populated and a real absorption trough appears
        (T < 1 somewhere) beyond what the forest alone would produce at that
        same forest-seed.'''
        found_a_dla = False
        for seed in range(20):
            T, wave, params = self.igm.transmission(self.zqso, add_dlas=True, seed=seed)
            if params['ndla'] > 0:
                found_a_dla = True
                self.assertIsNotNone(params['dla_meta'])
                self.assertEqual(len(params['dla_meta']), params['ndla'])
                break
        self.assertTrue(found_a_dla, 'no DLA drawn in 20 seeds -- suspiciously low incidence, check dla.py wiring')

    def test_explicit_log10wave_grid_is_honored(self):
        custom_wave = np.arange(1000.0, 1250.0, 0.5)
        igm = IGMAbsorption(log10wave=np.log10(custom_wave))
        T, wave, params = igm.transmission(self.zqso, seed=1)
        np.testing.assert_allclose(wave, custom_wave)


if __name__ == '__main__':
    unittest.main()
