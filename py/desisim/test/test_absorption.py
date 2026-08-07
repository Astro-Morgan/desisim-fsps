import unittest
import numpy as np

from desisim.absorption import AbsorptionSpectrum


class TestAbsorptionSpectrum(unittest.TestCase):
    '''Unit/integration tests for handoff Sec 1.3's non-stellar ISM/CGM
    absorption channel (Na I D, Ca II H&K, Mg II absorption). No external
    DESI data products required -- this module only needs a caller-supplied
    continuum array, which these tests synthesize directly.
    '''

    def setUp(self):
        self.wave = np.arange(2000.0, 10000.0, 0.5)
        self.flux = np.full_like(self.wave, 1e-16)  # flat continuum
        self.ab = AbsorptionSpectrum(minwave=2000.0, maxwave=10000.0)

    def _row(self, line_table, name):
        row = line_table[line_table['name'] == name]
        self.assertEqual(len(row), 1, f'expected exactly one {name} row')
        return row

    def test_all_six_lines_present_by_default(self):
        _, _, line = self.ab.spectrum(self.wave, self.flux, seed=1)
        self.assertEqual(set(line['name']), set(AbsorptionSpectrum.LINE_NAMES))

    def test_include_lines_subsets_correctly(self):
        subset = ['MgII_2796_abs', 'MgII_2803_abs']
        ab = AbsorptionSpectrum(minwave=2000.0, maxwave=10000.0, include_lines=subset)
        _, _, line = ab.spectrum(self.wave, self.flux, seed=1)
        self.assertEqual(set(line['name']), set(subset))

    def test_unknown_include_line_raises(self):
        with self.assertRaises(ValueError):
            AbsorptionSpectrum(include_lines=['NotARealLine'])

    def test_absorption_flux_is_nonpositive_and_finite(self):
        '''Physical requirement: this is a flux *deficit* by construction
        (continuum * (exp(-tau) - 1) with tau >= 0), so it must never be
        positive, and must never be NaN/Inf for reasonable draws.'''
        for seed in range(5):
            absflux, wave, line = self.ab.spectrum(self.wave, self.flux, seed=seed)
            self.assertTrue(np.all(np.isfinite(absflux)))
            self.assertTrue(np.all(absflux <= 1e-25))  # allow tiny fp noise above exact 0

    def test_zero_tau0_gives_zero_deficit_everywhere(self):
        zero_tau = {n: 0.0 for n in AbsorptionSpectrum.LINE_NAMES}
        absflux, wave, line = self.ab.spectrum(self.wave, self.flux, tau0=zero_tau, seed=1)
        np.testing.assert_allclose(absflux, 0.0, atol=1e-30)

    def test_explicit_tau0_always_wins(self):
        explicit = {n: 0.5 for n in AbsorptionSpectrum.LINE_NAMES}
        _, _, line = self.ab.spectrum(self.wave, self.flux, tau0=explicit, seed=1)
        for name in AbsorptionSpectrum.LINE_NAMES:
            self.assertAlmostEqual(float(self._row(line, name)['tau0'][0]), 0.5, places=6)

    def test_seed_reproducible_different_seed_not(self):
        a1, _, line1 = self.ab.spectrum(self.wave, self.flux, seed=5)
        a2, _, line2 = self.ab.spectrum(self.wave, self.flux, seed=5)
        a3, _, line3 = self.ab.spectrum(self.wave, self.flux, seed=6)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(line1['tau0'].data, line2['tau0'].data)
        self.assertFalse(np.array_equal(a1, a3))

    def test_explicit_sigma_kms_always_wins(self):
        _, _, line = self.ab.spectrum(self.wave, self.flux, sigma_kms=123.0, seed=1)
        np.testing.assert_allclose(line['sigma_kms'].data, 123.0)

    def test_sigma_kms_draw_within_range_and_reproducible(self):
        _, _, lineA = self.ab.spectrum(self.wave, self.flux, seed=10)
        _, _, lineA2 = self.ab.spectrum(self.wave, self.flux, seed=10)
        sigma_A = float(lineA['sigma_kms'][0])
        sigma_A2 = float(lineA2['sigma_kms'][0])
        self.assertAlmostEqual(sigma_A, sigma_A2, places=10)
        self.assertGreaterEqual(sigma_A, AbsorptionSpectrum.SIGMA_KMS_RANGE[0])
        self.assertLessEqual(sigma_A, AbsorptionSpectrum.SIGMA_KMS_RANGE[1])

    def test_tau0_priors_override(self):
        zero_sigma_priors = {
            name: dict(mean=np.log10(0.2), sigma=0.0) for name in AbsorptionSpectrum.LINE_NAMES
        }
        _, _, line1 = self.ab.spectrum(self.wave, self.flux, tau0_priors=zero_sigma_priors, seed=1)
        _, _, line2 = self.ab.spectrum(self.wave, self.flux, tau0_priors=zero_sigma_priors, seed=99)
        for name in AbsorptionSpectrum.LINE_NAMES:
            self.assertAlmostEqual(float(self._row(line1, name)['tau0'][0]), 0.2, places=6)
            self.assertAlmostEqual(float(self._row(line2, name)['tau0'][0]), 0.2, places=6)

    def test_recovers_exact_transmission_at_line_center(self):
        '''Direct physical check: continuum + absorption_flux at a strong,
        isolated line's center must equal continuum * exp(-tau0) (the
        Gaussian is exactly at peak there), for a fine enough wavelength
        grid and narrow enough window that no other line contaminates it.'''
        name = 'CaII_K'
        tau0_val = 2.0
        # Fine grid (2 km/s/pixel) so the nearest output pixel to the exact
        # line center is well within a small fraction of the 50 km/s line
        # width -- otherwise pixelization alone (default cdelt_kms=20)
        # introduces a few-tenths-of-a-sigma offset from the true peak,
        # which is a grid-resolution artifact, not a bug in the formula.
        ab = AbsorptionSpectrum(minwave=3000.0, maxwave=5000.0, cdelt_kms=2.0,
                                 include_lines=[name])
        absflux, wave, line = ab.spectrum(self.wave, self.flux, tau0={name: tau0_val},
                                           sigma_kms=50.0, seed=1)
        center_wave = AbsorptionSpectrum.LINE_WAVE_VACUUM[name]
        idx = np.argmin(np.abs(wave - center_wave))
        continuum_here = np.interp(wave[idx], self.wave, self.flux)
        total_at_center = continuum_here + absflux[idx]
        expected = continuum_here * np.exp(-tau0_val)
        self.assertAlmostEqual(total_at_center / expected, 1.0, places=2)

    def test_mgii_absorption_reuses_emission_line_rest_wavelengths(self):
        '''Internal-consistency check: Mg II absorption rest wavelengths
        must exactly match the pre-existing Mg II emission rows in
        forbidden_lines.ecsv (same physical transition).'''
        from desisim.templates import EMSpectrum
        em = EMSpectrum(minwave=2000.0, maxwave=10000.0, include_mgii=True)
        emission_2796 = float(em.line[em.line['name'] == 'MgII_2800a']['wave'][0])
        emission_2803 = float(em.line[em.line['name'] == 'MgII_2800b']['wave'][0])
        self.assertAlmostEqual(AbsorptionSpectrum.LINE_WAVE_VACUUM['MgII_2796_abs'], emission_2796, places=3)
        self.assertAlmostEqual(AbsorptionSpectrum.LINE_WAVE_VACUUM['MgII_2803_abs'], emission_2803, places=3)

    def test_lines_outside_window_are_flagged_and_contribute_nothing(self):
        '''A narrow output window that excludes every active line must
        still run without error, report in_window=False for all rows, and
        produce zero absorption flux everywhere.'''
        ab = AbsorptionSpectrum(minwave=6000.0, maxwave=6100.0)  # excludes all 6 lines
        absflux, wave, line = ab.spectrum(self.wave, self.flux, seed=1)
        self.assertTrue(np.all(~line['in_window'].data))
        np.testing.assert_allclose(absflux, 0.0, atol=1e-30)

    def test_output_grid_matches_constructor_window(self):
        absflux, wave, line = self.ab.spectrum(self.wave, self.flux, seed=1)
        self.assertEqual(absflux.shape, wave.shape)
        self.assertAlmostEqual(wave.min(), 2000.0, delta=1.0)
        self.assertLess(wave.max(), 10000.0)

    def test_torch_and_numpy_backends_agree(self):
        kwargs = dict(seed=1)
        a_np, _, _ = self.ab.spectrum(self.wave, self.flux, backend='numpy', **kwargs)
        a_torch, _, _ = self.ab.spectrum(self.wave, self.flux, backend='torch', device='cpu', **kwargs)
        np.testing.assert_allclose(a_np, a_torch, rtol=1e-6, atol=1e-30)

    def test_explicit_log10wave_grid_is_respected(self):
        custom_grid = np.linspace(np.log10(3000.0), np.log10(6000.0), 5000)
        ab = AbsorptionSpectrum(log10wave=custom_grid)
        absflux, wave, line = ab.spectrum(self.wave, self.flux, seed=1)
        np.testing.assert_array_equal(wave, 10**custom_grid)

    def test_outflow_velshift_off_by_default_is_noop(self):
        '''Backward-compatibility guarantee: with include_outflow_velshift
        left at its default (False), velshift_kms must be forced to
        exactly 0.0 whenever not explicitly passed, for every pre-existing
        caller that doesn't know this parameter exists.'''
        for seed in range(5):
            _, _, line = self.ab.spectrum(self.wave, self.flux, seed=seed)
            np.testing.assert_allclose(line['velshift_kms'].data, 0.0)

    def test_include_outflow_velshift_draws_within_range_and_reproducible(self):
        ab = AbsorptionSpectrum(minwave=2000.0, maxwave=10000.0,
                                 include_outflow_velshift=True)
        _, _, lineA = ab.spectrum(self.wave, self.flux, seed=7)
        _, _, lineA2 = ab.spectrum(self.wave, self.flux, seed=7)
        v = float(lineA['velshift_kms'][0])
        v2 = float(lineA2['velshift_kms'][0])
        self.assertAlmostEqual(v, v2, places=10)
        self.assertGreaterEqual(v, AbsorptionSpectrum.VELSHIFT_KMS_RANGE[0])
        self.assertLessEqual(v, AbsorptionSpectrum.VELSHIFT_KMS_RANGE[1])

    def test_include_outflow_velshift_varies_across_seeds(self):
        ab = AbsorptionSpectrum(minwave=2000.0, maxwave=10000.0,
                                 include_outflow_velshift=True)
        draws = set()
        for seed in range(10):
            _, _, line = ab.spectrum(self.wave, self.flux, seed=seed)
            draws.add(round(float(line['velshift_kms'][0]), 6))
        self.assertGreater(len(draws), 1)

    def test_explicit_velshift_kms_always_wins_regardless_of_flag(self):
        '''An explicit velshift_kms must be respected whether or not
        include_outflow_velshift was set on the constructor.'''
        for flag in (False, True):
            ab = AbsorptionSpectrum(minwave=2000.0, maxwave=10000.0,
                                     include_outflow_velshift=flag)
            _, _, line = ab.spectrum(self.wave, self.flux, velshift_kms=-42.0, seed=1)
            np.testing.assert_allclose(line['velshift_kms'].data, -42.0)

    def test_velshift_shifts_absorption_trough_bluewad(self):
        '''A negative (blueshifted) velshift_kms should move the observed
        absorption trough to shorter wavelength relative to velshift=0,
        mirroring how zshift moves associated-absorber line centers
        (see test_associated_absorption.py's test_zshift_moves_line_centers).'''
        name = 'CaII_K'
        ab = AbsorptionSpectrum(minwave=3000.0, maxwave=5000.0, cdelt_kms=2.0,
                                 include_lines=[name])
        d0, wave0, _ = ab.spectrum(self.wave, self.flux, tau0={name: 2.0},
                                    sigma_kms=50.0, velshift_kms=0.0, seed=1)
        d1, wave1, _ = ab.spectrum(self.wave, self.flux, tau0={name: 2.0},
                                    sigma_kms=50.0, velshift_kms=-200.0, seed=1)
        np.testing.assert_array_equal(wave0, wave1)
        idx0 = np.argmin(d0)
        idx1 = np.argmin(d1)
        self.assertLess(wave0[idx1], wave0[idx0])

    def test_velshift_kms_backend_agreement(self):
        ab = AbsorptionSpectrum(minwave=2000.0, maxwave=10000.0,
                                 include_outflow_velshift=True)
        kwargs = dict(seed=3, velshift_kms=-150.0)
        a_np, _, _ = ab.spectrum(self.wave, self.flux, backend='numpy', **kwargs)
        a_torch, _, _ = ab.spectrum(self.wave, self.flux, backend='torch', device='cpu', **kwargs)
        np.testing.assert_allclose(a_np, a_torch, rtol=1e-6, atol=1e-30)


if __name__ == '__main__':
    unittest.main()
