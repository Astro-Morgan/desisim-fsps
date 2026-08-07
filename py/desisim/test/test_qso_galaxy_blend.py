import unittest
import numpy as np

from desisim.qso_galaxy_blend import blend_qso_galaxy


class TestBlendQsoGalaxy(unittest.TestCase):
    '''Tests for the PyQSOFit-style composite QSO/galaxy blend
    (frac*QSO + (1-frac)*GALAXY). Pure bookkeeping on top of
    decompose.combine_into_channels -- no new physics, so these tests
    focus on the blending arithmetic, frac resolution/validation, and
    provenance, not on any individual physical channel (already tested
    elsewhere: test_decompose.py, test_dust.py, etc.).
    '''

    def setUp(self):
        self.wave = np.arange(3000.0, 10000.0, 1.0)
        self.npix = self.wave.shape[0]

    def _const(self, value):
        return np.full(self.npix, value)

    def _minimal_galaxy(self, stellar=2.0):
        return dict(continuum_stellar=self._const(stellar))

    def _minimal_qso(self, agn=3.0):
        return dict(continuum_stellar=self._const(0.0), continuum_agn=self._const(agn))

    def test_frac_zero_is_pure_galaxy(self):
        galaxy = self._minimal_galaxy(stellar=2.0)
        qso = self._minimal_qso(agn=3.0)
        out = blend_qso_galaxy(self.wave, galaxy, qso, frac=0.0)
        np.testing.assert_allclose(out['continuum'], 2.0)
        np.testing.assert_allclose(out['total'], 2.0)
        self.assertEqual(out['frac'], 0.0)

    def test_frac_one_is_pure_qso(self):
        galaxy = self._minimal_galaxy(stellar=2.0)
        qso = self._minimal_qso(agn=3.0)
        out = blend_qso_galaxy(self.wave, galaxy, qso, frac=1.0)
        np.testing.assert_allclose(out['continuum'], 3.0)
        np.testing.assert_allclose(out['total'], 3.0)

    def test_intermediate_frac_is_linear_combination(self):
        galaxy = self._minimal_galaxy(stellar=2.0)
        qso = self._minimal_qso(agn=3.0)
        out = blend_qso_galaxy(self.wave, galaxy, qso, frac=0.25)
        expected = 0.25 * 3.0 + 0.75 * 2.0
        np.testing.assert_allclose(out['continuum'], expected)

    def test_default_frac_drawn_uniform_and_seed_reproducible(self):
        galaxy = self._minimal_galaxy()
        qso = self._minimal_qso()
        out1 = blend_qso_galaxy(self.wave, galaxy, qso, seed=5)
        out2 = blend_qso_galaxy(self.wave, galaxy, qso, seed=5)
        out3 = blend_qso_galaxy(self.wave, galaxy, qso, seed=6)
        self.assertAlmostEqual(out1['frac'], out2['frac'], places=12)
        self.assertNotAlmostEqual(out1['frac'], out3['frac'], places=6)
        self.assertGreaterEqual(out1['frac'], 0.0)
        self.assertLessEqual(out1['frac'], 1.0)

    def test_frac_out_of_range_raises(self):
        galaxy = self._minimal_galaxy()
        qso = self._minimal_qso()
        with self.assertRaises(ValueError):
            blend_qso_galaxy(self.wave, galaxy, qso, frac=1.5)
        with self.assertRaises(ValueError):
            blend_qso_galaxy(self.wave, galaxy, qso, frac=-0.1)

    def test_total_equals_sum_of_three_blended_buckets(self):
        rand = np.random.RandomState(0)
        galaxy = dict(
            continuum_stellar=rand.uniform(1, 2, self.npix),
            narrow_emission=rand.uniform(0, 0.5, self.npix),
            ism_absorption=-rand.uniform(0, 0.3, self.npix),
            dust_flux=-rand.uniform(0, 0.3, self.npix),
        )
        qso = dict(
            continuum_stellar=np.zeros(self.npix),
            continuum_agn=rand.uniform(1, 3, self.npix),
            broad_emission=rand.uniform(0, 1.0, self.npix),
            associated_absorption_flux=-rand.uniform(0, 0.2, self.npix),
            dust_flux=-rand.uniform(0, 0.4, self.npix),
        )
        out = blend_qso_galaxy(self.wave, galaxy, qso, frac=0.6)
        np.testing.assert_allclose(out['total'], out['continuum'] + out['emission'] + out['absorption'])

    def test_galaxy_and_qso_channels_are_kept_and_unblended(self):
        '''The returned 'galaxy'/'qso' sub-dicts must be each component's
        own full combine_into_channels() output, NOT scaled by frac --
        i.e. ground-truth provenance is preserved at full (frac=1)
        strength for each side independently.'''
        galaxy = self._minimal_galaxy(stellar=2.0)
        qso = self._minimal_qso(agn=3.0)
        out = blend_qso_galaxy(self.wave, galaxy, qso, frac=0.3)
        np.testing.assert_allclose(out['galaxy']['continuum'], 2.0)
        np.testing.assert_allclose(out['qso']['continuum'], 3.0)

    def test_qso_side_accepts_independent_narrow_emission(self):
        '''Per the PI's explicit conditional direction, the QSO side must
        be able to carry its own independently-drawn narrow_emission
        (e.g. AGN-NLR lines with BPT-distinct ratios), separate from the
        galaxy side's narrow_emission -- exercising the documented future
        extension path without requiring any code change here.'''
        galaxy = self._minimal_galaxy(stellar=1.0)
        galaxy['narrow_emission'] = self._const(0.1)
        qso = self._minimal_qso(agn=1.0)
        qso['narrow_emission'] = self._const(0.05)
        out = blend_qso_galaxy(self.wave, galaxy, qso, frac=0.5)
        expected_emission = 0.5 * 0.05 + 0.5 * 0.1
        np.testing.assert_allclose(out['emission'], expected_emission)

    def test_mismatched_length_array_raises(self):
        galaxy = self._minimal_galaxy()
        qso = self._minimal_qso()
        qso['broad_emission'] = np.zeros(self.npix - 5)
        with self.assertRaises(ValueError):
            blend_qso_galaxy(self.wave, galaxy, qso, frac=0.5)

    def test_output_arrays_have_wave_length(self):
        galaxy = self._minimal_galaxy()
        qso = self._minimal_qso()
        out = blend_qso_galaxy(self.wave, galaxy, qso, frac=0.5)
        for key in ('continuum', 'emission', 'absorption', 'total'):
            self.assertEqual(out[key].shape[0], self.npix)

    def test_additive_linearity_across_full_frac_grid(self):
        '''Sanity/consistency check across the full frac range: total(frac)
        must vary linearly between the frac=0 and frac=1 endpoints for a
        component with only continuum set (closed-form check).'''
        galaxy = self._minimal_galaxy(stellar=5.0)
        qso = self._minimal_qso(agn=-3.0)
        for frac in (0.0, 0.1, 0.5, 0.9, 1.0):
            out = blend_qso_galaxy(self.wave, galaxy, qso, frac=frac)
            expected = frac * (-3.0) + (1.0 - frac) * 5.0
            np.testing.assert_allclose(out['continuum'], expected)


if __name__ == '__main__':
    unittest.main()
