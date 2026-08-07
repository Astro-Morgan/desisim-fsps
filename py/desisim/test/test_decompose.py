import unittest
import numpy as np

from desisim.decompose import combine_into_channels


class TestCombineIntoChannels(unittest.TestCase):
    '''Tests for the pure-bookkeeping 3-bucket (continuum/emission/
    absorption) grouping helper. No new physics here -- just verifying the
    sums/defaults/provenance are exactly what the module docstring claims.
    '''

    def setUp(self):
        self.wave = np.arange(3000.0, 10000.0, 1.0)
        self.npix = self.wave.shape[0]

    def _const(self, value):
        return np.full(self.npix, value)

    def test_continuum_stellar_only_minimal_call(self):
        cont = self._const(2.0)
        out = combine_into_channels(self.wave, cont)
        np.testing.assert_array_equal(out['continuum'], cont)
        np.testing.assert_array_equal(out['emission'], np.zeros(self.npix))
        np.testing.assert_array_equal(out['absorption'], np.zeros(self.npix))
        np.testing.assert_array_equal(out['total'], cont)
        self.assertFalse(out['components']['continuum_agn'])
        self.assertTrue(out['components']['continuum_stellar'])

    def test_continuum_bucket_sums_stellar_and_agn(self):
        cont_stellar = self._const(2.0)
        cont_agn = self._const(0.5)
        out = combine_into_channels(self.wave, cont_stellar, continuum_agn=cont_agn)
        np.testing.assert_array_equal(out['continuum'], cont_stellar + cont_agn)
        self.assertTrue(out['components']['continuum_agn'])

    def test_emission_bucket_sums_narrow_broad_and_dust_scatter(self):
        cont = self._const(1.0)
        narrow = self._const(0.1)
        broad = self._const(0.2)
        scatter = self._const(0.05)
        out = combine_into_channels(self.wave, cont, narrow_emission=narrow,
                                     broad_emission=broad, dust_scatter_excess=scatter)
        np.testing.assert_array_equal(out['emission'], narrow + broad + scatter)
        self.assertTrue(out['components']['narrow_emission'])
        self.assertTrue(out['components']['broad_emission'])
        self.assertTrue(out['components']['dust_scatter_excess'])

    def test_absorption_bucket_sums_ism_associated_dust_and_bal(self):
        '''The PI's clarified grouping: ISM/CGM absorption, the stochastic
        multi-system associated-absorption channel, BAL, and dust
        attenuation deficit all land in the SAME output bucket, even
        though they remain four independently-computed input arrays.'''
        cont = self._const(1.0)
        ism = self._const(-0.1)
        associated = self._const(-0.03)
        dust = self._const(-0.2)
        bal = self._const(-0.05)
        out = combine_into_channels(self.wave, cont, ism_absorption=ism,
                                     associated_absorption_flux=associated,
                                     dust_flux=dust, bal_flux=bal)
        np.testing.assert_array_equal(out['absorption'], ism + associated + dust + bal)
        self.assertTrue(out['components']['ism_absorption'])
        self.assertTrue(out['components']['associated_absorption_flux'])
        self.assertTrue(out['components']['dust_flux'])
        self.assertTrue(out['components']['bal_flux'])

    def test_total_equals_sum_of_three_buckets(self):
        rand = np.random.RandomState(0)
        cont = rand.uniform(1, 2, self.npix)
        narrow = rand.uniform(0, 0.5, self.npix)
        broad = rand.uniform(0, 0.5, self.npix)
        ism = -rand.uniform(0, 0.3, self.npix)
        dust = -rand.uniform(0, 0.3, self.npix)
        out = combine_into_channels(self.wave, cont, narrow_emission=narrow,
                                     broad_emission=broad, ism_absorption=ism,
                                     dust_flux=dust)
        np.testing.assert_allclose(out['total'], out['continuum'] + out['emission'] + out['absorption'])
        np.testing.assert_allclose(out['total'], cont + narrow + broad + ism + dust)

    def test_missing_optional_channels_default_to_zero_and_flagged_false(self):
        cont = self._const(1.0)
        out = combine_into_channels(self.wave, cont)
        for key in ('continuum_agn', 'narrow_emission', 'broad_emission',
                    'dust_scatter_excess', 'ism_absorption', 'associated_absorption_flux',
                    'dust_flux', 'bal_flux'):
            self.assertFalse(out['components'][key])

    def test_mismatched_length_array_raises(self):
        cont = self._const(1.0)
        bad = np.zeros(self.npix - 5)
        with self.assertRaises(ValueError):
            combine_into_channels(self.wave, cont, narrow_emission=bad)

    def test_output_arrays_have_wave_length(self):
        cont = self._const(1.0)
        out = combine_into_channels(self.wave, cont)
        for key in ('continuum', 'emission', 'absorption', 'total'):
            self.assertEqual(out[key].shape[0], self.npix)


if __name__ == '__main__':
    unittest.main()
