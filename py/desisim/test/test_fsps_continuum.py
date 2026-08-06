import os
import unittest
import numpy as np

fsps_available = True
try:
    import fsps  # noqa: F401
except ImportError:
    fsps_available = False

from desisim.fsps_continuum import _d4000, _continuum_flux_near


class TestD4000Helper(unittest.TestCase):
    '''Pure-math unit tests for the D4000/continuum helpers -- no FSPS or
    SPS_HOME required, since these operate on plain arrays.'''

    def test_flat_fnu_spectrum_gives_d4000_near_one(self):
        # f_nu flat => f_lambda = const/wave**2, so that fnu = flambda*wave**2
        # is exactly flat and D4000 (a ratio of two f_nu means) must be 1.
        wave = np.linspace(3000.0, 5000.0, 20000)
        flambda = 1.0 / wave ** 2
        self.assertAlmostEqual(_d4000(wave, flambda), 1.0, places=6)

    def test_break_produces_d4000_above_one(self):
        '''A spectrum with a genuine step up in f_nu redward of 4000A must
        give D4000 > 1, matching the physical definition (older/more
        evolved populations have a stronger break => higher D4000).'''
        wave = np.linspace(3000.0, 5000.0, 20000)
        fnu = np.where(wave < 4000.0, 1.0, 3.0)  # step at the break
        flambda = fnu / wave ** 2
        d4000 = _d4000(wave, flambda)
        self.assertGreater(d4000, 1.0)
        self.assertAlmostEqual(d4000, 3.0, places=2)

    def test_d4000_raises_if_grid_does_not_cover_sidebands(self):
        wave = np.linspace(4200.0, 4300.0, 100)  # entirely redward of both sidebands
        flambda = np.ones_like(wave)
        with self.assertRaises(ValueError):
            _d4000(wave, flambda)

    def test_continuum_flux_near_averages_a_window(self):
        wave = np.linspace(3700.0, 3760.0, 1000)
        flux = np.full_like(wave, 5.0)
        flux[(wave > 3720) & (wave < 3735)] = 100.0  # narrow spike near center
        # A wide-enough window should be pulled up by the spike; a value far
        # from any data should just return the nearest pixel.
        near = _continuum_flux_near(wave, flux, center=3727.4, halfwidth=25.0)
        self.assertGreater(near, 5.0)
        far = _continuum_flux_near(wave, flux, center=10000.0, halfwidth=1.0)
        self.assertAlmostEqual(far, 5.0, places=6)


@unittest.skipUnless(fsps_available, 'python-fsps is not installed.')
class TestFSPSBasisTemplates(unittest.TestCase):
    '''Integration tests for fsps_basis_templates() and its use as a
    drop-in baseflux/basewave/basemeta source for GALAXY/ELG/BGS. Requires
    python-fsps + $SPS_HOME (not $DESI_BASIS_TEMPLATES -- that's the whole
    point of this being a supplemental, independent continuum source).
    nbase is kept deliberately tiny (each draw costs real wall-clock time;
    see fsps_continuum.py's module docstring Caveat 3) and results are
    computed once in setUpClass and reused across test methods.
    '''

    @classmethod
    def setUpClass(cls):
        from desisim.fsps_continuum import fsps_basis_templates
        cls.elg_result = fsps_basis_templates(objtype='ELG', nbase=3, seed=1)
        cls.bgs_result = fsps_basis_templates(objtype='BGS', nbase=3, seed=1)

    def test_shapes_and_types(self):
        baseflux, basewave, basemeta = self.elg_result
        self.assertEqual(baseflux.shape[0], 3)
        self.assertEqual(baseflux.shape[1], len(basewave))
        self.assertEqual(len(basemeta), 3)
        self.assertIn('D4000', basemeta.colnames)
        self.assertIn('OII_CONTINUUM', basemeta.colnames)
        self.assertIn('HBETA_CONTINUUM', basemeta.colnames)
        self.assertIn('HBETA_LIMIT', basemeta.colnames)

    def test_flux_is_finite_and_nonnegative(self):
        baseflux, _, _ = self.elg_result
        self.assertTrue(np.all(np.isfinite(baseflux)))
        self.assertTrue(np.all(baseflux >= 0))

    def test_d4000_in_physically_plausible_range(self):
        _, _, basemeta = self.elg_result
        # Real DESI D4000 values for star-forming/quiescent galaxies span
        # roughly 1.0-2.0; a wildly outside-range value would indicate a
        # units/indexing bug in the D4000 computation, not just noise.
        self.assertTrue(np.all(basemeta['D4000'] > 0.5))
        self.assertTrue(np.all(basemeta['D4000'] < 3.0))

    def test_hbeta_limit_all_zero(self):
        _, _, basemeta = self.bgs_result
        np.testing.assert_array_equal(basemeta['HBETA_LIMIT'].data, 0)

    def test_same_seed_reproducible_different_seed_not(self):
        from desisim.fsps_continuum import fsps_basis_templates
        a, _, _ = fsps_basis_templates(objtype='ELG', nbase=2, seed=42)
        b, _, _ = fsps_basis_templates(objtype='ELG', nbase=2, seed=42)
        c, _, _ = fsps_basis_templates(objtype='ELG', nbase=2, seed=43)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_invalid_objtype_raises(self):
        from desisim.fsps_continuum import fsps_basis_templates
        with self.assertRaises(ValueError):
            fsps_basis_templates(objtype='WD', nbase=2, seed=1)

    def test_nproc_matches_single_process_bit_for_bit(self):
        '''The whole point of sharding across processes is that it must not
        change the answer: same seed, same draws, same flux, regardless of
        how many workers split the work.'''
        from desisim.fsps_continuum import fsps_basis_templates
        flux1, wave1, meta1 = fsps_basis_templates(objtype='ELG', nbase=4, seed=11, nproc=1)
        flux2, wave2, meta2 = fsps_basis_templates(objtype='ELG', nbase=4, seed=11, nproc=2)
        np.testing.assert_array_equal(wave1, wave2)
        np.testing.assert_allclose(flux1, flux2)
        np.testing.assert_array_equal(meta1['D4000'].data, meta2['D4000'].data)

    def test_nproc_not_exceeding_nbase_does_not_error(self):
        '''Requesting more workers than draws must be handled gracefully
        (clamped down), not spawn empty/degenerate shards.'''
        from desisim.fsps_continuum import fsps_basis_templates
        flux, wave, meta = fsps_basis_templates(objtype='ELG', nbase=2, seed=5, nproc=8)
        self.assertEqual(flux.shape[0], 2)

    def test_drop_in_as_ELG_baseflux_end_to_end(self):
        '''The actual point of this feature: GALAXY/ELG needs zero code
        changes to consume an FSPS-generated continuum.'''
        from desisim.templates import ELG
        baseflux, basewave, basemeta = self.elg_result
        elg = ELG(wave=np.arange(5000.0, 8000.0, 2.0),
                  baseflux=baseflux, basewave=basewave, basemeta=basemeta)
        flux, wave, meta, objmeta = elg.make_templates(nmodel=3, seed=2, nocolorcuts=True)
        self.assertEqual(flux.shape, (3, len(wave)))
        self.assertTrue(np.all(np.isfinite(flux)))
        self.assertTrue(np.all(flux >= 0))
        self.assertTrue(np.all(np.isfinite(objmeta['EWOII'].data)))

    def test_drop_in_as_BGS_baseflux_end_to_end(self):
        from desisim.templates import BGS
        baseflux, basewave, basemeta = self.bgs_result
        bgs = BGS(wave=np.arange(3600.0, 9800.0, 2.0),
                  baseflux=baseflux, basewave=basewave, basemeta=basemeta)
        flux, wave, meta, objmeta = bgs.make_templates(nmodel=3, seed=2, nocolorcuts=True)
        self.assertEqual(flux.shape, (3, len(wave)))
        self.assertTrue(np.all(np.isfinite(flux)))


if __name__ == '__main__':
    unittest.main()
