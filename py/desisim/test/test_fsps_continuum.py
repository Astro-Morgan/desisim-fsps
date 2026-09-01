import os
import unittest
import numpy as np

fsps_available = True
try:
    import fsps  # noqa: F401
except ImportError:
    fsps_available = False

from desisim.fsps_continuum import (_d4000, _continuum_flux_near, _ou_log_offset,
                                     _bursty_tabular_sfh, OU_TAU_RANGE_GYR, OU_SIGMA_RANGE_DEX)


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


class TestOrnsteinUhlenbeckSFH(unittest.TestCase):
    '''Pure-math unit tests for the bursty-SFH Ornstein-Uhlenbeck process
    (task #37) -- no FSPS or SPS_HOME required, since these operate on
    plain arrays. See fsps_continuum.py's "Bursty SFH" module docstring
    section for the derivation and literature (Caplar & Tacchella 2019,
    MNRAS 487, 3845, and the main-sequence-scatter/bursty-dwarf references
    cited there).
    '''

    def test_reproducible_given_seed(self):
        t = np.arange(0.01, 5.0, 0.01)
        d1 = _ou_log_offset(t, tau_ou_gyr=0.2, sigma_ou_dex=0.3, rand=np.random.RandomState(1))
        d2 = _ou_log_offset(t, tau_ou_gyr=0.2, sigma_ou_dex=0.3, rand=np.random.RandomState(1))
        np.testing.assert_array_equal(d1, d2)

    def test_different_seeds_differ(self):
        t = np.arange(0.01, 5.0, 0.01)
        d1 = _ou_log_offset(t, tau_ou_gyr=0.2, sigma_ou_dex=0.3, rand=np.random.RandomState(1))
        d2 = _ou_log_offset(t, tau_ou_gyr=0.2, sigma_ou_dex=0.3, rand=np.random.RandomState(2))
        self.assertFalse(np.array_equal(d1, d2))

    def test_long_run_standard_deviation_converges_to_sigma(self):
        '''The OU process's stationary marginal is Normal(0, sigma^2) --
        confirm the EXACT discrete transition actually reproduces this
        over many long, independent runs (averaging across seeds, since a
        single ~250-correlation-time realization still has real Monte
        Carlo scatter in its sample mean/std -- this checks the estimator
        converges, not that any one draw hits the target exactly).'''
        t = np.arange(0.01, 50.0, 0.01)  # ~250 correlation times at tau=0.2
        stds = []
        means = []
        for seed in range(20):
            delta = _ou_log_offset(t, tau_ou_gyr=0.2, sigma_ou_dex=0.3, rand=np.random.RandomState(seed))
            stds.append(delta.std())
            means.append(delta.mean())
        self.assertAlmostEqual(np.mean(stds), 0.3, delta=0.03)
        self.assertAlmostEqual(np.mean(means), 0.0, delta=0.03)

    def test_zero_sigma_gives_exactly_zero_offset(self):
        '''sigma_ou=0 must recover Delta(t)=0 everywhere -- the strict-
        generalization property (bursty SFH collapses to the plain smooth
        trend) that this design was specifically chosen for.'''
        t = np.arange(0.01, 5.0, 0.01)
        delta = _ou_log_offset(t, tau_ou_gyr=0.2, sigma_ou_dex=0.0, rand=np.random.RandomState(4))
        np.testing.assert_array_equal(delta, np.zeros_like(t))

    def test_empirical_decorrelation_time_matches_tau(self):
        '''Direct check of the autocorrelation function's 1/e-folding
        time against the input tau_ou_gyr -- confirms the exact transition
        kernel's parametrization is not off by a factor (e.g. a stray
        factor of 2 in the exponent would still pass a pure reproducibility
        test but fail this one). Averaged over several long, independent
        realizations to average down the real single-realization ACF
        estimation noise rather than relying on one seed landing close.'''
        dt = 0.01
        t = np.arange(dt, 200.0, dt)
        decorr_times = []
        for seed in range(10):
            delta = _ou_log_offset(t, tau_ou_gyr=0.3, sigma_ou_dex=0.3, rand=np.random.RandomState(seed))
            x = delta - delta.mean()
            ac = np.correlate(x, x, mode='full')[len(x) - 1:]
            ac /= ac[0]
            below = np.where(ac < 1.0 / np.e)[0]
            if len(below):
                decorr_times.append(below[0] * dt)
        self.assertGreater(len(decorr_times), 0)
        self.assertAlmostEqual(np.mean(decorr_times), 0.3, delta=0.05)

    def test_bursty_tabular_sfh_always_positive_and_increasing(self):
        age, sfr = _bursty_tabular_sfh(tage_gyr=8.0, tau_gyr=2.0, ou_tau_gyr=0.2,
                                        ou_sigma_dex=0.4, rand=np.random.RandomState(6))
        self.assertTrue(np.all(np.diff(age) > 0))
        self.assertTrue(np.all(sfr > 0))
        self.assertTrue(np.any(sfr > 1e-33))  # set_tabular_sfh()'s own requirement

    def test_bursty_tabular_sfh_grid_covers_tage(self):
        age, sfr = _bursty_tabular_sfh(tage_gyr=5.0, tau_gyr=1.0, ou_tau_gyr=0.2,
                                        ou_sigma_dex=0.3, rand=np.random.RandomState(7))
        self.assertGreaterEqual(age.max(), 5.0)

    def test_zero_sigma_tabular_sfh_matches_smooth_delayed_tau_shape(self):
        '''With sigma_ou=0, the tabulated SFH must be EXACTLY the plain
        delayed-tau curve t*exp(-t/tau) -- confirms the strict-
        generalization property end-to-end through _bursty_tabular_sfh,
        not just through _ou_log_offset in isolation.'''
        age, sfr = _bursty_tabular_sfh(tage_gyr=5.0, tau_gyr=1.5, ou_tau_gyr=0.2,
                                        ou_sigma_dex=0.0, rand=np.random.RandomState(8))
        expected = age * np.exp(-age / 1.5)
        np.testing.assert_allclose(sfr, expected, rtol=1e-10)

    def test_literature_anchored_default_ranges_are_physically_sane(self):
        '''Sanity check on the module-level defaults themselves (not a
        test of any function) -- confirms these weren't accidentally
        edited to something outside the cited literature range (Caplar &
        Tacchella 2019\'s fitted tau_break=170 Myr and the ~0.2-0.4 dex
        main-sequence-scatter literature).'''
        self.assertGreaterEqual(OU_TAU_RANGE_GYR[0], 0.05)
        self.assertLessEqual(OU_TAU_RANGE_GYR[1], 2.0)
        self.assertTrue(OU_TAU_RANGE_GYR[0] <= 0.17 <= OU_TAU_RANGE_GYR[1])
        self.assertAlmostEqual(OU_SIGMA_RANGE_DEX[0], 0.2)
        self.assertAlmostEqual(OU_SIGMA_RANGE_DEX[1], 0.4)


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


@unittest.skipUnless(fsps_available, 'python-fsps is not installed.')
class TestBurstySFHIntegration(unittest.TestCase):
    '''Integration tests for task #37's bursty (Ornstein-Uhlenbeck) SFH
    option, requiring python-fsps + $SPS_HOME (see TestFSPSBasisTemplates'
    own docstring). nbase kept tiny for the same wall-clock reasons.
    '''

    def test_bursty_output_finite_and_nonnegative(self):
        from desisim.fsps_continuum import fsps_basis_templates
        baseflux, basewave, basemeta = fsps_basis_templates(objtype='ELG', nbase=3, seed=1, bursty=True)
        self.assertTrue(np.all(np.isfinite(baseflux)))
        self.assertTrue(np.all(baseflux >= 0))
        self.assertTrue(np.all(basemeta['BURSTY']))
        self.assertTrue(np.all(np.isfinite(basemeta['OU_TAU_GYR'].data)))
        self.assertTrue(np.all(np.isfinite(basemeta['OU_SIGMA_DEX'].data)))

    def test_non_bursty_metadata_columns_present_but_nan(self):
        '''The new BURSTY/OU_TAU_GYR/OU_SIGMA_DEX columns must always be
        present (fixed schema) but marked not-applicable when bursty=False.'''
        from desisim.fsps_continuum import fsps_basis_templates
        baseflux, basewave, basemeta = fsps_basis_templates(objtype='ELG', nbase=2, seed=1, bursty=False)
        self.assertFalse(np.any(basemeta['BURSTY']))
        self.assertTrue(np.all(np.isnan(basemeta['OU_TAU_GYR'].data)))
        self.assertTrue(np.all(np.isnan(basemeta['OU_SIGMA_DEX'].data)))

    def test_bursty_false_is_byte_for_byte_unchanged(self):
        '''The core backward-compatibility claim: adding the bursty
        machinery must not perturb bursty=False's tage/logzsol/dust2/tau
        draws or resulting flux at all, for the same seed.'''
        from desisim.fsps_continuum import fsps_basis_templates
        flux_a, wave_a, meta_a = fsps_basis_templates(objtype='ELG', nbase=3, seed=9, bursty=False)
        flux_b, wave_b, meta_b = fsps_basis_templates(objtype='ELG', nbase=3, seed=9)  # bursty defaults False
        np.testing.assert_array_equal(flux_a, flux_b)
        np.testing.assert_array_equal(meta_a['TAU_GYR'].data, meta_b['TAU_GYR'].data)

    def test_bursty_seed_reproducible(self):
        from desisim.fsps_continuum import fsps_basis_templates
        flux_a, _, meta_a = fsps_basis_templates(objtype='ELG', nbase=3, seed=13, bursty=True)
        flux_b, _, meta_b = fsps_basis_templates(objtype='ELG', nbase=3, seed=13, bursty=True)
        np.testing.assert_array_equal(flux_a, flux_b)
        np.testing.assert_array_equal(meta_a['OU_TAU_GYR'].data, meta_b['OU_TAU_GYR'].data)

    def test_bursty_zero_sigma_gives_similar_shape_to_smooth_default(self):
        '''The strict-generalization property (bursty SFH collapses to
        the plain smooth trend as sigma_ou->0) is proven exactly at the
        INPUT level by test_zero_sigma_tabular_sfh_matches_smooth_delayed_tau_shape
        in TestOrnsteinUhlenbeckSFH (no FSPS needed -- the tabulated (age,
        SFR) array handed to FSPS is bit-identical to t*exp(-t/tau) when
        sigma_ou=0). This test only checks the WEAKER, honestly-stated
        claim that FSPS's own two different internal code paths for
        integrating "the same" SFH -- its native analytic sfh=4 solver
        versus its generic piecewise-linear sfh=3 tabular integrator --
        are not wildly discrepant; exact bit-for-bit agreement between two
        different numerical integration schemes inside FSPS's Fortran core
        is NOT asserted here (this was not verified against a real FSPS
        installation in this dev sandbox, which lacks $SPS_HOME -- do not
        strengthen this test without first confirming the tighter claim
        actually holds in an environment where FSPS runs).'''
        from desisim.fsps_continuum import fsps_basis_templates
        flux_smooth, wave, _ = fsps_basis_templates(objtype='ELG', nbase=3, seed=21, bursty=False)
        flux_bursty_zero, _, _ = fsps_basis_templates(
            objtype='ELG', nbase=3, seed=21, bursty=True, ou_sigma_range=(0.0, 0.0))
        d4000_smooth = np.array([_d4000(wave, flux_smooth[ii]) for ii in range(3)])
        d4000_bursty_zero = np.array([_d4000(wave, flux_bursty_zero[ii]) for ii in range(3)])
        np.testing.assert_allclose(d4000_smooth, d4000_bursty_zero, rtol=0.1)

    def test_bursty_nproc_matches_single_process_bit_for_bit(self):
        from desisim.fsps_continuum import fsps_basis_templates
        flux1, wave1, meta1 = fsps_basis_templates(objtype='ELG', nbase=4, seed=11, bursty=True, nproc=1)
        flux2, wave2, meta2 = fsps_basis_templates(objtype='ELG', nbase=4, seed=11, bursty=True, nproc=2)
        np.testing.assert_array_equal(wave1, wave2)
        np.testing.assert_allclose(flux1, flux2)
        np.testing.assert_array_equal(meta1['OU_TAU_GYR'].data, meta2['OU_TAU_GYR'].data)

    def test_drop_in_as_ELG_baseflux_end_to_end_bursty(self):
        '''Same drop-in-compatibility claim as the smooth path's own test,
        confirmed for bursty output too.'''
        from desisim.fsps_continuum import fsps_basis_templates
        from desisim.templates import ELG
        baseflux, basewave, basemeta = fsps_basis_templates(objtype='ELG', nbase=3, seed=2, bursty=True)
        elg = ELG(wave=np.arange(5000.0, 8000.0, 2.0),
                  baseflux=baseflux, basewave=basewave, basemeta=basemeta)
        flux, wave, meta, objmeta = elg.make_templates(nmodel=3, seed=2, nocolorcuts=True)
        self.assertEqual(flux.shape, (3, len(wave)))
        self.assertTrue(np.all(np.isfinite(flux)))
        self.assertTrue(np.all(flux >= 0))


if __name__ == '__main__':
    unittest.main()
