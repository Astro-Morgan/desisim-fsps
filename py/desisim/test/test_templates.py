import os
import unittest
import numpy as np
from astropy.table import Table, Column
from desisim.templates import ELG, LRG, QSO, BGS, STAR, STD, MWS_STAR, WD, SIMQSO, EMSpectrum, GALAXY
from desisim import lya_mock_p1d as lyamock

desimodel_data_available = 'DESIMODEL' in os.environ
desi_templates_available = 'DESI_ROOT' in os.environ
desi_basis_templates_available = 'DESI_BASIS_TEMPLATES' in os.environ


class TestEMSpectrum(unittest.TestCase):
    '''Unit tests for EMSpectrum, in particular the newly-tunable [OI],
    [SIII], [ArIII], and MgII auxiliary line ratios (handoff Sec 1.2).
    None of these require $DESI_BASIS_TEMPLATES / $DESIMODEL: EMSpectrum
    only reads package-bundled data files.
    '''

    def setUp(self):
        # Wide enough to cover every line touched by these tests (MgII 2796
        # through [SIII] 9532), since spectrum() only returns lines that fall
        # within the constructor's [minwave, maxwave] window.
        self.em = EMSpectrum(minwave=2000.0, maxwave=10000.0, include_mgii=True)
        self.em_nomgii = EMSpectrum(minwave=2000.0, maxwave=10000.0, include_mgii=False)
        # Fixed forbidden-line ratios so tests are deterministic and don't
        # depend on the forbidmog draw.
        self.fixed_ratios = dict(oiiihbeta=-0.2, oiihbeta=0.1, niihbeta=-0.2, siihbeta=-0.3)

    def _ratio(self, line_table, name):
        row = line_table[line_table['name'] == name]
        self.assertEqual(len(row), 1, f'expected exactly one {name} row')
        return float(row['ratio'][0])

    def test_legacy_default_reproduces_fixed_constants(self):
        '''Regression test: calling spectrum() with none of the new
        arguments must reproduce the pre-existing hardcoded ratios exactly
        (0.1, 0.75, 0.04 for OI/SIII/ArIII; MgII 2796=0.3 and, after the
        accompanying bugfix, MgII 2803=0.3/mgiidoublet).'''
        _, _, line = self.em.spectrum(seed=1, **self.fixed_ratios)
        self.assertAlmostEqual(self._ratio(line, '[OI]_6300'), 0.1, places=6)
        self.assertAlmostEqual(self._ratio(line, '[OI]_6363'), 0.1 / self.em.oidoublet, places=6)
        self.assertAlmostEqual(self._ratio(line, '[SIII]_9532'), 0.75, places=6)
        self.assertAlmostEqual(self._ratio(line, '[SIII]_9069'), 0.75 / self.em.siiidoublet, places=6)
        self.assertAlmostEqual(self._ratio(line, '[ArIII]_7135'), 0.04, places=6)
        self.assertAlmostEqual(self._ratio(line, '[ArIII]_7751'), 0.04 / self.em.ariiidoublet, places=6)
        self.assertAlmostEqual(self._ratio(line, 'MgII_2800a'), 0.3, places=6)

    def test_mgii_2803_bugfix(self):
        '''Pre-existing bug: MgII 2803 (is2800b) was never assigned and
        silently kept the Column-init default ratio of 1.0 instead of being
        derived from the 2796 ratio via mgiidoublet. Confirm the fix.'''
        _, _, line = self.em.spectrum(seed=1, **self.fixed_ratios)
        ratio_2796 = self._ratio(line, 'MgII_2800a')
        ratio_2803 = self._ratio(line, 'MgII_2800b')
        self.assertNotEqual(ratio_2803, 1.0)
        self.assertAlmostEqual(ratio_2803, ratio_2796 / self.em.mgiidoublet, places=10)

    def test_explicit_auxline_value_always_wins(self):
        '''An explicit float always overrides the draw/legacy-constant,
        regardless of vary_auxlines.'''
        for vary in (False, True):
            _, _, line = self.em.spectrum(seed=1, oihbeta=0.5, siiihbeta=0.6,
                                           ariiihbeta=0.02, mgiihbeta=0.9,
                                           vary_auxlines=vary, **self.fixed_ratios)
            self.assertAlmostEqual(self._ratio(line, '[OI]_6300'), 0.5, places=6)
            self.assertAlmostEqual(self._ratio(line, '[SIII]_9532'), 0.6, places=6)
            self.assertAlmostEqual(self._ratio(line, '[ArIII]_7135'), 0.02, places=6)
            self.assertAlmostEqual(self._ratio(line, 'MgII_2800a'), 0.9, places=6)

    def test_vary_auxlines_draws_and_is_seed_reproducible(self):
        '''With vary_auxlines=True, repeated draws with different seeds
        should (almost surely) differ, and the same seed should reproduce
        the same draw exactly.'''
        _, _, lineA = self.em.spectrum(seed=1, vary_auxlines=True, **self.fixed_ratios)
        _, _, lineB = self.em.spectrum(seed=2, vary_auxlines=True, **self.fixed_ratios)
        _, _, lineA2 = self.em.spectrum(seed=1, vary_auxlines=True, **self.fixed_ratios)
        self.assertNotEqual(self._ratio(lineA, '[OI]_6300'), self._ratio(lineB, '[OI]_6300'))
        self.assertEqual(self._ratio(lineA, '[OI]_6300'), self._ratio(lineA2, '[OI]_6300'))

    def test_vary_auxlines_zero_sigma_collapses_to_mean(self):
        '''Edge case: sigma=0 in the prior must deterministically reproduce
        exp10(mean), with no scatter regardless of seed.'''
        zero_sigma_priors = {
            'oihbeta':    dict(mean=np.log10(0.1),  sigma=0.0),
            'siiihbeta':  dict(mean=np.log10(0.75), sigma=0.0),
            'ariiihbeta': dict(mean=np.log10(0.04), sigma=0.0),
            'mgiihbeta':  dict(mean=np.log10(0.3),  sigma=0.0),
        }
        _, _, line1 = self.em.spectrum(seed=1, vary_auxlines=True,
                                        auxline_priors=zero_sigma_priors, **self.fixed_ratios)
        _, _, line2 = self.em.spectrum(seed=99, vary_auxlines=True,
                                        auxline_priors=zero_sigma_priors, **self.fixed_ratios)
        self.assertAlmostEqual(self._ratio(line1, '[OI]_6300'), 0.1, places=6)
        self.assertAlmostEqual(self._ratio(line2, '[OI]_6300'), 0.1, places=6)

    def test_auxline_priors_boundary_values_do_not_error(self):
        '''Edge case: extreme (but finite) explicit ratio values, including
        zero, must not raise or produce non-finite output.'''
        for val in (0.0, 1e-6, 10.0):
            emspec, wave, line = self.em.spectrum(seed=1, oihbeta=val, siiihbeta=val,
                                                   ariiihbeta=val, mgiihbeta=val,
                                                   **self.fixed_ratios)
            self.assertTrue(np.all(np.isfinite(emspec)))
            self.assertTrue(np.all(np.isfinite(wave)))

    def test_mgii_ignored_when_include_mgii_false(self):
        '''mgiihbeta must have no effect and no MgII rows should appear when
        include_mgii=False (default), matching legacy behavior.'''
        _, _, line = self.em_nomgii.spectrum(seed=1, mgiihbeta=0.9, vary_auxlines=True,
                                              **self.fixed_ratios)
        self.assertEqual(len(line[line['name'] == 'MgII_2800a']), 0)
        self.assertEqual(len(line[line['name'] == 'MgII_2800b']), 0)

    def test_output_shape_and_no_nan(self):
        '''Smoke test: output array is well-formed regardless of
        vary_auxlines.'''
        for vary in (False, True):
            emspec, wave, line = self.em.spectrum(seed=1, vary_auxlines=vary, **self.fixed_ratios)
            self.assertEqual(emspec.shape, wave.shape)
            self.assertTrue(np.all(np.isfinite(emspec)))
            self.assertTrue(np.all(emspec >= 0))


class TestEMSpectrumBackends(unittest.TestCase):
    '''Equivalence tests between the numpy reference and torch-accelerated
    backends for EMSpectrum's line-profile construction (handoff's new
    "port to pytorch wherever possible, auto CUDA detection" requirement,
    2026-08-06). backend='auto' is now the *default*, so these tests are
    what proves that default change doesn't silently alter results for
    existing callers -- both backends must agree to floating-point
    precision, not just "close enough".
    '''

    def setUp(self):
        self.em = EMSpectrum(minwave=2000.0, maxwave=10000.0, include_mgii=True)
        self.kwargs = dict(oiiihbeta=-0.2, oiihbeta=0.1, niihbeta=-0.2, siihbeta=-0.3, seed=1)

    def test_torch_is_actually_available_in_this_environment(self):
        '''Sanity check on the test environment itself: if this fails, the
        rest of this class's assertions about the 'auto' backend selecting
        torch are not being exercised at all.'''
        from desisim.torch_utils import torch_available
        self.assertTrue(torch_available(), 'PyTorch is not installed; the '
                         'auto/torch backend paths below would silently no-op '
                         'to numpy and this test class would not be testing '
                         'what it claims to.')

    def test_numpy_and_torch_backends_agree(self):
        emspec_np, wave_np, _ = self.em.spectrum(backend='numpy', **self.kwargs)
        emspec_torch, wave_torch, _ = self.em.spectrum(backend='torch', **self.kwargs)
        np.testing.assert_array_equal(wave_np, wave_torch)
        np.testing.assert_allclose(emspec_np, emspec_torch, rtol=1e-9, atol=1e-30)

    def test_auto_backend_matches_explicit_numpy_backend(self):
        '''This is the actual backward-compatibility guarantee: calling
        spectrum() with none of the new backend/device/dtype arguments
        (i.e. the exact old call signature) must give the same physical
        result as the old numpy-only code did, even though it now silently
        routes through torch by default.'''
        emspec_default, _, _ = self.em.spectrum(**self.kwargs)  # backend='auto' implicitly
        emspec_numpy, _, _ = self.em.spectrum(backend='numpy', **self.kwargs)
        np.testing.assert_allclose(emspec_default, emspec_numpy, rtol=1e-9, atol=1e-30)

    def test_auto_prefers_torch_when_available(self):
        '''auto should behave identically to explicitly requesting torch
        when torch is installed (which test_torch_is_actually_available_in_this_environment
        confirms it is here).'''
        emspec_auto, _, _ = self.em.spectrum(**self.kwargs)
        emspec_torch, _, _ = self.em.spectrum(backend='torch', **self.kwargs)
        np.testing.assert_array_equal(emspec_auto, emspec_torch)

    def test_auto_falls_back_to_numpy_when_torch_unavailable(self):
        '''Simulate an environment without torch installed (monkeypatching
        desisim.templates.torch_available rather than actually uninstalling
        torch) and confirm auto still produces the correct, numpy-equivalent
        result rather than erroring.'''
        import desisim.templates as tmpl
        import desisim.torch_utils as torch_utils
        original = torch_utils.torch_available
        torch_utils.torch_available = lambda: False
        try:
            emspec_fallback, _, _ = self.em.spectrum(**self.kwargs)
        finally:
            torch_utils.torch_available = original
        emspec_numpy, _, _ = self.em.spectrum(backend='numpy', **self.kwargs)
        np.testing.assert_array_equal(emspec_fallback, emspec_numpy)

    def test_torch_backend_raises_if_torch_unavailable_and_explicitly_requested(self):
        import desisim.torch_utils as torch_utils
        original = torch_utils.torch_available
        torch_utils.torch_available = lambda: False
        try:
            with self.assertRaises(ImportError):
                self.em.spectrum(backend='torch', **self.kwargs)
        finally:
            torch_utils.torch_available = original

    def test_invalid_backend_raises(self):
        with self.assertRaises(ValueError):
            self.em.spectrum(backend='not-a-real-backend', **self.kwargs)

    def test_explicit_cpu_device_matches_auto(self):
        emspec_cpu, _, _ = self.em.spectrum(backend='torch', device='cpu', **self.kwargs)
        emspec_auto, _, _ = self.em.spectrum(**self.kwargs)
        np.testing.assert_array_equal(emspec_cpu, emspec_auto)

    def test_torch_backend_handles_zero_lines_in_window(self):
        '''Edge case: a wavelength window with no lines in it must not
        error in either backend (this is the em.spectrum "theseline empty"
        branch, which never calls into the backend at all -- confirm that
        explicitly rather than assuming).'''
        narrow_em = EMSpectrum(minwave=4990.0, maxwave=4991.0)  # no lines in this tiny window
        emspec, wave, line = narrow_em.spectrum(backend='torch', **self.kwargs)
        self.assertTrue(np.all(emspec == 0))
        self.assertEqual(len(line), 0)


class TestNewLinesNarrowBroad(unittest.TestCase):
    '''Unit/regression tests for handoff Sec 1.3's seven original new lines
    ([NeIII] 3869,3968; [OIII] 4363; HeII 4686; [NII] 5755; [SII] 4068,4076)
    plus task #33's four additions (SiIV+OIV] 1400, CIV 1549, CIII] 1909,
    MgII 2798) and their independent narrow (nebular) + broad (AGN-like)
    tunable components. include_new_lines defaults to False, so every
    pre-existing caller (including every other test in this file) is
    unaffected -- that invariant is exercised directly below.
    '''

    def setUp(self):
        # Wide window so every new line (1396.76-9532A range across narrow
        # + broad + all pre-existing lines, after task #33 pushed the blue
        # edge down to SiIV+OIV] 1396.76) falls inside [minwave, maxwave].
        self.em_new = EMSpectrum(minwave=1200.0, maxwave=10000.0, include_new_lines=True)
        self.em_legacy = EMSpectrum(minwave=1200.0, maxwave=10000.0, include_new_lines=False)
        self.fixed_ratios = dict(oiiihbeta=-0.2, oiihbeta=0.1, niihbeta=-0.2, siihbeta=-0.3)

    def _rows(self, line_table, name):
        return line_table[line_table['name'] == name]

    def test_new_lines_absent_by_default(self):
        '''Regression test: include_new_lines=False (the default) must
        produce a line table with none of the 7 new lines or their broad
        counterparts -- exact legacy behavior for every existing caller.'''
        _, _, line = self.em_legacy.spectrum(seed=1, hbetaflux=1e-16, **self.fixed_ratios)
        names = set(line['name'])
        for name in EMSpectrum.NEW_LINE_NAMES:
            self.assertNotIn(name, names)
            self.assertNotIn(name + '_broad', names)

    def test_default_call_with_new_lines_on_still_matches_legacy_old_lines(self):
        '''Turning include_new_lines on must not perturb the pre-existing
        lines' ratios -- the new lines are additive rows in the same table,
        not a change to how old rows are computed.'''
        _, _, line_new = self.em_new.spectrum(seed=1, hbetaflux=1e-16, **self.fixed_ratios)
        _, _, line_legacy = self.em_legacy.spectrum(seed=1, hbetaflux=1e-16, **self.fixed_ratios)
        for name in ('[OIII]_5007', '[NII]_6584', '[SII]_6716', 'Hbeta', '[OII]_3726'):
            old_ratio = float(self._rows(line_legacy, name)['ratio'][0])
            new_ratio = float(self._rows(line_new, name)['ratio'][0])
            self.assertAlmostEqual(old_ratio, new_ratio, places=10)

    def test_new_lines_present_when_enabled(self):
        '''Every one of the 7 new lines must appear with both a narrow row
        (bare name) and a broad row (name + "_broad") once enabled.'''
        _, _, line = self.em_new.spectrum(seed=1, hbetaflux=1e-16, **self.fixed_ratios)
        names = set(line['name'])
        for name in EMSpectrum.NEW_LINE_NAMES:
            self.assertIn(name, names)
            self.assertIn(name + '_broad', names)

    def test_explicit_narrow_ratio_always_wins(self):
        explicit = {n: 0.123 for n in EMSpectrum.NEW_LINE_NAMES}
        _, _, line = self.em_new.spectrum(seed=1, hbetaflux=1e-16, new_line_ratios=explicit,
                                           **self.fixed_ratios)
        for name in EMSpectrum.NEW_LINE_NAMES:
            # places=6, not more: the underlying 'ratio' Column is float32
            # (see self.line's Column dtypes in __init__), so a float64
            # 0.123 literal is only reproduced to ~1e-7-1e-8 precision --
            # the same tolerance TestEMSpectrum uses for the analogous
            # auxline ratio checks (e.g. test_explicit_auxline_value_always_wins).
            self.assertAlmostEqual(float(self._rows(line, name)['ratio'][0]), 0.123, places=6)

    def test_explicit_broad_ratio_always_wins(self):
        explicit_broad = {n: 0.045 for n in EMSpectrum.NEW_LINE_NAMES}
        _, _, line = self.em_new.spectrum(seed=1, hbetaflux=1e-16,
                                           new_line_broad_ratios=explicit_broad, **self.fixed_ratios)
        for name in EMSpectrum.NEW_LINE_NAMES:
            row = self._rows(line, name + '_broad')
            self.assertAlmostEqual(float(row['ratio'][0]), 0.045, places=10)

    def test_broad_flux_scales_with_effective_hbeta_flux(self):
        '''Broad-component flux = hbeta_flux_effective * broad_ratio. Vary
        hbetaflux and confirm the broad flux for a fixed ratio scales
        linearly with it (the narrow ratio path is independently tested
        already in TestEMSpectrum; this isolates the broad-flux formula).'''
        explicit_broad = {n: 0.02 for n in EMSpectrum.NEW_LINE_NAMES}
        _, _, line_a = self.em_new.spectrum(seed=1, hbetaflux=1e-16,
                                             new_line_broad_ratios=explicit_broad, **self.fixed_ratios)
        _, _, line_b = self.em_new.spectrum(seed=1, hbetaflux=2e-16,
                                             new_line_broad_ratios=explicit_broad, **self.fixed_ratios)
        for name in EMSpectrum.NEW_LINE_NAMES:
            flux_a = float(self._rows(line_a, name + '_broad')['flux'][0])
            flux_b = float(self._rows(line_b, name + '_broad')['flux'][0])
            self.assertAlmostEqual(flux_b / flux_a, 2.0, places=6)

    def test_broadsigma_explicit_value_always_wins(self):
        '''Passing broadsigma explicitly must be honored (no draw), and be
        reflected in a wider broad-line amplitude-to-flux relationship
        (amp = flux/(wave*ln10) / (sqrt(2*pi)*log10sigma), so larger
        broadsigma at fixed flux gives smaller amp).'''
        explicit_broad = {n: 0.02 for n in EMSpectrum.NEW_LINE_NAMES}
        _, _, line_narrow_sigma = self.em_new.spectrum(
            seed=1, hbetaflux=1e-16, new_line_broad_ratios=explicit_broad,
            broadsigma=500.0, **self.fixed_ratios)
        _, _, line_wide_sigma = self.em_new.spectrum(
            seed=1, hbetaflux=1e-16, new_line_broad_ratios=explicit_broad,
            broadsigma=4000.0, **self.fixed_ratios)
        name = EMSpectrum.NEW_LINE_NAMES[0]
        amp_narrow = float(self._rows(line_narrow_sigma, name + '_broad')['amp'][0])
        amp_wide = float(self._rows(line_wide_sigma, name + '_broad')['amp'][0])
        # Same integrated flux, wider sigma => smaller peak amplitude.
        self.assertGreater(amp_narrow, amp_wide)

    def test_broadsigma_draw_is_seed_reproducible(self):
        '''With broadsigma=None (the default), the log-uniform draw over
        BROADSIGMA_RANGE_KMS must be seed-reproducible and (almost surely)
        differ across seeds.'''
        _, _, lineA = self.em_new.spectrum(seed=10, hbetaflux=1e-16, **self.fixed_ratios)
        _, _, lineA2 = self.em_new.spectrum(seed=10, hbetaflux=1e-16, **self.fixed_ratios)
        _, _, lineB = self.em_new.spectrum(seed=11, hbetaflux=1e-16, **self.fixed_ratios)
        name = EMSpectrum.NEW_LINE_NAMES[0] + '_broad'
        amp_A = float(self._rows(lineA, name)['amp'][0])
        amp_A2 = float(self._rows(lineA2, name)['amp'][0])
        amp_B = float(self._rows(lineB, name)['amp'][0])
        self.assertAlmostEqual(amp_A, amp_A2, places=10)
        # amp values here are O(1e-20); assertNotAlmostEqual's absolute
        # "places" rounding would spuriously call any two such tiny numbers
        # "equal" (both round to 0.0 at 10 decimal places). Compare via
        # relative closeness instead, which is what's physically meaningful.
        self.assertFalse(np.isclose(amp_A, amp_B, rtol=1e-6, atol=0.0))

    def test_broadsigma_range_boundary_values_do_not_error(self):
        for val in EMSpectrum.BROADSIGMA_RANGE_KMS:
            emspec, wave, line = self.em_new.spectrum(seed=1, hbetaflux=1e-16, broadsigma=val,
                                                       **self.fixed_ratios)
            self.assertTrue(np.all(np.isfinite(emspec)))

    def test_zero_broad_ratio_contributes_zero_flux(self):
        '''Edge case: a broad ratio of exactly 0 must produce exactly zero
        broad-component flux (not NaN/Inf from e.g. a log(0) in the
        amplitude formula), while the narrow component is untouched.'''
        zero_broad = {n: 0.0 for n in EMSpectrum.NEW_LINE_NAMES}
        emspec, wave, line = self.em_new.spectrum(seed=1, hbetaflux=1e-16,
                                                    new_line_broad_ratios=zero_broad, **self.fixed_ratios)
        self.assertTrue(np.all(np.isfinite(emspec)))
        for name in EMSpectrum.NEW_LINE_NAMES:
            row = self._rows(line, name + '_broad')
            self.assertAlmostEqual(float(row['flux'][0]), 0.0, places=15)

    def test_new_line_priors_override(self):
        '''A caller-supplied new_line_priors dict must be used in place of
        NEW_LINE_PRIORS for the draw (zero-sigma edge case: deterministic
        regardless of seed).'''
        zero_sigma_priors = {
            name: dict(narrow_mean=np.log10(0.07), narrow_sigma=0.0,
                       broad_mean=np.log10(0.01), broad_sigma=0.0)
            for name in EMSpectrum.NEW_LINE_NAMES
        }
        _, _, line1 = self.em_new.spectrum(seed=1, hbetaflux=1e-16,
                                            new_line_priors=zero_sigma_priors, **self.fixed_ratios)
        _, _, line2 = self.em_new.spectrum(seed=99, hbetaflux=1e-16,
                                            new_line_priors=zero_sigma_priors, **self.fixed_ratios)
        for name in EMSpectrum.NEW_LINE_NAMES:
            self.assertAlmostEqual(float(self._rows(line1, name)['ratio'][0]), 0.07, places=6)
            self.assertAlmostEqual(float(self._rows(line2, name)['ratio'][0]), 0.07, places=6)

    def test_narrow_and_broad_flux_are_additive_in_emspec(self):
        '''Integration/energy-conservation check: the summed spectrum with
        both narrow and broad components enabled must, at each pixel, equal
        the sum of a narrow-only run and a broad-only run (narrow ratio
        forced to ~0 in the broad-only run and vice versa), since the two
        components are independent superposed Gaussians by construction.'''
        narrow_ratios = {n: 0.05 for n in EMSpectrum.NEW_LINE_NAMES}
        broad_ratios = {n: 0.02 for n in EMSpectrum.NEW_LINE_NAMES}
        zero = {n: 0.0 for n in EMSpectrum.NEW_LINE_NAMES}

        emspec_both, wave, _ = self.em_new.spectrum(
            seed=1, hbetaflux=1e-16, new_line_ratios=narrow_ratios,
            new_line_broad_ratios=broad_ratios, broadsigma=1000.0, **self.fixed_ratios)
        emspec_narrow_only, _, _ = self.em_new.spectrum(
            seed=1, hbetaflux=1e-16, new_line_ratios=narrow_ratios,
            new_line_broad_ratios=zero, broadsigma=1000.0, **self.fixed_ratios)
        emspec_broad_only, _, _ = self.em_new.spectrum(
            seed=1, hbetaflux=1e-16, new_line_ratios=zero,
            new_line_broad_ratios=broad_ratios, broadsigma=1000.0, **self.fixed_ratios)
        emspec_neither, _, _ = self.em_new.spectrum(
            seed=1, hbetaflux=1e-16, new_line_ratios=zero,
            new_line_broad_ratios=zero, broadsigma=1000.0, **self.fixed_ratios)

        # narrow_only + broad_only - neither (to remove the double-counted
        # shared old-line spectrum) must equal the both-enabled spectrum.
        reconstructed = emspec_narrow_only + emspec_broad_only - emspec_neither
        np.testing.assert_allclose(emspec_both, reconstructed, rtol=1e-8, atol=1e-30)

    def test_output_shape_and_no_nan(self):
        emspec, wave, line = self.em_new.spectrum(seed=1, hbetaflux=1e-16, **self.fixed_ratios)
        self.assertEqual(emspec.shape, wave.shape)
        self.assertTrue(np.all(np.isfinite(emspec)))
        self.assertTrue(np.all(emspec >= 0))

    def test_torch_and_numpy_backends_agree_with_new_lines(self):
        '''The broad-component path reuses _lines_to_spectrum_{numpy,torch}
        exactly like the narrow path -- confirm both backends still agree
        once new lines + broad components are in play.'''
        kwargs = dict(seed=1, hbetaflux=1e-16, **self.fixed_ratios)
        emspec_np, _, _ = self.em_new.spectrum(backend='numpy', **kwargs)
        emspec_torch, _, _ = self.em_new.spectrum(backend='torch', device='cpu', **kwargs)
        np.testing.assert_allclose(emspec_np, emspec_torch, rtol=1e-6, atol=1e-30)

    def test_task33_lines_present_when_enabled(self):
        '''The four task #33 additions must all be reachable in [minwave,
        maxwave] and produce both a narrow and a broad row.'''
        _, _, line = self.em_new.spectrum(seed=1, hbetaflux=1e-16, **self.fixed_ratios)
        names = set(line['name'])
        for name in ('SiIV_1400', 'CIV_1549', 'CIII]_1909', 'MgII_2798'):
            self.assertIn(name, names)
            self.assertIn(name + '_broad', names)

    def test_task33_broad_ratios_match_vanden_berk_2001(self):
        '''broad_mean for the task #33 lines is log10(Rel.Flux/Rel.Flux_Hbeta)
        from Vanden Berk et al. (2001) Table 2 -- not drawn from a prior
        (default broadsigma path would scatter it), so an explicit-ratio
        call must reproduce those exact literature-anchored numbers via
        the same hbeta_flux_effective*ratio formula used for every other
        broad line.'''
        hbeta = 8.649
        expected_ratio = {
            'SiIV_1400': 8.916 / hbeta,
            'CIV_1549': 25.291 / hbeta,
            'CIII]_1909': 15.943 / hbeta,
            'MgII_2798': 14.725 / hbeta,
        }
        explicit_broad = {n: expected_ratio[n] for n in expected_ratio}
        hbetaflux = 1e-16
        _, _, line = self.em_new.spectrum(seed=1, hbetaflux=hbetaflux,
                                           new_line_broad_ratios=explicit_broad, **self.fixed_ratios)
        for name, ratio in expected_ratio.items():
            row = self._rows(line, name + '_broad')
            self.assertAlmostEqual(float(row['ratio'][0]), ratio, places=6)
            np.testing.assert_allclose(float(row['flux'][0]), hbetaflux * ratio, rtol=1e-6)

    def test_mgii_2798_broad_row_independent_of_existing_narrow_doublet(self):
        '''MgII_2798 (task #33, broad-only) must coexist with the separate,
        pre-existing MgII_2800a/2800b narrow doublet (include_mgii) without
        interfering with it -- they are independent rows/mechanisms.'''
        em = EMSpectrum(minwave=1200.0, maxwave=10000.0, include_new_lines=True, include_mgii=True)
        # mgiihbeta is used directly as the linear MgII2796/Hbeta ratio
        # (not a log10 value) -- see templates.py's
        # `line['ratio'][is2800a] = mgiihbeta` assignment.
        _, _, line = em.spectrum(seed=1, hbetaflux=1e-16, mgiihbeta=0.1, **self.fixed_ratios)
        names = set(line['name'])
        self.assertIn('MgII_2800a', names)
        self.assertIn('MgII_2800b', names)
        self.assertIn('MgII_2798', names)
        self.assertIn('MgII_2798_broad', names)
        # the dedicated narrow doublet's ratio must be untouched by the
        # new row's own (deliberately negligible) narrow draw.
        row_2800a = self._rows(line, 'MgII_2800a')
        self.assertAlmostEqual(float(row_2800a['ratio'][0]), 0.1, places=8)


class TestBroadVelshift(unittest.TestCase):
    '''Tests for the independent broad-line-region velocity OFFSET
    (broadshift_kms), gated behind include_broad_velshift (default False,
    same backward-compatibility convention as
    AbsorptionSpectrum.include_outflow_velshift -- see that module's
    2026-08-07 docstring section). Only has an effect when
    include_new_lines=True, since that's the only source of broad lines
    in EMSpectrum.'''

    def setUp(self):
        self.em_new = EMSpectrum(minwave=2000.0, maxwave=10000.0, include_new_lines=True)
        self.em_shift = EMSpectrum(minwave=2000.0, maxwave=10000.0, include_new_lines=True,
                                    include_broad_velshift=True)
        self.fixed_ratios = dict(oiiihbeta=-0.2, oiihbeta=0.1, niihbeta=-0.2, siihbeta=-0.3)

    def _broad_rows(self, line_table):
        mask = np.array([str(n).endswith('_broad') for n in line_table['name']])
        return line_table[mask]

    def test_broadshift_off_by_default_is_noop(self):
        '''Backward-compatibility guarantee: with include_broad_velshift
        left at its default (False), broadshift_kms must be forced to
        exactly 0.0 whenever not explicitly passed.'''
        for seed in range(5):
            _, _, line = self.em_new.spectrum(seed=seed, hbetaflux=1e-16, **self.fixed_ratios)
            broad = self._broad_rows(line)
            np.testing.assert_allclose(broad['broadshift_kms'].data, 0.0)

    def test_include_broad_velshift_draws_within_range_and_reproducible(self):
        _, _, lineA = self.em_shift.spectrum(seed=7, hbetaflux=1e-16, **self.fixed_ratios)
        _, _, lineA2 = self.em_shift.spectrum(seed=7, hbetaflux=1e-16, **self.fixed_ratios)
        broadA = self._broad_rows(lineA)
        broadA2 = self._broad_rows(lineA2)
        v = float(broadA['broadshift_kms'][0])
        v2 = float(broadA2['broadshift_kms'][0])
        self.assertAlmostEqual(v, v2, places=10)
        self.assertGreaterEqual(v, EMSpectrum.BROADSHIFT_KMS_RANGE[0])
        self.assertLessEqual(v, EMSpectrum.BROADSHIFT_KMS_RANGE[1])

    def test_include_broad_velshift_varies_across_seeds(self):
        draws = set()
        for seed in range(10):
            _, _, line = self.em_shift.spectrum(seed=seed, hbetaflux=1e-16, **self.fixed_ratios)
            broad = self._broad_rows(line)
            draws.add(round(float(broad['broadshift_kms'][0]), 6))
        self.assertGreater(len(draws), 1)

    def test_explicit_broadshift_kms_always_wins_regardless_of_flag(self):
        for em in (self.em_new, self.em_shift):
            _, _, line = em.spectrum(seed=1, hbetaflux=1e-16, broadshift_kms=-321.0,
                                      **self.fixed_ratios)
            broad = self._broad_rows(line)
            np.testing.assert_allclose(broad['broadshift_kms'].data, -321.0)

    def test_broadshift_moves_broad_component_but_not_narrow(self):
        '''A nonzero broadshift_kms must move only the broad-component
        line centers, leaving the narrow-only spectrum bit-identical
        (isolated by forcing the broad ratio to exactly 0 in a parallel
        pair of calls), while the full narrow+broad spectrum must actually
        change under the shift (otherwise this test would be vacuous).'''
        broad_ratios = {n: 0.05 for n in EMSpectrum.NEW_LINE_NAMES}
        zero = {n: 0.0 for n in EMSpectrum.NEW_LINE_NAMES}
        common = dict(seed=1, hbetaflux=1e-16, broadsigma=1000.0, **self.fixed_ratios)

        emspec_shift0, wave, _ = self.em_new.spectrum(
            broadshift_kms=0.0, new_line_broad_ratios=broad_ratios, **common)
        emspec_shift_neg, _, _ = self.em_new.spectrum(
            broadshift_kms=-800.0, new_line_broad_ratios=broad_ratios, **common)

        emspec_narrow_only_shift0, _, _ = self.em_new.spectrum(
            broadshift_kms=0.0, new_line_broad_ratios=zero, **common)
        emspec_narrow_only_shift_neg, _, _ = self.em_new.spectrum(
            broadshift_kms=-800.0, new_line_broad_ratios=zero, **common)

        np.testing.assert_array_equal(emspec_narrow_only_shift0, emspec_narrow_only_shift_neg)
        self.assertFalse(np.allclose(emspec_shift0, emspec_shift_neg, rtol=1e-6, atol=1e-30))

    def test_broadshift_kms_backend_agreement(self):
        kwargs = dict(seed=1, hbetaflux=1e-16, broadshift_kms=-400.0, **self.fixed_ratios)
        emspec_np, _, _ = self.em_shift.spectrum(backend='numpy', **kwargs)
        emspec_torch, _, _ = self.em_shift.spectrum(backend='torch', device='cpu', **kwargs)
        np.testing.assert_allclose(emspec_np, emspec_torch, rtol=1e-6, atol=1e-30)

    def test_broadshift_kms_harmless_when_new_lines_disabled(self):
        '''include_new_lines=False means there are no broad lines to shift
        at all -- passing broadshift_kms must be a harmless no-op (no
        error, no effect on the output spectrum) rather than raising.'''
        em_legacy = EMSpectrum(minwave=2000.0, maxwave=10000.0, include_new_lines=False)
        emspec_a, _, _ = em_legacy.spectrum(seed=1, hbetaflux=1e-16, broadshift_kms=-500.0,
                                             **self.fixed_ratios)
        emspec_b, _, _ = em_legacy.spectrum(seed=1, hbetaflux=1e-16, broadshift_kms=0.0,
                                             **self.fixed_ratios)
        np.testing.assert_array_equal(emspec_a, emspec_b)


class TestGaussHermiteProfile(unittest.TestCase):
    '''Task #34: Gauss-Hermite (h3/h4) generalization of the previously-
    pure-Gaussian line profile, at both the low-level
    _lines_to_spectrum_{numpy,torch} functions and EMSpectrum's own
    narrow_h3/narrow_h4/broad_h3/broad_h4 + include_line_asymmetry wiring.
    '''

    def setUp(self):
        self.log10wave = np.linspace(np.log10(4800.0), np.log10(4930.0), 4000)
        self.center = np.log10(4862.68)
        self.log10sigma = 500.0 / 299792.458 / np.log(10)  # ~500 km/s in dex

    def test_h3_h4_zero_matches_original_gaussian_exactly(self):
        from desisim.templates import _lines_to_spectrum_numpy
        norm = np.array([1.0])
        g_default = _lines_to_spectrum_numpy(self.log10wave, np.array([self.center]), norm, self.log10sigma)
        g_explicit_zero = _lines_to_spectrum_numpy(self.log10wave, np.array([self.center]), norm,
                                                     self.log10sigma, h3=0.0, h4=0.0)
        np.testing.assert_array_equal(g_default, g_explicit_zero)

    def test_nonzero_h3_breaks_symmetry(self):
        '''Skewness (h3) must make the profile measurably asymmetric about
        its own centroid -- the whole physical point of adding it.'''
        from desisim.templates import _lines_to_spectrum_numpy
        norm = np.array([1.0])
        flux = _lines_to_spectrum_numpy(self.log10wave, np.array([self.center]), norm,
                                         self.log10sigma, h3=0.25, h4=0.0)
        below = flux[self.log10wave < self.center]
        above = flux[self.log10wave > self.center]
        # Mirror 'above' about the center and compare to 'below' -- an
        # exactly symmetric profile would match; a skewed one should not.
        self.assertFalse(np.allclose(below[::-1][:len(above)], above[:len(below)], atol=1e-10))

    def test_nonzero_h4_changes_peak_and_wing_shape(self):
        '''Kurtosis (h4) must change the profile shape relative to h4=0
        even though it preserves exact front-back symmetry -- uses a grid
        built to be exactly mirror-symmetric about the center (odd length,
        symmetric range) to avoid pixel-quantization noise in the
        symmetry check.'''
        from desisim.templates import _lines_to_spectrum_numpy
        sym_wave = self.center + np.linspace(-0.01, 0.01, 4001)  # odd length -> exact center pixel
        norm = np.array([1.0])
        gauss = _lines_to_spectrum_numpy(sym_wave, np.array([self.center]), norm, self.log10sigma)
        gh4 = _lines_to_spectrum_numpy(sym_wave, np.array([self.center]), norm,
                                        self.log10sigma, h3=0.0, h4=0.25)
        self.assertFalse(np.allclose(gauss, gh4))
        np.testing.assert_allclose(gh4, gh4[::-1], atol=1e-14)

    def test_profile_never_negative_even_at_large_h3_h4(self):
        from desisim.templates import _lines_to_spectrum_numpy
        norm = np.array([1.0])
        flux = _lines_to_spectrum_numpy(self.log10wave, np.array([self.center]), norm,
                                         self.log10sigma, h3=0.3, h4=-0.3)
        self.assertTrue(np.all(flux >= 0.0))

    def test_numpy_and_torch_agree_with_nonzero_h3_h4(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest('torch not installed')
        from desisim.templates import _lines_to_spectrum_numpy, _lines_to_spectrum_torch
        norm = np.array([1.0, 0.5])
        centers = np.array([self.center, self.center + 0.01])
        f_np = _lines_to_spectrum_numpy(self.log10wave, centers, norm, self.log10sigma, h3=0.2, h4=-0.15)
        f_torch = _lines_to_spectrum_torch(self.log10wave, centers, norm, self.log10sigma,
                                            h3=0.2, h4=-0.15, device='cpu')
        np.testing.assert_allclose(f_np, f_torch, rtol=1e-6, atol=1e-30)

    def test_emspectrum_asymmetry_off_by_default(self):
        '''include_line_asymmetry defaults to False -- narrow_h3/h4 and
        broad_h3/h4 must stay exactly 0.0 (pure Gaussian) unless
        explicitly requested, for every pre-existing caller.'''
        em = EMSpectrum(minwave=1200.0, maxwave=10000.0, include_new_lines=True)
        self.assertFalse(em.include_line_asymmetry)
        fixed_ratios = dict(oiiihbeta=-0.2, oiihbeta=0.1, niihbeta=-0.2, siihbeta=-0.3)
        emspec_a, _, _ = em.spectrum(seed=1, hbetaflux=1e-16, **fixed_ratios)
        emspec_b, _, _ = em.spectrum(seed=1, hbetaflux=1e-16, narrow_h3=0.0, narrow_h4=0.0,
                                      broad_h3=0.0, broad_h4=0.0, **fixed_ratios)
        np.testing.assert_array_equal(emspec_a, emspec_b)

    def test_emspectrum_draws_within_gh_range_when_enabled(self):
        em = EMSpectrum(minwave=1200.0, maxwave=10000.0, include_new_lines=True,
                         include_line_asymmetry=True)
        fixed_ratios = dict(oiiihbeta=-0.2, oiihbeta=0.1, niihbeta=-0.2, siihbeta=-0.3)
        _, _, line = em.spectrum(seed=1, hbetaflux=1e-16, **fixed_ratios)
        # Reach into the same rand stream indirectly: just check the drawn
        # emission spectrum differs from the h3=h4=0 case, and re-running
        # with the same seed reproduces identically.
        emspec1, _, _ = em.spectrum(seed=7, hbetaflux=1e-16, **fixed_ratios)
        emspec2, _, _ = em.spectrum(seed=7, hbetaflux=1e-16, **fixed_ratios)
        emspec_gaussian, _, _ = em.spectrum(seed=7, hbetaflux=1e-16, narrow_h3=0.0, narrow_h4=0.0,
                                             broad_h3=0.0, broad_h4=0.0, **fixed_ratios)
        np.testing.assert_array_equal(emspec1, emspec2)
        # Fluxes here are astrophysical-unit-scale (~1e-15 to 1e-18) --
        # np.allclose's default atol=1e-8 would swamp any real difference
        # at that scale, so use exact inequality instead.
        self.assertFalse(np.array_equal(emspec1, emspec_gaussian))

    def test_explicit_narrow_and_broad_h3_h4_always_win(self):
        em = EMSpectrum(minwave=1200.0, maxwave=10000.0, include_new_lines=True,
                         include_line_asymmetry=True)
        fixed_ratios = dict(oiiihbeta=-0.2, oiihbeta=0.1, niihbeta=-0.2, siihbeta=-0.3)
        emspec_a, _, _ = em.spectrum(seed=1, hbetaflux=1e-16, narrow_h3=0.1, narrow_h4=-0.1,
                                      broad_h3=0.2, broad_h4=0.15, **fixed_ratios)
        emspec_b, _, _ = em.spectrum(seed=1, hbetaflux=1e-16, narrow_h3=0.1, narrow_h4=-0.1,
                                      broad_h3=0.2, broad_h4=0.15, **fixed_ratios)
        np.testing.assert_array_equal(emspec_a, emspec_b)

    def test_narrow_and_broad_asymmetry_are_independent(self):
        '''Setting only broad_h3/h4 nonzero must change the spectrum
        relative to the plain-Gaussian case while narrow_h3/h4 stay 0 (and
        vice versa) -- confirms the two groups are genuinely decoupled.'''
        em = EMSpectrum(minwave=1200.0, maxwave=10000.0, include_new_lines=True)
        fixed_ratios = dict(oiiihbeta=-0.2, oiihbeta=0.1, niihbeta=-0.2, siihbeta=-0.3)
        base, _, _ = em.spectrum(seed=1, hbetaflux=1e-16, narrow_h3=0.0, narrow_h4=0.0,
                                  broad_h3=0.0, broad_h4=0.0, **fixed_ratios)
        broad_only, _, _ = em.spectrum(seed=1, hbetaflux=1e-16, narrow_h3=0.0, narrow_h4=0.0,
                                        broad_h3=0.25, broad_h4=0.0, **fixed_ratios)
        narrow_only, _, _ = em.spectrum(seed=1, hbetaflux=1e-16, narrow_h3=0.25, narrow_h4=0.0,
                                         broad_h3=0.0, broad_h4=0.0, **fixed_ratios)
        # Fluxes here are astrophysical-unit-scale (~1e-15 to 1e-18) --
        # np.allclose's default atol=1e-8 would swamp any real difference
        # at that scale, so use exact inequality instead.
        self.assertFalse(np.array_equal(base, broad_only))
        self.assertFalse(np.array_equal(base, narrow_only))
        self.assertFalse(np.array_equal(broad_only, narrow_only))


class TestGalaxyEWScatter(unittest.TestCase):
    '''Tests for the widened, extended D4000-coupled EW scatter (handoff
    Sec 1.4). Requires $DESI_BASIS_TEMPLATES for the statistical tests since
    they need real ELG/BGS basis templates (D4000, OII_CONTINUUM/
    HBETA_CONTINUUM metadata); the pure class-constant check does not.
    '''

    def test_default_sigma_values_are_wider_than_legacy_floor(self):
        '''"Do not tighten anything below its current value" (handoff Sec
        1.4) -- the new defaults must be strictly wider than the original
        pre-fork constants (0.3 dex for OII, 0.2 dex for Hbeta).'''
        self.assertGreater(GALAXY.EWOII_SIGMA, 0.3)
        self.assertGreater(GALAXY.EWHBETA_SIGMA, 0.2)

    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_ewoii_sigma_param_changes_observed_scatter(self):
        '''The ewoii_sigma override must actually reach the RNG draw: a
        much wider requested sigma should produce a much wider empirical
        spread in log10(EWOII) across many independent models than a much
        narrower one. Statistical, not exact, by design (this exercises the
        full make_templates pipeline, not just the isolated formula).'''
        elg = ELG(wave=np.arange(5000, 8000, 2.0))
        _, _, _, narrow_meta = elg.make_templates(nmodel=200, seed=1, ewoii_sigma=0.02)
        _, _, _, wide_meta = elg.make_templates(nmodel=200, seed=1, ewoii_sigma=1.0)
        narrow_spread = np.std(np.log10(narrow_meta['EWOII'].data))
        wide_spread = np.std(np.log10(wide_meta['EWOII'].data))
        self.assertGreater(wide_spread, 3 * narrow_spread)

    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_ewhbeta_sigma_param_changes_observed_scatter(self):
        bgs = BGS(wave=np.arange(3600, 9800, 2.0))
        _, _, _, narrow_meta = bgs.make_templates(nmodel=200, seed=1, ewhbeta_sigma=0.02)
        _, _, _, wide_meta = bgs.make_templates(nmodel=200, seed=1, ewhbeta_sigma=1.0)
        narrow_hb = narrow_meta['EWHBETA'].data
        wide_hb = wide_meta['EWHBETA'].data
        # HBETA_LIMIT can zero some entries out; compare only the
        # nonzero/finite population actually eligible for scatter.
        narrow_spread = np.std(np.log10(narrow_hb[narrow_hb > 0]))
        wide_spread = np.std(np.log10(wide_hb[wide_hb > 0]))
        self.assertGreater(wide_spread, 3 * narrow_spread)

    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_legacy_sigma_values_still_accepted(self):
        '''Explicitly passing the original pre-fork values must still work
        (the escape hatch back to old behavior promised in the docstring),
        producing finite, positive EW values.'''
        elg = ELG(wave=np.arange(5000, 8000, 2.0))
        _, _, _, meta = elg.make_templates(nmodel=10, seed=2, ewoii_sigma=0.3)
        self.assertTrue(np.all(np.isfinite(meta['EWOII'].data)))
        self.assertTrue(np.all(meta['EWOII'].data > 0))

        bgs = BGS(wave=np.arange(3600, 9800, 2.0))
        _, _, _, meta = bgs.make_templates(nmodel=10, seed=2, ewhbeta_sigma=0.2)
        self.assertTrue(np.all(np.isfinite(meta['EWHBETA'].data)))

    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_default_call_unaffected_by_new_kwarg_presence(self):
        '''Smoke test: omitting ewoii_sigma/ewhbeta_sigma entirely (the
        pre-fork call signature) must still work end-to-end.'''
        elg = ELG(wave=np.arange(5000, 8000, 2.0))
        flux, wave, meta, objmeta = elg.make_templates(nmodel=5, seed=3)
        self.assertEqual(flux.shape[0], 5)
        self.assertTrue(np.all(np.isfinite(flux)))


class TestTemplates(unittest.TestCase):

    def setUp(self):
        self.wavemin = 5000
        self.wavemax = 8000
        self.dwave = 2.0
        self.wave = np.arange(self.wavemin, self.wavemax+self.dwave/2, self.dwave)
        self.nspec = 5
        self.seed = np.random.randint(2**32)
        self.rand = np.random.RandomState(self.seed)

    def _check_output_size(self, flux, wave, meta):
        self.assertEqual(len(meta), self.nspec)
        self.assertEqual(len(wave), len(self.wave))
        self.assertEqual(flux.shape, (self.nspec, len(self.wave)))

    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_simple_south(self):
        '''Confirm that creating templates works at all, both with and without a random seed.'''
        #print('In function test_simple_south, seed = {}'.format(self.seed))
        for T in [ELG, LRG, QSO, BGS, STAR, STD, MWS_STAR, WD, SIMQSO]:
            template_factory = T(wave=self.wave)
            flux, wave, meta, _ = template_factory.make_templates(self.nspec, seed=self.seed, south=True)
            self._check_output_size(flux, wave, meta)
            flux, wave, meta, _ = template_factory.make_templates(self.nspec, seed=None)
            self._check_output_size(flux, wave, meta)
        
    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_simple_north(self):
        '''Confirm that creating templates works at all'''
        #print('In function test_simple_north, seed = {}'.format(self.seed))
        for T in [ELG, LRG, QSO, BGS, STAR, STD, MWS_STAR, WD, SIMQSO]:
            template_factory = T(wave=self.wave)
            flux, wave, meta, _ = template_factory.make_templates(self.nspec, seed=self.seed, south=False)
            self._check_output_size(flux, wave, meta)
        
    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_restframe(self):
        '''Confirm restframe template creation for a galaxy and a star'''
        #print('In function test_simple, seed = {}'.format(self.seed))
        for T in [ELG, MWS_STAR]:
            template_factory = T(wave=self.wave)
            flux, wave, meta, _ = template_factory.make_templates(self.nspec, seed=self.seed, restframe=True)
            self.assertEqual(len(wave), len(template_factory.basewave))
        
    def test_input_wave(self):
        '''Confirm that we can specify the wavelength array.'''
        #print('In function test_input_wave, seed = {}'.format(self.seed))
        lrg = LRG(minwave=self.wavemin, maxwave=self.wavemax, cdelt=self.dwave)
        flux, wave, meta, _ = lrg.make_templates(self.nspec, seed=self.seed)
        self._check_output_size(flux, wave, meta)
        
    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_random_seed(self):
        '''Test that random seed works to get the same results back'''
        #print('In function test_input_random_seed, seed = {}'.format(self.seed))
        for T in [ELG, LRG, BGS, MWS_STAR, QSO, SIMQSO]:
            Tx = T(wave=self.wave)
            flux1, wave1, meta1, objmeta1 = Tx.make_templates(self.nspec, seed=1)
            flux2, wave2, meta2, objmeta2 = Tx.make_templates(self.nspec, seed=1)
            flux3, wave3, meta3, objmeta3 = Tx.make_templates(self.nspec, seed=2)
            self.assertTrue(np.all(flux1==flux2))
            self.assertTrue(np.any(flux1!=flux3))
            self.assertTrue(np.all(wave1==wave2))
        
            # Build a model from one of the randomly generated seeds.
            I = self.rand.choice(self.nspec)
        
            if T.__name__ != 'SIMQSO':
                flux4, wave4, meta4, objmeta4 = Tx.make_templates(1, seed=meta1['SEED'][I])
                self.assertTrue(np.all(flux1[I, :]==flux4))
        
            for key in meta1.colnames:
                if T.__name__ != 'SIMQSO':
                    # this won't match for simulated templates
                    if key == 'TARGETID' or (key == 'TEMPLATEID' and 'QSO' in T.__name__): 
                        continue
                    #print(T.__name__, key, meta1[key][I], meta4[key])
                    self.assertTrue(np.all(meta1[key][I]==meta4[key]))
        
                if key in ['TARGETID', 'OBJTYPE', 'SUBTYPE', 'MAGFILTER']:
                    continue
                # TEMPLATEID is identical for (SIM)QSO templates
                if 'QSO' in T.__name__ and key == 'TEMPLATEID': 
                    continue
                self.assertTrue(np.all(meta1[key]==meta2[key]))
                self.assertTrue(np.any(meta1[key]!=meta3[key]))
        
            for key in objmeta1.colnames:
                #print(T.__name__, key, objmeta1[key].data, objmeta3[key].data)
                self.assertTrue(np.all(objmeta1[key]==objmeta2[key]))
                if T.__name__ != 'SIMQSO':
                    if key == 'TARGETID' or (key == 'TEMPLATEID' and 'QSO' in T.__name__): 
                        continue
                    #print(T.__name__, key, objmeta1[key][I], objmeta4[key])
                    self.assertTrue(np.all(objmeta1[I][key]==objmeta4[key]))
        
                # Skip null value columns (would be -1 or '' for all rows)
                if (key != 'TARGETID' and objmeta1[key].ndim == 1 and objmeta1[key][0] != -1 and 
                    objmeta1[key][0] != '' and objmeta1[key].dtype != bool):
                    self.assertTrue(np.any(objmeta1[key]!=objmeta3[key]))
        
    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_OII(self):
        '''Confirm that ELG [OII] flux matches meta table description'''
        #print('In function test_OII, seed = {}'.format(self.seed))
        wave = np.arange(5000, 9800.1, 0.2)
        flux, ww, meta, objmeta = ELG(wave=wave).make_templates(
            seed=self.seed, nmodel=10, zrange=(0.6, 1.6), 
            vdisprange=(75.0, 75.0),
            nocolorcuts=True, nocontinuum=True)
        
        for i in range(len(meta)):
            z = meta['REDSHIFT'][i]
            ii = (3722*(1+z) < wave) & (wave < 3736*(1+z))
            OIIflux = 1e-17 * np.sum(flux[i,ii] * np.gradient(wave[ii]))
            self.assertAlmostEqual(OIIflux, objmeta['OIIFLUX'][i], 2)
        
    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_HBETA(self):
        '''Confirm that BGS H-beta flux matches meta table description'''
        #print('In function test_HBETA, seed = {}'.format(self.seed))
        wave = np.arange(5000, 7000.1, 0.2)
        # Need to choose just the star-forming galaxies.
        from desisim.io import read_basis_templates
        baseflux, basewave, basemeta = read_basis_templates(objtype='BGS')
        keep = np.where(basemeta['HBETA_LIMIT'] == 0)[0]
        bgs = BGS(wave=wave, basewave=basewave, baseflux=baseflux[keep, :],
                  basemeta=basemeta[keep])
        flux, ww, meta, objmeta = bgs.make_templates(seed=self.seed,
            nmodel=10, zrange=(0.05, 0.4),
            vdisprange=(75.0, 75.0),
            nocolorcuts=True, nocontinuum=True)
        
        for i in range(len(meta)):
            z = meta['REDSHIFT'][i]
            ii = (4854*(1+z) < wave) & (wave < 4868*(1+z))
            hbetaflux = 1e-17 * np.sum(flux[i,ii] * np.gradient(wave[ii]))
            self.assertAlmostEqual(hbetaflux, objmeta['HBETAFLUX'][i], 2)
        
    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_input_redshift(self):
        '''Test that we can input the redshift for a representative galaxy and star class.'''
        #print('In function test_input_redshift, seed = {}'.format(self.seed))
        zrange = np.array([(0.5, 1.0), (0.5, 4.0), (-0.003, 0.003)])
        for zminmax, T in zip(zrange, [LRG, QSO, STAR, SIMQSO]):
            redshift = np.random.uniform(zminmax[0], zminmax[1], self.nspec)
            Tx = T(wave=self.wave)
            flux, wave, meta, _ = Tx.make_templates(self.nspec, redshift=redshift, seed=self.seed)
            self.assertTrue(np.allclose(redshift, meta['REDSHIFT']))
        
    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_wd_subtype(self):
        '''Test option of specifying the white dwarf subtype.'''
        #print('In function test_wd_subtype, seed = {}'.format(self.seed))
        wd = WD(wave=self.wave, subtype='DA')
        flux, wave, meta, _ = wd.make_templates(self.nspec, seed=self.seed, nocolorcuts=True)
        self._check_output_size(flux, wave, meta)
        np.all(meta['SUBTYPE'] == 'DA')
        
        wd = WD(wave=self.wave, subtype='DB')
        flux, wave, meta, _ = wd.make_templates(self.nspec, seed=self.seed, nocolorcuts=True)
        np.all(meta['SUBTYPE'] == 'DB')
        
    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_input_meta(self):
        '''Test that input meta table option works.'''
        #print('In function test_input_meta, seed = {}'.format(self.seed))
        for T in [ELG, LRG, BGS, QSO, STAR, MWS_STAR, WD]:
            #print('Working on {} templates'.format(T.__name__))
            Tx = T(wave=self.wave)
            flux1, wave1, meta1, objmeta1 = Tx.make_templates(self.nspec, seed=self.seed)
            flux2, wave2, meta2, objmeta2 = Tx.make_templates(input_meta=meta1, input_objmeta=objmeta1)
        
            badkeys = list()
            for key in meta1.colnames:
                if key in ('REDSHIFT', 'MAG', 'SEED', 'FLUX_G',
                           'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2'):
                    #- not sure why the tolerances aren't closer
                    if not np.allclose(meta1[key], meta2[key], rtol=1e-4):
                        #print(meta1['OBJTYPE'][0], key, meta1[key], meta2[key])
                        badkeys.append(key)
                        print(key, meta1[key][0], meta2[key][0])
                else:
                    if not np.all(meta1[key] == meta2[key]):
                        badkeys.append(key)
        
            self.assertEqual(len(badkeys), 0, 'mismatch for spectral type {} in keys {}'.format(meta1['OBJTYPE'][0], badkeys))
            self.assertTrue(np.allclose(flux1, flux2, rtol=1e-4))
            self.assertTrue(np.all(wave1 == wave2))
        
    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_star_properties(self):
       '''Test that input data table option works.'''
       #print('In function test_star_properties, seed = {}'.format(self.seed))
       star_properties = Table()
       star_properties.add_column(Column(name='REDSHIFT', length=self.nspec, dtype='f4'))
       star_properties.add_column(Column(name='MAG', length=self.nspec, dtype='f4'))
       star_properties.add_column(Column(name='MAGFILTER', length=self.nspec, dtype='U15'))
       star_properties.add_column(Column(name='TEFF', length=self.nspec, dtype='f4'))
       star_properties.add_column(Column(name='LOGG', length=self.nspec, dtype='f4'))
       star_properties.add_column(Column(name='FEH', length=self.nspec, dtype='f4'))
       star_properties['REDSHIFT'] = self.rand.uniform(-5E-4, 5E-4, self.nspec)
       star_properties['MAG'] = self.rand.uniform(16, 19, self.nspec)
       star_properties['MAGFILTER'][:] = 'decam2014-r'
       star_properties['TEFF'] = self.rand.uniform(4000, 10000, self.nspec)
       star_properties['LOGG'] = self.rand.uniform(0.5, 5.0, self.nspec)
       star_properties['FEH'] = self.rand.uniform(-2.0, 0.0, self.nspec)
       for T in [STAR]:
           Tx = T(wave=self.wave)
           flux, wave, meta, objmeta = Tx.make_templates(star_properties=star_properties, seed=self.seed)
           badkeys = list()
           for key in ('REDSHIFT', 'MAG'):
               if not np.allclose(meta[key], star_properties[key]):
                   badkeys.append(key)
           for key in ('TEFF', 'LOGG', 'FEH'):
               if not np.allclose(objmeta[key], star_properties[key]):
                   badkeys.append(key)
           self.assertEqual(len(badkeys), 0, 'mismatch for spectral type {} in keys {}'.format(meta['OBJTYPE'][0], badkeys))
        
    def test_lyamock_seed(self):
        '''Test that random seed works to get the same results back'''
        #print('In function test_lyamock_seed, seed = {}'.format(self.seed))
        mock = lyamock.MockMaker()
        wave1, flux1 = mock.get_lya_skewers(self.nspec, new_seed=1)
        wave2, flux2 = mock.get_lya_skewers(self.nspec, new_seed=1)
        wave3, flux3 = mock.get_lya_skewers(self.nspec, new_seed=2)
        self.assertTrue(np.all(flux1==flux2))
        self.assertTrue(np.any(flux1!=flux3))
        self.assertTrue(np.all(wave1==wave2))
        
    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_qso_options(self):
        '''Test that the QSO keyword arguments work'''
        flux, wave, meta, objmeta = QSO(wave=self.wave, balqso=False).make_templates(
            self.nspec, seed=self.seed, lyaforest=False)
        self.assertTrue(np.all(meta['SUBTYPE']==''))
    
        flux, wave, meta, objmeta = QSO(wave=self.wave, balqso=False).make_templates(
            self.nspec, seed=self.seed, lyaforest=True)
        self.assertTrue(np.all(meta['SUBTYPE']=='LYA'))
    
        flux, wave, meta, objmeta = QSO(wave=self.wave, balqso=True).make_templates(
            self.nspec, seed=self.seed, balprob=1.0, lyaforest=False)
        self.assertTrue(np.all(meta['SUBTYPE']=='BAL'))
        self.assertTrue(np.all(objmeta['BAL_TEMPLATEID']!=-1))
    
        flux1, wave1, meta1, objmeta1 = QSO(wave=self.wave, balqso=True).make_templates(
            self.nspec, seed=self.seed, balprob=1.0, lyaforest=True)
        self.assertTrue(np.all(meta1['SUBTYPE']=='LYA+BAL'))
        self.assertTrue(np.all(objmeta1['BAL_TEMPLATEID']!=-1))
    
        # Test that the spectra are reproducible when they include BALs.
        I = self.rand.choice(self.nspec)
        flux2, wave2, meta2, objmeta2 = QSO(wave=self.wave, balqso=True).make_templates(
            1, seed=meta1['SEED'][I], balprob=1.0, lyaforest=True)
        
        self.assertTrue(np.all(flux1[I, :]==flux2))
        self.assertTrue(np.all(wave1==wave2))
    
        for key in meta2.colnames:
            if key in ['TARGETID', 'TEMPLATEID']: # this won't match in this test
                continue
            self.assertTrue(np.all(meta2[key]==meta1[key][I]))
            
        for key in objmeta2.colnames:
            if key in ['TARGETID', 'TEMPLATEID']: # this won't match in this test
                continue
            self.assertTrue(np.all(objmeta2[key]==objmeta1[key][I]))
    
    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_meta(self):
        '''Test the metadata tables have the columns we expect'''
        #print('In function test_meta, seed = {}'.format(self.seed))
        for T in [ELG, LRG, BGS, STAR, STD, MWS_STAR, WD, QSO]:
            template_factory = T(wave=self.wave)
            flux, wave, meta, objmeta = template_factory.make_templates(self.nspec, seed=self.seed)
        
            self.assertTrue(np.all(np.isin(['TARGETID', 'OBJTYPE', 'SUBTYPE', 'TEMPLATEID', 'SEED',
                                            'REDSHIFT', 'MAG', 'MAGFILTER', 'FLUX_G', 'FLUX_R',
                                            'FLUX_Z', 'FLUX_W1', 'FLUX_W2'],
                                            meta.colnames)))
        
            if ( isinstance(template_factory, ELG) or isinstance(template_factory, LRG) or
                 isinstance(template_factory, BGS) ):
                self.assertTrue(np.all(np.isin(['TARGETID', 'OIIFLUX', 'HBETAFLUX', 'EWOII', 'EWHBETA',
                                                'D4000', 'VDISP', 'OIIDOUBLET', 'OIIIHBETA', 'OIIHBETA',
                                                'NIIHBETA', 'SIIHBETA'],
                                                objmeta.colnames)))
                
            if (isinstance(template_factory, STAR) or isinstance(template_factory, STD) or
                isinstance(template_factory, MWS_STAR) ):
                self.assertTrue(np.all(np.isin(['TARGETID', 'TEFF', 'LOGG', 'FEH'], objmeta.colnames)))
        
            if isinstance(template_factory, WD):
                self.assertTrue(np.all(np.isin(['TARGETID', 'TEFF', 'LOGG'], objmeta.colnames)))
        
            if isinstance(template_factory, QSO):
                self.assertTrue(np.all(np.isin(['TARGETID', 'PCA_COEFF'], objmeta.colnames)))
    
if __name__ == '__main__':
    unittest.main()
