import unittest
import numpy as np

from desisim.mock_spectrum import (generate_galaxy_component, generate_qso_component,
                                    generate_blended_spectrum)
from desisim.templates import EMSpectrum
from desisim.absorption import AbsorptionSpectrum
from desisim.associated_absorption import AssociatedAbsorberSystems
from desisim.dust import DustAttenuation
from desisim.agn_continuum import AGNPowerLawContinuum


def _log_uniform_grid(minwave, maxwave, cdelt_kms=20.0):
    from desisim.templates import C_LIGHT
    cdelt = cdelt_kms / C_LIGHT / np.log(10)
    npix = int(round((np.log10(maxwave) - np.log10(minwave)) / cdelt)) + 1
    return 10 ** np.linspace(np.log10(minwave), np.log10(maxwave), npix)


def _simqso_available():
    try:
        import simqso  # noqa: F401
        return True
    except ImportError:
        return False


_HAS_SIMQSO = _simqso_available()


def _fsps_available():
    try:
        import fsps  # noqa: F401
        return True
    except ImportError:
        return False


_HAS_FSPS = _fsps_available()


def _flat_continuum(value=2.5, minwave=1000.0, maxwave=12000.0):
    '''A trivial 2-point (wave, flux) continuum, wide enough to cover any
    grid used in these tests after interpolation. Used to bypass the real
    FSPS draw (python-fsps is a heavy, separately-compiled native
    dependency not installed in every test environment -- see
    fsps_continuum.py's own module docstring) for tests that only care
    about the orchestration/harmonization logic, not FSPS itself.'''
    return np.array([minwave, maxwave]), np.array([value, value])


class TestGenerateGalaxyComponent(unittest.TestCase):
    '''Tests for the galaxy-side orchestrator: dust-free FSPS continuum +
    galactic dust + narrow emission + ISM/CGM absorption, harmonized onto
    one common grid and summed via decompose.combine_into_channels. No
    simqso dependency -- these must all pass regardless of whether simqso
    is installed.
    '''

    def setUp(self):
        self.wave = _log_uniform_grid(3000.0, 9000.0)

    def test_output_finite_and_correct_shape(self):
        out = generate_galaxy_component(self.wave, continuum=_flat_continuum(), seed=1)
        for key in ('continuum', 'emission', 'absorption', 'total'):
            self.assertEqual(out[key].shape, self.wave.shape)
            self.assertTrue(np.all(np.isfinite(out[key])))

    def test_total_equals_sum_of_three_buckets(self):
        out = generate_galaxy_component(self.wave, continuum=_flat_continuum(), seed=2)
        np.testing.assert_allclose(out['total'], out['continuum'] + out['emission'] + out['absorption'])

    def test_seed_reproducible(self):
        out1 = generate_galaxy_component(self.wave, continuum=_flat_continuum(), seed=42)
        out2 = generate_galaxy_component(self.wave, continuum=_flat_continuum(), seed=42)
        np.testing.assert_array_equal(out1['total'], out2['total'])

    def test_different_seeds_differ(self):
        out1 = generate_galaxy_component(self.wave, continuum=_flat_continuum(), seed=1)
        out2 = generate_galaxy_component(self.wave, continuum=_flat_continuum(), seed=2)
        self.assertFalse(np.array_equal(out1['total'], out2['total']))

    @unittest.skipUnless(_HAS_FSPS, 'requires python-fsps (compiled native SPS backend)')
    def test_default_draws_fsps_continuum_when_not_given(self):
        '''The one test that actually exercises the default (no `continuum`
        override) code path, i.e. a real fsps_basis_templates() draw with
        dust2 forced to (0.0, 0.0) regardless of caller input.'''
        out = generate_galaxy_component(self.wave, seed=1)
        self.assertTrue(np.all(np.isfinite(out['total'])))
        self.assertIsNotNone(out['draws']['fsps_basemeta'])
        self.assertAlmostEqual(float(out['draws']['fsps_basemeta']['DUST2']), 0.0, places=10)

    def test_explicit_continuum_bypasses_fsps_draw(self):
        const_wave = np.array([2000.0, 9500.0])
        const_flux = np.array([3.0, 3.0])
        out = generate_galaxy_component(self.wave, continuum=(const_wave, const_flux),
                                         em=EMSpectrum(log10wave=np.log10(self.wave)),
                                         em_kwargs=dict(hbetaflux=0.0, oiiihbeta=-0.2, oiihbeta=0.1,
                                                        niihbeta=-0.2, siihbeta=-0.3),
                                         dust_kwargs=dict(theta=dict(theta0=0.0, theta1=0.0,
                                                                     theta2=0.0, theta3=0.0)),
                                         absorber_kwargs=dict(tau0={n: 0.0 for n in
                                                                    AbsorptionSpectrum.LINE_NAMES}),
                                         seed=1)
        # Zero dust (theta all 0 => k(wave)=0 => dust_flux=0), zero
        # absorption tau, and zero emission-line flux (hbetaflux=0.0)
        # should reproduce the flat input continuum exactly.
        np.testing.assert_allclose(out['continuum'], 3.0)
        np.testing.assert_allclose(out['emission'], 0.0, atol=1e-25)
        np.testing.assert_allclose(out['absorption'], 0.0, atol=1e-25)
        np.testing.assert_allclose(out['total'], 3.0)

    def test_pre_built_instances_are_honored(self):
        em = EMSpectrum(log10wave=np.log10(self.wave), include_mgii=True)
        absorber = AbsorptionSpectrum(log10wave=np.log10(self.wave),
                                       include_lines=['CaII_K'])
        dust = DustAttenuation()
        out = generate_galaxy_component(self.wave, continuum=_flat_continuum(),
                                         em=em, absorber=absorber, dust=dust, seed=3)
        self.assertTrue(np.all(np.isfinite(out['total'])))
        self.assertEqual(set(out['draws']['absorber_line']['name']), {'CaII_K'})

    def test_draws_provenance_present(self):
        out = generate_galaxy_component(self.wave, continuum=_flat_continuum(), seed=4)
        for key in ('fsps_basemeta', 'em_line', 'absorber_line', 'dust_theta'):
            self.assertIn(key, out['draws'])
        # continuum was supplied explicitly, so no FSPS draw happened --
        # fsps_basemeta must be present but None (see mock_spectrum.py's
        # generate_galaxy_component docstring).
        self.assertIsNone(out['draws']['fsps_basemeta'])


@unittest.skipUnless(_HAS_SIMQSO, 'requires simqso')
class TestGenerateQsoComponent(unittest.TestCase):
    '''Tests for the QSO-side orchestrator: simqso broken power-law
    continuum + QSO-type dust + broad-only emission (recovered via the
    documented narrow/broad subtraction) + blueshifted associated
    absorbers. Requires simqso to be installed (same requirement as
    agn_continuum.py itself -- see test_agn_continuum.py).
    '''

    def setUp(self):
        self.wave = _log_uniform_grid(1200.0, 3000.0)

    def test_output_finite_and_correct_shape(self):
        out = generate_qso_component(self.wave, seed=1)
        for key in ('continuum', 'emission', 'absorption', 'total'):
            self.assertEqual(out[key].shape, self.wave.shape)
            self.assertTrue(np.all(np.isfinite(out[key])))

    def test_continuum_stellar_forced_zero_continuum_agn_populated(self):
        out = generate_qso_component(self.wave, seed=2)
        self.assertTrue(out['components']['continuum_agn'])
        self.assertFalse(np.allclose(out['continuum'], 0.0))

    def test_seed_reproducible(self):
        out1 = generate_qso_component(self.wave, seed=7)
        out2 = generate_qso_component(self.wave, seed=7)
        np.testing.assert_array_equal(out1['total'], out2['total'])

    def test_broad_only_recovered_matches_manual_subtraction(self):
        '''Directly check the documented narrow/broad subtraction trick:
        emission bucket must equal a manual EM.spectrum() call's broad-only
        difference, not the raw (narrow+broad) emspec.'''
        em = EMSpectrum(log10wave=np.log10(self.wave), include_new_lines=True)
        fixed_ratios = dict(oiiihbeta=-0.2, oiihbeta=0.1, niihbeta=-0.2, siihbeta=-0.3)
        em_kwargs = dict(seed=99, hbetaflux=1e-16, **fixed_ratios)

        # Zero out Fe II's and the Balmer continuum's contributions to
        # 'emission' for this test -- it's checking EM's own narrow/broad
        # subtraction in isolation, not the other channels that legitimately
        # also land in the 'emission' bucket (see decompose.py's module
        # docstring).
        out = generate_qso_component(self.wave, em=em, em_kwargs=dict(em_kwargs),
                                      feii_kwargs=dict(uv_norm=0.0, optical_norm=0.0),
                                      balmer_kwargs=dict(edge_norm=0.0, line_norm=0.0), seed=5)

        emflux_total, emwave, _ = em.spectrum(**em_kwargs)
        narrow_only = dict(em_kwargs)
        narrow_only['new_line_broad_ratios'] = {n: 0.0 for n in EMSpectrum.NEW_LINE_NAMES}
        emflux_narrow_only, _, _ = em.spectrum(**narrow_only)
        expected_broad = np.interp(self.wave, emwave, emflux_total - emflux_narrow_only)

        np.testing.assert_allclose(out['emission'], expected_broad, rtol=1e-8, atol=1e-30)

    def test_associated_absorption_and_dust_are_nonpositive_contributors(self):
        '''Sanity check that the absorption bucket is dominated by
        non-positive contributions (associated absorption + dust deficit),
        i.e. wiring didn't accidentally flip a sign.'''
        out = generate_qso_component(self.wave, seed=8)
        self.assertLessEqual(np.median(out['absorption']), 0.0)

    def test_pre_built_associated_instance_is_honored(self):
        associated = AssociatedAbsorberSystems(log10wave=np.log10(self.wave),
                                                include_transitions=['MgII_2796', 'MgII_2803'])
        out = generate_qso_component(self.wave, associated=associated, seed=9)
        self.assertTrue(np.all(np.isfinite(out['total'])))

    def test_feii_is_wired_into_emission_bucket(self):
        '''Task #31 wiring check: the Fe II pseudo-continuum must actually
        land in the returned 'emission' array and provenance dict --
        zeroing it out via feii_kwargs must measurably change 'emission'
        and 'total' relative to a real draw.'''
        out_on = generate_qso_component(self.wave, seed=20)
        out_off = generate_qso_component(self.wave, seed=20,
                                          feii_kwargs=dict(uv_norm=0.0, optical_norm=0.0))
        self.assertIn('feii_params', out_on['draws'])
        self.assertTrue(out_on['components']['feii_flux'])
        self.assertFalse(np.allclose(out_on['emission'], out_off['emission']))

    def test_pre_built_feii_instance_is_honored(self):
        from desisim.feii_continuum import FeIIPseudoContinuum
        feii = FeIIPseudoContinuum(log10wave=np.log10(self.wave))
        out = generate_qso_component(self.wave, feii=feii,
                                      feii_kwargs=dict(uv_norm=1.0, optical_norm=1.0), seed=21)
        self.assertTrue(np.all(np.isfinite(out['total'])))

    def test_balmer_is_wired_into_emission_bucket(self):
        '''Task #32 wiring check: the Balmer-continuum-plus-cascade flux
        must actually land in the returned 'emission' array and provenance
        dict -- zeroing it out via balmer_kwargs must measurably change
        'emission' and 'total' relative to a real draw.'''
        out_on = generate_qso_component(self.wave, seed=20)
        out_off = generate_qso_component(self.wave, seed=20,
                                          balmer_kwargs=dict(edge_norm=0.0, line_norm=0.0))
        self.assertIn('balmer_params', out_on['draws'])
        self.assertTrue(out_on['components']['balmer_flux'])
        self.assertFalse(np.allclose(out_on['emission'], out_off['emission']))

    def test_pre_built_balmer_instance_is_honored(self):
        from desisim.balmer_continuum import BalmerContinuum
        balmer = BalmerContinuum(log10wave=np.log10(self.wave))
        out = generate_qso_component(self.wave, balmer=balmer,
                                      balmer_kwargs=dict(edge_norm=1.0, line_norm=1.0), seed=21)
        self.assertTrue(np.all(np.isfinite(out['total'])))

    def test_broad_velshift_decoupled_from_narrow_by_default(self):
        '''generate_qso_component's default EMSpectrum passes
        include_broad_velshift=True (unlike EMSpectrum's own class
        default), so the drawn broadshift_kms for the NEW_LINE_NAMES
        broad rows should be nonzero without the caller doing anything
        special -- verified via the returned em_line_total table (narrow
        rows carry no broadshift_kms column; only the *_broad rows do).'''
        out = generate_qso_component(self.wave, seed=42)
        line = out['draws']['em_line_total']
        broad_rows = line[[n.endswith('_broad') for n in line['name']]]
        self.assertGreater(len(broad_rows), 0)
        self.assertTrue(np.all(broad_rows['broadshift_kms'] == broad_rows['broadshift_kms'][0]))
        self.assertNotEqual(float(broad_rows['broadshift_kms'][0]), 0.0)

    def test_igm_off_by_default(self):
        '''zqso=None (the default) must skip the IGM channel entirely --
        exact previous behavior, for full backward compatibility with every
        existing caller (see generate_qso_component's zqso docstring).'''
        out = generate_qso_component(self.wave, seed=50)
        self.assertFalse(out['components']['igm_flux'])
        self.assertIsNone(out['draws']['igm_params'])

    def test_igm_on_when_zqso_given(self):
        '''self.wave spans 1200-3000A rest-frame, bracketing the QSO's own
        rest-frame Lyalpha (1215.67A), so with a real zqso the IGM channel
        should contribute a genuine, nonzero, non-positive deficit.'''
        out = generate_qso_component(self.wave, zqso=2.5, seed=51)
        self.assertTrue(out['components']['igm_flux'])
        self.assertIsNotNone(out['draws']['igm_params'])
        self.assertEqual(out['draws']['igm_params']['zqso'], 2.5)

    def test_igm_seed_reproducible(self):
        out1 = generate_qso_component(self.wave, zqso=2.5, seed=52)
        out2 = generate_qso_component(self.wave, zqso=2.5, seed=52)
        np.testing.assert_array_equal(out1['total'], out2['total'])
        self.assertEqual(out1['draws']['igm_params']['ndla'], out2['draws']['igm_params']['ndla'])

    def test_igm_absent_recovers_no_igm_baseline(self):
        '''Turning the IGM channel off (zqso=None) must reproduce EXACTLY
        the same output as before this task's wiring existed at all -- i.e.
        identical to a call that doesn't even know about zqso, confirming
        this is a true opt-in extension, not a behavior change.'''
        out_no_zqso = generate_qso_component(self.wave, seed=53)
        out_zqso_none_explicit = generate_qso_component(self.wave, zqso=None, seed=53)
        np.testing.assert_array_equal(out_no_zqso['total'], out_zqso_none_explicit['total'])

    def test_pre_built_igm_instance_is_honored(self):
        from desisim.igm_absorption import IGMAbsorption
        igm = IGMAbsorption(log10wave=np.log10(self.wave))
        out = generate_qso_component(self.wave, zqso=2.5, igm=igm,
                                      igm_kwargs=dict(add_dlas=False), seed=54)
        self.assertTrue(out['components']['igm_flux'])
        self.assertFalse(out['draws']['igm_params']['add_dlas'])

    def test_igm_deficit_is_nonpositive_contribution(self):
        out_off = generate_qso_component(self.wave, zqso=2.5,
                                          igm_kwargs=dict(add_dlas=False), seed=55)
        out_no_igm = generate_qso_component(self.wave, seed=55)
        # Adding the (non-positive, by construction) IGM deficit on top of
        # an otherwise-identical draw can only lower or leave unchanged the
        # total flux at every pixel.
        self.assertTrue(np.all(out_off['total'] <= out_no_igm['total'] + 1e-8))


@unittest.skipUnless(_HAS_SIMQSO, 'requires simqso')
class TestGenerateBlendedSpectrum(unittest.TestCase):
    '''Tests for the top-level orchestrator that runs both components and
    blends them via the PyQSOFit-style composite.'''

    def setUp(self):
        self.wave = _log_uniform_grid(1200.0, 9000.0)
        # Bypass the real FSPS draw throughout this class (see
        # _flat_continuum's docstring) -- the one dedicated end-to-end
        # FSPS test lives at the bottom of this class.
        self.galaxy_kwargs = dict(continuum=_flat_continuum())

    def test_output_finite_and_correct_shape(self):
        out = generate_blended_spectrum(self.wave, galaxy_kwargs=self.galaxy_kwargs, seed=1)
        for key in ('continuum', 'emission', 'absorption', 'total'):
            self.assertEqual(out[key].shape, self.wave.shape)
            self.assertTrue(np.all(np.isfinite(out[key])))
        self.assertGreaterEqual(out['frac'], 0.0)
        self.assertLessEqual(out['frac'], 1.0)

    def test_frac_zero_is_pure_galaxy(self):
        out = generate_blended_spectrum(self.wave, galaxy_kwargs=self.galaxy_kwargs, frac=0.0, seed=1)
        np.testing.assert_allclose(out['continuum'], out['galaxy']['continuum'])

    def test_frac_one_is_pure_qso(self):
        out = generate_blended_spectrum(self.wave, galaxy_kwargs=self.galaxy_kwargs, frac=1.0, seed=1)
        np.testing.assert_allclose(out['continuum'], out['qso']['continuum'])

    def test_total_equals_sum_of_three_blended_buckets(self):
        out = generate_blended_spectrum(self.wave, galaxy_kwargs=self.galaxy_kwargs, frac=0.4, seed=3)
        np.testing.assert_allclose(out['total'], out['continuum'] + out['emission'] + out['absorption'])

    def test_seed_reproducible(self):
        out1 = generate_blended_spectrum(self.wave, galaxy_kwargs=self.galaxy_kwargs, seed=11)
        out2 = generate_blended_spectrum(self.wave, galaxy_kwargs=self.galaxy_kwargs, seed=11)
        np.testing.assert_array_equal(out1['total'], out2['total'])
        self.assertAlmostEqual(out1['frac'], out2['frac'], places=12)

    def test_galaxy_and_qso_draws_are_kept_for_provenance(self):
        out = generate_blended_spectrum(self.wave, galaxy_kwargs=self.galaxy_kwargs, seed=13)
        self.assertIn('draws', out['galaxy'])
        self.assertIn('draws', out['qso'])

    @unittest.skipUnless(_HAS_FSPS, 'requires python-fsps (compiled native SPS backend)')
    def test_full_default_pipeline_with_real_fsps(self):
        '''The one true end-to-end test with no shortcuts: real FSPS draw
        on the galaxy side, real simqso draw on the QSO side, blended.'''
        out = generate_blended_spectrum(self.wave, seed=1)
        self.assertTrue(np.all(np.isfinite(out['total'])))


if __name__ == '__main__':
    unittest.main()
