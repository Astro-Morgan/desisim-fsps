import os
import unittest
import numpy as np
from astropy.table import Table, Column
from desisim.templates import ELG, LRG, QSO, BGS, STAR, STD, MWS_STAR, WD, SIMQSO, EMSpectrum
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
