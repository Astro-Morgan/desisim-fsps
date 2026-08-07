import unittest
import numpy as np

from desisim.associated_absorption import AssociatedAbsorberSystems


class TestAssociatedAbsorberSystems(unittest.TestCase):
    '''Tests for the stochastic multi-system narrow associated-absorption
    model: Poisson-process system count/placement blueward of systemic,
    Maxwell-Boltzmann per-system velocity dispersion, shared kinematics
    across each system's own transition set. See
    associated_absorption.py's module docstring for the full statistical
    derivation and why this replaces the earlier (incorrect)
    parametric-BAL / two-component-dust tasks.
    '''

    def setUp(self):
        self.wave = np.linspace(700.0, 3000.0, 20000)
        self.flux = np.full_like(self.wave, 1e-16)
        self.sys = AssociatedAbsorberSystems()

    def test_output_finite_and_nonpositive(self):
        for seed in range(10):
            dflux, wave, table = self.sys.spectrum(self.wave, self.flux, seed=seed)
            self.assertTrue(np.all(np.isfinite(dflux)))
            self.assertTrue(np.all(dflux <= 1e-25))
            # dflux/wave live on the object's own internal log-uniform
            # grid (like AbsorptionSpectrum), not the caller's continuum
            # grid -- compare shapes against `wave`, not self.wave.
            self.assertEqual(dflux.shape, wave.shape)

    def test_zero_systems_gives_zero_deficit_and_empty_table(self):
        dflux, _, table = self.sys.spectrum(self.wave, self.flux, n_systems=0, seed=1)
        np.testing.assert_allclose(dflux, 0.0, atol=1e-30)
        self.assertEqual(len(table), 0)

    def test_explicit_n_systems_produces_expected_row_count(self):
        _, _, table = self.sys.spectrum(self.wave, self.flux, n_systems=3, seed=1)
        self.assertEqual(len(table), 3 * len(self.sys.include_transitions))
        self.assertEqual(set(table['system'].tolist()), {0, 1, 2})

    def test_poisson_count_distribution_matches_mean(self):
        '''Empirical sanity check on the Poisson-process count: over many
        draws, the mean number of systems should be close to
        MEAN_N_SYSTEMS (statistical test, generous tolerance).'''
        n_draws = 400
        counts = []
        for seed in range(n_draws):
            _, _, table = self.sys.spectrum(self.wave, self.flux, seed=seed)
            counts.append(len(set(table['system'].tolist())) if len(table) else 0)
        mean_count = np.mean(counts)
        # Poisson(1.5) std over 400 draws of the sample mean is
        # sqrt(1.5/400) ~ 0.06 -- use a generous 5-sigma-ish window.
        self.assertAlmostEqual(mean_count, AssociatedAbsorberSystems.MEAN_N_SYSTEMS, delta=0.3)

    def test_velocities_are_negative_blueward_and_within_v_max(self):
        for seed in range(20):
            _, _, table = self.sys.spectrum(self.wave, self.flux, n_systems=4, seed=seed)
            velocities = np.unique(table['velocity_kms'].data)
            self.assertTrue(np.all(velocities <= 0.0))
            self.assertTrue(np.all(velocities >= -AssociatedAbsorberSystems.V_MAX_KMS))

    def test_explicit_velocities_kms_override(self):
        v = np.array([-200.0, -1500.0])
        _, _, table = self.sys.spectrum(self.wave, self.flux, n_systems=2, velocities_kms=v, seed=1)
        got = sorted(np.unique(table['velocity_kms'].data).tolist())
        self.assertEqual(got, sorted(v.tolist()))

    def test_mismatched_velocities_length_raises(self):
        with self.assertRaises(ValueError):
            self.sys.spectrum(self.wave, self.flux, n_systems=2,
                               velocities_kms=np.array([-100.0]), seed=1)

    def test_explicit_sigma_kms_override(self):
        sigma = np.array([75.0])
        _, _, table = self.sys.spectrum(self.wave, self.flux, n_systems=1,
                                         sigma_kms=sigma, seed=1)
        self.assertTrue(np.allclose(np.unique(table['sigma_kms'].data), sigma))

    def test_mismatched_sigma_length_raises(self):
        with self.assertRaises(ValueError):
            self.sys.spectrum(self.wave, self.flux, n_systems=2,
                               sigma_kms=np.array([50.0]), seed=1)

    def test_sigma_kms_nonnegative_maxwell_draw(self):
        '''Maxwell-Boltzmann is supported on [0, inf) -- every drawn
        sigma_kms must be strictly positive.'''
        for seed in range(20):
            _, _, table = self.sys.spectrum(self.wave, self.flux, n_systems=3, seed=seed)
            self.assertTrue(np.all(table['sigma_kms'].data > 0.0))

    def test_each_system_shares_kinematics_across_its_transitions(self):
        '''Within one system, every transition must share the exact same
        velocity offset and sigma_kms -- only tau0 varies per transition
        (this is the "same physical cloud, different ions" convention).'''
        _, _, table = self.sys.spectrum(self.wave, self.flux, n_systems=1, seed=1)
        self.assertEqual(len(set(table['velocity_kms'].data.tolist())), 1)
        self.assertEqual(len(set(table['sigma_kms'].data.tolist())), 1)
        self.assertGreater(len(set(table['tau0'].data.tolist())), 1)

    def test_include_transitions_subsets_correctly(self):
        sys_subset = AssociatedAbsorberSystems(include_transitions=['MgII_2796', 'MgII_2803'])
        _, _, table = sys_subset.spectrum(self.wave, self.flux, n_systems=2, seed=1)
        self.assertEqual(set(table['transition'].tolist()), {'MgII_2796', 'MgII_2803'})
        self.assertEqual(len(table), 4)

    def test_unknown_transition_name_raises(self):
        with self.assertRaises(ValueError):
            AssociatedAbsorberSystems(include_transitions=['NotARealLine'])

    def test_explicit_tau0_override_per_system(self):
        tau0 = [dict(MgII_2796=0.9, MgII_2803=0.45)]
        sys_subset = AssociatedAbsorberSystems(include_transitions=['MgII_2796', 'MgII_2803'])
        _, _, table = sys_subset.spectrum(self.wave, self.flux, n_systems=1, tau0=tau0, seed=1)
        row_2796 = table[table['transition'] == 'MgII_2796']
        row_2803 = table[table['transition'] == 'MgII_2803']
        self.assertAlmostEqual(float(row_2796['tau0'][0]), 0.9, places=10)
        self.assertAlmostEqual(float(row_2803['tau0'][0]), 0.45, places=10)

    def test_seed_reproducible_different_seed_not(self):
        # Fix n_systems so the comparison exercises the velocity/sigma/
        # tau0 draws specifically, rather than risking both seeds
        # independently drawing n_systems=0 (a real ~22% outcome at
        # MEAN_N_SYSTEMS=1.5, which would make d1==d3 trivially).
        d1, _, t1 = self.sys.spectrum(self.wave, self.flux, n_systems=2, seed=11)
        d2, _, t2 = self.sys.spectrum(self.wave, self.flux, n_systems=2, seed=11)
        d3, _, t3 = self.sys.spectrum(self.wave, self.flux, n_systems=2, seed=12)
        np.testing.assert_array_equal(d1, d2)
        self.assertFalse(np.array_equal(d1, d3))

    def test_torch_and_numpy_backends_agree(self):
        a_np, _, _ = self.sys.spectrum(self.wave, self.flux, n_systems=2, seed=5, backend='numpy')
        a_torch, _, _ = self.sys.spectrum(self.wave, self.flux, n_systems=2, seed=5,
                                           backend='torch', device='cpu')
        np.testing.assert_allclose(a_np, a_torch, rtol=1e-6, atol=1e-30)

    def test_zshift_moves_line_centers(self):
        '''A nonzero (small) QSO systemic zshift should shift the
        observed trough redward relative to zshift=0, on top of whatever
        blueward velocity offset is applied. Uses a small zshift so the
        shifted Mg II 2796A line stays within this object's rest-frame-ish
        [700,3000]A output window (a large zshift, e.g. 1.0, would push
        Mg II to ~5592A -- entirely outside the window, which is a grid-
        design limitation to respect in the test, not a bug to probe).'''
        v = np.array([-500.0])
        sys_subset = AssociatedAbsorberSystems(include_transitions=['MgII_2796'])
        d0, wave0, _ = sys_subset.spectrum(self.wave, self.flux, n_systems=1,
                                            velocities_kms=v, zshift=0.0, seed=1)
        d1, wave1, _ = sys_subset.spectrum(self.wave, self.flux, n_systems=1,
                                            velocities_kms=v, zshift=0.01, seed=1)
        np.testing.assert_array_equal(wave0, wave1)
        idx0 = np.argmin(d0)
        idx1 = np.argmin(d1)
        self.assertGreater(wave0[idx1], wave0[idx0])

    def test_default_priors_respect_nonnegativity_and_scale_constants(self):
        self.assertGreater(AssociatedAbsorberSystems.MEAN_N_SYSTEMS, 0.0)
        self.assertGreater(AssociatedAbsorberSystems.V_MAX_KMS, 0.0)
        self.assertGreater(AssociatedAbsorberSystems.MB_SCALE_KMS, 0.0)


if __name__ == '__main__':
    unittest.main()
