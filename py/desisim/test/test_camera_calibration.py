import unittest
import numpy as np

from desisim.camera_calibration import CameraCalibration


class TestCameraCalibration(unittest.TestCase):
    '''Unit/integration tests for handoff task 9: free-parameter, per-camera
    Legendre-polynomial calibration artifacts (Anand et al. 2024, AJ 168,
    124, Eqn 1). No external DESI data products required.
    '''

    def setUp(self):
        # Full DESI-like span, fine enough to resolve camera boundaries
        # precisely.
        self.wave = np.arange(3600.0, 9824.0, 0.8)
        self.cc = CameraCalibration()

    def _rows(self, table, camera):
        return table[table['camera'] == camera]

    def test_default_camera_ranges_match_literature(self):
        self.assertEqual(CameraCalibration.CAMERA_WAVE_RANGES['b'], (3600.0, 5800.0))
        self.assertEqual(CameraCalibration.CAMERA_WAVE_RANGES['r'], (5760.0, 7620.0))
        self.assertEqual(CameraCalibration.CAMERA_WAVE_RANGES['z'], (7520.0, 9824.0))

    def test_assignment_boundaries_split_overlaps_at_midpoint(self):
        bounds = self.cc._camera_assignment_boundaries()
        self.assertEqual(bounds['b'], (3600.0, 5780.0))   # midpoint(5760,5800)=5780
        self.assertEqual(bounds['r'], (5780.0, 7570.0))   # midpoint(7520,7620)=7570
        self.assertEqual(bounds['z'], (7570.0, 9824.0))

    def test_default_legendre_degree_is_one(self):
        '''Literature default: 2 terms per camera (constant + slope),
        matching Anand et al. 2024's actual implementation.'''
        self.assertEqual(CameraCalibration.DEFAULT_LEGENDRE_DEGREE, 1)
        _, _, table = self.cc.spectrum(self.wave, seed=1)
        for cam in ('b', 'r', 'z'):
            self.assertEqual(len(self._rows(table, cam)), 2)

    def test_output_shape_and_finite(self):
        calib, wave, table = self.cc.spectrum(self.wave, seed=1)
        self.assertEqual(calib.shape, self.wave.shape)
        self.assertTrue(np.all(np.isfinite(calib)))
        np.testing.assert_array_equal(wave, self.wave)

    def test_zero_coeffs_give_zero_calibration_flux(self):
        zero = {cam: np.zeros(2) for cam in CameraCalibration.CAMERA_WAVE_RANGES}
        calib, _, _ = self.cc.spectrum(self.wave, coeffs=zero, seed=1)
        np.testing.assert_allclose(calib, 0.0, atol=1e-30)

    def test_explicit_coeffs_always_win(self):
        explicit = {'b': np.array([1.5, -0.3]), 'r': np.array([0.2, 0.4]), 'z': np.array([-1.0, 0.0])}
        _, _, table = self.cc.spectrum(self.wave, coeffs=explicit, seed=1)
        for cam, vals in explicit.items():
            rows = self._rows(table, cam)
            np.testing.assert_allclose(rows['coeff'].data, vals)

    def test_wrong_length_explicit_coeffs_raises(self):
        with self.assertRaises(ValueError):
            self.cc.spectrum(self.wave, coeffs={'b': np.array([1.0, 2.0, 3.0])}, seed=1)

    def test_constant_term_reproduces_flat_offset(self):
        '''i=0 Legendre polynomial P_0(x)=1 everywhere, so a pure constant
        coefficient must give a flat additive offset across the whole
        camera, independent of wavelength.'''
        coeffs = {'b': np.array([2.5, 0.0]), 'r': np.zeros(2), 'z': np.zeros(2)}
        calib, wave, _ = self.cc.spectrum(self.wave, coeffs=coeffs, seed=1)
        bounds = self.cc._camera_assignment_boundaries()
        bmask = (wave >= bounds['b'][0]) & (wave < bounds['b'][1])
        np.testing.assert_allclose(calib[bmask], 2.5)

    def test_slope_term_matches_reduced_wavelength_formula(self):
        '''i=1 Legendre polynomial P_1(x)=x, so a pure slope coefficient of
        1.0 must reproduce the exact reduced-wavelength formula
        lambda' = 2*(lambda-lambda_min)/(lambda_max-lambda_min) - 1 from
        Anand et al. 2024's footnote 8, using the camera's TRUE (not
        assignment-split) wavelength range.'''
        coeffs = {'b': np.array([0.0, 1.0]), 'r': np.zeros(2), 'z': np.zeros(2)}
        calib, wave, _ = self.cc.spectrum(self.wave, coeffs=coeffs, seed=1)
        bounds = self.cc._camera_assignment_boundaries()
        bmask = (wave >= bounds['b'][0]) & (wave < bounds['b'][1])
        wmin, wmax = CameraCalibration.CAMERA_WAVE_RANGES['b']
        expected = 2.0 * (wave[bmask] - wmin) / (wmax - wmin) - 1.0
        np.testing.assert_allclose(calib[bmask], expected, atol=1e-10)

    def test_wave_outside_every_camera_gives_zero(self):
        wave = np.array([1000.0, 2000.0, 3599.0, 9825.0, 20000.0])
        calib, _, _ = self.cc.spectrum(wave, seed=1)
        np.testing.assert_allclose(calib, 0.0)

    def test_seed_reproducible_different_seed_not(self):
        c1, _, t1 = self.cc.spectrum(self.wave, seed=5)
        c2, _, t2 = self.cc.spectrum(self.wave, seed=5)
        c3, _, t3 = self.cc.spectrum(self.wave, seed=6)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(t1['coeff'].data, t2['coeff'].data)
        self.assertFalse(np.array_equal(c1, c3))

    def test_coeff_sigma_override(self):
        '''Zero sigma must deterministically draw all-zero coefficients
        (and hence all-zero calibration flux) regardless of seed.'''
        cc = CameraCalibration(coeff_sigma=0.0)
        calib1, _, table1 = cc.spectrum(self.wave, seed=1)
        calib2, _, table2 = cc.spectrum(self.wave, seed=99)
        np.testing.assert_allclose(table1['coeff'].data, 0.0)
        np.testing.assert_allclose(table2['coeff'].data, 0.0)
        np.testing.assert_allclose(calib1, 0.0)
        np.testing.assert_allclose(calib2, 0.0)

    def test_custom_legendre_degree(self):
        cc = CameraCalibration(legendre_degree=3)
        _, _, table = cc.spectrum(self.wave, seed=1)
        for cam in ('b', 'r', 'z'):
            self.assertEqual(len(self._rows(table, cam)), 4)

    def test_custom_camera_wave_ranges_no_overlap(self):
        '''Non-overlapping custom ranges must produce assignment boundaries
        identical to the input ranges (no midpoint-splitting needed).'''
        custom = {'b': (4000.0, 5000.0), 'r': (5000.0, 6000.0)}
        cc = CameraCalibration(camera_wave_ranges=custom)
        bounds = cc._camera_assignment_boundaries()
        self.assertEqual(bounds['b'], (4000.0, 5000.0))
        self.assertEqual(bounds['r'], (5000.0, 6000.0))

    def test_torch_and_numpy_backends_agree(self):
        kwargs = dict(seed=1)
        c_np, _, _ = self.cc.spectrum(self.wave, backend='numpy', **kwargs)
        c_torch, _, _ = self.cc.spectrum(self.wave, backend='torch', device='cpu', **kwargs)
        np.testing.assert_allclose(c_np, c_torch, rtol=1e-6, atol=1e-12)

    def test_coeff_table_columns(self):
        _, _, table = self.cc.spectrum(self.wave, seed=1)
        self.assertEqual(set(table.colnames), {'camera', 'order', 'coeff'})
        self.assertEqual(len(table), 6)  # 3 cameras x 2 (default degree=1) terms


if __name__ == '__main__':
    unittest.main()
