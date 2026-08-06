"""
desisim.camera_calibration
===========================

Free-parameter, per-camera Legendre-polynomial calibration artifacts, as an
independent ADDITIVE ground-truth channel -- handoff task 9.

--------------------------------------------------------------------------
Motivation and source of the model (read before changing anything here)
--------------------------------------------------------------------------
Anand et al. 2024 ("Archetype-based Redshift Estimation for DESI", AJ 168,
124; DOI 10.3847/1538-3881/ad60c2; arXiv:2405.19288) describes DESI's
existing redshift pipeline (Redrock) as occasionally producing unphysical
fits because it cannot separate real spectral features from instrumental
calibration defects -- specifically "discontinuities in the spectrum due
to CCD bias and zero [point] issues, and gradient-like throughput offsets
caused by chromatic distortions of the corrector" (their Sec 3.1). Their
fix is to add a per-camera Legendre-polynomial correction term to their
fitting model:

    S_G(lambda) = alpha_k * T_{G,k}(lambda) + sum_j sum_i a_{i,j} * P_i(lambda'_j)

(their Eqn 1), where j indexes DESI's three cameras (b, r, z), P_i is the
Legendre polynomial of degree i, and lambda'_j is the *reduced* wavelength
within camera j:

    lambda'_j = 2 * (lambda - lambda_min_j) / (lambda_max_j - lambda_min_j) - 1  in [-1, 1]

with lambda_min_j/lambda_max_j the min/max observed wavelength of camera j
(their footnote 8). Critically, a_{i,j} are real-valued (can be positive or
negative) and enter *additively* -- this is explicitly not a multiplicative
throughput correction (their Sec 3.2: "we want that polynomial correction
should only correct for the 'additive' terms caused by the CCD bias
issue", as opposed to genuinely multiplicative broadband/spectral-slope
effects, which they deliberately suppress via the coefficient prior -- see
below).

This module inverts that fitting model into a forward model: rather than
*solving for* a_{i,j} to absorb defects in real data, we *draw* a_{i,j}
from a prior and inject the resulting per-camera Legendre correction as a
known ground-truth calibration artifact into a mock spectrum, to be
NPE-calibrated later against the coefficient distributions real Redrock/
archetype fits actually recover -- the same "independent free parameter
now, real covariance later" pattern already used throughout this fork's
emission-line and absorption work.

--------------------------------------------------------------------------
Design choices carried over directly from the paper (cited, not guessed)
--------------------------------------------------------------------------
- DESI's three camera wavelength ranges (their Sec 2.2): b 3600-5800 A,
  r 5760-7620 A, z 7520-9824 A. Note the real ranges *overlap* at the
  edges (b/r share 5760-5800 A; r/z share 7520-7620 A) -- CAMERA_WAVE_RANGES
  below preserves these exact literature values because the reduced-
  wavelength formula depends on each camera's true (lambda_min, lambda_max).
  Since this module must still assign exactly one calibration value per
  output pixel (unlike the paper, which fits each camera's own separately-
  extracted spectrum), pixels in an overlap region are assigned to
  whichever camera's range midpoint is closer -- see
  _camera_assignment_boundaries(). This is a deterministic simplification
  of the real coadded/dual-camera-coverage geometry, not a derivation, and
  is flagged as such rather than silently resolved.
- Legendre degree: the paper's actual implementation "use[s] only the
  first two Legendre polynomials, i.e. a constant and slope term" per
  camera (their Sec 3.1, describing the CCD-bias/throughput-offset use
  case that motivated this feature) even though "the code can accommodate
  several Legendre terms." DEFAULT_LEGENDRE_DEGREE=1 (2 terms: i=0,1)
  reproduces that literature default; legendre_degree is a constructor
  argument for anyone who wants to explore higher-order artifacts.
- Coefficient prior: the paper regularizes a_{i,j} with a zero-mean
  Gaussian prior of width sigma_a added to their chi^2 (their Eqn 3,
  Sec 3.2), specifically to prevent genuinely multiplicative continuum-
  slope effects (e.g. real QSO power-law continua) from being absorbed as
  spurious "calibration" -- i.e. sigma_a is deliberately kept small enough
  that only genuinely small, additive CCD-type defects get modeled this
  way. They quote sigma_a=0.1 in one of their worked comparisons (Sec 4.3/
  Table 4), in the same units as the coefficients, which they state
  explicitly are "proportional to the calibrated flux" -- i.e. absolute
  flux-density units, the standard DESI/desisim convention of
  1e-17 erg/s/cm^2/Angstrom. LEGENDRE_COEFF_SIGMA=0.1 reuses that quoted
  value directly. It is still flagged (Y) MAGIC: it is the paper's
  empirically-chosen *inference* prior width for fitting real DESI defects
  (which are presumably small), not a first-principles derivation, and
  there is no guarantee it is the right *injection* amplitude for this
  project's synthetic forward model -- pending the project's planned NPE
  calibration against real recovered coefficient distributions, per the
  same convention used everywhere else in this fork.

--------------------------------------------------------------------------
Scope
--------------------------------------------------------------------------
Purely a forward-model ground-truth generator, independently testable
(exactly like fsps_continuum.py and absorption.py before it) -- NOT wired
into GALAXY/QSO.make_templates() here. Injection point and stacking order
relative to continuum/emission/absorption is a design decision that has
been discussed with the PI (see git history / PR discussion): since this
term is additive in the same sense as AbsorptionSpectrum's flux deficit,
it is intended to be summed in at the same stage as emission
(total = continuum + emission + absorption_flux + calibration_flux),
*not* treated as a multiplicative correction like bal.py's QSO BAL
troughs, which remain an entirely separate mechanism (whole trough
templates, QSO-only, physically a real optical-depth effect rather than
an instrumental artifact). Actual wiring into the per-object generation
pipeline is left for a follow-up commit.
"""

import numpy as np

from desisim.templates import _use_torch_backend


def _legendre_eval_numpy(x, coeffs):
    """Evaluate sum_i coeffs[i] * P_i(x) via numpy's own Legendre-series
    evaluator (numerically exact reference implementation; no need to
    hand-roll this path)."""
    from numpy.polynomial import legendre as L
    return L.legval(x, coeffs)


def _legendre_eval_torch(x, coeffs, device=None, dtype=None):
    """Vectorized torch equivalent of _legendre_eval_numpy, evaluated on
    `device` (auto CPU/CUDA-detected via desisim.torch_utils.get_device if
    device is None). Implements the standard three-term Legendre recurrence

        P_0(x) = 1
        P_1(x) = x
        P_n(x) = ((2n-1) x P_{n-1}(x) - (n-1) P_{n-2}(x)) / n

    since torch has no built-in Legendre evaluator. Numerically equivalent
    to _legendre_eval_numpy to floating-point summation-order precision
    (see test_camera_calibration.py's backend-agreement test).
    """
    import torch
    from desisim.torch_utils import get_device

    dev = get_device(device)
    dt = dtype if dtype is not None else torch.float64

    x_t = torch.as_tensor(np.asarray(x), device=dev, dtype=dt)
    coeffs = np.asarray(coeffs, dtype=np.float64)

    result = torch.zeros_like(x_t)
    p_prev2 = torch.ones_like(x_t)   # P_0(x)
    result = result + coeffs[0] * p_prev2
    if len(coeffs) > 1:
        p_prev1 = x_t.clone()        # P_1(x)
        result = result + coeffs[1] * p_prev1
        for n in range(2, len(coeffs)):
            p_n = ((2 * n - 1) * x_t * p_prev1 - (n - 1) * p_prev2) / n
            result = result + coeffs[n] * p_n
            p_prev2, p_prev1 = p_prev1, p_n

    # NOTE: same numpy<2/torch ABI workaround as templates.py's
    # _lines_to_spectrum_torch -- see that function's comment for why
    # Tensor.numpy() is avoided here.
    return np.asarray(result.detach().to('cpu').tolist(), dtype=np.float64)


class CameraCalibration(object):
    """Free-parameter, per-camera Legendre-polynomial additive calibration
    artifacts. See module docstring for the full derivation from
    Anand et al. 2024's Eqn 1.
    """

    # DESI's three spectrograph camera coverage ranges [Angstrom], exactly
    # as quoted in Anand et al. 2024 Sec 2.2 (citing Guy et al. 2023).
    # Deliberately preserves the real edge overlaps (b/r: 5760-5800 A;
    # r/z: 7520-7620 A) -- see module docstring and
    # _camera_assignment_boundaries().
    CAMERA_WAVE_RANGES = {
        'b': (3600.0, 5800.0),
        'r': (5760.0, 7620.0),
        'z': (7520.0, 9824.0),
    }

    # Literature default (Anand et al. 2024 Sec 3.1): "we use only the
    # first two Legendre polynomials, i.e. a constant and slope term" per
    # camera. degree=1 means 2 terms (i=0,1).
    DEFAULT_LEGENDRE_DEGREE = 1

    # (Y) MAGIC (but literature-informed, not guessed): Anand et al. 2024's
    # own quoted prior width sigma_a=0.1 (Sec 4.3/Table 4 discussion), in
    # the same absolute flux-density units as the coefficients themselves
    # (1e-17 erg/s/cm^2/Angstrom, the standard DESI/desisim flux
    # convention) -- see module docstring's "Coefficient prior" paragraph
    # for why this is a reasonable starting point and not a derivation.
    LEGENDRE_COEFF_SIGMA = 0.1

    def __init__(self, camera_wave_ranges=None, legendre_degree=None, coeff_sigma=None):
        """
        Args:
            camera_wave_ranges (dict, optional): override CAMERA_WAVE_RANGES,
                mapping camera name -> (wave_min, wave_max) [Angstrom].
                Default: DESI's literature b/r/z ranges.
            legendre_degree (int, optional): highest Legendre polynomial
                degree per camera (default: DEFAULT_LEGENDRE_DEGREE=1, i.e.
                2 terms per camera, matching the paper's actual usage).
            coeff_sigma (float, optional): shared Gaussian prior sigma for
                all Legendre coefficients (default: LEGENDRE_COEFF_SIGMA).
        """
        self.camera_wave_ranges = dict(camera_wave_ranges) if camera_wave_ranges is not None \
            else dict(self.CAMERA_WAVE_RANGES)
        self.legendre_degree = legendre_degree if legendre_degree is not None else self.DEFAULT_LEGENDRE_DEGREE
        self.coeff_sigma = coeff_sigma if coeff_sigma is not None else self.LEGENDRE_COEFF_SIGMA
        # Sorted by wave_min so overlap-boundary logic only ever needs to
        # look at each camera's immediate neighbors.
        self.camera_names = sorted(self.camera_wave_ranges, key=lambda c: self.camera_wave_ranges[c][0])

    def _camera_assignment_boundaries(self):
        """Return {camera: (lo, hi)}, a non-overlapping partition of the
        full wavelength span covered by self.camera_wave_ranges, used only
        to decide which single camera's polynomial applies to a given
        output pixel. Overlap regions between adjacent cameras are split
        at their midpoint. See module docstring for why this is a
        deliberate simplification rather than a real dual-camera coverage
        model.
        """
        names = self.camera_names
        ranges = [self.camera_wave_ranges[c] for c in names]
        bounds = {}
        for idx, cam in enumerate(names):
            wmin, wmax = ranges[idx]
            lo, hi = wmin, wmax
            if idx > 0:
                prev_wmin, prev_wmax = ranges[idx - 1]
                if prev_wmax > wmin:
                    lo = 0.5 * (wmin + prev_wmax)
            if idx < len(names) - 1:
                next_wmin, next_wmax = ranges[idx + 1]
                if next_wmin < wmax:
                    hi = 0.5 * (wmax + next_wmin)
            bounds[cam] = (lo, hi)
        return bounds

    def spectrum(self, wave, coeffs=None, seed=None, backend='auto', device=None, dtype=None):
        """Build the additive per-camera Legendre calibration-artifact array.

        Args:
            wave (ndarray): observed-frame wavelength grid [Angstrom] to
                evaluate the calibration artifact on (arbitrary grid;
                pixels outside every camera's range get exactly 0).
            coeffs (dict, optional): explicit per-camera coefficient
                override, {camera_name: array[legendre_degree+1]}. Unlisted
                active cameras draw independently from
                N(0, coeff_sigma) per Legendre order (matching the paper's
                single shared sigma_a across all i,j -- see module
                docstring). Explicit values always win.
            seed (int, optional): RNG seed for reproducibility.
            backend, device, dtype: identical torch/numpy dispatch
                convention as EMSpectrum.spectrum()/AbsorptionSpectrum.spectrum()
                (see those docstrings) -- 'auto' (default) prefers torch
                when available.

        Returns:
            Tuple of (calib_flux, wave, coeff_table), where calib_flux is
            an additive flux-density array [npix] (same units as the
            spectrum's continuum, e.g. 1e-17 erg/s/cm^2/A), 0 outside every
            camera's assigned range; wave is the input wave array,
            unchanged; coeff_table is a Table with one row per
            (camera, Legendre order) giving the coefficient used.
        """
        from astropy.table import Table

        rand = np.random.RandomState(seed)
        if coeffs is None:
            coeffs = {}

        wave = np.asarray(wave, dtype=float)
        calib_flux = np.zeros_like(wave)
        boundaries = self._camera_assignment_boundaries()

        rows = []
        for cam in self.camera_names:
            wmin, wmax = self.camera_wave_ranges[cam]  # TRUE camera range, used for lambda' below
            if cam in coeffs:
                c = np.asarray(coeffs[cam], dtype=float)
                if len(c) != self.legendre_degree + 1:
                    raise ValueError(
                        'coeffs[{!r}] has length {}, expected legendre_degree+1={}'.format(
                            cam, len(c), self.legendre_degree + 1))
            else:
                c = rand.normal(0.0, self.coeff_sigma, size=self.legendre_degree + 1)

            lo, hi = boundaries[cam]  # assignment-split range, used only to select pixels
            mask = (wave >= lo) & (wave < hi)
            if np.any(mask):
                # Reduced wavelength lambda' per the paper's footnote 8,
                # using the camera's TRUE (wmin, wmax) -- not the
                # assignment-split (lo, hi) -- so a coefficient's meaning
                # matches the paper's definition exactly regardless of how
                # this module resolves overlap-pixel assignment.
                xprime = 2.0 * (wave[mask] - wmin) / (wmax - wmin) - 1.0
                if _use_torch_backend(backend):
                    calib_flux[mask] = _legendre_eval_torch(xprime, c, device=device, dtype=dtype)
                else:
                    calib_flux[mask] = _legendre_eval_numpy(xprime, c)

            for order, coeff_val in enumerate(c):
                rows.append((cam, order, coeff_val))

        coeff_table = Table(rows=rows, names=('camera', 'order', 'coeff'))
        return calib_flux, wave, coeff_table
