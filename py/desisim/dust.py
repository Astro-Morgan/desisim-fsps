"""
desisim.dust
=============

Free-parameter dust attenuation, as an independent additive ground-truth
channel (handoff task: dust, the last of the five Stage-A decomposition
channels -- continuum, emission, absorption, dust, calibration artifacts).

--------------------------------------------------------------------------
Parametric form (as specified directly by the PI, not derived here)
--------------------------------------------------------------------------
The project originally considered adopting Noll et al. (2009)'s attenuation
law wholesale, but that was left unsettled pending a simplification: rather
than separately modeling host-galaxy vs. AGN-torus dust (which is not
identifiable from an observed spectrum alone -- "the observed spectrum only
ever gives you the combined effect"), this module uses a single shared
4-parameter family fit to whatever mixed intrinsic dust signature is
present in a given mock:

    k(lambda; theta) = theta0 * (lambda/lambda_V)^(-theta1)
                        + theta2 * D(lambda; lambda_0, gamma)
                        + theta3,          theta0, theta2, theta3 >= 0

where D(lambda; lambda_0, gamma) is the standard Drude profile used for the
UV dust bump (Fitzpatrick & Massa 1986; also used inside Noll et al. 2009's
own attenuation law):

    D(lambda; lambda_0, gamma) = (lambda^2 gamma^2) / [(lambda^2 - lambda_0^2)^2 + lambda^2 gamma^2]

normalized so D(lambda_0) = 1 (peak amplitude exactly 1 at the bump
center), independent of gamma. lambda_V (the V-band reference wavelength)
and lambda_0, gamma (the bump's center and width) are held FIXED at their
standard literature values (see LAMBDA_V/LAMBDA_BUMP/BUMP_WIDTH below) --
they are not part of theta, which is deliberately exactly 4-dimensional
(theta0=power-law normalization, theta1=power-law slope, theta2=bump
amplitude, theta3=grey/wavelength-independent floor) per the PI's
specification. This single family spans SMC-like curves (steep power law,
theta1 large, no bump, theta2~0) through flat/AGN-torus-like reddening
(theta1~0, large theta3 grey floor), covering the rest-frame range this
project cares about (~450-9824A after redshifting); Milky-Way foreground
dust is handled upstream, separately, and is out of scope here.

k(lambda; theta) is taken directly as the attenuation A(lambda) in
magnitudes (there is no separate A_V multiplier beyond theta0 itself --
theta0 already plays that normalization role). Transmission is therefore

    T(lambda) = 10^(-0.4 * k(lambda; theta))

--------------------------------------------------------------------------
Additive-flux-deficit convention (same pattern as absorption.py)
--------------------------------------------------------------------------
Physically, dust is a multiplicative attenuation of whatever light reaches
it -- both continuum and emission-line photons, since both originate deep
enough in/near the galaxy to be affected by the same foreground dust
column (unlike absorption.py's AbsorptionSpectrum, which by design only
attenuates the continuum -- see that module's docstring). To keep this
module's output an ADDITIVE ground-truth channel consistent with the rest
of this project (per the same "spectra decomposable into additive
components" goal), this module returns the flux *deficit*

    dust_flux(lambda) = flux_in(lambda) * (T(lambda) - 1)

which is <= 0 everywhere (T in (0, 1] since k >= 0 by the theta0,theta2,
theta3 >= 0 constraints), where flux_in is the caller-supplied intrinsic
(continuum + emission) spectrum -- i.e. dust is meant to be summed in
*after* continuum and emission are already combined, not applied to each
separately, so that

    total = continuum + emission + dust_flux(continuum + emission) + ...

reproduces (continuum + emission) * T(lambda) for those two components
exactly, while the separately-generated AbsorptionSpectrum/CameraCalibration
channels are unaffected by this module (each documents its own, independent
scope -- see their module docstrings).

--------------------------------------------------------------------------
Scope
--------------------------------------------------------------------------
Purely a forward-model ground-truth generator, independently testable like
every other new component in this fork -- NOT wired into
GALAXY/QSO.make_templates() here. theta itself has no covariance model
(per the same "independent free parameters now, NPE-calibrated later"
convention used throughout this fork) -- every theta_i is drawn
independently from THETA_PRIORS, which are (Y) MAGIC-flagged starting
ranges, not derived from data.
"""

import numpy as np

from desisim.templates import _use_torch_backend


def _drude_numpy(wave, lambda0, gamma):
    """Standard Drude profile, normalized so D(lambda0) = 1."""
    return (wave**2 * gamma**2) / ((wave**2 - lambda0**2)**2 + wave**2 * gamma**2)


def _dust_curve_numpy(wave, theta, lambda_v, lambda_bump, bump_width):
    """k(lambda; theta) -- see module docstring for the closed form."""
    term1 = theta['theta0'] * (wave / lambda_v) ** (-theta['theta1'])
    term2 = theta['theta2'] * _drude_numpy(wave, lambda_bump, bump_width)
    term3 = theta['theta3']
    return term1 + term2 + term3


def _dust_curve_torch(wave, theta, lambda_v, lambda_bump, bump_width, device=None, dtype=None):
    """Torch equivalent of _dust_curve_numpy, evaluated on `device` (auto
    CPU/CUDA-detected via desisim.torch_utils.get_device if device is
    None). This is an O(npix) elementwise closed-form evaluation (unlike
    the O(nline*npix) Gaussian-line-sum kernels elsewhere in this fork),
    so the torch backend buys little for a single spectrum -- it is
    provided for API consistency with the rest of the project's
    backend='auto'/'torch'/'numpy' convention and for future batched
    (many-mock) vectorization.
    """
    import torch
    from desisim.torch_utils import get_device

    dev = get_device(device)
    dt = dtype if dtype is not None else torch.float64

    wave_t = torch.as_tensor(np.asarray(wave), device=dev, dtype=dt)
    term1 = theta['theta0'] * (wave_t / lambda_v) ** (-theta['theta1'])
    drude = (wave_t**2 * bump_width**2) / ((wave_t**2 - lambda_bump**2)**2 + wave_t**2 * bump_width**2)
    term2 = theta['theta2'] * drude
    k = term1 + term2 + theta['theta3']

    # NOTE: same numpy<2/torch ABI workaround as templates.py's
    # _lines_to_spectrum_torch -- see that function's comment for why
    # Tensor.numpy() is avoided here.
    return np.asarray(k.detach().to('cpu').tolist(), dtype=np.float64)


class DustAttenuation(object):
    """Free-parameter, single-shared-family dust attenuation. See module
    docstring for the full derivation and the additive-flux-deficit
    convention used to return its contribution.
    """

    # Fixed constants of the functional form (NOT part of theta -- theta is
    # deliberately exactly 4-dimensional per the PI's specification).
    LAMBDA_V = 5500.0      # V-band reference wavelength [Angstrom] (Calzetti 2000; Noll et al. 2009 convention).
    LAMBDA_BUMP = 2175.0   # UV dust-bump central wavelength [Angstrom] (Fitzpatrick & Massa 1986).
    # (Y) MAGIC: fixed Drude bump width [Angstrom]. Not one of the 4 free
    # parameters (theta only controls the bump's *amplitude*, theta2);
    # real MW/LMC-like bump widths span roughly 300-500A (Fitzpatrick &
    # Massa 2007) -- 350A is a representative fixed value, not a fit.
    BUMP_WIDTH = 350.0

    # (Y) MAGIC: independent uniform priors for each free parameter,
    # chosen to span "SMC-like through flat/AGN-torus-like reddening laws"
    # per the project's stated goal, not derived from data. Pending the
    # project's planned NPE/normalizing-flow calibration against real
    # spectra, same convention as every other new component in this fork.
    THETA_PRIORS = {
        # Power-law normalization: an A_V-like mag-scale amplitude, from
        # negligible to heavily-obscured.
        'theta0': (0.01, 3.0),
        # Power-law slope: 0 => flat/grey (AGN-torus-like); ~1-2 => steep
        # SMC/starburst-like curves (c.f. Calzetti's k(lambda) ~ lambda^-0.7
        # to SMC bar's ~lambda^-1.2 to -1.6 in various parametrizations).
        'theta1': (0.0, 2.0),
        # UV bump amplitude: 0 => bump-free (SMC/starburst-like); >0 =>
        # MW/LMC-like curves with a real 2175A feature.
        'theta2': (0.0, 2.0),
        # Grey/wavelength-independent attenuation floor.
        'theta3': (0.0, 1.0),
    }

    def __init__(self, theta_priors=None, lambda_v=None, lambda_bump=None, bump_width=None):
        """
        Args:
            theta_priors (dict, optional): override THETA_PRIORS, mapping
                'theta0'..'theta3' -> (low, high) uniform-draw bounds.
            lambda_v, lambda_bump, bump_width (float, optional): override
                the fixed functional-form constants LAMBDA_V/LAMBDA_BUMP/
                BUMP_WIDTH.
        """
        self.theta_priors = dict(theta_priors) if theta_priors is not None else dict(self.THETA_PRIORS)
        self.lambda_v = lambda_v if lambda_v is not None else self.LAMBDA_V
        self.lambda_bump = lambda_bump if lambda_bump is not None else self.LAMBDA_BUMP
        self.bump_width = bump_width if bump_width is not None else self.BUMP_WIDTH

    def spectrum(self, wave, flux_in, theta=None, seed=None, backend='auto', device=None, dtype=None):
        """Build the additive dust-attenuation flux deficit.

        Args:
            wave (ndarray): wavelength grid [Angstrom] flux_in is defined
                on (rest or observed frame -- this module is frame-
                agnostic; lambda_V/lambda_bump should be specified in
                whatever frame `wave` is in).
            flux_in (ndarray): the intrinsic (continuum + emission) flux
                this attenuation applies to.
            theta (dict, optional): explicit override for any of
                'theta0'..'theta3'; unlisted parameters draw independently
                from theta_priors. Explicit values always win.
            seed (int, optional): RNG seed for reproducibility.
            backend, device, dtype: identical torch/numpy dispatch
                convention as the rest of this fork's new components (see
                EMSpectrum.spectrum()'s docstring).

        Returns:
            Tuple of (dust_flux, wave, theta_table), where dust_flux is an
            additive flux-deficit array [npix] (<=0 everywhere, same units
            as flux_in) such that flux_in + dust_flux reproduces
            flux_in * 10**(-0.4*k(wave;theta)); wave is the input wave
            array, unchanged; theta_table is a Table with the 4 resolved
            theta values used.
        """
        from astropy.table import Table

        rand = np.random.RandomState(seed)
        if theta is None:
            theta = {}

        resolved = {}
        for name in ('theta0', 'theta1', 'theta2', 'theta3'):
            if name in theta:
                resolved[name] = theta[name]
            else:
                lo, hi = self.theta_priors[name]
                resolved[name] = rand.uniform(lo, hi)

        wave = np.asarray(wave, dtype=float)
        flux_in = np.asarray(flux_in, dtype=float)

        if _use_torch_backend(backend):
            k = _dust_curve_torch(wave, resolved, self.lambda_v, self.lambda_bump, self.bump_width,
                                   device=device, dtype=dtype)
        else:
            k = _dust_curve_numpy(wave, resolved, self.lambda_v, self.lambda_bump, self.bump_width)

        transmission = 10.0 ** (-0.4 * k)
        dust_flux = flux_in * (transmission - 1.0)

        theta_table = Table(rows=[(name, resolved[name]) for name in ('theta0', 'theta1', 'theta2', 'theta3')],
                             names=('param', 'value'))
        return dust_flux, wave, theta_table
