"""
desisim.dust
=============

Free-parameter dust attenuation, as an independent additive ground-truth
channel (handoff task: dust, the last of the five Stage-A decomposition
channels -- continuum, emission, absorption, dust, calibration artifacts).

--------------------------------------------------------------------------
Parametric form (base family as specified directly by the PI)
--------------------------------------------------------------------------
The project originally considered adopting Noll et al. (2009)'s attenuation
law wholesale, but that was left unsettled pending a simplification: rather
than separately modeling host-galaxy vs. AGN-torus dust (which is not
identifiable from an observed spectrum alone -- "the observed spectrum only
ever gives you the combined effect"), this module uses a single shared
family fit to whatever mixed intrinsic dust signature is present in a
given mock. The base (legacy-default) form is exactly 4-dimensional:

    k(lambda; theta) = theta0 * (lambda/lambda_V)^(-theta1)
                        + theta2 * D(lambda; lambda_0, gamma)
                        + theta3,          theta0, theta2, theta3 >= 0

where D(lambda; lambda_0, gamma) is the standard Drude profile used for the
UV dust bump (Fitzpatrick & Massa 1986):

    D(lambda; lambda_0, gamma) = (lambda^2 gamma^2) / [(lambda^2 - lambda_0^2)^2 + lambda^2 gamma^2]

normalized so D(lambda_0) = 1. lambda_V, lambda_0, gamma default to fixed
literature values (LAMBDA_V/LAMBDA_BUMP/BUMP_WIDTH below).

--------------------------------------------------------------------------
2026-08-06 generalization: toward a genuinely universal family
--------------------------------------------------------------------------
Direct test against the real Calzetti (2000) curve showed the base
4-parameter family (a single power law, bump off) approximates Calzetti to
only ~10% typical / ~23% worst-case relative error in k(lambda) -- adequate
as a starting point, but explicitly NOT what "capable of recovering all
empirical laws" requires. Per PI direction: the dust treatment should be
"truly universal in capability," spanning real empirical attenuation/
extinction laws for BOTH galaxies and QSOs (each with its own independent
draw, applied to its own contribution once QSO/galaxy blending exists --
see "Universality across galaxies and QSOs" below), with "the onus of
physical exactness fall[ing] on the downstream NPE" -- i.e. this module's
job is *reach* (can it get close to any real curve shape somewhere in its
parameter space?), not built-in physical correctness of any one default.

Two OPT-IN extensions widen that reach, both off by default (so every
existing caller -- including this module's own original test suite -- is
bit-for-bit unaffected unless it explicitly asks for the new behavior):

1. `vary_bump_shape=True`: lets the bump's center (lambda_0) and width
   (gamma) themselves be free/drawn parameters (via theta keys
   'lambda_bump'/'bump_width'), instead of fixed literature constants.
   Real Milky-Way-type sightlines show real bump-to-bump scatter in both
   quantities (Fitzpatrick & Massa 2007's UV extinction curve atlas), not
   just amplitude -- fixing them at one value undersells how much a
   "universal" family should cover.

2. `include_fuv_curvature=True`: adds the Fitzpatrick & Massa (1990, FM90)
   far-UV curvature term

       F(x) = 0.5392*(x - 5.9)^2 + 0.05644*(x - 5.9)^3,   x = 1/lambda_um >= 5.9
       F(x) = 0,                                           x < 5.9

   scaled by a new free amplitude theta4 >= 0, active only blueward of
   ~1695 Angstrom (x=5.9 um^-1 is FM90's own fixed pivot, not a free
   parameter -- this is the standard, universally-used FM90 convention,
   not a project-specific choice). This is the specific feature responsible
   for the steep upturn Milky-Way/LMC-type curves show in the far-UV that
   neither Calzetti nor a plain power law reproduce; FM90's 4-6 term basis
   (of which this module implements the bump + curvature terms, the power
   law standing in for FM90's own linear c1+c2*x term) is the standard
   "universal" basis used to fit essentially any documented Galactic/
   Magellanic Cloud extinction curve in the literature.

With both extensions on, k(lambda; theta) becomes:

    k(lambda; theta) = theta0*(lambda/lambda_V)^(-theta1)
                        + theta2*D(lambda; lambda_0, gamma)
                        + theta3
                        + theta4*F(1/lambda_um)

theta0, theta2, theta3, theta4 >= 0; lambda_0, gamma free within
BUMP_CENTER_RANGE/BUMP_WIDTH_RANGE when vary_bump_shape=True.

Verified empirically (not assumed) that this remains a strict superset of
the base family's reach: with vary_bump_shape=False and
include_fuv_curvature=False (both defaults), k(lambda;theta) is
numerically identical to the original 4-parameter form for the same
theta0..theta3 draw (see test_dust.py's
test_generalization_flags_default_off_reproduce_legacy_family).

Also verified empirically (bounds-constrained scipy.optimize.curve_fit,
1250-9000A, against real reference laws from the `dust_extinction`
package, same method used for the earlier Calzetti check):
  - G03_SMCBar (steep, bump-free, SMC/starburst/AGN-like): base family
    ALONE (bump/curvature both off) already achieves ~2.7% typical / ~16%
    worst-case error, with the fitted bump amplitude theta2 pinned near 0
    -- confirming AGN/SMC-like curves need none of the new extensions, per
    the "Universality across galaxies and QSOs" section above.
  - CCM89 Rv=3.1 (Milky-Way-like, real 2175A bump): enabling both
    extensions improves the near-UV/bump region (1695-3000A) to ~2.6%
    typical/~6.0% worst and the far-UV region (<1695A, where the FM90 term
    is active) to ~5.5%/~6.3%, vs. the base family's fixed-bump fit over
    the same ranges. The single largest residual in EITHER family, however,
    is NOT in the UV at all: it is in the red/near-IR (>5000A, up to ~38%
    relative error), where CCM89's characteristic long-wavelength
    flattening is a genuinely different functional shape than a single
    power law can trace. This is an honest, currently-unaddressed
    limitation of the whole theta0*(lambda/lambda_V)^-theta1 term (not
    something the bump/curvature extensions target or fix) -- flagged here
    rather than papered over, consistent with this module's "reach over
    built-in exactness" mandate; closing it (e.g. a second, independent
    power-law segment or a smooth break parameter for the red/NIR) is a
    candidate future extension, not yet implemented.

--------------------------------------------------------------------------
Universality across galaxies and QSOs
--------------------------------------------------------------------------
This module deliberately does NOT special-case "galaxy dust" vs "QSO/AGN
dust" as two different classes or functional forms. The literature on AGN
reddening (Gaskell et al. 2004; Czerny et al. 2004) finds AGN dust curves
are typically SMC-like in steepness but with little-to-no 2175A bump and
sometimes flatter/greyer than starburst curves (larger effective grain
sizes) -- i.e. already comfortably within the base family's reach
(theta2~0, theta1 moderate-to-steep, optional theta3 floor for flatter
cases) without needing the new extensions at all. Galaxy attenuation laws
spanning Calzetti/starburst through Milky-Way/LMC-bump-bearing curves
benefit from the two extensions above. The intended usage once QSO/galaxy
spectral blending exists (tracked separately) is to call
DustAttenuation.spectrum() TWICE per blended mock -- once on the galaxy's
own (continuum+emission) with an independent theta draw appropriate to
galaxy dust, once on the QSO's own (continuum+emission) with a second,
independent theta draw appropriate to AGN dust -- and sum both flux
deficits into the final composite. Nothing about this module's API needs
to change to support that: it already operates on a caller-supplied
(wave, flux_in) pair with no assumption about what produced flux_in.

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

which is <= 0 everywhere (T in (0, 1] since k >= 0 by construction), where
flux_in is the caller-supplied intrinsic (continuum + emission) spectrum
for whichever single component (galaxy or QSO) is being attenuated -- so
that

    total = continuum + emission + dust_flux(continuum + emission) + ...

reproduces (continuum + emission) * T(lambda) for that component exactly,
while the separately-generated AbsorptionSpectrum/CameraCalibration
channels are unaffected by this module (each documents its own,
independent scope -- see their module docstrings).

--------------------------------------------------------------------------
Scope
--------------------------------------------------------------------------
Purely a forward-model ground-truth generator, independently testable like
every other new component in this fork -- NOT wired into
GALAXY/QSO.make_templates() here, and QSO/galaxy blending itself (the
"respective contributions in the blend" scenario this module is designed
to support) is tracked as a separate, not-yet-implemented item. theta
itself has no covariance model (per the same "independent free parameters
now, NPE-calibrated later" convention used throughout this fork) -- every
theta_i is drawn independently from THETA_PRIORS, which are (Y) MAGIC-
flagged starting ranges, not derived from data; the onus of matching real,
correlated, physically-consistent dust-law populations is explicitly left
to the project's planned NPE/normalizing-flow calibration, not this
forward model.
"""

import numpy as np

from desisim.templates import _use_torch_backend

# FM90 (Fitzpatrick & Massa 1990) far-UV curvature pivot [1/micron]. A
# fixed convention of the FM90 parametrization itself (not a free
# parameter, not project-specific) -- the curvature term is defined to be
# exactly zero at x < FM90_FUV_PIVOT_INV_UM and follow the quadratic+cubic
# form above it.
FM90_FUV_PIVOT_INV_UM = 5.9


def _drude_numpy(wave, lambda0, gamma):
    """Standard Drude profile, normalized so D(lambda0) = 1."""
    return (wave**2 * gamma**2) / ((wave**2 - lambda0**2)**2 + wave**2 * gamma**2)


def _fm90_fuv_curvature_numpy(wave):
    """FM90 far-UV curvature shape F(x), x=1/lambda[micron]. Zero below the
    fixed pivot x=5.9 (lambda > ~1695A); this is FM90's own convention, not
    a free parameter."""
    x = 1.0e4 / wave  # wave in Angstrom -> x in 1/micron
    fx = np.where(x >= FM90_FUV_PIVOT_INV_UM,
                  0.5392 * (x - FM90_FUV_PIVOT_INV_UM) ** 2 + 0.05644 * (x - FM90_FUV_PIVOT_INV_UM) ** 3,
                  0.0)
    return fx


def _dust_curve_numpy(wave, theta, lambda_v, lambda_bump, bump_width, include_fuv_curvature):
    """k(lambda; theta) -- see module docstring for the closed form."""
    term1 = theta['theta0'] * (wave / lambda_v) ** (-theta['theta1'])
    term2 = theta['theta2'] * _drude_numpy(wave, lambda_bump, bump_width)
    term3 = theta['theta3']
    k = term1 + term2 + term3
    if include_fuv_curvature:
        k = k + theta['theta4'] * _fm90_fuv_curvature_numpy(wave)
    return k


def _dust_curve_torch(wave, theta, lambda_v, lambda_bump, bump_width, include_fuv_curvature,
                       device=None, dtype=None):
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

    if include_fuv_curvature:
        x = 1.0e4 / wave_t
        zero = torch.zeros((), device=dev, dtype=dt)
        fx = torch.where(x >= FM90_FUV_PIVOT_INV_UM,
                          0.5392 * (x - FM90_FUV_PIVOT_INV_UM) ** 2 + 0.05644 * (x - FM90_FUV_PIVOT_INV_UM) ** 3,
                          zero)
        k = k + theta['theta4'] * fx

    # NOTE: same numpy<2/torch ABI workaround as templates.py's
    # _lines_to_spectrum_torch -- see that function's comment for why
    # Tensor.numpy() is avoided here.
    return np.asarray(k.detach().to('cpu').tolist(), dtype=np.float64)


class DustAttenuation(object):
    """Free-parameter dust attenuation, generalizable toward a universal
    family via two opt-in flags. See module docstring for the full
    derivation, the additive-flux-deficit convention, and how this class
    is intended to be called independently per component (galaxy vs. QSO)
    once spectral blending exists.
    """

    # Fixed constants of the base functional form (NOT part of theta by
    # default -- theta is exactly 4-dimensional unless the opt-in
    # extensions below are enabled).
    LAMBDA_V = 5500.0      # V-band reference wavelength [Angstrom] (Calzetti 2000; Noll et al. 2009 convention).
    LAMBDA_BUMP = 2175.0   # UV dust-bump central wavelength [Angstrom] (Fitzpatrick & Massa 1986).
    # (Y) MAGIC: fixed Drude bump width [Angstrom] used when
    # vary_bump_shape=False (the default). Real MW/LMC-like bump widths
    # span roughly 300-500A (Fitzpatrick & Massa 2007) -- 350A is a
    # representative fixed value, not a fit.
    BUMP_WIDTH = 350.0

    # (Y) MAGIC: draw ranges for the bump's center/width when
    # vary_bump_shape=True, informed by the same Fitzpatrick & Massa
    # (2007) UV extinction atlas's sightline-to-sightline scatter in both
    # quantities (not a fit to that specific dataset -- a broad range
    # meant to be a superset of real scatter, per this module's "reach,
    # not exactness" goal).
    BUMP_CENTER_RANGE = (2100.0, 2250.0)
    BUMP_WIDTH_RANGE = (250.0, 500.0)

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
        # UV bump amplitude: 0 => bump-free (SMC/starburst/most-AGN-like);
        # >0 => MW/LMC-like curves with a real 2175A feature.
        'theta2': (0.0, 2.0),
        # Grey/wavelength-independent attenuation floor.
        'theta3': (0.0, 1.0),
        # (Y) MAGIC: FM90 far-UV curvature amplitude (only used when
        # include_fuv_curvature=True). Range informed by typical c4 values
        # in Fitzpatrick & Massa-style fits to real MW/LMC sightlines
        # (order-unity in FM90's own normalization convention).
        'theta4': (0.0, 1.5),
    }

    def __init__(self, theta_priors=None, lambda_v=None, lambda_bump=None, bump_width=None,
                 vary_bump_shape=False, include_fuv_curvature=False,
                 bump_center_range=None, bump_width_range=None):
        """
        Args:
            theta_priors (dict, optional): override THETA_PRIORS, mapping
                'theta0'..'theta4' -> (low, high) uniform-draw bounds
                ('theta4' only consulted if include_fuv_curvature=True).
            lambda_v (float, optional): override LAMBDA_V.
            lambda_bump, bump_width (float, optional): fixed bump center/
                width used when vary_bump_shape=False (the default);
                override LAMBDA_BUMP/BUMP_WIDTH.
            vary_bump_shape (bool, optional): default False (legacy,
                fixed-shape bump, exactly reproducing the original
                4-parameter family). If True, the bump's center and width
                become per-call free parameters (theta keys 'lambda_bump'/
                'bump_width'), drawn from bump_center_range/
                bump_width_range unless explicitly given in `theta`.
            include_fuv_curvature (bool, optional): default False (legacy
                behavior, exactly reproducing the original family with no
                curvature term). If True, adds theta4*F(x) (see module
                docstring) to k(lambda).
            bump_center_range, bump_width_range (tuple, optional): override
                BUMP_CENTER_RANGE/BUMP_WIDTH_RANGE (only consulted if
                vary_bump_shape=True).
        """
        self.theta_priors = dict(theta_priors) if theta_priors is not None else dict(self.THETA_PRIORS)
        self.lambda_v = lambda_v if lambda_v is not None else self.LAMBDA_V
        self.lambda_bump = lambda_bump if lambda_bump is not None else self.LAMBDA_BUMP
        self.bump_width = bump_width if bump_width is not None else self.BUMP_WIDTH
        self.vary_bump_shape = vary_bump_shape
        self.include_fuv_curvature = include_fuv_curvature
        self.bump_center_range = bump_center_range if bump_center_range is not None else self.BUMP_CENTER_RANGE
        self.bump_width_range = bump_width_range if bump_width_range is not None else self.BUMP_WIDTH_RANGE

    def spectrum(self, wave, flux_in, theta=None, seed=None, backend='auto', device=None, dtype=None):
        """Build the additive dust-attenuation flux deficit.

        Args:
            wave (ndarray): wavelength grid [Angstrom] flux_in is defined
                on (rest or observed frame -- this module is frame-
                agnostic; lambda_V/lambda_bump should be specified in
                whatever frame `wave` is in).
            flux_in (ndarray): the intrinsic (continuum + emission) flux
                this attenuation applies to, for ONE component of the mock
                (e.g. the galaxy alone, or the QSO alone -- see module
                docstring's "Universality across galaxies and QSOs").
            theta (dict, optional): explicit override for any of
                'theta0'..'theta4' (theta4 only meaningful if
                include_fuv_curvature=True), plus 'lambda_bump'/
                'bump_width' if vary_bump_shape=True; unlisted parameters
                draw independently from theta_priors/bump_center_range/
                bump_width_range. Explicit values always win.
            seed (int, optional): RNG seed for reproducibility.
            backend, device, dtype: identical torch/numpy dispatch
                convention as the rest of this fork's new components (see
                EMSpectrum.spectrum()'s docstring).

        Returns:
            Tuple of (dust_flux, wave, theta_table), where dust_flux is an
            additive flux-deficit array [npix] (<=0 everywhere, same units
            as flux_in) such that flux_in + dust_flux reproduces
            flux_in * 10**(-0.4*k(wave;theta)); wave is the input wave
            array, unchanged; theta_table is a Table with the resolved
            parameter values used (4 rows by default, up to 6 with both
            extensions enabled).
        """
        from astropy.table import Table

        rand = np.random.RandomState(seed)
        if theta is None:
            theta = {}

        param_names = ['theta0', 'theta1', 'theta2', 'theta3']
        if self.include_fuv_curvature:
            param_names.append('theta4')

        resolved = {}
        for name in param_names:
            if name in theta:
                resolved[name] = theta[name]
            else:
                lo, hi = self.theta_priors[name]
                resolved[name] = rand.uniform(lo, hi)

        if self.vary_bump_shape:
            if 'lambda_bump' in theta:
                lambda_bump_used = theta['lambda_bump']
            else:
                lambda_bump_used = rand.uniform(*self.bump_center_range)
            if 'bump_width' in theta:
                bump_width_used = theta['bump_width']
            else:
                bump_width_used = rand.uniform(*self.bump_width_range)
        else:
            lambda_bump_used = self.lambda_bump
            bump_width_used = self.bump_width

        wave = np.asarray(wave, dtype=float)
        flux_in = np.asarray(flux_in, dtype=float)

        if _use_torch_backend(backend):
            k = _dust_curve_torch(wave, resolved, self.lambda_v, lambda_bump_used, bump_width_used,
                                   self.include_fuv_curvature, device=device, dtype=dtype)
        else:
            k = _dust_curve_numpy(wave, resolved, self.lambda_v, lambda_bump_used, bump_width_used,
                                   self.include_fuv_curvature)

        transmission = 10.0 ** (-0.4 * k)
        dust_flux = flux_in * (transmission - 1.0)

        table_rows = [(name, resolved[name]) for name in param_names]
        if self.vary_bump_shape:
            table_rows += [('lambda_bump', lambda_bump_used), ('bump_width', bump_width_used)]
        theta_table = Table(rows=table_rows, names=('param', 'value'))
        return dust_flux, wave, theta_table
