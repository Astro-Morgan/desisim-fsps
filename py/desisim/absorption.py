"""
desisim.absorption
===================

Non-stellar (ISM-like) absorption-line ground truth, as an independent
additive channel: Na I D (5891, 5898 vac. A), Ca II H&K (3935, 3970 vac. A),
and Mg II 2796,2804 (vac. A) *absorption* (a separate physical channel from
the pre-existing Mg II *emission* rows in recombination_lines.ecsv, and from
bal.py's QSO-only BAL trough templates -- see module scope note below).

--------------------------------------------------------------------------
Why this exists / where it fits (handoff Sec 1.3, second half)
--------------------------------------------------------------------------
The project's stated goal is mock spectra decomposable into KNOWN additive
ground-truth components (continuum, nebular emission, dust, ...). Stellar
photospheric absorption is intentionally bundled into the continuum (see
templates.py's EMSpectrum module-level docs / the fork's handoff Sec 0) --
no SPS code exposes a continuum-only, absorption-free stellar SED, so
"continuum" already means "full emergent stellar SED, absorption included."

Interstellar-medium and circumgalactic-medium resonant-line absorption
(Na I D from cold neutral gas, Ca II H&K from warm/cold gas, Mg II from CGM
gas -- all well-documented in real galaxy/QSO spectra) is a genuinely
*separate* physical origin from the photosphere, even though its observed
wavelengths can and do overlap with stellar absorption features already
baked into the continuum. Per explicit project direction, this overlap is
not a problem to avoid: as long as this module's contribution is returned
as its own explicit array, a downstream decomposer can be trained against
the true, separately-known ISM contribution regardless of what the
continuum happens to look like at that wavelength.

--------------------------------------------------------------------------
Design: additive-flux formulation of a physically multiplicative process
--------------------------------------------------------------------------
Line-of-sight absorption is physically multiplicative: transmitted flux =
incident_flux * exp(-tau(lambda)), for an optical-depth profile tau(lambda)
built here as a sum of Gaussians in log10-wavelength (the same convention
EMSpectrum uses for emission lines), each with its own peak optical depth
tau0 and a *shared* velocity width sigma_kms (one common ISM-kinematic
scale per object -- see SIGMA_KMS_RANGE below, the same pattern as
EMSpectrum's shared `linesigma`/`broadsigma`).

To keep this an ADDITIVE component (per the project's explicit "spectra
decomposable into additive components" goal) rather than a multiplicative
correction to the continuum, this module returns the *flux deficit*

    absorption_flux(lambda) = continuum(lambda) * (exp(-tau(lambda)) - 1)

which is <= 0 everywhere by construction (tau >= 0), and satisfies

    total_flux(lambda) = continuum(lambda) + emission(lambda) + absorption_flux(lambda)

exactly reproducing continuum(lambda) * exp(-tau(lambda)) for the
continuum's own contribution while leaving the (separately-generated)
nebular emission untouched by this absorption -- i.e. only the continuum
is assumed to sit behind the absorbing gas. This is a deliberate scope
choice (not derived from radiative-transfer geometry, which would require
knowing the actual relative geometry of the emitting and absorbing gas --
out of scope here) and should be flagged as an open question if downstream
work needs emission-line photons to also transit the absorber.

--------------------------------------------------------------------------
Scope / explicitly NOT covered here
--------------------------------------------------------------------------
- BAL (broad absorption line) troughs: already covered by bal.py as
  QSO-only, whole-trough-template multiplicative absorption. Untouched by
  this module; the two are complementary (BAL = QSO broad-outflow troughs,
  this module = narrower ISM/CGM resonant-line absorption applicable to
  any object type, galaxy or QSO).
- Wiring into GALAXY/QSO.make_templates(): NOT done in this module (still
  a standalone, independently testable class, analogous to how
  fsps_continuum.py's fsps_basis_templates() exists independently of
  GALAXY before being passed in as baseflux/basewave/basemeta), but the
  design question of *where* it plugs in has been resolved by the PI:
  since this module's output is already additive (see "Design" above),
  it sums in at the same generation stage as EMSpectrum's emission flux
  (total = continuum + emission + absorption_flux), not as a post-hoc
  multiplicative correction. BAL troughs remain a separate mechanism (see
  next bullet) precisely because they are physically multiplicative,
  which is why they cannot be merged into this additive stage. How a
  given mock's QSO-vs-galaxy blend fraction should modulate which of
  this module's 6 lines are active/how strong is explicitly deferred to
  the project's planned NPE calibration rather than hardcoded here (per
  PI direction: "QSO vs galaxy blend fraction correlation with emission/
  absorption components should be solvable via NPE downstream"). Actual
  code wiring into make_templates() is left for a follow-up commit.
- Real covariance between the 6 lines' tau0/sigma_kms and any other
  per-object property (metallicity, inclination, D4000, ...): explicitly
  out of scope per project direction -- every tau0/sigma_kms here is an
  independent free parameter with a MAGIC-flagged prior, pending the
  project's planned NPE/normalizing-flow calibration against real spectra.

--------------------------------------------------------------------------
Line list / vacuum wavelengths
--------------------------------------------------------------------------
Mg II 2796.352 / 2803.530 vac. A reuse the *exact* values already present
in desisim/data/forbidden_lines.ecsv (MgII_2800a/b, added for Sec 1.2's
tunable MgII emission) for internal consistency -- same transition, same
rest wavelength, whether seen in emission or absorption.

Ca II K/H and Na I D vacuum wavelengths are standard atomic values (NIST
ASD; air wavelengths converted to vacuum via the standard IAU/Ciddor-type
air-to-vacuum relation, as is already the convention for every other line
in this codebase's recombination_lines.ecsv/forbidden_lines.ecsv):
    Ca II K : air 3933.663 -> vac 3934.777
    Ca II H : air 3968.469 -> vac 3969.591
    Na I D2 : air 5889.951 -> vac 5891.583  (bluer, stronger component)
    Na I D1 : air 5895.924 -> vac 5897.558
"""

import numpy as np

from desisim.templates import _use_torch_backend, _lines_to_spectrum_numpy, _lines_to_spectrum_torch, C_LIGHT


class AbsorptionSpectrum(object):
    """Independent, additive ISM/CGM-like resonant absorption-line ground
    truth (Na I D, Ca II H&K, Mg II absorption). See module docstring for
    the full physical/design rationale.
    """

    LINE_NAMES = ['CaII_K', 'CaII_H', 'NaID_5891', 'NaID_5897',
                  'MgII_2796_abs', 'MgII_2803_abs']

    # Vacuum wavelengths [Angstrom]. See module docstring for sourcing.
    LINE_WAVE_VACUUM = {
        'CaII_K':        3934.777,
        'CaII_H':        3969.591,
        'NaID_5891':     5891.583,
        'NaID_5897':     5897.558,
        'MgII_2796_abs': 2796.352,  # identical rest wavelength to the
        'MgII_2803_abs': 2803.530,  # pre-existing MgII *emission* rows.
    }

    # ⚠ MAGIC: independent log-normal priors on peak optical depth tau0 for
    # each line, in log10(tau0) space. No derivation from data exists yet
    # (per explicit project direction: independent free parameters for now,
    # real values/covariances deferred to NPE/normalizing-flow calibration
    # against real spectra). Central values are order-of-magnitude choices
    # consistent with these lines being routinely detected-but-not-always-
    # saturated features in real galaxy/QSO spectra (tau0 ~ 0.1-1 typical,
    # occasionally higher for strong CGM/ISM absorbers) -- not a fit to any
    # specific dataset.
    TAU0_PRIORS = {
        'CaII_K':        dict(mean=np.log10(0.3), sigma=0.5),
        'CaII_H':        dict(mean=np.log10(0.3 * 0.5), sigma=0.5),  # H weaker than K (f-value ratio ~1:2)
        'NaID_5891':     dict(mean=np.log10(0.3), sigma=0.5),
        'NaID_5897':     dict(mean=np.log10(0.3 * 0.7), sigma=0.5),  # D1 weaker than D2 (f-value ratio ~1:2, saturating)
        'MgII_2796_abs': dict(mean=np.log10(0.5), sigma=0.5),
        'MgII_2803_abs': dict(mean=np.log10(0.5 * 0.7), sigma=0.5),
    }

    # ⚠ MAGIC: shared ISM/CGM-absorber velocity width, log-uniform draw
    # range [km/s]. One common value applied to all active lines in a given
    # spectrum() call (same pattern as EMSpectrum's shared linesigma /
    # broadsigma) -- a single characteristic absorber-kinematics scale per
    # object, not six independent ones. Range spans narrow individual
    # ISM clouds (~10s km/s Doppler b-parameters) through broader CGM/
    # rotational-blending scales (~few x 100 km/s); does not extend into
    # AGN-outflow/BAL territory (that's bal.py's domain, with its own much
    # broader trough widths).
    SIGMA_KMS_RANGE = (10.0, 300.0)

    def __init__(self, minwave=2000.0, maxwave=10000.0, cdelt_kms=20.0,
                 log10wave=None, include_lines=None):
        """
        Args:
            minwave, maxwave (float): rest-frame output grid bounds [A].
                Only used if log10wave is not provided.
            cdelt_kms (float): output-grid pixel size [km/s] (log-uniform
                grid, same convention as EMSpectrum). Only used if
                log10wave is not provided.
            log10wave (ndarray, optional): explicit output log10(wave) grid,
                to match an external continuum/emission grid exactly.
            include_lines (list of str, optional): subset of LINE_NAMES to
                activate (default: all 6). Lets a caller opt into e.g. only
                Mg II without a combinatorial explosion of per-line boolean
                flags.
        """
        if log10wave is None:
            cdelt_loglam = cdelt_kms / C_LIGHT / np.log(10)
            log10wave = np.arange(np.log10(minwave), np.log10(maxwave), cdelt_loglam)
        self.log10wave = log10wave

        if include_lines is None:
            include_lines = list(self.LINE_NAMES)
        else:
            unknown = set(include_lines) - set(self.LINE_NAMES)
            if unknown:
                raise ValueError('Unknown line name(s) in include_lines: {}'.format(sorted(unknown)))
        self.include_lines = include_lines

    def spectrum(self, continuum_wave, continuum_flux, tau0=None, sigma_kms=None,
                 tau0_priors=None, zshift=0.0, seed=None,
                 backend='auto', device=None, dtype=None):
        """Build the additive absorption-flux-deficit array.

        Args:
            continuum_wave (ndarray): wavelength grid [A] of the continuum
                this absorption is applied to (e.g. a basis-template or
                FSPS-generated continuum).
            continuum_flux (ndarray): continuum flux density on
                continuum_wave, same units as the desired output.
            tau0 (dict, optional): explicit peak optical depth per line
                name (keys from self.include_lines); unlisted active lines
                are drawn from tau0_priors. Explicit values always win.
            sigma_kms (float, optional): explicit shared absorber velocity
                width [km/s]; default None draws log-uniformly from
                SIGMA_KMS_RANGE.
            tau0_priors (dict, optional): override TAU0_PRIORS.
            zshift (float): redshift applied to line centers only (matches
                EMSpectrum.spectrum's zshift convention); the output grid
                (self.log10wave) is unchanged.
            seed (int, optional): RNG seed for reproducibility.
            backend, device, dtype: see EMSpectrum.spectrum's docstring for
                the identical torch/numpy dispatch convention -- this
                function reuses the exact same _lines_to_spectrum_{numpy,torch}
                primitives (interpreting their Gaussian "norm" parameter as
                peak optical depth tau0 rather than flux amplitude; see
                module docstring's "Design" section).

        Returns:
            Tuple of (absorption_flux, wave, line), where absorption_flux
            is an array [npix] of flux-deficit values (<=0 everywhere,
            same units as continuum_flux) such that
            continuum_on_output_grid + absorption_flux reproduces
            continuum_on_output_grid * exp(-tau(lambda)); wave is
            10**self.log10wave; line is a Table of per-line
            name/wave/tau0/sigma_kms used.
        """
        from astropy.table import Table

        rand = np.random.RandomState(seed)

        if tau0_priors is None:
            tau0_priors = self.TAU0_PRIORS
        if tau0 is None:
            tau0 = {}
        if sigma_kms is None:
            # ⚠ MAGIC: log-uniform draw over SIGMA_KMS_RANGE; see that
            # constant's definition above for the physical reasoning.
            sigma_kms = 10**rand.uniform(np.log10(self.SIGMA_KMS_RANGE[0]),
                                          np.log10(self.SIGMA_KMS_RANGE[1]))

        names = list(self.include_lines)
        tau0_resolved = np.empty(len(names))
        for i, name in enumerate(names):
            if name in tau0:
                tau0_resolved[i] = tau0[name]
            else:
                prior = tau0_priors[name]
                tau0_resolved[i] = 10**rand.normal(prior['mean'], prior['sigma'])

        wave_rest = np.array([self.LINE_WAVE_VACUUM[name] for name in names])
        log10sigma = sigma_kms / C_LIGHT / np.log(10)
        centers = np.log10(wave_rest * (1.0 + zshift))

        in_window = (centers > self.log10wave.min()) & (centers < self.log10wave.max())

        wave_out = 10**self.log10wave
        tau_total = np.zeros_like(self.log10wave)
        if np.any(in_window):
            if _use_torch_backend(backend):
                tau_total = _lines_to_spectrum_torch(self.log10wave, centers[in_window],
                                                      tau0_resolved[in_window], log10sigma,
                                                      device=device, dtype=dtype)
            else:
                tau_total = _lines_to_spectrum_numpy(self.log10wave, centers[in_window],
                                                      tau0_resolved[in_window], log10sigma)

        # Interpolate the supplied continuum onto our output grid. Constant
        # (edge-value) extrapolation outside continuum_wave's range is
        # np.interp's default behavior -- acceptable here since real calls
        # are expected to supply a continuum that already spans this
        # module's [minwave, maxwave].
        continuum_on_grid = np.interp(wave_out, continuum_wave, continuum_flux)

        absorption_flux = continuum_on_grid * (np.exp(-tau_total) - 1.0)

        line = Table()
        line['name'] = names
        line['wave'] = wave_rest
        line['tau0'] = tau0_resolved
        line['sigma_kms'] = np.full(len(names), sigma_kms)
        line['in_window'] = in_window

        return absorption_flux, wave_out, line
