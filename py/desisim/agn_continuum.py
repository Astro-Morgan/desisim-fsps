"""
desisim.agn_continuum
======================

Wraps simqso's own AGN accretion-disk broken-power-law continuum
(simqso.sqgrids.BrokenPowerLawContinuumVar) as an independent, additive
"continuum" channel for this fork's decomposition scheme -- per explicit
PI direction: "final ground-truth continuum is the dust-free FSPS stellar
SED + the simqso broken power-law." This reuses simqso's own architecture
rather than reimplementing an independent power-law model, since simqso's
continuum is already a genuinely separable object (confirmed below) and
is already the literature-calibrated model driving this fork's existing
SIMQSO.make_templates() QSO path internally.

--------------------------------------------------------------------------
Why wrapping (not reimplementing) simqso's continuum
--------------------------------------------------------------------------
Confirmed by direct inspection of simqso's source
(simqso.sqgrids.BrokenPowerLawContinuumVar), 2026-08-06:
  - It IS a genuinely independent, additive flux generator:
    `.render(wave, z, slopes, fluxNorm=None)` returns a plain continuum
    flux array with no emission-line or dust dependence baked in.
  - `simqso.sqrun.buildQsoSpectrum` combines continuum and emission
    features as continuum*(1+sum_of_EW_shapes) (multiplicative), but that
    combination happens entirely OUTSIDE BrokenPowerLawContinuumVar --
    calling .render() directly (as this module does) sidesteps it
    completely, so this module's output is a pure, standalone continuum
    array by construction, not entangled with simqso's emission-line
    convention.
  - The fork's existing SIMQSO.make_templates() path (templates.py)
    already uses this exact class, with this exact functional form, and
    (for the default 'BOSS_DR9_PLEpivot' model) these exact slope/break-
    point priors, via simqso.sqmodels.get_BossDr9_model_vars's
    'BossDr9_fiducial_continuum'. DEFAULT_SLOPE_PRIORS/DEFAULT_BREAK_POINTS
    below are copied verbatim from that object, reused here for numerical
    consistency with the model already driving the rest of this fork's
    QSO generation, rather than introducing a second, independently-tuned
    power-law family.
  - Also confirmed: the default model carries no baked-in dust/extinction
    feature (unlike FSPS's dust_type=2 default before the 2026-08-06 fix),
    so there is no double-counting risk between this continuum and
    DustAttenuation.

--------------------------------------------------------------------------
A real upstream quirk: simqso's own `seed=` kwarg does not work here
--------------------------------------------------------------------------
simqso.sqgrids.QsoSimVar.__call__ (the base class) sets
np.random.seed(self.seed) before sampling if a seed was given at
construction -- but BrokenPowerLawContinuumVar's method-resolution order
(BrokenPowerLawContinuumVar -> ContinuumVar -> MultiDimVar -> QsoSimVar ->
...) resolves __call__ to MultiDimVar.__call__, which does NOT call
np.random.seed() at all. Verified empirically (2026-08-06): constructing
the same BrokenPowerLawContinuumVar with the same seed= kwarg and calling
it twice gives DIFFERENT slope draws each time -- the kwarg is silently a
no-op for this class as of the current github.com/imcgreer/simqso HEAD.
This is an upstream simqso limitation, not something introduced by this
fork. To provide the reproducible seed=... behavior every other module in
this fork guarantees, AGNPowerLawContinuum.spectrum() below seeds the
GLOBAL numpy RNG (np.random.seed(seed)) immediately before calling the
underlying var -- verified empirically to produce identical draws for a
repeated seed (GaussianSampler/CdfSampler.sample() draw from
np.random.random() directly, the global state, not a per-instance
RandomState) -- rather than trusting simqso's own (broken) per-instance
seed argument. Callers running many draws in a shared process should be
aware this reseeds the global RNG each time seed is given explicitly.

--------------------------------------------------------------------------
Scope
--------------------------------------------------------------------------
Purely a continuum generator -- no dust, no emission lines, no IGM
absorption. Per PI direction, the final ground-truth continuum channel is
the dust-free FSPS stellar SED (fsps_continuum.py, apply_dust=False)
PLUS this module's AGN power-law continuum (for QSO or QSO-blend
components) -- see decompose.py's combine_into_channels(
continuum_stellar=..., continuum_agn=...). Independently testable, not
wired into GALAXY/QSO.make_templates() here (same standalone-module
convention as dust.py/absorption.py/camera_calibration.py). Requires
simqso to be installed (already a hard dependency of this fork's existing
SIMQSO QSO-generation path).
"""

import numpy as np


# Default slopes/break points: exactly simqso.sqmodels's
# BossDr9_fiducial_continuum, the model already driving this fork's
# existing SIMQSO.make_templates() QSO path (via get_BossDr9_model_vars)
# -- reused here for consistency rather than introducing a second,
# independently-tuned continuum model. Five Gaussian-sampled f_nu
# power-law slopes (alpha_nu, where f_nu ~ nu^alpha_nu) across four break
# points [Angstrom, rest-frame], following the Vanden Berk et al. (2001)
# SDSS QSO composite-spectrum segmentation.
DEFAULT_SLOPE_PRIORS = [
    dict(mean=-1.50, sigma=0.3),
    dict(mean=-0.50, sigma=0.3),
    dict(mean=-0.37, sigma=0.3),
    dict(mean=-1.70, sigma=0.3),
    dict(mean=-1.03, sigma=0.3),
]
DEFAULT_BREAK_POINTS = [1100.0, 5700.0, 9730.0, 22300.0]


class AGNPowerLawContinuum(object):
    """Wraps simqso.sqgrids.BrokenPowerLawContinuumVar as an independent,
    additive AGN accretion-disk continuum channel. See module docstring
    for the full rationale and the seed-reproducibility caveat.
    """

    def __init__(self, slope_priors=None, break_points=None):
        """
        Args:
            slope_priors (list of dict, optional): override
                DEFAULT_SLOPE_PRIORS; each dict needs 'mean'/'sigma' for a
                Gaussian draw on alpha_nu (f_nu ~ nu^alpha_nu), one more
                entry than break_points.
            break_points (list of float, optional): override
                DEFAULT_BREAK_POINTS [Angstrom, rest-frame].
        """
        self.slope_priors = list(slope_priors) if slope_priors is not None else list(DEFAULT_SLOPE_PRIORS)
        self.break_points = list(break_points) if break_points is not None else list(DEFAULT_BREAK_POINTS)
        if len(self.slope_priors) != len(self.break_points) + 1:
            raise ValueError('slope_priors must have exactly one more entry than break_points '
                              '(got {} slope_priors, {} break_points)'.format(
                                  len(self.slope_priors), len(self.break_points)))

    def _build_var(self):
        try:
            from simqso.sqgrids import BrokenPowerLawContinuumVar, GaussianSampler
        except ImportError:
            raise ImportError('AGNPowerLawContinuum requires simqso: '
                               'please install https://github.com/imcgreer/simqso')
        samplers = [GaussianSampler(p['mean'], p['sigma']) for p in self.slope_priors]
        return BrokenPowerLawContinuumVar(samplers, self.break_points)

    def spectrum(self, wave, z=0.0, slopes=None, flux_norm=None, seed=None):
        """Build the additive AGN power-law continuum flux array.

        Args:
            wave (ndarray): wavelength grid to evaluate on. Pass z=0.0
                with a rest-frame wave for a rest-frame continuum
                (matching this fork's other new modules' convention).
            z (float): redshift passed through to
                BrokenPowerLawContinuumVar.render (0.0 for rest-frame
                output).
            slopes (ndarray, optional): explicit (1, nseg) array of
                alpha_nu slopes overriding a fresh draw; nseg =
                len(break_points)+1.
            flux_norm (dict, optional): passed through to simqso's
                .render() as `fluxNorm`; {'wavelength':.., 'M_AB':..,
                'DM':<callable(z)->distmod>} to normalize to a physical
                absolute magnitude. If None (default), the output is
                arbitrarily normalized -- simqso's own convention when no
                fluxNorm is given.
            seed (int, optional): RNG seed. NOTE: seeds the GLOBAL numpy
                RNG (np.random.seed), not a local RandomState -- see
                module docstring's "upstream quirk" section. Only applied
                if `slopes` is not explicitly given.

        Returns:
            Tuple of (continuum_flux, wave, slope_table), where
            slope_table is a Table with the nseg resolved alpha_nu slopes
            actually used (drawn or explicit).
        """
        from astropy.table import Table

        var = self._build_var()

        if slopes is None:
            if seed is not None:
                np.random.seed(seed)
            slopes = var(1)
        else:
            slopes = np.atleast_2d(slopes)

        wave = np.asarray(wave, dtype=float)
        continuum_flux = var.render(wave, z, slopes, fluxNorm=flux_norm)

        slope_table = Table()
        slope_table['segment'] = np.arange(slopes.shape[1])
        slope_table['alpha_nu'] = np.asarray(slopes[0], dtype=float)

        return continuum_flux, wave, slope_table
