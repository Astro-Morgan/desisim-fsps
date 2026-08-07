"""
desisim.decompose
==================

Groups this fork's already-independent additive ground-truth channels
(continuum, emission, ISM/CGM absorption, dust) into the 3-bucket
decomposition scheme the PI specified directly, for final delivery to the
downstream NPE:

    continuum = stellar SED (FSPS or real basis templates)
                + AGN accretion-disk power-law continuum (simqso)
    emission  = narrow lines + broad lines
                (+ any dust-scattering excess pushed above the continuum
                 -- physical mechanism not yet finalized, see Open items)
    absorption = ISM/CGM narrow absorption + BAL troughs
                 + dust attenuation (flux deficit)

--------------------------------------------------------------------------
Why dust moves into the "absorption" bucket here
--------------------------------------------------------------------------
This is a re-grouping decision, not a change to how dust is computed.
DustAttenuation (dust.py) still returns its own independently-generated,
separately-labeled flux-deficit array exactly as before -- nothing about
its functional form, priors, or per-component (galaxy vs. QSO) usage
changes. What changes here is only which *output bucket* that array gets
summed into for the final 3-channel product: previously implied to be its
own 4th/5th channel, now folded into "absorption" alongside
AbsorptionSpectrum's ISM/CGM lines and (once parametrized) BAL troughs,
because all three are, physically, sources of *non-stellar light removal*
along the line of sight -- exactly the PI's stated grouping rationale.
The dust channel's own internal ground truth (theta0..theta5, lambda_bump,
lambda_break, etc.) is NOT lost -- callers that want the disaggregated
per-physical-process arrays (e.g. for a more granular NPE target) should
keep using dust.py/absorption.py/bal.py directly; this module only builds
the coarser 3-bucket VIEW on top of them.

--------------------------------------------------------------------------
What already exists vs. what's still open
--------------------------------------------------------------------------
Currently exposable as genuine, independent additive channels (fully
implemented and tested elsewhere in this fork):
  - continuum_stellar: FSPS (fsps_continuum.py, dust2=0 by default per the
    2026-08-06 double-counting fix) or real DESI basis templates.
  - narrow_emission, broad_emission: EMSpectrum (templates.py).
  - ism_absorption: AbsorptionSpectrum (absorption.py).
  - dust_flux: DustAttenuation (dust.py) -- called once per blended
    component (galaxy, QSO) per its own module docstring.

Two pieces of the PI's scheme are NOT yet implemented anywhere in this
fork and are passed into this module as optional (default None/zero)
placeholders until built:
  - continuum_agn: the AGN accretion-disk power-law continuum. Confirmed
    (2026-08-06, by reading simqso's source directly) that this is a
    genuinely separable object in simqso's own architecture --
    `simqso.sqgrids.BrokenPowerLawContinuumVar` is evaluated independently
    of emission lines via `.render(wave, z, slopes, fluxNorm)`, and
    `simqso.sqrun.buildQsoSpectrum(..., save_components=True)` already
    returns a components dict separating the continuum's raw contribution
    from each emission feature's (multiplicative, EW-based) contribution.
    Also confirmed: the fork's current default QSO model
    (`simqso.sqmodels.get_BossDr9_model_vars`, used by
    `templates.SIMQSO._make_simqso_templates`) does NOT include simqso's
    own optional SMC/Calzetti dust-extinction feature -- so, unlike the
    FSPS/Calzetti case, there is currently no double-counting risk between
    simqso's QSO continuum and this fork's DustAttenuation. Exposing
    continuum_agn as an independent callable channel (mirroring how
    fsps_continuum.py exposes the stellar side) is a genuinely new, but
    now well-scoped, integration task -- not started yet, and needs a
    design decision on whether to call into simqso's own
    BrokenPowerLawContinuumVar directly (tied to whichever QSO model is
    configured) or reimplement an independent, simqso-decoupled power-law
    continuum generator (in the same spirit as the SBPL machinery just
    built for dust.py, reusable even where simqso isn't installed).
  - dust_scatter_excess: the "dust scattering into the line of sight [that]
    pushes [emission] above the continuum" effect the PI flagged. This is
    real physics (documented, e.g., in polar-scattered/Type-2 AGN spectra
    and resonant-line radiative transfer such as Lyman-alpha), but the
    PI's exact intended mechanism was not specified precisely enough here
    to commit to one parametric form without risking building the wrong
    thing -- flagged for PI clarification rather than guessed at (see the
    two candidate mechanisms in this module's accompanying discussion:
    (a) resonant-line scattering wings vs. (b) a scattered-light pseudo-
    continuum/pseudo-emission excess). Not started.

BAL troughs are intentionally NOT summed by this module yet: bal.py's
existing mechanism selects and multiplies a whole real-observed BAL
transmission template onto the spectrum (a multiplicative correction, not
an additive flux-deficit array), so it cannot be summed alongside the
other three additive absorption terms without first computing a residual
(BAL-on spectrum minus BAL-off spectrum) the way simqso's own multiplicative
emission features are reconciled to additive form (see below) -- or,
preferably, once task #17's independent parametric/stochastic BAL model
(a genuinely additive channel, in the spirit of AbsorptionSpectrum) lands.
A `bal_flux` keyword is provided for forward-compatibility but is None by
default and simply omitted from the absorption sum if not supplied.

--------------------------------------------------------------------------
Note on simqso's own additive/multiplicative convention
--------------------------------------------------------------------------
Internally, simqso's `buildQsoSpectrum` combines its own continuum and
emission features as `continuum * (1 + sum_of_line_EW_shapes)`, NOT
additively -- a different internal convention from this fork's `EMSpectrum`
(which is additive throughout). This does not block using simqso's
continuum in the "continuum" bucket here (the continuum var itself is
already a plain additive flux array, independent of how simqso combines
it with lines downstream) -- but if simqso's own broad-line templates
(via `BossDr9_EmLineTemplate`/`generateBEffEmissionLines`) are ever used
as this fork's QSO broad-line ground truth instead of `EMSpectrum`'s own
lines, their true per-line additive flux contribution must be
reconstructed as `continuum_agn(wave) * emspec_line_fraction(wave)`
(exact, not an approximation) before being placed in the "emission"
bucket -- not yet needed since this fork currently generates its own
broad lines via EMSpectrum, independent of simqso's.

--------------------------------------------------------------------------
Scope
--------------------------------------------------------------------------
Pure bookkeeping/grouping layer -- computes no new physics, applies no
new priors, and does not change any existing module's output. Every input
array must already be on the same common wavelength grid (this module
does not resample); combine_into_channels does not know or care how each
piece was generated.
"""

import numpy as np


def combine_into_channels(wave, continuum_stellar, narrow_emission=None, broad_emission=None,
                           ism_absorption=None, dust_flux=None,
                           continuum_agn=None, dust_scatter_excess=None, bal_flux=None):
    """Group already-generated additive channels into the PI-specified
    3-bucket decomposition (continuum / emission / absorption), plus the
    total. See module docstring for exactly what belongs in each bucket
    and which pieces (continuum_agn, dust_scatter_excess, bal_flux) are
    not yet implemented elsewhere in this fork.

    Args:
        wave (ndarray): common wavelength grid [npix] every array below is
            defined on. Not used for resampling -- purely for the returned
            table/shape bookkeeping.
        continuum_stellar (ndarray): stellar SED continuum [npix]
            (FSPS or real basis templates). Required -- every mock has one.
        narrow_emission, broad_emission (ndarray, optional): additive
            emission-line flux from EMSpectrum. Default: zero.
        ism_absorption (ndarray, optional): additive ISM/CGM absorption
            flux deficit from AbsorptionSpectrum (<=0 convention). Default:
            zero.
        dust_flux (ndarray, optional): additive dust attenuation flux
            deficit from DustAttenuation (<=0 convention). If attenuating a
            QSO/galaxy blend, sum each component's own DustAttenuation
            call's output before passing it in here (see dust.py's
            "Universality across galaxies and QSOs"). Default: zero.
        continuum_agn (ndarray, optional): AGN accretion-disk power-law
            continuum. NOT YET IMPLEMENTED anywhere in this fork -- see
            module docstring's "Open items". Default: zero.
        dust_scatter_excess (ndarray, optional): dust-scattering-into-LOS
            excess pushing emission above the continuum. NOT YET
            IMPLEMENTED -- mechanism pending PI clarification (see module
            docstring). Default: zero.
        bal_flux (ndarray, optional): forward-compatible slot for an
            eventual additive parametric BAL channel (task #17). bal.py's
            *current* whole-template multiplicative mechanism should NOT
            be passed here (see module docstring's BAL note). Default: zero
            (omitted from the sum).

    Returns:
        dict with keys 'continuum', 'emission', 'absorption', 'total'
        (each an ndarray [npix]), plus 'components' -- a dict recording
        exactly which named arrays were summed into each bucket (with
        zero-filled placeholders explicitly marked), for provenance.
    """
    wave = np.asarray(wave)
    npix = wave.shape[0]
    zero = np.zeros(npix)

    def _resolve(name, arr):
        if arr is None:
            return zero, False
        arr = np.asarray(arr)
        if arr.shape[0] != npix:
            raise ValueError('{} has {} pixels, expected {} (len(wave))'.format(
                name, arr.shape[0], npix))
        return arr, True

    continuum_stellar_arr, _ = _resolve('continuum_stellar', continuum_stellar)
    continuum_agn_arr, has_continuum_agn = _resolve('continuum_agn', continuum_agn)
    narrow_emission_arr, has_narrow = _resolve('narrow_emission', narrow_emission)
    broad_emission_arr, has_broad = _resolve('broad_emission', broad_emission)
    dust_scatter_arr, has_dust_scatter = _resolve('dust_scatter_excess', dust_scatter_excess)
    ism_absorption_arr, has_ism = _resolve('ism_absorption', ism_absorption)
    dust_flux_arr, has_dust = _resolve('dust_flux', dust_flux)
    bal_flux_arr, has_bal = _resolve('bal_flux', bal_flux)

    continuum = continuum_stellar_arr + continuum_agn_arr
    emission = narrow_emission_arr + broad_emission_arr + dust_scatter_arr
    absorption = ism_absorption_arr + dust_flux_arr + bal_flux_arr
    total = continuum + emission + absorption

    components = {
        'continuum_stellar': True,
        'continuum_agn': has_continuum_agn,
        'narrow_emission': has_narrow,
        'broad_emission': has_broad,
        'dust_scatter_excess': has_dust_scatter,
        'ism_absorption': has_ism,
        'dust_flux': has_dust,
        'bal_flux': has_bal,
    }

    return {
        'continuum': continuum,
        'emission': emission,
        'absorption': absorption,
        'total': total,
        'components': components,
    }
