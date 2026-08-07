"""
desisim.qso_galaxy_blend
=========================

Task #16 of the fork's handoff: a composite QSO/galaxy spectral blend that
combines a galaxy component and a QSO component into a single object's
mock spectrum, while keeping the whole thing exactly additively
decomposable -- both at the coarse 3-bucket level (continuum/emission/
absorption, decompose.py's convention) and, if wanted, down to each
component's own per-physical-process channels.

--------------------------------------------------------------------------
Why this exists / design approved by the PI (2026-08-07)
--------------------------------------------------------------------------
Real quasar spectra are frequently host-galaxy contaminated (especially at
low-to-moderate AGN luminosity / low redshift, where the host galaxy's
starlight is not negligible next to the AGN's own continuum), and this
fork's mocks should be able to represent that whole continuum of
"pure galaxy" to "pure QSO" objects, not just the two endpoints.

The blending functional form adopted here is the standard PyQSOFit-style
composite decomposition used for exactly this problem in the literature
(e.g. Shen et al.-style QSO host-decomposition fitting, going back to
Vanden Berk et al. 2006's own composite-fraction approach):

    total(lambda) = frac * QSO_component(lambda) + (1 - frac) * GALAXY_component(lambda)

with frac ~ Uniform(0, 1) drawn per-object by default (no preference for
any particular QSO/galaxy mix -- the downstream NPE is expected to learn
the real population distribution; this fork consistently prioritizes
*reachability* of the full parameter range over matching any specific
prior, per the same "recall over precision" philosophy already used for
e.g. the widened D4000-EW-scatter task).

Since QSO_component and GALAXY_component are each ALREADY fully additive
sums of their own sub-channels (continuum + emission + absorption, via
decompose.combine_into_channels), and a linear combination of additive
decompositions is trivially still additive, this blend introduces no new
physics and no risk to the project's core "spectra decomposable into
known additive components" requirement -- frac simply becomes one more
known ground-truth number alongside every other free parameter already
tracked per mock.

--------------------------------------------------------------------------
Per-component physics assignment (PI direction, 2026-08-07)
--------------------------------------------------------------------------
This module does not compute any spectra itself -- it is a pure combining
layer, exactly like decompose.py, deliberately kept that way so each
physical process stays independently testable (same pattern as every
other module in this fork; see e.g. absorption.py's module docstring on
why GALAXY/QSO.make_templates() wiring is deferred). The caller is
responsible for generating each side's channels using the PI's specified
assignment:

    GALAXY_component:
      - continuum_stellar: FSPS (or real basis template) stellar SED.
      - narrow_emission:   EMSpectrum's narrow (nebular) lines.
      - ism_absorption:    AbsorptionSpectrum (Na I D, Ca II H&K, Mg II
                            absorption), with its own
                            include_outflow_velshift galactic-outflow
                            treatment if wanted.
      - dust_flux:         DustAttenuation called with GALACTIC-type
                            priors/parameters (e.g. an MW/SMC/Calzetti-like
                            draw -- DustAttenuation itself is a universal
                            functional family per its own module docstring;
                            "galactic-type" here is a matter of which prior
                            draw the caller uses, not a separate class).

    QSO_component:
      - continuum_agn:     AGNPowerLawContinuum (wraps simqso's broken
                            power law).
      - broad_emission:    EMSpectrum's broad (AGN-like) lines, with its
                            own include_broad_velshift outflow treatment
                            if wanted (see templates.py's
                            BROADSHIFT_KMS_RANGE).
      - associated_absorption_flux: AssociatedAbsorberSystems' stochastic,
                            multi-system blueshifted narrow absorbers
                            (Mg II/C IV/Si IV/etc.).
      - dust_flux:         DustAttenuation called with QSO-type
                            priors/parameters (e.g. an SMC-like draw, the
                            literature-standard choice for AGN/QSO
                            reddening).
      - bal_flux:          reserved for a future additive BAL channel (see
                            decompose.py's BAL note) -- not populated by
                            any existing module yet.

    Narrow-line simplification (explicitly flagged, PI-accepted pending
    NPE performance, 2026-08-07): narrow emission lines are only drawn
    for the galaxy side in this default assignment -- i.e. this module
    does not, by itself, force a caller to draw a second, AGN-specific
    narrow-line region (NLR) contribution for the QSO side (BPT-diagram-
    distinct AGN narrow-line ratios are a real physical effect, and higher
    fidelity than the single shared narrow_emission array used here). Per
    the PI's explicit direction: "if the downstream NPE is capable of
    overcoming the narrow-line simplification then the simplification is
    fine, if not, we can add AGN associated narrow lines to the quasar
    component as well, drawn independently from the galactic
    contributions." This module's `qso` dict already accepts its own
    independent `narrow_emission` key (see blend_qso_galaxy's Args) for
    exactly that future extension -- a caller wanting the higher-fidelity
    treatment today can already pass a second, independently-drawn
    EMSpectrum narrow-line array into qso['narrow_emission'] without any
    change to this module; nothing here hardcodes the simplification.

--------------------------------------------------------------------------
Scope
--------------------------------------------------------------------------
Pure bookkeeping/combining layer, same scope discipline as decompose.py:
computes no new physics, applies no new priors beyond frac itself, and
does not resample -- every array in `galaxy` and `qso` must already be on
the same common wavelength grid (each contributing module already
resamples the caller-supplied continuum onto its own internal grid via
np.interp; harmonizing all of them onto one final output grid is left to
the caller / a future pipeline-orchestration layer, exactly as noted in
absorption.py and associated_absorption.py's own "not yet wired into
make_templates()" scope notes). Wiring this into GALAXY/QSO.make_templates()
itself is left for a follow-up commit.
"""

import numpy as np

from desisim.decompose import combine_into_channels


def blend_qso_galaxy(wave, galaxy, qso, frac=None, seed=None):
    """Blend an already-decomposed galaxy component and QSO component into
    a single additive composite spectrum, frac*QSO + (1-frac)*GALAXY.

    Args:
        wave (ndarray): common wavelength grid [npix] every array in
            `galaxy` and `qso` is defined on (see module docstring's
            Scope section -- not used for resampling).
        galaxy (dict): keyword arguments for decompose.combine_into_channels
            describing the galaxy-only component (must include
            'continuum_stellar'; see module docstring's per-component
            physics assignment for the recommended contents of each key).
        qso (dict): keyword arguments for decompose.combine_into_channels
            describing the QSO-only component (must include
            'continuum_stellar' -- pass an all-zero array, since the QSO
            side's own continuum lives in 'continuum_agn' instead; see
            module docstring). May also include its own independent
            'narrow_emission' (see module docstring's "Narrow-line
            simplification" note) for the higher-fidelity AGN-NLR
            extension, if/when it's needed.
        frac (float, optional): QSO fraction, 0 (pure galaxy) to 1 (pure
            QSO). Default None draws uniformly from [0, 1] (PyQSOFit-style
            composite fraction -- see module docstring). Passing an
            explicit value always wins (no draw).
        seed (int, optional): RNG seed, used only if frac is None.

    Returns:
        dict with keys 'continuum', 'emission', 'absorption', 'total'
        (each an ndarray [npix], the frac-blended composite), 'frac' (the
        resolved blend fraction actually used), and 'galaxy'/'qso' (each
        the full dict returned by combine_into_channels for that
        component alone, i.e. the *unblended*, frac=1-normalized
        per-component decomposition -- kept for ground-truth provenance
        and so a downstream NPE target can include the disaggregated
        per-component channels if wanted, not just the blended total).
    """
    if frac is None:
        rand = np.random.RandomState(seed)
        # Uniform(0, 1) composite fraction: no preference for any
        # particular QSO/galaxy mix by default -- see module docstring's
        # "recall over precision" rationale.
        frac = rand.uniform(0.0, 1.0)
    if not (0.0 <= frac <= 1.0):
        raise ValueError('frac must be in [0, 1], got {}'.format(frac))

    galaxy_out = combine_into_channels(wave, **galaxy)
    qso_out = combine_into_channels(wave, **qso)

    continuum = frac * qso_out['continuum'] + (1.0 - frac) * galaxy_out['continuum']
    emission = frac * qso_out['emission'] + (1.0 - frac) * galaxy_out['emission']
    absorption = frac * qso_out['absorption'] + (1.0 - frac) * galaxy_out['absorption']
    total = continuum + emission + absorption

    return {
        'continuum': continuum,
        'emission': emission,
        'absorption': absorption,
        'total': total,
        'frac': frac,
        'galaxy': galaxy_out,
        'qso': qso_out,
    }
