"""
desisim.mock_spectrum
=======================

Task #30: a standalone orchestrator that actually composes this fork's
independently-built, independently-tested additive physics modules
(fsps_continuum, AGNPowerLawContinuum, EMSpectrum, AbsorptionSpectrum,
AssociatedAbsorberSystems, DustAttenuation, blend_qso_galaxy) into one
fully ground-truth-decomposed mock spectrum -- WITHOUT touching
GALAXY/QSO.make_templates() at all.

--------------------------------------------------------------------------
Why a new, separate entry point instead of retrofitting GALAXY/QSO
--------------------------------------------------------------------------
As of 2026-08-25, none of dust.py, absorption.py, associated_absorption.py,
agn_continuum.py, or qso_galaxy_blend.py were reachable from
GALAXY/QSO.make_templates() at all -- those entry points still ran the
pre-fork pipeline unmodified. Retrofitting them directly was considered
and explicitly rejected per PI direction: GALAXY/QSO carry a lot of real
DESI-production-specific machinery this project doesn't need (imaging
color cuts, north/south photometric-system normalization, iterative
per-object template-rejection loops), and the PI's stated plan is to
substantially refactor this fork's whole generation pipeline again once
the project's normalizing-flow parameter sampler is trained and verified.
The right design now is therefore whatever requires the LEAST rework once
that happens, not whatever is most "integrated" with the legacy classes.

--------------------------------------------------------------------------
Design principle: separate "how a parameter is chosen" from "how a
spectrum is built from parameters"
--------------------------------------------------------------------------
Every stochastic parameter accepted anywhere in this module is either
None (draw from that submodule's own already-existing, already-tested
MAGIC-flagged prior -- exactly today's behavior) or an explicit
value/dict (used exactly as given, no draw at all). This is not a new
convention -- it's exactly the convention every submodule (EMSpectrum,
AbsorptionSpectrum, AssociatedAbsorberSystems, DustAttenuation,
AGNPowerLawContinuum) already uses internally. This module just composes
them without breaking that property.

The consequence: once the normalizing-flow sampler exists, swapping it in
means calling this exact same function with an explicit parameter dict
sampled from the flow instead of leaving it None. NONE of the
orchestration, wavelength-grid harmonization, or additive-combination
logic below needs to change -- only the caller's choice of what to pass
for `em_kwargs`, `dust_kwargs`, etc.

--------------------------------------------------------------------------
The one genuinely new piece of plumbing: wavelength-grid harmonization
--------------------------------------------------------------------------
Every contributing physics module already independently interpolates the
caller's continuum onto its OWN internal wavelength grid (see e.g.
absorption.py's, associated_absorption.py's own np.interp calls); none of
them harmonize onto a single common OUTPUT grid, and
decompose.combine_into_channels / qso_galaxy_blend.blend_qso_galaxy both
explicitly punt on this ("this module does not resample" -- see their own
module docstrings). That harmonization is the one piece of real, new
orchestration work this module provides: every component's own returned
(flux, wave) pair is np.interp'd onto the caller-specified common `wave`
grid before being combined. This is done defensively (always
interpolating, never assuming grid identity) even though most components
here are constructed to already share the grid exactly, since relying on
silent grid-identity assumptions is exactly the kind of thing that breaks
quietly during a future refactor.

--------------------------------------------------------------------------
Known limitation carried over from EMSpectrum, not fixed here
--------------------------------------------------------------------------
EMSpectrum.spectrum() returns a single combined narrow+broad `emspec`
array -- it does not separate the two internally (see templates.py's own
source). To recover broad_only for the QSO side's `combine_into_channels`
call, generate_qso_component() calls EM.spectrum() a second time with
new_line_broad_ratios forced to exactly zero (same seed => identical
narrow-ratio draws) and subtracts; the additive linearity this relies on
is already verified directly by
test_templates.py::TestNewLinesNarrowBroad::test_narrow_and_broad_flux_are_additive_in_emspec.
This is a workaround, not a fix -- if EMSpectrum's own return contract is
ever extended to separate narrow/broad natively, this can be simplified.

--------------------------------------------------------------------------
Per-component physics assignment
--------------------------------------------------------------------------
Follows the approved design plan (galaxy: dust-free FSPS SED + galactic-
type dust + narrow emission + ISM/CGM absorption; QSO: simqso broken
power-law + QSO-type dust + broad AGN-like emission + blueshifted
associated absorbers) exactly as documented in qso_galaxy_blend.py's own
module docstring -- see that file for the full rationale, including the
explicitly-accepted narrow-line simplification (QSO side carries no
independent AGN-NLR narrow lines yet).

--------------------------------------------------------------------------
Scope
--------------------------------------------------------------------------
Rest-frame only. No redshifting, resampling onto an observed-frame
instrumental grid, or noise/instrumental effects -- those remain DESI
pipeline concerns (specsim/quicksim), entirely separate from source-frame
ground truth. `wave` should be a log-uniform (constant-velocity-spacing)
grid, matching the convention EMSpectrum/AbsorptionSpectrum/
AssociatedAbsorberSystems already require internally.
"""

import numpy as np

from desisim.templates import EMSpectrum
from desisim.absorption import AbsorptionSpectrum
from desisim.associated_absorption import AssociatedAbsorberSystems
from desisim.dust import DustAttenuation
from desisim.agn_continuum import AGNPowerLawContinuum
from desisim.feii_continuum import FeIIPseudoContinuum
from desisim.decompose import combine_into_channels
from desisim.qso_galaxy_blend import blend_qso_galaxy


def _harmonize(common_wave, wave, flux):
    """np.interp `flux` (defined on `wave`) onto `common_wave`. Always
    applied, even when the two grids are expected to already match
    exactly (see module docstring) -- cheap, and removes any silent
    dependency on grid-identity assumptions holding across a future
    refactor."""
    return np.interp(common_wave, wave, flux)


def _child_seeds(seed, n):
    """Derive n independent child seeds from a single top-level seed, so
    a caller gets full reproducibility from one seed without correlating
    different components' draws. seed=None propagates as n Nones (every
    submodule already treats seed=None as "use fresh entropy")."""
    if seed is None:
        return [None] * n
    rand = np.random.RandomState(seed)
    return [int(s) for s in rand.randint(0, 2**31 - 1, size=n)]


def generate_galaxy_component(wave, continuum=None, fsps_kwargs=None,
                               em=None, em_kwargs=None,
                               absorber=None, absorber_kwargs=None,
                               dust=None, dust_kwargs=None, seed=None):
    """Assemble one galaxy-side additive spectrum: dust-free FSPS stellar
    SED + galactic-type dust attenuation + narrow emission + ISM/CGM
    absorption.

    Args:
        wave (ndarray): common rest-frame output grid [npix], log-uniform.
        continuum (tuple, optional): explicit (continuum_wave,
            continuum_flux) to use instead of drawing a fresh FSPS
            template. Default None draws one via fsps_basis_templates.
        fsps_kwargs (dict, optional): forwarded to
            fsps_continuum.fsps_basis_templates() when `continuum` is not
            given. dust2_range is forced to (0.0, 0.0) regardless of what
            is passed here (see fsps_continuum.py's own double-counting
            note) -- this component's own `dust` step is the only place
            galactic dust attenuation should be applied, so the FSPS SED
            fed into it must be dust-free.
        em (EMSpectrum, optional): pre-built instance. Default None builds
            EMSpectrum(log10wave=np.log10(wave)) (include_new_lines=False
            -- the galaxy side only ever carries narrow lines here).
        em_kwargs (dict, optional): forwarded to em.spectrum().
        absorber (AbsorptionSpectrum, optional): pre-built instance.
            Default None builds AbsorptionSpectrum(log10wave=np.log10(wave)).
        absorber_kwargs (dict, optional): forwarded to absorber.spectrum().
        dust (DustAttenuation, optional): pre-built instance, expected to
            carry galactic-type priors if a non-default one is wanted.
            Default None builds DustAttenuation().
        dust_kwargs (dict, optional): forwarded to dust.spectrum().
        seed (int, optional): top-level seed; derives independent child
            seeds for the FSPS draw, EMSpectrum, AbsorptionSpectrum, and
            DustAttenuation unless overridden inside the respective
            *_kwargs dict.

    Returns:
        dict: decompose.combine_into_channels()'s own return dict
        (continuum/emission/absorption/total/components), plus 'draws'
        -- a dict of the raw per-component tables/metadata returned by
        each submodule, for full ground-truth provenance.
    """
    fsps_kwargs = dict(fsps_kwargs) if fsps_kwargs else {}
    em_kwargs = dict(em_kwargs) if em_kwargs else {}
    absorber_kwargs = dict(absorber_kwargs) if absorber_kwargs else {}
    dust_kwargs = dict(dust_kwargs) if dust_kwargs else {}

    seed_fsps, seed_em, seed_absorber, seed_dust = _child_seeds(seed, 4)

    if continuum is None:
        from desisim.fsps_continuum import fsps_basis_templates
        fsps_kwargs['dust2_range'] = (0.0, 0.0)  # see docstring: dust-free by construction
        fsps_kwargs.setdefault('nbase', 1)
        fsps_kwargs.setdefault('seed', seed_fsps)
        baseflux, basewave, basemeta = fsps_basis_templates(**fsps_kwargs)
        continuum_wave, continuum_flux = basewave, baseflux[0]
    else:
        continuum_wave, continuum_flux = continuum
        basemeta = None
    continuum_stellar = _harmonize(wave, continuum_wave, continuum_flux)

    if em is None:
        em = EMSpectrum(log10wave=np.log10(wave))
    em_kwargs.setdefault('seed', seed_em)
    emflux, emwave, emline = em.spectrum(**em_kwargs)
    narrow_emission = _harmonize(wave, emwave, emflux)

    if absorber is None:
        absorber = AbsorptionSpectrum(log10wave=np.log10(wave))
    absorber_kwargs.setdefault('seed', seed_absorber)
    absflux, abswave, absline = absorber.spectrum(wave, continuum_stellar, **absorber_kwargs)
    ism_absorption = _harmonize(wave, abswave, absflux)

    if dust is None:
        dust = DustAttenuation()
    dust_kwargs.setdefault('seed', seed_dust)
    dustflux, dustwave, dusttable = dust.spectrum(wave, continuum_stellar, **dust_kwargs)
    dust_flux = _harmonize(wave, dustwave, dustflux)

    out = combine_into_channels(wave, continuum_stellar=continuum_stellar,
                                 narrow_emission=narrow_emission,
                                 ism_absorption=ism_absorption,
                                 dust_flux=dust_flux)
    out['draws'] = dict(fsps_basemeta=basemeta, em_line=emline, absorber_line=absline,
                         dust_theta=dusttable)
    return out


def generate_qso_component(wave, agn=None, agn_kwargs=None,
                            em=None, em_kwargs=None,
                            associated=None, associated_kwargs=None,
                            dust=None, dust_kwargs=None,
                            feii=None, feii_kwargs=None, seed=None):
    """Assemble one QSO-side additive spectrum: simqso broken power-law
    continuum + QSO-type dust attenuation + broad (AGN-like) emission +
    blueshifted associated-absorber systems + Fe II UV/optical pseudo-
    continuum.

    Narrow-line simplification (see qso_galaxy_blend.py's module
    docstring): this function discards em's own narrow-line component
    entirely (recovering only the broad component via the subtraction
    described in this module's docstring) -- it does not draw an
    independent AGN-NLR narrow-line contribution. A caller wanting that
    future extension should add their own independently-drawn
    narrow_emission array to this function's output before passing it
    into blend_qso_galaxy / combine_into_channels; nothing here hardcodes
    against that.

    Args:
        wave (ndarray): common rest-frame output grid [npix], log-uniform.
        agn (AGNPowerLawContinuum, optional): pre-built instance. Default
            None builds AGNPowerLawContinuum().
        agn_kwargs (dict, optional): forwarded to agn.spectrum().
        em (EMSpectrum, optional): pre-built instance. Default None builds
            EMSpectrum(log10wave=np.log10(wave), include_new_lines=True)
            -- broad lines require include_new_lines=True (see
            templates.py: these are currently the ONLY lines EMSpectrum
            can broaden).
        em_kwargs (dict, optional): forwarded to em.spectrum(). Do not
            pass new_line_broad_ratios here if you want it drawn; if you
            do pass it, that exact draw is honored (and re-used
            identically for the internal narrow-only subtraction call).
        associated (AssociatedAbsorberSystems, optional): pre-built
            instance. Default None builds
            AssociatedAbsorberSystems(log10wave=np.log10(wave)).
        associated_kwargs (dict, optional): forwarded to
            associated.spectrum().
        dust (DustAttenuation, optional): pre-built instance, expected to
            carry QSO-type (e.g. SMC-like) priors if a non-default one is
            wanted. Default None builds DustAttenuation().
        dust_kwargs (dict, optional): forwarded to dust.spectrum().
        feii (FeIIPseudoContinuum, optional): pre-built instance. Default
            None builds FeIIPseudoContinuum(log10wave=np.log10(wave)).
        feii_kwargs (dict, optional): forwarded to feii.spectrum() (e.g.
            uv_params/optical_params/uv_norm/optical_norm -- see
            feii_continuum.py for the independent-UV/optical-draw
            rationale).
        seed (int, optional): top-level seed; derives independent child
            seeds for each submodule unless overridden inside the
            respective *_kwargs dict.

    Returns:
        dict: decompose.combine_into_channels()'s own return dict
        (continuum/emission/absorption/total/components, with
        continuum_stellar forced to zero and continuum_agn populated),
        plus 'draws' for provenance.
    """
    agn_kwargs = dict(agn_kwargs) if agn_kwargs else {}
    em_kwargs = dict(em_kwargs) if em_kwargs else {}
    associated_kwargs = dict(associated_kwargs) if associated_kwargs else {}
    dust_kwargs = dict(dust_kwargs) if dust_kwargs else {}
    feii_kwargs = dict(feii_kwargs) if feii_kwargs else {}

    seed_agn, seed_em, seed_associated, seed_dust, seed_feii = _child_seeds(seed, 5)

    if agn is None:
        agn = AGNPowerLawContinuum()
    agn_kwargs.setdefault('seed', seed_agn)
    agnflux, agnwave, slopetable = agn.spectrum(wave, **agn_kwargs)
    continuum_agn = _harmonize(wave, agnwave, agnflux)

    if em is None:
        em = EMSpectrum(log10wave=np.log10(wave), include_new_lines=True)
    em_kwargs.setdefault('seed', seed_em)
    emflux_total, emwave, emline_total = em.spectrum(**em_kwargs)

    narrow_only_kwargs = dict(em_kwargs)
    narrow_only_kwargs['new_line_broad_ratios'] = {n: 0.0 for n in em.NEW_LINE_NAMES}
    emflux_narrow_only, _, _ = em.spectrum(**narrow_only_kwargs)
    emflux_broad_only = emflux_total - emflux_narrow_only
    broad_emission = _harmonize(wave, emwave, emflux_broad_only)

    if associated is None:
        associated = AssociatedAbsorberSystems(log10wave=np.log10(wave))
    associated_kwargs.setdefault('seed', seed_associated)
    assocflux, assocwave, assocline = associated.spectrum(wave, continuum_agn, **associated_kwargs)
    associated_absorption_flux = _harmonize(wave, assocwave, assocflux)

    if dust is None:
        dust = DustAttenuation()
    dust_kwargs.setdefault('seed', seed_dust)
    dustflux, dustwave, dusttable = dust.spectrum(wave, continuum_agn, **dust_kwargs)
    dust_flux = _harmonize(wave, dustwave, dustflux)

    if feii is None:
        feii = FeIIPseudoContinuum(log10wave=np.log10(wave))
    feii_kwargs.setdefault('seed', seed_feii)
    feiiflux, feiiwave, feiiparams = feii.spectrum(**feii_kwargs)
    feii_flux = _harmonize(wave, feiiwave, feiiflux)

    out = combine_into_channels(wave, continuum_stellar=np.zeros_like(wave),
                                 continuum_agn=continuum_agn,
                                 broad_emission=broad_emission,
                                 associated_absorption_flux=associated_absorption_flux,
                                 dust_flux=dust_flux,
                                 feii_flux=feii_flux)
    out['draws'] = dict(agn_slopes=slopetable, em_line_total=emline_total,
                         associated_line=assocline, dust_theta=dusttable,
                         feii_params=feiiparams)
    return out


def generate_blended_spectrum(wave, galaxy_kwargs=None, qso_kwargs=None, frac=None, seed=None):
    """Full pipeline: generate a galaxy component and a QSO component on
    the same common grid, then blend them via qso_galaxy_blend's
    PyQSOFit-style composite (frac*QSO + (1-frac)*GALAXY).

    Args:
        wave (ndarray): common rest-frame output grid [npix], log-uniform.
        galaxy_kwargs (dict, optional): forwarded to
            generate_galaxy_component() (excluding `wave`).
        qso_kwargs (dict, optional): forwarded to generate_qso_component()
            (excluding `wave`).
        frac (float, optional): forwarded to blend_qso_galaxy (default
            None draws Uniform(0,1)).
        seed (int, optional): top-level seed; derives independent child
            seeds for the galaxy component, QSO component, and frac draw.

    Returns:
        dict: blend_qso_galaxy()'s own return dict (continuum/emission/
        absorption/total/frac/galaxy/qso), where 'galaxy' and 'qso' are
        each the full generate_*_component() output (including 'draws')
        for that side.
    """
    galaxy_kwargs = dict(galaxy_kwargs) if galaxy_kwargs else {}
    qso_kwargs = dict(qso_kwargs) if qso_kwargs else {}

    seed_galaxy, seed_qso, seed_frac = _child_seeds(seed, 3)
    galaxy_kwargs.setdefault('seed', seed_galaxy)
    qso_kwargs.setdefault('seed', seed_qso)

    galaxy_out = generate_galaxy_component(wave, **galaxy_kwargs)
    qso_out = generate_qso_component(wave, **qso_kwargs)

    # blend_qso_galaxy calls combine_into_channels itself; since galaxy_out/
    # qso_out are already combine_into_channels() dicts (not raw kwargs),
    # re-derive the plain continuum/emission/absorption arrays it expects
    # rather than re-summing through combine_into_channels a second time.
    frac_resolved = frac
    if frac_resolved is None:
        frac_resolved = np.random.RandomState(seed_frac).uniform(0.0, 1.0)
    if not (0.0 <= frac_resolved <= 1.0):
        raise ValueError('frac must be in [0, 1], got {}'.format(frac_resolved))

    continuum = frac_resolved * qso_out['continuum'] + (1.0 - frac_resolved) * galaxy_out['continuum']
    emission = frac_resolved * qso_out['emission'] + (1.0 - frac_resolved) * galaxy_out['emission']
    absorption = frac_resolved * qso_out['absorption'] + (1.0 - frac_resolved) * galaxy_out['absorption']
    total = continuum + emission + absorption

    return {
        'continuum': continuum,
        'emission': emission,
        'absorption': absorption,
        'total': total,
        'frac': frac_resolved,
        'galaxy': galaxy_out,
        'qso': qso_out,
    }
