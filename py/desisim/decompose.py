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
    absorption = ISM/CGM narrow absorption + associated absorber systems
                 + BAL troughs + dust attenuation (flux deficit)

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
  - continuum_agn: AGNPowerLawContinuum (agn_continuum.py), wrapping
    simqso's own BrokenPowerLawContinuumVar. Per explicit PI direction
    ("final ground-truth continuum is the dust-free FSPS stellar SED + the
    simqso broken power-law"), the target continuum bucket is exactly
    continuum_stellar + continuum_agn -- no independent reimplementation,
    no double-counting risk (confirmed the default QSO model carries no
    baked-in dust extinction feature).
  - narrow_emission, broad_emission: EMSpectrum (templates.py).
  - ism_absorption: AbsorptionSpectrum (absorption.py).
  - associated_absorption_flux: AssociatedAbsorberSystems
    (associated_absorption.py, added 2026-08-06) -- stochastic, multi-
    system narrow QSO associated absorption (Mg II/C IV/Si IV/etc.,
    Poisson-process count and velocity placement, Maxwell-Boltzmann
    per-system velocity dispersion). A genuinely different statistical
    object from ism_absorption (which is one shared-kinematic draw of a
    fixed line set), landing in the same "absorption" bucket since both
    are non-stellar, non-dust LOS light removal.
  - dust_flux: DustAttenuation (dust.py) -- called once per blended
    component (galaxy, QSO) per its own module docstring. As of
    2026-08-06, DustAttenuation's own optional `include_scattered_light`
    flag folds the dust-scattering-into-LOS excess (see below) directly
    into this same returned array, so dust_flux is no longer guaranteed
    non-positive when that flag is on -- this is intentional, not a bug.
  - feii_flux: FeIIPseudoContinuum (feii_continuum.py, added 2026-08-25)
    -- the Fe II UV+optical pseudo-continuum (a quasi-continuous blend of
    tens of thousands of overlapping Fe II transitions, indistinguishable
    from a genuine continuum at real broad-line-region velocity widths).
    Landed in the "emission" bucket alongside narrow_emission/
    broad_emission since it is, physically, emitted photons (fluorescence/
    recombination/collisional excitation), not a LOS light-removal
    process -- despite the "continuum" in its name, it does not belong in
    the "continuum" bucket, which is reserved for the stellar SED + AGN
    power-law per the PI's original 3-bucket specification.
  - balmer_flux: BalmerContinuum (balmer_continuum.py, added 2026-08-26)
    -- the hydrogen recombination cascade's free-bound Balmer continuum
    (Grandi 1982 edge at 3646A) plus a broadened Lyalpha+Balmer(n=3..50)
    line series sharing one (T_e, log n_e) physical state. Same reasoning
    as feii_flux: lands in the "emission" bucket despite "continuum" in
    the name, since both pieces are real emitted photons, not LOS light
    removal.
  - igm_flux: IGMAbsorption (igm_absorption.py, added 2026-08-27) --
    reconciles this fork's pre-existing but previously-disconnected
    Lyman-alpha-forest (lya_mock_p1d.MockMaker) and DLA (dla.py) physics
    into a single additive flux-deficit channel, QSO-only, requiring the
    QSO's actual redshift (zqso) as an input since IGM absorption is not
    a rest-frame-intrinsic property of the source (see igm_absorption.py's
    module docstring for the full architectural discussion). Lands in the
    "absorption" bucket alongside ism_absorption/associated_absorption_flux/
    dust_flux, since it is, physically, the same kind of non-stellar,
    non-dust LOS light removal. Metal-line intervening absorption is
    explicitly deferred (see igm_absorption.py).

RESOLVED (2026-08-06): the dust-scattering-into-LOS excess -- dust/
electron scattering redirecting light back INTO the line of sight (e.g.
polar-scattered light in Type 2 AGN), appearing as a positive flux excess
-- no longer needs a separate `dust_scatter_excess` input. Per PI
direction, this was resolved by grounding it empirically (searched
2026-08-06: NGC 1068's scattered/polarized flux is ~1% of total, Type 2
quasar polar scattering ~3% of intrinsic continuum -- see dust.py's
"scattered light back into the line of sight" docstring section for
sources) and pairing it directly with DustAttenuation itself, as a
fraction f_scat of the same |dust_flux| light already being removed from
a component's LOS -- self-consistent by construction (can never scatter
back more than was removed at the reference wavelength) and sidestepping
the continuum-vs-emission classification question entirely: the net dust
effect (attenuation deficit plus scattered excess) stays in the
"absorption" bucket via the SAME `dust_flux` argument already accepted
here, simply no longer guaranteed non-positive when
DustAttenuation(include_scattered_light=True) is used. The
`dust_scatter_excess` keyword below is kept only for forward-compatible
generality (e.g. a future, non-dust-coupled scattering mechanism) and is
not expected to be needed for this specific effect.

RESOLVED (2026-08-27, task #36): bal_flux is now genuinely populated.
bal.py's original real-template mechanism remains untouched (it multiplies
a whole observed BAL transmission template from DESI's private
$DESI_BASIS_TEMPLATES tree onto flux -- a multiplicative correction, not
an additive array, and not obtainable in this project's dev environment;
see bal_trough.py's module docstring), but per PI direction this fork
now has its OWN from-scratch, additive, physically-parametrized BAL
trough model: bal_trough.BALTrough (mirrors dust.py's multiplicative-to-
additive reformulation pattern exactly, same as igm_absorption.py did for
task #35). Its free parameters were calibrated against the real, public
SDSS DR14Q balnicity-index (BI) distribution (Paris et al. 2018) rather
than left as unvalidated guesses -- see bal_trough.py's module docstring
for the full empirical-backtest writeup. BAL (broad, blended, several-
thousand-km/s troughs) and associated absorption (narrow, discrete,
per-system kinematics, associated_absorption.py) remain physically and
computationally distinct channels, both landing in the "absorption"
bucket via their own independently-computed arrays. The `bal_flux`
keyword below is None by default (omitted from the sum) for callers not
using BALTrough.

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
                           ism_absorption=None, associated_absorption_flux=None, dust_flux=None,
                           continuum_agn=None, dust_scatter_excess=None, bal_flux=None,
                           feii_flux=None, balmer_flux=None, igm_flux=None):
    """Group already-generated additive channels into the PI-specified
    3-bucket decomposition (continuum / emission / absorption), plus the
    total. See module docstring for exactly what belongs in each bucket
    and which pieces (dust_scatter_excess, bal_flux) are not yet
    implemented elsewhere in this fork.

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
        associated_absorption_flux (ndarray, optional): additive stochastic
            multi-system QSO associated-absorption flux deficit from
            AssociatedAbsorberSystems.spectrum() (associated_absorption.py,
            <=0 convention). Default: zero.
        dust_flux (ndarray, optional): additive dust attenuation flux
            deficit from DustAttenuation (<=0 convention). If attenuating a
            QSO/galaxy blend, sum each component's own DustAttenuation
            call's output before passing it in here (see dust.py's
            "Universality across galaxies and QSOs"). Default: zero.
        continuum_agn (ndarray, optional): AGN accretion-disk power-law
            continuum, from AGNPowerLawContinuum.spectrum() (agn_continuum.py).
            Per PI direction, the target continuum bucket is exactly
            continuum_stellar + continuum_agn. Default: zero.
        dust_scatter_excess (ndarray, optional): forward-compatible slot
            for a future, non-dust-coupled scattering mechanism (the
            dust-coupled case is already handled via DustAttenuation's own
            include_scattered_light flag -- see module docstring's
            "RESOLVED" note). Default: zero.
        bal_flux (ndarray, optional): additive BAL trough flux deficit from
            BALTrough.spectrum() (bal_trough.py, added 2026-08-27,
            <=0 convention). Lands in the "absorption" bucket (see module
            docstring). bal.py's original whole-template multiplicative
            mechanism should NOT be passed here. Default: zero.
        feii_flux (ndarray, optional): additive Fe II UV+optical pseudo-
            continuum flux from FeIIPseudoContinuum.spectrum()
            (feii_continuum.py). Lands in the "emission" bucket (see
            module docstring). Default: zero.
        balmer_flux (ndarray, optional): additive Balmer-continuum-plus-
            cascade flux from BalmerContinuum.spectrum() (balmer_continuum.py).
            Lands in the "emission" bucket (see module docstring). Default:
            zero.
        igm_flux (ndarray, optional): additive Lyman-alpha-forest + DLA
            intervening-absorption flux deficit from
            IGMAbsorption.spectrum() (igm_absorption.py, added
            2026-08-27, <=0 convention). Lands in the "absorption" bucket
            (see module docstring). Default: zero.

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
    associated_absorption_arr, has_associated = _resolve('associated_absorption_flux', associated_absorption_flux)
    dust_flux_arr, has_dust = _resolve('dust_flux', dust_flux)
    bal_flux_arr, has_bal = _resolve('bal_flux', bal_flux)
    feii_flux_arr, has_feii = _resolve('feii_flux', feii_flux)
    balmer_flux_arr, has_balmer = _resolve('balmer_flux', balmer_flux)
    igm_flux_arr, has_igm = _resolve('igm_flux', igm_flux)

    continuum = continuum_stellar_arr + continuum_agn_arr
    emission = narrow_emission_arr + broad_emission_arr + dust_scatter_arr + feii_flux_arr + balmer_flux_arr
    absorption = ism_absorption_arr + associated_absorption_arr + dust_flux_arr + bal_flux_arr + igm_flux_arr
    total = continuum + emission + absorption

    components = {
        'continuum_stellar': True,
        'continuum_agn': has_continuum_agn,
        'narrow_emission': has_narrow,
        'broad_emission': has_broad,
        'dust_scatter_excess': has_dust_scatter,
        'feii_flux': has_feii,
        'balmer_flux': has_balmer,
        'ism_absorption': has_ism,
        'associated_absorption_flux': has_associated,
        'dust_flux': has_dust,
        'bal_flux': has_bal,
        'igm_flux': has_igm,
    }

    return {
        'continuum': continuum,
        'emission': emission,
        'absorption': absorption,
        'total': total,
        'components': components,
    }
