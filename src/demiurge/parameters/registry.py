"""
Canonical registry of every Tier 2/3 NPE-parameter in demiurge.

Per the refactor charter (HANDOFF3 Sec. 5-6): a Tier 1 quantity is exact
physics (a closed-form/first-principles relation, no free parameter beyond
physical constants and inputs) and does not belong here. A Tier 2 quantity
has a real empirical measurement or relation in the literature that sets its
*default prior*, but the actual per-mock value is a genuine NPE-parameter --
drawn from that prior today, and (once a trained NPE exists) drawn from/
conditioned by the NPE instead. A Tier 3 quantity has no exact or empirical
source at all -- still an NPE-parameter, still explicitly labeled, but its
default distribution is a stated judgment call pending real NPE calibration.

Every entry below is `physical=True` (has direct astrophysical meaning a user
might reasonably want to specify or condition generation on -- HANDOFF3
Sec. 5.2). Nothing extracted for this first pass was a *non-physical*
(instrumental/generative-model-only) parameter; that category's tentative
example is camera/instrument calibration coefficients, which live in a
not-yet-ported module (`camera_calibration.py` on `main`) -- expect the first
non-physical entries when that module is ported.

This registry is populated from the pre-refactor reference implementation on
`main` (`py/desisim/{balmer_continuum,agn_continuum,dust,feii_continuum,
igm_absorption,bal_trough,associated_absorption,absorption,fsps_continuum,
templates,mock_spectrum}.py`) -- every citation and tier classification below
was read from that source directly, not reconstructed from memory. Where the
old code's own comments explicitly flagged a value "MAGIC" (its own
established convention for "no citation, order-of-magnitude judgment call"),
that self-labeling is trusted here and the parameter is Tier 3 -- this pass
faithfully catalogs the existing classification, it does not re-derive or
upgrade citations (that is separate future work, e.g. HANDOFF3 Sec. 6.4's
note on the Lyalpha linewidth needing real literature research before
defaulting to a free Tier 3 parameter).

Deliberately NOT ported into this registry (each is a real exclusion
decision, not an oversight):

- `templates.py`'s `EMSpectrum.forbidmog` (a Gaussian-mixture-model, fit to
  data and loaded from `forbidden_mogs.fits`, jointly drawing four forbidden-
  line ratios `[oiiihbeta, oiihbeta, niihbeta, siihbeta]`). This registry's
  schema is one independent `Distribution` per named parameter; a joint,
  correlated, data-fit prior over several parameters at once is a different
  and currently unbuilt construct (a `JointPrior`/correlated-parameter-group
  concept). Flagged as a known gap, not silently dropped -- revisit once/if
  such a construct is designed.
- `bal_trough.py`'s `WIDTH_SCALE`/`DEPTH_SCALE`: real, cited, backtest-
  calibrated coefficients (Sec. "bal_trough" below cites the same DR14Q
  backtest as the parameters that *are* registered from that module) -- but
  they are fixed, population-level coefficients of a deterministic
  strength -> width/depth transform, not values drawn per mock. Tier 2/3 as
  defined by the charter is specifically about per-mock-drawn NPE-parameters;
  a fixed calibrated transform coefficient doesn't fit that definition (it
  isn't Tier 1 either -- it's empirically fit, not derived). Left as a cited
  constant at its eventual definition site rather than forced into this
  schema as a degenerate one-point "distribution".
- `templates.py`'s legacy `QSO.balprob=0.12` (an uncited duplicate of
  `bal_trough.BALTrough.BALPROB=0.16`, which *is* registered below, cited).
  `QSO`/`GALAXY` are the legacy classes `mock_spectrum.py`'s orchestrator was
  explicitly built to bypass (HANDOFF1 Sec. 3.3) -- registering the bypassed
  class's own uncited duplicate would just create a confusing second entry
  for the same physical quantity.
- Survey/population sample-design ranges tied only to the bypassed legacy
  classes (`GALAXY`/`QSO`/`STAR` family `zrange`/`magrange`/`vdisprange`/
  `vrad_meansig`, `GALAXY.lineratios`'s `oiidoublet_meansig`) and
  `EMSpectrum.spectrum()`'s plain fixed-default kwargs
  (`oiidoublet=0.73`, `siidoublet=1.3`, `linesigma=75.0`) that are not drawn
  by default -- not part of the active `mock_spectrum.py` orchestrator path
  this registry is scoped to.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .distributions import (
    Bernoulli,
    DiscreteUniform,
    Distribution,
    Gamma,
    LogNormal,
    LogUniform,
    MaxwellBoltzmann,
    Normal,
    Poisson,
    Uniform,
)


@dataclass(frozen=True)
class NPEParameter:
    """One catalogued NPE-parameter: its default prior plus the metadata that
    makes the registry do real work (HANDOFF3 Sec. 6.2) -- `PriorSampler`
    reads `distribution`; the conditioning-input validator reads `physical`;
    the `.tex`/`.bib` completeness check reads `citation`.
    """

    name: str
    owner: str
    tier: int
    physical: bool
    distribution: Distribution
    description: str
    rationale: str
    citation: Optional[str] = None
    units: Optional[str] = None

    def __post_init__(self) -> None:
        if self.tier not in (2, 3):
            raise ValueError(
                f"{self.name}: tier must be 2 or 3 (Tier 1 constants are exact physics and do not "
                f"belong in this registry), got {self.tier}"
            )
        if self.tier == 2 and not self.citation:
            raise ValueError(f"{self.name}: Tier 2 parameters require a citation")
        if not self.rationale:
            raise ValueError(f"{self.name}: every parameter requires a rationale (HANDOFF3 Sec. 5.2 process)")


_PARAMETERS: list[NPEParameter] = []


def _add(*params: NPEParameter) -> None:
    _PARAMETERS.extend(params)


# =============================================================================
# balmer_continuum -- BalmerContinuum (case-B recombination lines + free-bound
# Balmer-edge continuum)
# =============================================================================
_CITE_DIETRICH = (
    "Dietrich et al. (2002, ApJ 581, 912; 2003, ApJ 596, 817) case-B recombination grid -- "
    "their tabulated T_e / log(n_e) / tau_BE ranges."
)
_CITE_STOREY_HUMMER = "Storey & Hummer (1995, MNRAS 272, 41) case-B recombination tables -- their full tabulated log(n_e) support."

_add(
    NPEParameter(
        name="balmer_continuum.electron_temperature_k",
        owner="balmer_continuum",
        tier=2,
        physical=True,
        distribution=Uniform(10000.0, 20000.0),
        units="K",
        citation=_CITE_DIETRICH,
        description="BLR electron temperature T_e, feeds the case-B line-ratio grid and the free-bound Balmer-edge Planck shape.",
        rationale="Range is the Dietrich et al. grid's own tabulated support, not an independent fit to a dataset.",
    ),
    NPEParameter(
        name="balmer_continuum.log_electron_density",
        owner="balmer_continuum",
        tier=2,
        physical=True,
        distribution=Uniform(5.0, 14.0),
        units="dex (log10 cm^-3)",
        citation=_CITE_STOREY_HUMMER,
        description="log10(n_e) feeding the case-B line-ratio grid.",
        rationale="Deliberately spans the grid's full tabulated range rather than a narrower BLR-typical subset (PI direction, not itself an empirical BLR-density fit).",
    ),
    NPEParameter(
        name="balmer_continuum.free_bound_optical_depth",
        owner="balmer_continuum",
        tier=2,
        physical=True,
        distribution=LogUniform(0.1, 2.0),
        units="unitless (optical depth at the Balmer edge)",
        citation=_CITE_DIETRICH,
        description="tau_BE, optical depth at the Balmer series limit (3646 A) setting the free-bound edge's sharpness.",
        rationale="Range is the Dietrich et al. grid's own tabulated support (fiducial tau_BE=1), not an independent fit.",
    ),
    NPEParameter(
        name="balmer_continuum.line_sigma_kms",
        owner="balmer_continuum",
        tier=3,
        physical=True,
        distribution=LogUniform(425.0, 4250.0),
        units="km/s",
        description="Shared Gaussian velocity width for every case-B recombination line drawn by this module.",
        rationale="No independent citation -- cross-referenced to EMSpectrum's own BROADSIGMA_RANGE_KMS (also Tier 3, see below), itself a rough FWHM~1000-10000 km/s AGN-broad-line conversion, not a fit.",
    ),
    NPEParameter(
        name="balmer_continuum.line_velshift_kms",
        owner="balmer_continuum",
        tier=3,
        physical=True,
        distribution=Uniform(-1000.0, 200.0),
        units="km/s",
        description="Shared bulk velocity shift for every case-B recombination line drawn by this module.",
        rationale="No independent citation beyond the same BROADSHIFT_KMS_RANGE cross-reference as EMSpectrum (also Tier 3).",
    ),
)

# =============================================================================
# agn_continuum -- AGNPowerLawContinuum (broken power-law AGN accretion-disk
# continuum, 5 segments)
# =============================================================================
_CITE_VANDENBERK_CONTINUUM = (
    "simqso's BossDr9_fiducial_continuum model (Vanden Berk et al. 2001, AJ 122, 549 SDSS QSO "
    "composite-spectrum segmentation/slopes)."
)
_AGN_SLOPE_SEGMENTS = [
    ("segment1_lt1100a", -1.50, "<1100 A"),
    ("segment2_1100_5700a", -0.50, "1100-5700 A"),
    ("segment3_5700_9730a", -0.37, "5700-9730 A"),
    ("segment4_9730_22300a", -1.70, "9730-22300 A"),
    ("segment5_gt22300a", -1.03, "beyond 22300 A"),
]
_add(
    *[
        NPEParameter(
            name=f"agn_continuum.slope_{seg_name}",
            owner="agn_continuum",
            tier=2,
            physical=True,
            distribution=Normal(mean, 0.3),
            units="unitless (spectral index alpha_nu, f_nu ~ nu**alpha_nu)",
            citation=_CITE_VANDENBERK_CONTINUUM,
            description=f"AGN power-law continuum spectral index for the {wave_range} segment.",
            rationale="Mean and functional segmentation directly from the cited model; sigma=0.3 is that same model's own adopted per-segment scatter.",
        )
        for seg_name, mean, wave_range in _AGN_SLOPE_SEGMENTS
    ]
)

# =============================================================================
# dust -- DustAttenuation (Noll et al. 2009-family parametric attenuation law)
# =============================================================================
_add(
    NPEParameter(
        name="dust.theta0_amplitude",
        owner="dust",
        tier=3,
        physical=True,
        distribution=Uniform(0.01, 3.0),
        units="mag-scale amplitude",
        description="Overall attenuation-curve amplitude (A_V-like).",
        rationale="Uncited (\"MAGIC\" in source): negligible to heavily-obscured, order-of-magnitude bracket only.",
    ),
    NPEParameter(
        name="dust.theta1_slope",
        owner="dust",
        tier=3,
        physical=True,
        distribution=Uniform(0.0, 2.0),
        units="unitless",
        description="Attenuation-curve slope: 0 = flat/grey (AGN-torus-like), ~1-2 = steep SMC/starburst-like.",
        rationale="Uncited (\"MAGIC\" in source), order-of-magnitude bracket only.",
    ),
    NPEParameter(
        name="dust.theta2_bump_amplitude",
        owner="dust",
        tier=3,
        physical=True,
        distribution=Uniform(0.0, 2.0),
        units="unitless",
        description="2175 A UV bump amplitude: 0 = bump-free, >0 = MW/LMC-like.",
        rationale="Uncited (\"MAGIC\" in source), order-of-magnitude bracket only.",
    ),
    NPEParameter(
        name="dust.theta3_grey_floor",
        owner="dust",
        tier=3,
        physical=True,
        distribution=Uniform(0.0, 1.0),
        units="unitless",
        description="Grey (wavelength-independent) attenuation floor.",
        rationale="Uncited (\"MAGIC\" in source), order-of-magnitude bracket only.",
    ),
    NPEParameter(
        name="dust.theta4_fuv_curvature",
        owner="dust",
        tier=3,
        physical=True,
        distribution=Uniform(0.0, 1.5),
        units="unitless (FM90 curvature amplitude)",
        description="Far-UV curvature amplitude, active only when the curve includes FM90-style FUV curvature.",
        rationale="\"MAGIC but literature-informed\" per source: range shaped by typical c4 values in Fitzpatrick & Massa-style MW/LMC sightline fits, but not itself a fit -- self-labeled MAGIC, kept Tier 3.",
    ),
    NPEParameter(
        name="dust.theta5_red_nir_slope",
        owner="dust",
        tier=3,
        physical=True,
        distribution=Uniform(0.0, 3.0),
        units="unitless",
        description="Red/NIR-side attenuation slope, active only when the curve includes an NIR break.",
        rationale="Uncited (\"MAGIC\" in source): wider than theta1's range since the true red/NIR behavior of real curves is less well constrained.",
    ),
    NPEParameter(
        name="dust.scattered_light_fraction",
        owner="dust",
        tier=2,
        physical=True,
        distribution=Uniform(0.0, 0.05),
        units="unitless fraction",
        citation=(
            "NGC 1068's scattered/polarized Type 1 flux measured at ~1% of total; highly inclined "
            "Type 2 quasars show ~3% (polar-scattering literature, e.g. Antonucci, Hurt & Miller-era "
            "spectropolarimetry)."
        ),
        description="Fraction of light reaching the observer via scattering rather than direct transmission (f_scat), active only when scattered-light modeling is enabled.",
        rationale="Explicitly NOT flagged MAGIC in source -- range anchored directly on real scattered-flux-fraction measurements.",
    ),
    NPEParameter(
        name="dust.scattering_wavelength_exponent",
        owner="dust",
        tier=3,
        physical=True,
        distribution=Uniform(0.0, 2.0),
        units="unitless exponent",
        description="Wavelength-dependence exponent of the scattered-light term (p_scat=0 matches NGC 1068's wavelength-independent electron-scattering polarization; p_scat>0 adds blueward rise from separate dust/Rayleigh-type scattering).",
        rationale="Self-labeled \"MAGIC (partly literature-anchored)\" in source: the p_scat=0 endpoint matches real NGC 1068 behavior, but the upper bound (2.0) mirrors theta1's slope range rather than a dedicated measurement.",
    ),
    NPEParameter(
        name="dust.bump_center_aa",
        owner="dust",
        tier=2,
        physical=True,
        distribution=Uniform(2100.0, 2250.0),
        units="Angstrom",
        citation="Fitzpatrick & Massa (2007) UV extinction atlas -- sightline-to-sightline scatter in the 2175 A bump center.",
        description="Central wavelength of the 2175 A UV dust bump, active only when bump-shape variation is enabled.",
        rationale="Range informed by the cited atlas's real sightline scatter, not a direct numerical fit to that specific dataset.",
    ),
    NPEParameter(
        name="dust.bump_width_aa",
        owner="dust",
        tier=2,
        physical=True,
        distribution=Uniform(250.0, 500.0),
        units="Angstrom",
        citation="Fitzpatrick & Massa (2007) UV extinction atlas -- sightline-to-sightline scatter in the 2175 A bump width.",
        description="FWHM of the 2175 A UV dust bump, active only when bump-shape variation is enabled.",
        rationale="Range informed by the cited atlas's real sightline scatter, not a direct numerical fit to that specific dataset.",
    ),
    NPEParameter(
        name="dust.nir_break_wavelength_aa",
        owner="dust",
        tier=2,
        physical=True,
        distribution=Uniform(7000.0, 12000.0),
        units="Angstrom",
        citation="Cardelli, Clayton & Mathis (1989) optical/near-IR extinction-law transition point (x=1.1 um^-1, ~9091 A).",
        description="Wavelength of the optical-to-NIR attenuation-law break, active only when an NIR break is enabled.",
        rationale="Range anchored on CCM89's real transition wavelength (9091 A) then widened for reach -- the anchor is cited, the widening is a judgment call.",
    ),
)

# =============================================================================
# feii_continuum -- FeIIPseudoContinuum (CLOUDY-grid-based Fe II pseudo-
# continuum, optical + UV)
# =============================================================================
_CITE_FEII_CLOUDY_GRID = "CLOUDY Fe II template grid (Ashwani et al. 2018-lineage grid, arXiv:2401.18052) -- its documented tabulated parameter ranges."
_add(
    NPEParameter(
        name="feii_continuum.log_ionizing_flux",
        owner="feii_continuum",
        tier=2,
        physical=True,
        distribution=Uniform(17.0, 22.0),
        units="dex (log10 photons cm^-2 s^-1)",
        citation=_CITE_FEII_CLOUDY_GRID,
        description="log10(ionizing photon flux, Phi(H)) selecting a point on the CLOUDY Fe II template grid.",
        rationale="Range is the CLOUDY grid's own tabulated support.",
    ),
    NPEParameter(
        name="feii_continuum.log_hydrogen_density",
        owner="feii_continuum",
        tier=2,
        physical=True,
        distribution=Uniform(9.0, 14.0),
        units="dex (log10 cm^-3)",
        citation=_CITE_FEII_CLOUDY_GRID,
        description="log10(n_H) selecting a point on the CLOUDY Fe II template grid.",
        rationale="Range is the CLOUDY grid's own tabulated support.",
    ),
    NPEParameter(
        name="feii_continuum.microturbulence_kms",
        owner="feii_continuum",
        tier=2,
        physical=True,
        distribution=DiscreteUniform((0.0, 20.0, 50.0, 100.0)),
        units="km/s",
        citation=_CITE_FEII_CLOUDY_GRID,
        description="Microturbulent velocity selecting a point on the CLOUDY Fe II template grid (grid is only tabulated at these four discrete values).",
        rationale="Values are exactly the CLOUDY grid's own tabulated microturbulence points.",
    ),
    NPEParameter(
        name="feii_continuum.sed_shape",
        owner="feii_continuum",
        tier=2,
        physical=True,
        distribution=DiscreteUniform(("AGN_SED", "Intermediate_SED")),
        units="categorical",
        citation="Mathews & Ferland (1987)-like AGN SED vs. Jin et al. (2012) intermediate SED, the two ionizing-SED shapes tabulated in the CLOUDY grid.",
        description="Which ionizing continuum SED shape was used to generate the Fe II template grid point.",
        rationale="The two options are exactly the two SED families the cited CLOUDY grid was run with.",
    ),
    NPEParameter(
        name="feii_continuum.line_sigma_kms",
        owner="feii_continuum",
        tier=3,
        physical=True,
        distribution=LogUniform(425.0, 4250.0),
        units="km/s",
        description="Velocity broadening applied to the Fe II pseudo-continuum templates.",
        rationale="No independent citation -- anchored on the same BLR kinematic scale as EMSpectrum's BROADSIGMA_RANGE_KMS (Tier 3, Fe II is understood to arise from BLR-like gas but this specific range is not a fit).",
    ),
    NPEParameter(
        name="feii_continuum.velshift_kms",
        owner="feii_continuum",
        tier=2,
        physical=True,
        distribution=Uniform(-500.0, 500.0),
        units="km/s",
        citation="Kovacevic, Popovic & Dimitrijevic (2010) report a nonzero Fe II velocity shift relative to H-beta.",
        description="Bulk velocity shift of the Fe II pseudo-continuum relative to the broad Balmer lines.",
        rationale="Range is a broad superset of the cited measured shift scale, not a direct fit to it.",
    ),
    NPEParameter(
        name="feii_continuum.r_feii_optical_broad_hbeta",
        owner="feii_continuum",
        tier=2,
        physical=True,
        distribution=LogNormal(np.log10(0.6), 0.28),
        units="unitless ratio (R_FeII = optical Fe II flux / broad H-beta flux)",
        citation=(
            "Shen & Ho (2014, Nature 513, 210, arXiv:1409.2887) report a population peak/mode of R_FeII~0.6. "
            "Panda et al. (2020, arXiv:2001.08765, using the Shen et al. 2011 DR7 quasar catalog) report values "
            "up to R_FeII=6.56."
        ),
        description="Optical Fe II pseudo-continuum flux relative to broad H-beta flux.",
        rationale="Mean is a directly measured population statistic; sigma=0.28 is tuned (not independently fit) to jointly reproduce both cited statistics (~22% of the population above R_FeII=1, max ~6.5).",
    ),
    NPEParameter(
        name="feii_continuum.log_optical_to_uv_ratio",
        owner="feii_continuum",
        tier=2,
        physical=True,
        distribution=Normal(-0.8, 0.2),
        units="dex",
        citation=(
            "Sameshima, Kawara, Matsuoka, Oyabu, Asami & Ienaka (2010, MNRAS 410, 1018, arXiv:1008.2405, "
            "Figure 7 / Sec. 4.1) -- 884 SDSS quasars, FeII(4570)/FeII(UV) = 10**(-0.8 +/- 0.2 dex)."
        ),
        description="log10(optical Fe II / UV Fe II) flux ratio.",
        rationale="Both mean and sigma are directly measured population statistics from the cited sample -- not MAGIC.",
    ),
)

# =============================================================================
# igm_absorption -- IGMAbsorption (Lya forest + DLA incidence)
# =============================================================================
_add(
    NPEParameter(
        name="igm_absorption.dla_boost",
        owner="igm_absorption",
        tier=2,
        physical=True,
        distribution=LogUniform(1.0, 2.2),
        units="unitless multiplier on DLA incidence rate",
        citation=(
            "Prochaska et al. (2008) DLA incidence rate is the fully-cited floor (boost=1.0, pure DLA "
            "incidence, no SLLS contribution); legacy dla.py's boost=1.6 default has no independent "
            "citation for that specific multiplier."
        ),
        description="Multiplier on the Prochaska et al. (2008) DLA incidence rate, allowing extra sub-DLA/SLLS-driven absorption.",
        rationale="Lower bound (1.0) is the cited physical floor; upper bound (2.2) is a generous-but-not-absurd bound above the legacy uncited 1.6 point value, not itself an independent fit.",
    ),
)

# =============================================================================
# bal_trough -- BALTrough (broad absorption line troughs)
# =============================================================================
_CITE_DR14Q_BACKTEST = (
    "Calibrated (2026-08-27) against the real public SDSS DR14Q BI(CIV) distribution "
    "(Paris et al. 2018, VizieR VII/286) -- backtest-fitted, not a first-principles physical constant, "
    "but not arbitrary."
)
_add(
    NPEParameter(
        name="bal_trough.bal_probability",
        owner="bal_trough",
        tier=2,
        physical=True,
        distribution=Bernoulli(0.16),
        units="unitless probability",
        citation=(
            "quickquasars' own documented --balprob default, itself grounded in the Niu (2020) DR14-based "
            "template lineage; Hewett & Foltz (2003) independently report an intrinsic BAL fraction of "
            "order 15-22% at z=1.5-3.0."
        ),
        description="Probability that a given QSO mock is a BAL quasar.",
        rationale="Single cited population value used directly as the Bernoulli probability.",
    ),
    NPEParameter(
        name="bal_trough.strength",
        owner="bal_trough",
        tier=2,
        physical=True,
        distribution=Gamma(1.0, 0.041),
        units="unitless (latent BAL-strength scale, deterministically maps to trough width/depth)",
        citation=_CITE_DR14Q_BACKTEST,
        description="Latent per-BAL-QSO strength parameter driving the trough's width and depth via a deterministic transform.",
        rationale="Gamma shape/scale calibrated against the real DR14Q BI(CIV) distribution.",
    ),
    NPEParameter(
        name="bal_trough.width_jitter",
        owner="bal_trough",
        tier=2,
        physical=True,
        distribution=Uniform(0.7, 1.3),
        units="unitless multiplicative jitter",
        citation=_CITE_DR14Q_BACKTEST,
        description="Multiplicative scatter applied on top of the strength-to-width transform.",
        rationale="Calibrated against the real DR14Q BI(CIV) distribution alongside `strength`.",
    ),
    NPEParameter(
        name="bal_trough.depth_jitter",
        owner="bal_trough",
        tier=2,
        physical=True,
        distribution=Uniform(0.6, 1.15),
        units="unitless multiplicative jitter",
        citation=_CITE_DR14Q_BACKTEST,
        description="Multiplicative scatter applied on top of the strength-to-depth transform.",
        rationale="Calibrated against the real DR14Q BI(CIV) distribution alongside `strength`.",
    ),
    NPEParameter(
        name="bal_trough.max_outflow_velocity_kms",
        owner="bal_trough",
        tier=2,
        physical=True,
        distribution=Uniform(-3500.0, -3000.0),
        units="km/s",
        citation=_CITE_DR14Q_BACKTEST,
        description="Blueshifted velocity of the trough's high-velocity (blue) edge.",
        rationale="Calibrated against the real DR14Q BI(CIV) distribution.",
    ),
    NPEParameter(
        name="bal_trough.edge_smoothing_kms",
        owner="bal_trough",
        tier=3,
        physical=True,
        distribution=Uniform(300.0, 1200.0),
        units="km/s",
        description="Velocity-space smoothing scale applied to the trough's edges.",
        rationale="Self-labeled \"MAGIC\" in source: a resolution-motivated guess, not literature-measured.",
    ),
)

# =============================================================================
# associated_absorption -- AssociatedAbsorberSystems (intrinsic/associated QSO
# absorption systems)
# =============================================================================
_add(
    NPEParameter(
        name="associated_absorption.n_systems",
        owner="associated_absorption",
        tier=3,
        physical=True,
        distribution=Poisson(1.5),
        units="count",
        description="Number of independent associated-absorber systems along the line of sight.",
        rationale="Self-labeled \"MAGIC\" in source: Poisson-family count statistics for QSO absorber populations are well established in the literature, but this specific mean is a placeholder, not a fit to any dataset.",
    ),
    NPEParameter(
        name="associated_absorption.system_velocity_offset_kms",
        owner="associated_absorption",
        tier=3,
        physical=True,
        distribution=Uniform(0.0, 3000.0),
        units="km/s",
        description="Velocity offset of each associated-absorber system from the QSO systemic redshift.",
        rationale="Self-labeled \"MAGIC\" in source: matches the order-of-magnitude AGN-outflow velocity range used elsewhere in this project, not a fit to any dataset.",
    ),
    NPEParameter(
        name="associated_absorption.component_linewidth_kms",
        owner="associated_absorption",
        tier=3,
        physical=True,
        distribution=MaxwellBoltzmann(50.0),
        units="km/s",
        description="Individual absorption-component linewidth (Doppler b-parameter-like), Maxwell-Boltzmann distributed.",
        rationale="Self-labeled \"MAGIC\" in source: consistent with typical individual-component linewidths in the associated/mini-BAL literature, not a fit to a specific dataset.",
    ),
)

_ASSOCIATED_TAU0_LINES = {
    "SiII_1260": 0.25,
    "OI_1302": 0.15,
    "SiII_1304": 0.15,
    "CII_1334": 0.2,
    "SiIV_1393": 0.4,
    "SiIV_1402": 0.4 * 0.5,
    "SV_1501": 0.1,
    "SiII_1526": 0.2,
    "CIV_1548": 0.6,
    "CIV_1550": 0.6 * 0.5,
    "FeII_1608": 0.12,
    "AlII_1670": 0.2,
    "SiII_1808": 0.15,
    "AlIII_1854": 0.3,
    "AlIII_1862": 0.3 * 0.5,
    "MgI_2026": 0.1,
    "FeII_2344": 0.2,
    "FeII_2374": 0.08,
    "FeII_2382": 0.35,
    "FeII_2389": 0.08,
    "MgII_2796": 0.5,
    "MgII_2803": 0.5 * 0.7,
    "Lyb_1025": 0.3,
    "Lya_1215": 0.7,
    "NV_1238": 0.4,
    "NV_1242": 0.4 * 0.5,
    "CIII_1908": 0.2,
    "OVI_1031": 0.3,
    "OVI_1037": 0.3 * 0.5,
}
_CITE_ASSOCIATED_TAU0 = (
    "No dedicated fit exists; central values are informed by rough, literature-typical relative line "
    "strengths/oscillator strengths (order-of-magnitude placements, not a precise atomic-data fit). "
    "Doublet ratios (bluer:redder ~2:1, or per-doublet as tabulated) follow the standard alkali-like "
    "resonance-doublet oscillator-strength ratio already used for Na I D / Mg II in the ISM absorption module."
)
_add(
    *[
        NPEParameter(
            name=f"associated_absorption.tau0.{line}",
            owner="associated_absorption",
            tier=3,
            physical=True,
            distribution=LogNormal(np.log10(tau0_central), 0.6),
            units="unitless (resonance-line optical-depth proxy)",
            description=f"Per-system optical-depth-proxy for the {line} transition.",
            rationale=_CITE_ASSOCIATED_TAU0,
        )
        for line, tau0_central in _ASSOCIATED_TAU0_LINES.items()
    ]
)

# =============================================================================
# ism_absorption -- AbsorptionSpectrum (galaxy/QSO-host interstellar-medium
# absorption; module file is `absorption.py`, owner named `ism_absorption`
# to avoid confusion with the 3-bucket decomposition's own "absorption" name)
# =============================================================================
_ISM_TAU0_LINES = {
    "CaII_K": 0.3,
    "CaII_H": 0.3 * 0.5,
    "NaID_5891": 0.3,
    "NaID_5897": 0.3 * 0.7,
    "MgII_2796_abs": 0.5,
    "MgII_2803_abs": 0.5 * 0.7,
}
_CITE_ISM_TAU0 = (
    "No derivation from data exists yet. Central values are order-of-magnitude choices consistent with "
    "these lines being routinely detected-but-not-always-saturated features in real galaxy/QSO spectra "
    "(tau0 ~ 0.1-1 typical) -- not a fit to any specific dataset."
)
_add(
    *[
        NPEParameter(
            name=f"ism_absorption.tau0.{line}",
            owner="ism_absorption",
            tier=3,
            physical=True,
            distribution=LogNormal(np.log10(tau0_central), 0.5),
            units="unitless (resonance-line optical-depth proxy)",
            description=f"Optical-depth-proxy for the {line} transition.",
            rationale=_CITE_ISM_TAU0,
        )
        for line, tau0_central in _ISM_TAU0_LINES.items()
    ]
)
_add(
    NPEParameter(
        name="ism_absorption.line_sigma_kms",
        owner="ism_absorption",
        tier=3,
        physical=True,
        distribution=LogUniform(10.0, 300.0),
        units="km/s",
        description="Velocity width shared by every ISM absorption line drawn by this module.",
        rationale="Self-labeled \"MAGIC\" in source: spans narrow individual ISM clouds (~10s km/s) through broader CGM/rotational-blending scales (~few x 100 km/s), not a fit.",
    ),
    NPEParameter(
        name="ism_absorption.outflow_velshift_kms",
        owner="ism_absorption",
        tier=2,
        physical=True,
        distribution=Uniform(-300.0, 0.0),
        units="km/s",
        citation=(
            "Weiner et al. (2009) down-the-barrel outflow study: representative mean ~-166 +/- 130 km/s "
            "at z~2.2. Rubin et al. (2014) independently corroborates blueshifted ISM absorption from "
            "galactic outflows."
        ),
        description="Bulk blueshift of ISM absorption lines from galaxy-scale outflows, active only when outflow modeling is enabled.",
        rationale="Range anchored on the cited measured outflow velocity scale, not a direct numerical fit.",
    ),
)

# =============================================================================
# fsps_continuum -- stellar population synthesis (age/metallicity/SFH priors
# feeding the FSPS-based stellar continuum)
# =============================================================================
_CITE_SFH_SCATTER = (
    "Star-forming-main-sequence SFR scatter repeatedly measured at sigma~0.2-0.4 dex: Brinchmann et al. "
    "(2004, MNRAS 351, 1151); Daddi et al. (2007, ApJ 670, 156); Noeske et al. (2007, ApJ 660, L43); "
    "Whitaker et al. (2012, ApJ 754, L29); Speagle et al. (2014, ApJS 214, 15)."
)
_CITE_OU_TAU = (
    "Caplar & Tacchella (2019, MNRAS 487, 3845) fit an Ornstein-Uhlenbeck stochastic-SFH model directly "
    "to observed z~0, M*~1e10 Msun main-sequence scatter and find tau_break = 170 (+169/-85) Myr; their "
    "companion simulation comparison brackets tau_break between ~100 Myr (FIRE) and ~1000 Myr "
    "(IllustrisTNG); corroborated by UV-vs-nebular-line timescale studies (Weisz et al. 2012, ApJ 744, 44; "
    "Guo et al. 2016, ApJ 833, 37; Emami et al. 2019, ApJ 881, 71; Broussard et al. 2019, ApJ 873, 74)."
)
_add(
    NPEParameter(
        name="fsps_continuum.age_gyr",
        owner="fsps_continuum",
        tier=3,
        physical=True,
        distribution=Uniform(0.05, 13.0),
        units="Gyr",
        description="Stellar population age.",
        rationale="Self-labeled \"MAGIC\" in source: broadly spans what real star-forming/quiescent galaxy populations occupy (cf. Conroy 2013 ARAA review) but is not a calibrated fit.",
    ),
    NPEParameter(
        name="fsps_continuum.log_metallicity_solar",
        owner="fsps_continuum",
        tier=3,
        physical=True,
        distribution=Uniform(-1.0, 0.2),
        units="dex (log10 Z/Zsun)",
        description="Stellar metallicity relative to solar.",
        rationale="Self-labeled \"MAGIC\" in source: no calibration data justifies this specific range.",
    ),
    NPEParameter(
        name="fsps_continuum.dust2_optical_depth",
        owner="fsps_continuum",
        tier=3,
        physical=True,
        distribution=Uniform(0.0, 1.0),
        units="unitless (Calzetti-like V-band optical depth)",
        description="FSPS-internal dust optical depth. NOTE: the mock_spectrum.py orchestrator forces this to (0.0, 0.0) when building the stellar continuum, since dust attenuation is handled separately by the dedicated `dust` channel (3-bucket-scheme scope separation) -- registered here for completeness/standalone use of this module, not because the orchestrator draws it.",
        rationale="Self-labeled \"MAGIC\" in source.",
    ),
    NPEParameter(
        name="fsps_continuum.sfh_tau_gyr",
        owner="fsps_continuum",
        tier=3,
        physical=True,
        distribution=Uniform(0.1, 10.0),
        units="Gyr",
        description="Delayed-tau star-formation-history e-folding timescale.",
        rationale="Self-labeled \"MAGIC\" in source.",
    ),
    NPEParameter(
        name="fsps_continuum.bursty_sfh_ou_sigma_dex",
        owner="fsps_continuum",
        tier=2,
        physical=True,
        distribution=Uniform(0.2, 0.4),
        units="dex",
        citation=_CITE_SFH_SCATTER,
        description="Steady-state standard deviation of the Ornstein-Uhlenbeck bursty-SFH model (active only when bursty SFH is enabled).",
        rationale="Range matches the directly cited, repeatedly-measured main-sequence SFR scatter.",
    ),
    NPEParameter(
        name="fsps_continuum.bursty_sfh_ou_tau_gyr",
        owner="fsps_continuum",
        tier=2,
        physical=True,
        distribution=Uniform(0.1, 1.0),
        units="Gyr",
        citation=_CITE_OU_TAU,
        description="Correlation timescale of the Ornstein-Uhlenbeck bursty-SFH model (active only when bursty SFH is enabled).",
        rationale="Range brackets the cited fitted/simulated tau_break estimates.",
    ),
)

# =============================================================================
# emspectrum -- EMSpectrum (narrow + broad emission-line spectrum, shared by
# galaxy and QSO orchestration paths)
# =============================================================================
_add(
    NPEParameter(
        name="emspectrum.auxline_oi_6300_hbeta_ratio",
        owner="emspectrum",
        tier=3,
        physical=True,
        distribution=LogNormal(np.log10(0.1), 0.15),
        units="unitless (log10 ratio)",
        description="[OI] 6300 / H-beta flux ratio.",
        rationale="Self-labeled \"MAGIC\" in source: mean set to the log10 of the original hardcoded ratio, sigma a placeholder starting guess.",
    ),
    NPEParameter(
        name="emspectrum.auxline_siii_9069_hbeta_ratio",
        owner="emspectrum",
        tier=3,
        physical=True,
        distribution=LogNormal(np.log10(0.75), 0.15),
        units="unitless (log10 ratio)",
        description="[SIII] 9069 / H-beta flux ratio.",
        rationale="Self-labeled \"MAGIC\" in source: mean set to the log10 of the original hardcoded ratio, sigma a placeholder starting guess.",
    ),
    NPEParameter(
        name="emspectrum.auxline_ariii_7135_hbeta_ratio",
        owner="emspectrum",
        tier=3,
        physical=True,
        distribution=LogNormal(np.log10(0.04), 0.20),
        units="unitless (log10 ratio)",
        description="[ArIII] 7135 / H-beta flux ratio.",
        rationale="Self-labeled \"MAGIC\" in source: sigma set wider than the other auxlines since [ArIII] is weaker/more environmentally sensitive.",
    ),
    NPEParameter(
        name="emspectrum.auxline_mgii_hbeta_ratio",
        owner="emspectrum",
        tier=3,
        physical=True,
        distribution=LogNormal(np.log10(0.3), 0.30),
        units="unitless (log10 ratio)",
        description="Narrow Mg II / H-beta flux ratio.",
        rationale="Self-labeled \"MAGIC\" in source: Mg II is a resonant line whose strength is heavily modulated by outflows/AGN, sigma set wider accordingly.",
    ),
    NPEParameter(
        name="emspectrum.mgii_doublet_ratio",
        owner="emspectrum",
        tier=2,
        physical=True,
        distribution=LogNormal(np.log10(1.7), 0.25),
        units="unitless (F(2796)/F(2803))",
        citation=(
            "Intrinsic optically-thin ratio 2.013 from oscillator strengths f_2796=0.6155, f_2803=0.3058 "
            "(Morton 2003, ApJS 149, 205, Table 2). Real-population range ~0.3-2.7, median ~1.7 (Henry et "
            "al. 2018, ApJ 855, 96; Scarlata et al. 2024, ApJ 971, 184, arXiv:2310.17908)."
        ),
        description="Mg II 2796/2803 intra-doublet flux ratio.",
        rationale="Mean is the cited measured population median; sigma is a judgment call not itself independently fit.",
    ),
    NPEParameter(
        name="emspectrum.broad_line_sigma_kms",
        owner="emspectrum",
        tier=3,
        physical=True,
        distribution=LogUniform(425.0, 4250.0),
        units="km/s",
        description="Shared Gaussian velocity width for broad emission lines.",
        rationale="Self-labeled \"MAGIC\" in source: a rough FWHM~1000-10000 km/s (typical AGN broad-line FWHM) converted to sigma via FWHM/2.355, not a fit.",
    ),
    NPEParameter(
        name="emspectrum.broad_line_velshift_kms",
        owner="emspectrum",
        tier=3,
        physical=True,
        distribution=Uniform(-1000.0, 200.0),
        units="km/s",
        description="Shared bulk blueshift for high-ionization broad emission lines.",
        rationale=(
            "Qualitatively grounded (high-ionization lines like CIV/HeII are frequently blueshifted by "
            "100s-1000s km/s: Coatman et al. 2016 vs. Eddington ratio; Sulentic et al. 2000 eigenvector-1; "
            "Zamanov et al. 2002; Komossa et al. 2008) but the source itself states the specific numeric "
            "bounds are \"not a fit to any dataset\" -- kept Tier 3 despite the qualitative citation."
        ),
    ),
    NPEParameter(
        name="emspectrum.gauss_hermite_h3",
        owner="emspectrum",
        tier=2,
        physical=True,
        distribution=Uniform(-0.3, 0.3),
        units="unitless (Gauss-Hermite skewness coefficient)",
        citation="Zamfir, Sulentic, Marziani & Dultzin (2010, MNRAS 403, 1759, arXiv:0912.4306), Table 1 -- Population A asymmetry-index distribution.",
        description="Gauss-Hermite h3 (asymmetry) coefficient for narrow and broad line profiles.",
        rationale="+/-0.3 extremes correspond to ~3.5 sigma from the cited Pop A mean asymmetry index -- a reasoned outer bound, not a direct per-object fit.",
    ),
    NPEParameter(
        name="emspectrum.gauss_hermite_h4",
        owner="emspectrum",
        tier=2,
        physical=True,
        distribution=Uniform(-0.12, 0.3),
        units="unitless (Gauss-Hermite kurtosis coefficient)",
        citation="Zamfir, Sulentic, Marziani & Dultzin (2010, MNRAS 403, 1759, arXiv:0912.4306), Table 1 -- Population A kurtosis-index distribution.",
        description="Gauss-Hermite h4 (kurtosis) coefficient for narrow and broad line profiles.",
        rationale="h4=-0.12 alone predicts the same ~3.5 sigma deviation from the cited Pop A kurtosis-index mean as h3's own extremes -- range chosen for internal consistency with h3, not independently fit.",
    ),
)

_EMSPECTRUM_NEW_LINES_BOTH_MAGIC = {
    # name: (narrow_mean_ratio, narrow_sigma, broad_mean_ratio, broad_sigma)
    "neiii_3869": (0.15, 0.30, 0.02, 0.50),
    "neiii_3968": (0.15 * 0.3, 0.30, 0.02 * 0.3, 0.50),
    "oiii_4363": (0.02, 0.40, 0.005, 0.50),
    "heii_4686": (0.02, 0.40, 0.05, 0.60),
    "nii_5755": (0.01, 0.40, 0.002, 0.50),
    "sii_4068": (0.015, 0.35, 0.003, 0.50),
    "sii_4076": (0.010, 0.35, 0.002, 0.50),
}
_CITE_NEW_LINE_MAGIC = "Self-labeled \"MAGIC\" in source: order-of-magnitude HII-region/AGN-diagnostic-typical placement, no independent citation."
_add(
    *[
        param
        for line, (nm, ns, bm, bs) in _EMSPECTRUM_NEW_LINES_BOTH_MAGIC.items()
        for param in (
            NPEParameter(
                name=f"emspectrum.new_line.{line}.narrow_ratio",
                owner="emspectrum",
                tier=3,
                physical=True,
                distribution=LogNormal(np.log10(nm), ns),
                units="unitless (log10 ratio to H-beta)",
                description=f"Narrow-component {line}/H-beta flux ratio.",
                rationale=_CITE_NEW_LINE_MAGIC,
            ),
            NPEParameter(
                name=f"emspectrum.new_line.{line}.broad_ratio",
                owner="emspectrum",
                tier=3,
                physical=True,
                distribution=LogNormal(np.log10(bm), bs),
                units="unitless (log10 ratio to H-beta)",
                description=f"Broad-component {line}/H-beta flux ratio.",
                rationale=_CITE_NEW_LINE_MAGIC,
            ),
        )
    ]
)

_CITE_VANDENBERK_TABLE2 = "Vanden Berk et al. (2001, AJ 122, 549), Table 2 -- SDSS QSO composite equivalent widths, used here as line/H-beta flux ratios."
_add(
    # SiIV 1400: narrow uncited (Tier 3), broad mean cited (Tier 2, sigma uncited but mean anchors tier)
    NPEParameter(
        name="emspectrum.new_line.siiv_1400.narrow_ratio",
        owner="emspectrum",
        tier=3,
        physical=True,
        distribution=LogNormal(np.log10(8.916 / 8.649) - 1.5, 0.50),
        units="unitless (log10 ratio to H-beta)",
        description="Narrow-component Si IV 1400 / H-beta flux ratio.",
        rationale="Self-labeled \"MAGIC\" in source: offset of -1.5 dex from the broad value's cited mean is a judgment call, not itself cited.",
    ),
    NPEParameter(
        name="emspectrum.new_line.siiv_1400.broad_ratio",
        owner="emspectrum",
        tier=2,
        physical=True,
        distribution=LogNormal(np.log10(8.916 / 8.649), 0.30),
        units="unitless (log10 ratio to H-beta)",
        citation=_CITE_VANDENBERK_TABLE2,
        description="Broad-component Si IV 1400 / H-beta flux ratio.",
        rationale="Mean = log10(8.916/8.649), directly from the cited table's EWs; sigma=0.30 is MAGIC (analogy-based, no direct per-object fit for this line).",
    ),
    # CIV 1549: narrow uncited, broad fully cited (mean AND sigma)
    NPEParameter(
        name="emspectrum.new_line.civ_1549.narrow_ratio",
        owner="emspectrum",
        tier=3,
        physical=True,
        distribution=LogNormal(np.log10(25.291 / 8.649) - 1.5, 0.50),
        units="unitless (log10 ratio to H-beta)",
        description="Narrow-component C IV 1549 / H-beta flux ratio.",
        rationale="Self-labeled \"MAGIC\" in source: offset of -1.5 dex from the broad value's cited mean is a judgment call, not itself cited.",
    ),
    NPEParameter(
        name="emspectrum.new_line.civ_1549.broad_ratio",
        owner="emspectrum",
        tier=2,
        physical=True,
        distribution=LogNormal(np.log10(25.291 / 8.649), 0.28),
        units="unitless (log10 ratio to H-beta)",
        citation=(
            _CITE_VANDENBERK_TABLE2
            + " Sigma=0.28 derived via a validated quadrature method from Shen et al. (2011, ApJS 194, 45) "
            "SDSS DR7 quasar catalog (queried via VizieR TAP, table J/ApJS/194/45/catalog)."
        ),
        description="Broad-component C IV 1549 / H-beta flux ratio.",
        rationale="Mean = log10(25.291/8.649) from the cited table's EWs; sigma independently derived from a real per-object catalog -- fully cited, not MAGIC.",
    ),
    # CIII] 1909: narrow uncited, broad mean cited (sigma MAGIC)
    NPEParameter(
        name="emspectrum.new_line.ciii_1909.narrow_ratio",
        owner="emspectrum",
        tier=3,
        physical=True,
        distribution=LogNormal(np.log10(15.943 / 8.649) - 1.5, 0.50),
        units="unitless (log10 ratio to H-beta)",
        description="Narrow-component C III] 1909 / H-beta flux ratio.",
        rationale="Self-labeled \"MAGIC\" in source: offset of -1.5 dex from the broad value's cited mean is a judgment call, not itself cited.",
    ),
    NPEParameter(
        name="emspectrum.new_line.ciii_1909.broad_ratio",
        owner="emspectrum",
        tier=2,
        physical=True,
        distribution=LogNormal(np.log10(15.943 / 8.649), 0.30),
        units="unitless (log10 ratio to H-beta)",
        citation=_CITE_VANDENBERK_TABLE2,
        description="Broad-component C III] 1909 / H-beta flux ratio.",
        rationale="Mean = log10(15.943/8.649), directly from the cited table's EWs; sigma=0.30 is MAGIC (analogy-based -- no direct Shen et al. 2011 fit exists for this line).",
    ),
    # MgII 2798: narrow deliberately pinned negligible, broad fully cited
    NPEParameter(
        name="emspectrum.new_line.mgii_2798.narrow_ratio",
        owner="emspectrum",
        tier=3,
        physical=True,
        distribution=LogNormal(np.log10(1e-4), 0.50),
        units="unitless (log10 ratio to H-beta)",
        description="Narrow-component Mg II 2798 / H-beta flux ratio, as drawn within EMSpectrum's new-line block.",
        rationale=(
            "Deliberately pinned to a negligible value -- not a MAGIC guess about real narrow Mg II "
            "strength, but a structural choice to avoid double-counting against the separate, already-"
            "registered narrow Mg II line (`emspectrum.auxline_mgii_hbeta_ratio`)."
        ),
    ),
    NPEParameter(
        name="emspectrum.new_line.mgii_2798.broad_ratio",
        owner="emspectrum",
        tier=2,
        physical=True,
        distribution=LogNormal(np.log10(14.725 / 8.649), 0.26),
        units="unitless (log10 ratio to H-beta)",
        citation=(
            _CITE_VANDENBERK_TABLE2
            + " Sigma=0.26 is a direct per-object measurement from Shen et al. (2011) SDSS DR7 "
            "(0.35<z<0.89 subsample)."
        ),
        description="Broad-component Mg II 2798 / H-beta flux ratio.",
        rationale="Both mean and sigma directly measured/derived from cited real catalogs -- fully cited, not MAGIC.",
    ),
)

# =============================================================================
# emspectrum -- narrow emission-line equivalent-width scatter (currently only
# implemented on the legacy, bypassed `GALAXY.lineratios` path on `main` --
# registered here as the best current guess at owning channel, since the
# underlying D4000 -> EW(line) scatter is real astrophysics worth preserving
# even though which demiurge module will own it is not yet decided; see
# module docstring's exclusions list for what was NOT ported from GALAXY)
# =============================================================================
_add(
    NPEParameter(
        name="emspectrum.narrow_oii_ew_scatter_dex",
        owner="emspectrum",
        tier=3,
        physical=True,
        distribution=Normal(0.0, 0.6),
        units="dex (residual scatter around a deterministic D4000 -> EW([OII]) relation)",
        description="Residual scatter of narrow [OII] equivalent width around its D4000-break-predicted value.",
        rationale="Self-labeled \"MAGIC\" in source: no calibration data justifies this specific value, chosen as roughly double an earlier, even-less-justified floor.",
    ),
    NPEParameter(
        name="emspectrum.narrow_hbeta_ew_scatter_dex",
        owner="emspectrum",
        tier=3,
        physical=True,
        distribution=Normal(0.0, 0.45),
        units="dex (residual scatter around a deterministic D4000 -> EW(H-beta) relation)",
        description="Residual scatter of narrow H-beta equivalent width around its D4000-break-predicted value.",
        rationale="Self-labeled \"MAGIC\" in source: no calibration data justifies this specific value, chosen as roughly double an earlier, even-less-justified floor.",
    ),
)

# =============================================================================
# mock_spectrum -- orchestrator-level, shared across channels
# =============================================================================
_add(
    NPEParameter(
        name="mock_spectrum.hbeta_broad_narrow_ratio",
        owner="mock_spectrum",
        tier=3,
        physical=True,
        distribution=LogUniform(2.0, 15.0),
        units="unitless ratio",
        description="Broad-to-narrow H-beta flux ratio, shared across the Fe II and Balmer-continuum channels' own normalization.",
        rationale="No precisely-measured population distribution found in the literature search backing this parameter; (2.0, 15.0) is an order-of-magnitude bracket around qualitative evidence (narrow H-beta commonly ~10-20% of total in luminous Type 1 quasars).",
    ),
)


# =============================================================================
# Public accessors
# =============================================================================
def _build_registry() -> dict[str, NPEParameter]:
    registry: dict[str, NPEParameter] = {}
    for param in _PARAMETERS:
        if param.name in registry:
            raise ValueError(f"Duplicate NPE-parameter name: {param.name!r}")
        registry[param.name] = param
    return registry


REGISTRY: dict[str, NPEParameter] = _build_registry()


def get_parameter(name: str) -> NPEParameter:
    """Look up one catalogued NPE-parameter by name. Raises KeyError if unknown."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"{name!r} is not a registered NPE-parameter") from None


def list_parameters(
    *, tier: Optional[int] = None, physical: Optional[bool] = None, owner: Optional[str] = None
) -> list[NPEParameter]:
    """Filter the registry by tier, physical/non-physical, and/or owning module."""
    params = REGISTRY.values()
    if tier is not None:
        params = (p for p in params if p.tier == tier)
    if physical is not None:
        params = (p for p in params if p.physical == physical)
    if owner is not None:
        params = (p for p in params if p.owner == owner)
    return list(params)
