"""
Canonical registry scaffolding for every Tier 2/3 NPE-parameter in demiurge.

Per the refactor charter (HANDOFF3 Sec. 5-6): a Tier 1 quantity is exact
physics (a closed-form/first-principles relation, no free parameter beyond
physical constants and inputs) and does not belong here. A Tier 2 quantity
has a real empirical measurement or relation in the literature that sets its
*default prior*, but the actual per-mock value is a genuine NPE-parameter --
drawn from that prior today, and (once a trained NPE exists) drawn from/
conditioned by the NPE instead. A Tier 3 quantity has no exact or empirical
source at all -- still an NPE-parameter, still explicitly labeled, but its
default distribution is a stated judgment call pending real NPE calibration.

**Entries are added one channel at a time, alongside that channel's real
module** -- this is a ground-up rebuild, not an incremental patch of `main`.
`main` is a reference to consult for physics/citations when a channel is
actually being (re-)designed and built, not a manifest to bulk-port ahead of
that work; every entry below was added alongside real code for that specific
channel, with its own literature check at the time (not copied wholesale
from `main` or from an earlier research pass). (A full research extraction
of every Tier 2/3 parameter `main` currently defines exists as reference
material, kept outside this repo -- ask if you need to know where; it is not
meant to be transcribed in bulk.)

Every entry added here must be `physical=True` unless it's a genuine property
of the generative model/instrument rather than the astrophysical source
(HANDOFF3 Sec. 5.2) -- the tentative example is camera/instrument calibration
coefficients, not yet ported. Every Tier 2 entry requires a citation; every
entry requires a rationale (HANDOFF3 Sec. 5.2's edge-case process: record a
one-line documented reason at the definition site, not just the label).

Whenever a citation-bearing entry is added here, update the living
`docs/paper/methods.tex` section for that channel and add the corresponding
`docs/paper/refs.bib` entry in the *same commit* (HANDOFF3 Sec. 7) -- this is
the project's actual citation/rigor discipline, not a separate cleanup pass.

Example of the shape a real entry takes, once a channel actually needs one:

    _add(
        NPEParameter(
            name="<channel>.<parameter>",
            owner="<channel>",
            tier=2,  # or 3
            physical=True,
            distribution=Uniform(low, high),  # or LogUniform/Normal/LogNormal/etc.
            units="...",
            citation="Author et al. (Year, Journal Vol, Page) -- what it measured.",  # Tier 2 only
            description="What this parameter physically represents.",
            rationale="Why this tier/prior/physical classification, in one line.",
        ),
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .distributions import Dirichlet, Distribution, LogNormal, LogUniform, Normal, Uniform, ZeroInflated


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
# galaxy_continuum.sfh -- Dense-Basis-style continuous SFH generator
# (Iyer & Gawiser 2017, ApJ 838, 127; Iyer et al. 2019, ApJ 879, 116)
# =============================================================================
_add(
    NPEParameter(
        name="galaxy_continuum.sfh.total_stellar_mass",
        owner="galaxy_continuum.sfh",
        tier=3,
        physical=True,
        distribution=LogUniform(5.0e7, 5.0e12),
        units="Msun",
        description="Total stellar mass formed by the observation epoch (integral of SFR(t)).",
        rationale="Order-of-magnitude bracket spanning the real galaxy stellar-mass-function range, not fit to a specific survey's mass function -- MAGIC.",
    ),
    NPEParameter(
        name="galaxy_continuum.sfh.mass_quantile_gap_fractions",
        owner="galaxy_continuum.sfh",
        tier=3,
        physical=True,
        distribution=Dirichlet((2.0, 2.0, 2.0, 2.0)),
        units="unitless (4 fractions summing to 1, of the interval [0, t_obs])",
        description=(
            "Fractional time-gaps between the four intervals bounded by the mass-formation quantile "
            "times t25/t50/t75 and the endpoints t=0 (formation) and t=t_obs (observation) -- "
            "cumulative-summing these fractions (times t_obs) gives the three quantile constraint "
            "times fed to the SFH Gaussian-process reconstruction."
        ),
        rationale=(
            "Symmetric Dirichlet(alpha=2) chosen as a mildly-peaked-toward-even-spacing prior over the "
            "ordering-respecting simplex -- inspired by, but not independently verified against, the "
            "alpha=1 (uniform-simplex) choice Iglesias-Navarro et al. (2024, A&A 689, A58) used for a "
            "related (not identical) per-bin sSFR-fraction Dirichlet prior in their own Dense-Basis-based "
            "SBI model. MAGIC pending a dedicated check against Iyer et al.'s own default."
        ),
    ),
    NPEParameter(
        name="galaxy_continuum.sfh.gp_length_scale_fraction",
        owner="galaxy_continuum.sfh",
        tier=3,
        physical=True,
        distribution=Uniform(0.05, 0.5),
        units="unitless (fraction of t_obs)",
        description=(
            "Matern-3/2 GP length-scale, as a fraction of t_obs, controlling SFH smoothness/burstiness "
            "(Iyer et al. 2019's own description: 'the tension in a string that passes through all the "
            "constraints') -- small values allow burstier reconstructed SFHs, large values force smoother ones."
        ),
        rationale=(
            "Iyer et al. (2019, ApJ 879, 116) state this hyperparameter was calibrated against semi-"
            "analytic models to minimize reconstruction loss while avoiding unphysical negative SFR, but "
            "their exact calibrated value/range was not independently verified here -- MAGIC range pending "
            "that check."
        ),
    ),
)

# =============================================================================
# galaxy_continuum.metallicity -- closed-box chemical enrichment tied to the
# SFH's own cumulative mass fraction (Searle & Sargent 1972)
# =============================================================================
_add(
    NPEParameter(
        name="galaxy_continuum.metallicity.yield",
        owner="galaxy_continuum.metallicity",
        tier=3,
        physical=True,
        distribution=LogUniform(1.0e-3, 5.0e-2),
        units="unitless (absolute metal-mass yield per unit mass locked into stars)",
        description="Effective nucleosynthetic yield y in the closed-box relation Z(t) = y * ln(1/mu(t)).",
        rationale="Order-of-magnitude bracket consistent with typical effective-yield values discussed in the chemical-evolution literature -- not fit to a specific measured sample. MAGIC.",
    ),
    NPEParameter(
        name="galaxy_continuum.metallicity.star_formation_efficiency",
        owner="galaxy_continuum.metallicity",
        tier=3,
        physical=True,
        distribution=Uniform(0.01, 0.99),
        units="unitless (fraction of the initial gas reservoir ultimately locked into stars by t_obs)",
        description="Efficiency epsilon setting the closed-box gas reservoir size (M_gas,initial = total_stellar_mass / epsilon), via mu(t) = 1 - epsilon * F(t).",
        rationale="Bounded (0,1) by the closed-box construction itself; the specific prior shape (uniform) is not fit to real gas-fraction observations. MAGIC.",
    ),
)

# =============================================================================
# quasar_continuum.agnsed -- AGN accretion-disk continuum (Kubota & Done
# 2018, MNRAS 480, 1247; arXiv:1804.00171 -- KD18). Full-AGNSED reimplementation,
# not QSOSED's restricted special case -- see quasar_continuum/geometry.py's
# module docstring for the validation summary (exact match on radii/
# luminosities against the compiled official Fortran, ~1% match on the full
# spectral shape against KD18's own published Figure 9). Several entries
# below are parameters QSOSED (the paper's own simplified variant) pins at
# a single "typical" value; here they are real Tier 2 NPE-parameters because
# KD18 itself reports the measured population scatter around that typical
# value (their own Sec. 4.2/Table 2), not because the fixed value was wrong.
# =============================================================================
_add(
    NPEParameter(
        name="quasar_continuum.agnsed.black_hole_mass",
        owner="quasar_continuum.agnsed",
        tier=2,
        physical=True,
        distribution=LogUniform(1.0e6, 1.0e10),
        units="Msun",
        citation="Kubota & Done (2018, MNRAS 480, 1247) Sec. 5.1 -- the mass grid (1e6-1e10 Msun) their own full SED model was validated over.",
        description="Supermassive black hole mass.",
        rationale="Direct match to KD18's own explored/validated mass range, not an independently chosen bracket.",
    ),
    NPEParameter(
        name="quasar_continuum.agnsed.eddington_ratio",
        owner="quasar_continuum.agnsed",
        tier=2,
        physical=True,
        distribution=LogUniform(0.02, 1.0),
        units="unitless (Mdot / Mdot_Edd)",
        citation="Kubota & Done (2018, MNRAS 480, 1247) Sec. 5.1 (grid range mdot=0.03-1) and Lusso & Risaliti (2017, A&A 602, A79) -- the real SDSS quasar sample KD18 compare their model to spans mdot~0.03-1.",
        description="Eddington ratio Mdot/Mdot_Edd.",
        rationale="Widened slightly below KD18's own grid floor (0.02 vs 0.03) only to stay consistent with the hard_xray_luminosity_fraction prior's own low end; otherwise matches both KD18's grid and the real quasar sample it's benchmarked against.",
    ),
    NPEParameter(
        name="quasar_continuum.agnsed.spin",
        owner="quasar_continuum.agnsed",
        tier=3,
        physical=True,
        distribution=Uniform(-1.0, 0.998),
        units="unitless (dimensionless Kerr spin a*)",
        description="Dimensionless black hole spin.",
        rationale="MAGIC -- spans the full range AGNSED's own parameter file (lmodel_agnsed.dat) allows; no informed population-level spin distribution imposed. KD18's own worked examples fix astar=0 throughout, so there is no in-paper guidance on a realistic spin prior shape to cite here.",
    ),
    NPEParameter(
        name="quasar_continuum.agnsed.cosi",
        owner="quasar_continuum.agnsed",
        tier=2,
        physical=True,
        distribution=Uniform(0.0, 1.0),
        units="unitless (cosine of the disc/warm-Comptonisation inclination angle)",
        citation="Antonucci (1993, ARA&A 31, 473) and Netzer (2015, ARA&A 53, 365) -- the standard AGN unification argument treats the torus/disc symmetry axis as isotropically oriented across the general AGN population (uniform in cos i), with type-1 vs. type-2 classification an EMERGENT consequence of whether a given random sightline happens to intercept the torus, not a precondition on which systems get generated.",
        description="Cosine of the inclination angle applied to the disc and warm-Comptonisation components (agnsed.f's own cosi/0.5 geometric factor; the hot corona is treated as isotropic and unaffected).",
        rationale="Widened from an earlier Uniform(0.5, 1.0) (2026-09-09, restricted to Urry & Padovani (1995)'s type-1-selection argument, i <~ 60 deg) to the full isotropic Uniform(0.0, 1.0) (2026-09-10). That restriction was a workaround for not yet having any torus-obscuration machinery -- generation was limited to geometries that would already look type-1. Now that quasar_continuum.torus_reddening exists to model obscuration explicitly as a deterministic consequence of this same cosi draw (see that module), restricting cosi itself is no longer necessary or correct: the population should be generated isotropically, and the reddening layer -- not a pre-restricted prior -- is what should produce the type-1/type-2 appearance split. cosi_scale=cosi/0.5 (continuum.py) remains well-behaved across the full range (->0 as edge-on, the physically expected near-total dimming of direct disc/warm flux for a geometrically thin disc; hot corona stays isotropic and unaffected either way).",
    ),
    NPEParameter(
        name="quasar_continuum.agnsed.hard_xray_luminosity_fraction",
        owner="quasar_continuum.agnsed",
        tier=2,
        physical=True,
        distribution=LogUniform(0.01, 0.05),
        units="unitless (L_diss,hot / L_Edd)",
        citation="Kubota & Done (2018, MNRAS 480, 1247) Sec. 4.2 -- measured 0.02-0.04 L_Edd across their 3 fitted AGN (NGC 5548, Mrk 509, PG 1115+407); consistent with Jin, Ward, Done & Gelbord (2012a, MNRAS 420, 1825)'s 50-AGN sample, referenced in KD18 Sec. 4.2 as varying by only a factor 2-3 when stacked by Eddington ratio.",
        description="Intrinsic hard X-ray (hot-corona) dissipated luminosity, as a fraction of L_Edd -- generalizes QSOSED's own hardcoded 0.02 into a real Tier-2 parameter (see quasar_continuum/geometry.py's solve_geometry docstring for the r_hot inversion this drives).",
        rationale="LogUniform bracket set slightly wider than the 0.02-0.04 measured range to allow real population scatter beyond 3 objects, anchored directly on KD18's own reported numbers rather than an independently chosen range.",
    ),
    NPEParameter(
        name="quasar_continuum.agnsed.kte_hot",
        owner="quasar_continuum.agnsed",
        tier=2,
        physical=True,
        distribution=Uniform(40.0, 100.0),
        units="keV",
        citation="Fabian et al. (2015, MNRAS 451, 4375) and Lubinski et al. (2016, MNRAS 458, 2454) -- observed hot-corona electron temperature range kTe~40-100 keV, tau~1-2, cited in KD18 Sec. 1.",
        description="Hot-corona electron temperature.",
        rationale="Direct match to the observed range KD18 themselves cite; QSOSED's own fixed value (100 keV) is the upper edge of this range, not an independent choice.",
    ),
    NPEParameter(
        name="quasar_continuum.agnsed.kte_warm",
        owner="quasar_continuum.agnsed",
        tier=2,
        physical=True,
        distribution=LogUniform(0.1, 1.0),
        units="keV",
        citation="Magdziarz et al. (1998, MNRAS 301, 179), Czerny et al. (2003, A&A 412, 317), Gierlinski & Done (2004b, MNRAS 349, L7), and Porquet et al. (2004, A&A 422, 85) -- observed warm-Comptonisation electron temperature range kTe~0.1-1 keV, tau~10-25, cited in KD18 Sec. 1; consistent with KD18's own Table 2 per-object fits (0.17-0.50 keV across their 3 AGN).",
        description="Warm-Comptonisation electron temperature.",
        rationale="Direct match to the observed range KD18 cite; QSOSED's own fixed value (0.2 keV) sits well inside this range and inside KD18's own Table 2 fitted spread.",
    ),
    NPEParameter(
        name="quasar_continuum.agnsed.gamma_warm",
        owner="quasar_continuum.agnsed",
        tier=2,
        physical=True,
        distribution=Normal(2.5, 0.3),
        units="unitless (photon index)",
        citation="Petrucci et al. (2018, A&A 611, A59)'s passive-disc theoretical prediction Gamma_warm=2.5, and Kubota & Done (2018, MNRAS 480, 1247) Table 2's own per-object fits (2.28-3.06 across their 3 AGN).",
        description="Warm-Comptonisation photon index.",
        rationale="Normal centered on the passive-disc theoretical value (also QSOSED's own fixed default) with sigma chosen to bracket KD18's own measured per-object spread; KD18 Sec. 4.3 notes real objects deviate from the passive-disc prediction (steeper for higher mdot), consistent with real scatter rather than a single true value.",
    ),
    NPEParameter(
        name="quasar_continuum.agnsed.r_warm_over_r_hot",
        owner="quasar_continuum.agnsed",
        tier=2,
        physical=True,
        distribution=LogUniform(1.5, 4.0),
        units="unitless (R_warm / R_hot)",
        citation="Kubota & Done (2018, MNRAS 480, 1247) Table 2 -- their own per-object fits give R_warm/R_hot = 151/43=3.51 (NGC 5548), 40/21=1.90 (Mrk 509), 35/9.8=3.57 (PG 1115+407), Sec. 4.3.",
        description="Ratio of the warm-Comptonisation outer radius to the hot-corona outer radius (r_hot itself is derived from hard_xray_luminosity_fraction, not drawn directly -- see geometry.py).",
        rationale="KD18's own general-grid convention (r_warm=2*r_hot, Sec. 4.3, 'guided by the fits to individual objects') is a simplifying tie, not a measured law -- their own Table 2 per-object fits show real scatter around it (1.9-3.6), which is what this prior's range reflects directly rather than an independently chosen bracket.",
    ),
)

# =============================================================================
# quasar_continuum.torus_reddening -- the compact, nuclear-scale AGN dust
# screen from the classical unification-model torus (distinct from
# quasar_continuum.host_disk_reddening's host-galaxy-scale screen, and from
# any future galaxy-global diffuse ISM reddening -- three physically
# different screens). See torus_reddening.py's module docstring for the
# deterministic cosi-vs-covering_angle_cosine interception mechanism.
# =============================================================================
_add(
    NPEParameter(
        name="quasar_continuum.torus_reddening.covering_angle_cosine",
        owner="quasar_continuum.torus_reddening",
        tier=2,
        physical=True,
        distribution=Uniform(0.13, 0.47),
        units="unitless (cosine of the torus half-opening angle, measured from the pole)",
        citation="Ezhikode et al. (2017, MNRAS 472, 3492) -- mean torus covering factor f_c = 0.30 +/- 0.17 across 51 local type-1 AGN (IR/bolometric method).",
        description="quasar_continuum.agnsed.cosi intercepts the torus (torus reddening applies) iff cosi < this value; not intercepted otherwise (torus_reddening.py).",
        rationale="Uniform(mean-1sigma, mean+1sigma) from Ezhikode et al.'s own measured population scatter -- an approximation to their reported distribution shape using an existing family rather than adding a new bounded-shape distribution (e.g. Beta) for this one parameter alone.",
    ),
    NPEParameter(
        name="quasar_continuum.torus_reddening.theta0_amplitude",
        owner="quasar_continuum.torus_reddening",
        tier=3,
        physical=True,
        distribution=Uniform(0.1, 3.0),
        units="mag-scale amplitude (dust.curve.k_lambda's theta0, at lambda_v=5500A)",
        description="Torus-local reddening magnitude when the sightline intercepts the torus (see covering_angle_cosine above); irrelevant, unused, when it does not.",
        rationale="MAGIC -- matches main's own precedent order-of-magnitude bracket (negligible to heavily obscured) for this kind of amplitude; no dedicated covering-factor-to-E(B-V) calibration exists yet to derive a tighter, citable range.",
    ),
    NPEParameter(
        name="quasar_continuum.torus_reddening.theta1_slope",
        owner="quasar_continuum.torus_reddening",
        tier=2,
        physical=True,
        distribution=Uniform(0.0, 0.8),
        units="unitless (dust.curve.k_lambda's theta1, power-law slope)",
        citation="Gaskell, Goosmann, Antonucci & Whysong (2004, ApJ 616, 147) -- AGN nuclear reddening curves are significantly flatter in the UV than the local ISM/SMC; radio-loud (least host-contaminated) AGN curves specifically 'very flat'.",
        description="Power-law steepness of the torus-local reddening curve.",
        rationale="Upper bound well below main's general-reach theta1 range (0.0-2.0, SMC-like at the top) to reflect Gaskell et al.'s specific finding that nuclear (torus-scale) curves are flatter than SMC, not merely bounded by it -- host-galaxy-scale reddening (steeper, per their own radio-quiet-vs-radio-loud comparison) is handled separately by quasar_continuum.host_disk_reddening, consistent with that finding.",
    ),
)
# theta2 (UV bump) and theta3 (grey floor) are fixed at 0.0 here (not
# registered/drawn) -- Gaskell et al. (2004) find AGN nuclear reddening
# curves show no 2175A bump; a grey floor isn't needed to capture the
# flat/SMC-like family's reach with just an amplitude+slope. See
# torus_reddening.py.

# =============================================================================
# quasar_continuum.host_disk_reddening -- the host galaxy's OWN disk
# material along the specific nuclear sightline to the AGN (distinct from
# quasar_continuum.torus_reddening's compact nuclear-scale screen, and from
# any future galaxy-global diffuse ISM reddening). See
# host_disk_reddening.py's module docstring for the geometry.
# =============================================================================
_add(
    NPEParameter(
        name="quasar_continuum.host_disk_reddening.cosi_disk",
        owner="quasar_continuum.host_disk_reddening",
        tier=2,
        physical=True,
        distribution=Uniform(0.0, 1.0),
        units="unitless (cosine of the host-galaxy-disk inclination to our line of sight)",
        citation="Hopkins, Hernquist, Hayward & Narayanan (2012, MNRAS 425, 1121) -- the AGN/torus symmetry axis is not correlated with the host galaxy's own large-scale disk axis (independent angular-momentum transport at very different physical scales); standard isotropic-orientation argument for the Uniform-in-cosine shape itself.",
        description="Host-galaxy-disk inclination -- deliberately independent of quasar_continuum.agnsed.cosi (real AGN host disks are not assumed coplanar with the AGN disk/torus).",
        rationale="Drawn fully independently of agnsed.cosi per Hopkins et al.'s finding: if the two axes are randomly oriented relative to EACH OTHER, then drawing both isotropically relative to our fixed sightline is not a simplifying shortcut, it is what the actual 3D geometry gives once relative alignment is accounted for.",
    ),
    NPEParameter(
        name="quasar_continuum.host_disk_reddening.av_faceon",
        owner="quasar_continuum.host_disk_reddening",
        tier=3,
        physical=True,
        distribution=ZeroInflated(p_zero=0.3, base=LogUniform(0.05, 2.0)),
        units="mag-scale amplitude (dust.curve.k_lambda's theta0 at face-on, i.e. before the inclination path-length scaling in host_disk_reddening.py)",
        description="Face-on-equivalent host-disk reddening amplitude; the actual applied amplitude is this value scaled by a path-length factor depending on cosi_disk (host_disk_reddening.py).",
        rationale="ZeroInflated chosen specifically because a real, non-negligible fraction of hosts (e.g. gas-poor early-type disks) are genuinely dust-free, independent of viewing geometry -- LogUniform/LogNormal structurally cannot represent that. p_zero and the LogUniform bracket are both MAGIC (no dedicated per-quantity literature check performed yet) pending a real calibration pass.",
    ),
    NPEParameter(
        name="quasar_continuum.host_disk_reddening.theta1_slope",
        owner="quasar_continuum.host_disk_reddening",
        tier=3,
        physical=True,
        distribution=Uniform(0.0, 2.0),
        units="unitless (dust.curve.k_lambda's theta1, power-law slope)",
        description="Power-law steepness of the host-disk-local reddening curve -- ordinary host-galaxy ISM dust, not AGN-processed nuclear dust, so not restricted to torus_reddening's narrower flat/SMC-like range.",
        rationale="MAGIC -- reuses main's own general-reach theta1 bracket (Calzetti-like through steep SMC-like) since this is ordinary ISM dust, not yet independently re-derived for this specific host-disk-local context.",
    ),
)
# theta2 (UV bump) and theta3 (grey floor) are fixed at 0.0 here for this
# first pass too -- UNLIKE torus_reddening, this is NOT because of a
# bump-free physical finding (ordinary host ISM dust genuinely can show a
# real 2175A bump, main:py/desisim/dust.py's own vary_bump_shape extension
# exists for exactly this reason) -- it is a deliberate scope-narrowing
# choice for this first pass, honestly flagged rather than silently
# resolved: revisit if a specific host-disk channel need arises. See
# host_disk_reddening.py.

# =============================================================================
# Real channel parameters get added here, one channel at a time, alongside
# that channel's actual module -- see the module docstring above.
# =============================================================================


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
