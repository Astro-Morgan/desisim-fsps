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

from .distributions import Dirichlet, Distribution, LogUniform, Uniform


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
        distribution=LogUniform(1.0e8, 1.0e12),
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
