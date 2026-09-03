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

**This registry is deliberately empty of real entries right now.** The
refactor is a ground-up rebuild, not an incremental patch of `main` -- `main`
is a reference to consult for physics/citations when a channel is actually
being (re-)designed and built, not a manifest to bulk-port ahead of that
work. Add a parameter here only alongside the real channel module that owns
it, with its citation/rationale re-verified against the actual literature (or
against `main`'s own citation comments, re-checked, not assumed current) at
the time it's added -- not copied wholesale from an earlier research pass.
(A full research extraction of every Tier 2/3 parameter `main` currently
defines exists as reference material, kept outside this repo -- ask if you
need to know where; it is not meant to be transcribed in bulk.)

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

from .distributions import Distribution


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
# Real channel parameters get added here, one channel at a time, alongside
# that channel's actual module -- see the module docstring above. Empty for
# now: no demiurge channel modules have been built yet.
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
