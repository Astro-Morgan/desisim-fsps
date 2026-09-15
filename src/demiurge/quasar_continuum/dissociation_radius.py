"""
AGN-proximity dust-sublimation/dissociation radius: the distance inside which
the radiation field is harsh enough to sublimate dust grains (and, closer in,
photodissociate molecules) -- the boundary the ionization emulator's
metallicity self-consistency work needs (see project memory:
`project-dust-emission-planning-handoff`'s "AGN-proximity zone" section and
`project-ionization-emulator-design`'s "beyond Cue" item 5), not yet consumed
by anything since the emulator itself doesn't exist yet.

Foundational formula: Barvainis (1987, ApJ 320, 537) -- dust in radiative
equilibrium with the illuminating continuum, sublimating at a fixed grain
temperature:

    R_sub = prefactor * sqrt(L_UV / 1e46 erg/s)   [pc]

where `prefactor` (this module's own registered parameter) absorbs Barvainis's
own (T_sub/1500K)^-2.8 * (a/0.05um)^-0.5 grain-temperature/size dependence --
his paper quotes prefactor values of 0.4-1.3 pc across the grain assumptions
he considers, which is real (if uncertain) grain physics, not measurement
noise on one true value, hence registered as a single Tier-2 range rather
than exposing T_sub/grain-size as two separate degenerate free parameters.

Angle (cosi) dependence, PI direction 2026-09-15 -- deliberately NOT an
imported empirical correction: two independent, real literature findings say
the observed sublimation/torus radius is smaller than Barvainis's isotropic
formula predicts -- (1) Kawaguchi & Mori (2010, ApJL 724, L183; 2011, ApJ
737, 105) and Minezaki et al. (2019, ApJ 886, 150)'s empirically-confirmed
anisotropic disc illumination (the disc shines less brightly toward the
equator than the pole -- an inclination effect, present at any accretion
rate) and (2) Wang, Qiu, Du & Ho (2014, ApJ 797, 65)'s slim-disc
self-shadowing (at high Eddington ratio, the puffed-up inner disc advects
energy inward rather than radiating it, so less power escapes at all,
independent of viewing angle -- subsequently applied to observed torus
size-luminosity relations by later reverberation-mapping studies). These are
genuinely different mechanisms, but (1) is NOT imported here:
`continuum.py`'s AGNSED synthesis already applies
`cosi_scale = cosi/0.5` (KD18's own Lambertian disc/warm geometry) to
the disc+warm luminosity before it ever reaches this module -- bolting
Minezaki's population-averaged correction on top of luminosity we've already
reduced for the same physical reason would double-count the same
inclination effect. Instead, `compute_l_uv_erg_s` below takes the ALREADY
cosi-scaled disc+warm luminosity directly from AGNSED's own output, so
`R_sub` is angle-dependent through the same first-principles geometry this
project already computes, not a second, independently-fit correction.
Mechanism (2), the Eddington-ratio/self-shadowing effect, is a real, still-
missing piece of physics -- explicitly excluded from this pass (no citable
per-object formula was found, only population-level slope shifts; see
project memory) and left as a known simplification for a later improvement,
not implemented here.

L_UV definition: the disc+warm ("big blue bump") bolometric luminosity only,
cosi-scaled, EXCLUDING the isotropic hot-corona/X-ray tail -- the hot corona
is AGNSED's hard-X-ray component (KD18 Sec. 2.2), not the optical/UV
continuum that heats dust grains to sublimation, so it plays no role in this
formula (consistent with how "L_UV" is used throughout the AGN-torus
size-luminosity literature this module's docstring cites -- a bolometric-ish
UV/optical continuum proxy, not a narrowly-bounded photon-energy integral,
since no single source specifies exact integration limits for it either).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .continuum import KEV_TO_ERG
from ..parameters.samplers import ParameterSampler, PriorSampler

_SUBLIMATION_PREFACTOR = "quasar_continuum.dissociation_radius.sublimation_prefactor"


@dataclass(frozen=True)
class DissociationRadiusResult:
    sublimation_prefactor_pc: float
    l_uv_erg_s: float
    radius_pc: float


def compute_l_uv_erg_s(
    disk_rate_unscaled: np.ndarray,
    warm_rate_unscaled: np.ndarray,
    energy_kev: np.ndarray,
    cosi: float,
) -> float:
    """The disc+warm ('big blue bump') bolometric luminosity [erg/s], cosi-
    scaled EXACTLY as `continuum.py`'s own `cosi_scale = cosi/0.5` Lambertian
    geometry -- see this module's own docstring for why reusing that same
    factor (rather than importing a separate empirical anisotropy
    correction) is the deliberate design here. `disk_rate_unscaled`/
    `warm_rate_unscaled`/`energy_kev` are `QuasarContinuum.meta`'s own
    `"disk_rate_unscaled"`/`"warm_rate_unscaled"`/`"energy_kev"` entries
    (per-bin photon rates [photons/s] and bin-midpoint photon energies
    [keV]) -- pass those directly, not a re-derived approximation.

    `cosi=0.0` (exactly edge-on) gives `L_UV=0.0` here, since the Lambertian
    disc/warm geometry has genuinely zero direct disc/warm flux at that
    limit -- a known, accepted edge case of this simplified proxy (the real
    isotropic hot corona would still contribute some ionizing flux at any
    inclination, but the hot corona is deliberately excluded from this L_UV
    definition; see module docstring).
    """
    disk_rate_unscaled = np.asarray(disk_rate_unscaled, dtype=float)
    warm_rate_unscaled = np.asarray(warm_rate_unscaled, dtype=float)
    energy_kev = np.asarray(energy_kev, dtype=float)

    cosi_scale = cosi / 0.5
    photon_rate = (disk_rate_unscaled + warm_rate_unscaled) * cosi_scale
    energy_rate_erg_s = photon_rate * (energy_kev * KEV_TO_ERG)
    return float(np.sum(energy_rate_erg_s))


def draw_dissociation_radius(
    rng: np.random.Generator,
    l_uv_erg_s: float,
    *,
    sampler: Optional[ParameterSampler] = None,
) -> DissociationRadiusResult:
    """Draws `sublimation_prefactor` and evaluates
    `R_sub = prefactor * sqrt(l_uv_erg_s / 1e46 erg/s)` [pc]. `l_uv_erg_s`
    is this mock's own already-computed `compute_l_uv_erg_s(...)` result --
    computing it is this function's caller's responsibility (it needs the
    full AGNSED output, not just a scalar), keeping the registered-parameter
    draw here decoupled from AGNSED's own internals for easy testing.
    """
    if sampler is None:
        sampler = PriorSampler()
    draws = sampler.sample([_SUBLIMATION_PREFACTOR], rng=rng)
    prefactor = float(draws[_SUBLIMATION_PREFACTOR])
    radius_pc = prefactor * np.sqrt(max(l_uv_erg_s, 0.0) / 1.0e46)
    return DissociationRadiusResult(
        sublimation_prefactor_pc=prefactor,
        l_uv_erg_s=l_uv_erg_s,
        radius_pc=radius_pc,
    )
