"""
Metallicity history Z(t) for the galaxy continuum channel, tied to the
star-formation history rather than drawn as an independent free function of
time -- per project direction (2026-09-04): metallicity at any epoch is
downstream of the preceding SFH/enrichment history, not an independent
degree of freedom.

Uses the classic closed-box chemical evolution relation (Searle & Sargent
1972, ApJ 173, 25): Z(t) = y * ln(1 / mu(t)), where mu(t) is the remaining
gas-mass fraction and y is the effective nucleosynthetic yield. The gas
reservoir is sized by a star-formation efficiency epsilon (the fraction of
the *initial* gas reservoir ultimately locked into stars by the observation
epoch): mu(t) = 1 - epsilon * F(t), where F(t) is the SFH's own cumulative
stellar-mass fraction (`SFHResult.cumulative_mass_fraction`). Initial
metallicity is fixed at Z=0 (pristine gas) rather than a free parameter,
matching the classic closed-box assumption and keeping this channel's
parameter count minimal.

This is a deliberate first-pass simplification, not the final word: a more
complete treatment would solve a numerical, inflow/outflow-aware model
(e.g. Yin, Shen & Hao 2023, ApJ 958, 34) rather than assume a closed box
with no gas exchange -- left as a flagged future refinement, matching how
the IMF channel's own known-incomplete first pass is being handled.

Z(t) is monotonically non-decreasing by construction here (F(t) is
non-decreasing, epsilon in (0,1), and ln(1/mu) is increasing in F) -- no
separate enforcement needed for this generator. `validate_monotonic_z` is
for the *other* construction path, a fully user-supplied Z(t) array, where
that guarantee doesn't automatically hold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..parameters.samplers import ParameterSampler, PriorSampler


@dataclass(frozen=True)
class MetallicityResult:
    z_grid: np.ndarray
    yield_: float
    star_formation_efficiency: float


def draw_metallicity(
    rng: np.random.Generator,
    cumulative_mass_fraction: np.ndarray,
    *,
    sampler: Optional[ParameterSampler] = None,
) -> MetallicityResult:
    """Draw yield/efficiency from `sampler` and evaluate the closed-box Z(t)
    relation on the same time grid as the SFH's `cumulative_mass_fraction`.
    """
    if sampler is None:
        sampler = PriorSampler()

    draws = sampler.sample(
        ["galaxy_continuum.metallicity.yield", "galaxy_continuum.metallicity.star_formation_efficiency"],
        rng=rng,
    )
    yield_ = float(draws["galaxy_continuum.metallicity.yield"])
    efficiency = float(draws["galaxy_continuum.metallicity.star_formation_efficiency"])

    z_grid = evaluate_closed_box(cumulative_mass_fraction, yield_, efficiency)
    return MetallicityResult(z_grid=z_grid, yield_=yield_, star_formation_efficiency=efficiency)


def evaluate_closed_box(cumulative_mass_fraction: np.ndarray, yield_: float, efficiency: float) -> np.ndarray:
    """Z(t) = yield_ * ln(1 / mu(t)), mu(t) = 1 - efficiency * F(t)."""
    gas_fraction_remaining = 1.0 - efficiency * cumulative_mass_fraction
    if np.any(gas_fraction_remaining <= 0.0):
        raise ValueError(
            "star_formation_efficiency * cumulative_mass_fraction reached 1.0 (exhausted the closed-box "
            "gas reservoir) -- efficiency must stay low enough that mu(t) > 0 for all t."
        )
    return yield_ * np.log(1.0 / gas_fraction_remaining)


def validate_monotonic_z(z_grid: np.ndarray, *, enforce: bool = False) -> np.ndarray:
    """Check that a (typically user-supplied) Z(t) array is non-decreasing.

    Raises ValueError unless `enforce=True`, in which case it's coerced via
    a running maximum rather than rejected.
    """
    if np.any(np.diff(z_grid) < 0.0):
        if not enforce:
            raise ValueError(
                "Z(t) must be non-decreasing -- metallicity does not un-enrich over time in this model. "
                "Pass enforce=True to auto-correct via a running maximum, or supply a valid array."
            )
        z_grid = np.maximum.accumulate(z_grid)
    return z_grid
