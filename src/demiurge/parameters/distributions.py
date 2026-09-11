"""
Distribution families used as default priors for Tier 2/3 NPE-parameters.

Each distribution is a small, frozen, immutable spec: it knows its own support
and how to draw from itself given a `numpy.random.Generator` (see
`demiurge.rng` for how those are constructed/seeded -- these draw methods
never construct their own RNG, so every draw is reproducible from the
caller's seed). `demiurge.parameters.registry` uses these as the declared
default prior for each catalogued NPE-parameter; `demiurge.parameters.samplers`
is what actually calls `.draw(...)` on a caller's behalf.

Every family here is deliberately minimal -- add a new one only when a real
catalogued parameter needs it. Most families here cover parameters ported
from the pre-refactor reference implementation on `main` (see registry.py's
citations); `ZeroInflated` is a genuinely new addition (2026-09-10, the
quasar/galaxy blending + reddening channel) for parameters with real
probability mass at exactly zero, which no family ported from `main` needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np


@dataclass(frozen=True)
class Uniform:
    """U(low, high)."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not (self.low < self.high):
            raise ValueError(f"Uniform requires low < high, got low={self.low}, high={self.high}")

    @property
    def support(self) -> tuple[float, float]:
        return (self.low, self.high)

    def draw(self, rng: np.random.Generator, size: Optional[int] = None):
        return rng.uniform(self.low, self.high, size=size)


@dataclass(frozen=True)
class LogUniform:
    """10 ** U(log10(low), log10(high)) -- uniform in log10-space."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not (self.low > 0.0 and self.low < self.high):
            raise ValueError(f"LogUniform requires 0 < low < high, got low={self.low}, high={self.high}")

    @property
    def support(self) -> tuple[float, float]:
        return (self.low, self.high)

    def draw(self, rng: np.random.Generator, size: Optional[int] = None):
        log_low, log_high = np.log10(self.low), np.log10(self.high)
        return 10.0 ** rng.uniform(log_low, log_high, size=size)


@dataclass(frozen=True)
class Normal:
    """N(mean, sigma)."""

    mean: float
    sigma: float

    def __post_init__(self) -> None:
        if not (self.sigma > 0.0):
            raise ValueError(f"Normal requires sigma > 0, got sigma={self.sigma}")

    @property
    def support(self) -> tuple[float, float]:
        return (-np.inf, np.inf)

    def draw(self, rng: np.random.Generator, size: Optional[int] = None):
        return rng.normal(self.mean, self.sigma, size=size)


@dataclass(frozen=True)
class LogNormal:
    """10 ** N(mean, sigma) -- `mean`/`sigma` describe log10(value), not value itself."""

    mean: float
    sigma: float

    def __post_init__(self) -> None:
        if not (self.sigma > 0.0):
            raise ValueError(f"LogNormal requires sigma > 0, got sigma={self.sigma}")

    @property
    def support(self) -> tuple[float, float]:
        return (0.0, np.inf)

    def draw(self, rng: np.random.Generator, size: Optional[int] = None):
        return 10.0 ** rng.normal(self.mean, self.sigma, size=size)


@dataclass(frozen=True)
class DiscreteUniform:
    """Uniform choice among a fixed, finite set of values (categorical, incl. numeric grids)."""

    values: tuple

    def __post_init__(self) -> None:
        if len(self.values) == 0:
            raise ValueError("DiscreteUniform requires at least one value")

    @property
    def support(self) -> tuple:
        return self.values

    def draw(self, rng: np.random.Generator, size: Optional[int] = None):
        idx = rng.integers(0, len(self.values), size=size)
        if size is None:
            return self.values[int(idx)]
        arr = np.asarray(self.values)
        return arr[idx]


@dataclass(frozen=True)
class Bernoulli:
    """True with probability `p`, else False."""

    p: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.p <= 1.0):
            raise ValueError(f"Bernoulli requires 0 <= p <= 1, got p={self.p}")

    @property
    def support(self) -> tuple[bool, bool]:
        return (False, True)

    def draw(self, rng: np.random.Generator, size: Optional[int] = None):
        return rng.uniform(size=size) < self.p


@dataclass(frozen=True)
class Poisson:
    """Poisson(mean) -- e.g. a per-mock count of discrete sub-systems."""

    mean: float

    def __post_init__(self) -> None:
        if not (self.mean >= 0.0):
            raise ValueError(f"Poisson requires mean >= 0, got mean={self.mean}")

    @property
    def support(self) -> tuple[float, float]:
        return (0.0, np.inf)

    def draw(self, rng: np.random.Generator, size: Optional[int] = None):
        return rng.poisson(self.mean, size=size)


@dataclass(frozen=True)
class Gamma:
    """Gamma(shape, scale), NumPy's shape/scale parametrization (not shape/rate)."""

    shape: float
    scale: float

    def __post_init__(self) -> None:
        if not (self.shape > 0.0 and self.scale > 0.0):
            raise ValueError(f"Gamma requires shape > 0 and scale > 0, got shape={self.shape}, scale={self.scale}")

    @property
    def support(self) -> tuple[float, float]:
        return (0.0, np.inf)

    def draw(self, rng: np.random.Generator, size: Optional[int] = None):
        return rng.gamma(self.shape, self.scale, size=size)


@dataclass(frozen=True)
class MaxwellBoltzmann:
    """Maxwell-Boltzmann speed distribution with the given `scale` (numpy-only,
    no scipy dependency): a Maxwell-Boltzmann-distributed value is the Euclidean
    norm of a 3-vector of iid N(0, scale) components -- used here directly
    rather than pulling in `scipy.stats.maxwell` for one distribution family.
    """

    scale: float

    def __post_init__(self) -> None:
        if not (self.scale > 0.0):
            raise ValueError(f"MaxwellBoltzmann requires scale > 0, got scale={self.scale}")

    @property
    def support(self) -> tuple[float, float]:
        return (0.0, np.inf)

    def draw(self, rng: np.random.Generator, size: Optional[int] = None):
        shape = (3,) if size is None else (size, 3)
        components = rng.normal(0.0, self.scale, size=shape)
        return np.linalg.norm(components, axis=-1)


@dataclass(frozen=True)
class Dirichlet:
    """Dirichlet(alpha) over a k-simplex: draws a vector of k non-negative
    fractions summing to 1 (e.g. the fractional time-gaps between ordered
    mass-formation quantiles). `alpha` is a per-component concentration
    tuple; a symmetric Dirichlet uses the same value k times. Unlike the
    other families here, a single draw is itself a k-vector, not a scalar --
    `draw(rng)` returns shape (k,), `draw(rng, size=n)` returns (n, k).
    """

    alpha: tuple

    def __post_init__(self) -> None:
        if len(self.alpha) < 2:
            raise ValueError(f"Dirichlet requires at least 2 components, got {len(self.alpha)}")
        if not all(a > 0.0 for a in self.alpha):
            raise ValueError(f"Dirichlet requires all alpha > 0, got {self.alpha}")

    @property
    def support(self) -> tuple[float, float]:
        return (0.0, 1.0)

    def draw(self, rng: np.random.Generator, size: Optional[int] = None):
        return rng.dirichlet(self.alpha, size=size)


@dataclass(frozen=True)
class ZeroInflated:
    """Point mass at exactly 0.0 with probability `p_zero`, else a draw from
    `base`. Use only when a real catalogued parameter has genuine,
    non-negligible probability of being EXACTLY zero (e.g. "this sightline's
    host-disk dust is negligible") -- `LogUniform`/`LogNormal` structurally
    exclude 0, and `Uniform(0, hi)` assigns it zero density, not positive
    mass, so neither can represent this.

    Deliberately not modeled as a separate Bernoulli-gate parameter composed
    downstream: that split means anyone who queries the continuous `base`
    parameter in isolation (plotting, validation, a future NPESampler's
    density code) silently gets a nonzero value with no indication a gate
    exists. `ZeroInflated` is the full marginal by construction, so querying
    it alone is always correct.

    This is a mixed discrete+continuous family, the first one in this
    module -- `.support` reports the continuous part's support only (per
    the shared `Distribution` protocol's numeric-range convention); the
    atom at zero is a separate fact, exposed via `.point_mass`. Any future
    consumer that treats `.support` as the complete characterization of a
    distribution (e.g. a density-plotting routine) needs to also check
    `.point_mass` for a family like this one.
    """

    p_zero: float
    base: "Distribution"

    def __post_init__(self) -> None:
        if not (0.0 <= self.p_zero <= 1.0):
            raise ValueError(f"ZeroInflated requires 0 <= p_zero <= 1, got p_zero={self.p_zero}")

    @property
    def support(self) -> tuple[float, float]:
        lo, hi = self.base.support
        return (min(0.0, lo), hi)

    @property
    def point_mass(self) -> float:
        return 0.0

    def draw(self, rng: np.random.Generator, size: Optional[int] = None):
        if size is None:
            return 0.0 if rng.uniform() < self.p_zero else self.base.draw(rng)
        is_zero = rng.uniform(size=size) < self.p_zero
        return np.where(is_zero, 0.0, self.base.draw(rng, size=size))


Distribution = Union[
    Uniform,
    LogUniform,
    Normal,
    LogNormal,
    DiscreteUniform,
    Bernoulli,
    Poisson,
    Gamma,
    MaxwellBoltzmann,
    Dirichlet,
    ZeroInflated,
]
