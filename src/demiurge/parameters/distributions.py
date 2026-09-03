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
catalogued parameter needs it (this module currently covers every family
actually used by the parameters ported from the pre-refactor reference
implementation on `main`; see registry.py's citations).
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


Distribution = Union[
    Uniform, LogUniform, Normal, LogNormal, DiscreteUniform, Bernoulli, Poisson, Gamma, MaxwellBoltzmann
]
