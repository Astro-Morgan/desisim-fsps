"""
Reproducible, independent random-number streams for demiurge's generative pipeline.

Every stochastic draw in the simulator (a channel's own parameters, and the
NPE-parameter registry's priors) must be reproducible from a single top-level
seed, and every independently-drawn quantity needs its own statistically
independent stream -- one channel's draw sequence must not shift if a sibling
channel consumes a different number of random values first.

This module is the one place that decision is made, via
`numpy.random.SeedSequence.spawn()`: a `SeedSequence` deterministically
produces any number of *independent* child `SeedSequence`s from one seed, and
each spawned child seeds its own `numpy.random.Generator` (PCG64). Spawning
is order-stable and count-stable -- asking for the first 3 children and later
asking for 3 more always reproduces the same first 3, so composing/extending a
draw sequence never desyncs earlier draws. This directly replaces the old
fork's ad hoc hash-based `_child_seeds()` scheme with NumPy's own supported
mechanism for exactly this purpose, and sidesteps the class of bug documented
in HANDOFF2 Sec. 4 (two `EMSpectrum.spectrum()` calls silently desyncing
because pre-supplying a kwarg changed how many `rand.*` calls one branch
consumed before the other) -- spawned child streams are independent regardless
of how many draws any *sibling* stream consumes.

Uses `numpy.random.Generator`/`default_rng`, not the legacy `RandomState` API
the old fork used throughout -- `Generator` is NumPy's currently-recommended
API (better statistical properties from PCG64, and it's what `.spawn()`
requires).
"""
from __future__ import annotations

import numpy as np


def child_generators(seed: int | None, n: int) -> list[np.random.Generator]:
    """Derive `n` independent, reproducible `Generator`s from one top-level seed.

    `seed=None` draws fresh, non-reproducible entropy for every child (useful
    for exploratory/interactive use); an integer seed is fully reproducible --
    calling this again with the same `seed` and the same or a larger `n`
    reproduces the same generators in the same order.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    root = np.random.SeedSequence(seed)
    return [np.random.default_rng(child) for child in root.spawn(n)]


def child_generator(seed: int | None) -> np.random.Generator:
    """Convenience wrapper for the common single-stream case."""
    return child_generators(seed, 1)[0]
