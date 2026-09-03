"""
NPE-parameter sampling backends.

Per the refactor charter (HANDOFF3 Sec. 6.1): the *sampler* that supplies a
value for a given NPE-parameter is swappable, and the physics/spectrum-
building code that consumes those values never needs to know or care which
backend supplied them. This module builds only `PriorSampler` -- the backend
that draws each parameter independently from its registry default prior,
exactly like the pre-refactor fork's own `*_RANGE` + `rand.uniform`/
`rand.normal` convention, just centralized and registry-driven.

`NPESampler` (drawing from a trained normalizing-flow posterior instead) is
deliberately NOT built here. Per the charter's build-order decision: no
trained flow exists yet (Simulation Round 1, HANDOFF3 Sec. 0.4, hasn't run),
and dispatch/auto-detection logic with nothing real to validate against is
easy to get subtly wrong now and easy to get right later. `NPESampler` and
the mode-auto-selection dispatch land together, in the same commit, once a
real trained flow exists (HANDOFF3 Sec. 6.1).

`ParameterSampler` exists as an abstract base specifically so that later
addition is a small, contained change rather than a rewrite of every caller.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Optional, Sequence

import numpy as np

from .registry import get_parameter


class ParameterSampler(ABC):
    """Interface every NPE-parameter sampling backend implements."""

    @abstractmethod
    def sample(
        self,
        names: Sequence[str],
        *,
        rng: np.random.Generator,
        condition: Optional[Mapping[str, float]] = None,
        size: Optional[int] = None,
    ) -> dict[str, object]:
        """Return one value (or, if `size` is given, an array of `size` values)
        per requested parameter name.

        `condition` supplies fixed values for zero or more *physical*
        NPE-parameters (HANDOFF3 Sec. 5.2/6.3) -- implementations must reject
        any name in `condition` that the registry doesn't tag `physical`.
        """
        raise NotImplementedError


class PriorSampler(ParameterSampler):
    """Draws each requested NPE-parameter independently from its registry
    default prior. This is what Simulation Round 1 (HANDOFF3 Sec. 0.4) runs
    on in full, and what every other caller falls back to before a trained
    NPE exists.

    Independent-prior sampling means "conditioning" here can only mean
    "use this fixed value instead of drawing one" (real posterior
    conditioning, where fixing one parameter changes the *distribution* of
    the others, needs a trained NPE and is `NPESampler`'s job later) -- but
    the physical/non-physical gate (HANDOFF3 Sec. 5.2) is enforced here
    regardless, so callers and tests can rely on that rule from day one.
    """

    def sample(
        self,
        names: Sequence[str],
        *,
        rng: np.random.Generator,
        condition: Optional[Mapping[str, float]] = None,
        size: Optional[int] = None,
    ) -> dict[str, object]:
        condition = condition or {}
        for cond_name in condition:
            param = get_parameter(cond_name)
            if not param.physical:
                raise ValueError(
                    f"{cond_name!r} is not a physical NPE-parameter and cannot be supplied as a "
                    f"conditioning input (HANDOFF3 Sec. 5.2)."
                )

        result: dict[str, object] = {}
        for name in names:
            if name in condition:
                value = condition[name]
                result[name] = np.full(size, value) if size is not None else value
            else:
                param = get_parameter(name)
                result[name] = param.distribution.draw(rng, size=size)
        return result
