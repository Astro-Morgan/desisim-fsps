"""
Torus-local reddening for the quasar continuum: the compact, nuclear-scale
AGN dust screen from the classical unification-model torus (distinct from
`quasar_continuum.host_disk_reddening`'s host-galaxy-scale screen, and from
any future galaxy-global diffuse ISM reddening -- three physically different
screens along different parts of the same overall line of sight).

Obscuration is a DETERMINISTIC consequence of the same `cosi` value already
drawn for `quasar_continuum.agnsed.cosi` (`continuum.py`'s Lambertian
`cosi_scale` factor), not an independent draw: the sightline intercepts the
torus iff `cosi < covering_angle_cosine`, where `covering_angle_cosine`
(this module's own registered parameter, representing cos(theta_oc),
theta_oc the torus's half-opening angle measured from the pole) is drawn
once per mock, independently of `cosi`. This mirrors exactly how
`continuum.py` already computes `cosi_scale = cosi/0.5` deterministically
inside the physics module rather than via any sampler-level correlation --
`PriorSampler` only supports independent draws (see
`demiurge.parameters.samplers`'s module docstring); this gate is ordinary
physics-module logic operating on two already-independently-drawn values,
not a workaround for that constraint.

`quasar_continuum.agnsed.cosi` itself was widened (2026-09-10) from a
type-1-selection-restricted `Uniform(0.5, 1.0)` to the full isotropic
`Uniform(0.0, 1.0)` specifically because this module now exists to model
obscuration explicitly -- see that parameter's registry entry.

Reddening curve: the flat, bump-free family Gaskell, Goosmann, Antonucci &
Whysong (2004, ApJ 616, 147) find for AGN nuclear reddening -- significantly
flatter in the UV than the local ISM/SMC, with radio-loud AGN curves
specifically "very flat" (their more host-contaminated radio-quiet sample is
slightly steeper -- consistent with `host_disk_reddening.py` capturing
exactly that additional, distinct, steeper component separately). `theta2`
(UV bump) and `theta3` (grey floor) are fixed at 0.0 here (not drawn) per
that same bump-free finding -- see `registry.py`'s rationale.

`transmission()` uses `dust.curve.transmission_with_floor`, NOT the bare
`transmission` -- this curve is not applied below ~912A (see that module's
"Validity floor" note): quasar_continuum's native grid reaches into the
EUV/X-ray, far outside where any UV/optical dust curve means anything, and
naively extrapolating the power-law term there crushes the transmission to
numerical noise that has nothing to do with real physics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..dust.curve import k_lambda, transmission_with_floor
from ..parameters.samplers import ParameterSampler, PriorSampler

_COVERING_ANGLE_COSINE = "quasar_continuum.torus_reddening.covering_angle_cosine"
_THETA0_AMPLITUDE = "quasar_continuum.torus_reddening.theta0_amplitude"
_THETA1_SLOPE = "quasar_continuum.torus_reddening.theta1_slope"


@dataclass(frozen=True)
class TorusReddeningResult:
    """`intercepted=False` means the sightline misses the torus entirely --
    `transmission()` is then identically 1.0 (no reddening) at every
    wavelength, and `theta0_amplitude`/`theta1_slope` describe a draw that
    was made (see RNG-hygiene note on `draw_torus_reddening`) but never
    used."""

    intercepted: bool
    covering_angle_cosine: float
    theta0_amplitude: float
    theta1_slope: float
    cosi: float

    def transmission(self, wave: np.ndarray) -> np.ndarray:
        """T(lambda) for this draw."""
        wave = np.asarray(wave, dtype=float)
        if not self.intercepted:
            return np.ones_like(wave)
        k = k_lambda(wave, self.theta0_amplitude, self.theta1_slope, theta2=0.0, theta3=0.0)
        return transmission_with_floor(wave, k)


def draw_torus_reddening(
    rng: np.random.Generator,
    cosi: float,
    *,
    sampler: Optional[ParameterSampler] = None,
) -> TorusReddeningResult:
    """`cosi` must be the SAME value already drawn for this mock's
    `quasar_continuum.agnsed.cosi` (e.g.
    `quasar_result.meta["drawn_parameters"]["quasar_continuum.agnsed.cosi"]`)
    -- torus interception is a deterministic consequence of that geometry,
    not an independent draw.

    `covering_angle_cosine`/`theta0_amplitude`/`theta1_slope` are drawn
    unconditionally, before checking interception -- not only when the
    sightline turns out to be obscured. This is deliberate RNG hygiene: if
    the amplitude/slope draws were skipped whenever `intercepted` turns out
    False, a tiny change in some upstream parameter that flips `intercepted`
    for one mock in a batch would desync the RNG stream for every subsequent
    mock drawn under the same seed.
    """
    if sampler is None:
        sampler = PriorSampler()
    draws = sampler.sample(
        [_COVERING_ANGLE_COSINE, _THETA0_AMPLITUDE, _THETA1_SLOPE],
        rng=rng,
    )
    covering_angle_cosine = float(draws[_COVERING_ANGLE_COSINE])
    theta0_amplitude = float(draws[_THETA0_AMPLITUDE])
    theta1_slope = float(draws[_THETA1_SLOPE])
    return TorusReddeningResult(
        intercepted=cosi < covering_angle_cosine,
        covering_angle_cosine=covering_angle_cosine,
        theta0_amplitude=theta0_amplitude,
        theta1_slope=theta1_slope,
        cosi=cosi,
    )
