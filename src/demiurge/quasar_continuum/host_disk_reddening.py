"""
Host-disk-local reddening for the quasar continuum: the host galaxy's OWN
disk material along the specific nuclear sightline to the AGN -- distinct
from `quasar_continuum.torus_reddening`'s compact, nuclear-scale torus
screen, and from any future galaxy-global diffuse ISM reddening (the
galaxy's own stellar light self-attenuated by its own disk, a separate,
not-yet-built channel that would reuse this module's `cosi_disk` draw
rather than drawing its own -- same physical disk, same inclination).

Independent geometry from the torus: `cosi_disk` (this module's own
registered parameter) is drawn completely independently of
`quasar_continuum.agnsed.cosi`. Hopkins, Hernquist, Hayward & Narayanan
(2012, MNRAS 425, 1121) find the AGN/torus symmetry axis is NOT correlated
with the host galaxy's own large-scale disk axis -- angular-momentum
transport happens independently at very different physical scales -- so
drawing the two inclinations independently is not a simplifying shortcut,
it is what the actual 3D geometry gives once random relative alignment
between the two axes is accounted for.

Unlike `torus_reddening`'s hard geometric threshold, this is a smooth
path-length effect: a foreground dust layer the nuclear sightline crosses
en route out of the host disk, with optical depth scaling roughly as
sec(i) = 1/cos(i) for an inclined disk screen -- the standard geometric
shape real radiative-transfer disk-attenuation models (e.g. Tuffs, Popescu,
Voelk, Kylafis & Dopita 2004, A&A 419, 821) qualitatively confirm saturates
rather than truly diverging as i -> 90deg. Not reproduced here as an exact
fit to those models -- just their qualitative saturating shape, via a hard
cap on the path-length factor.

Zero reddening needs a real point mass, not just low density near it: a
continuously-drawn `cosi_disk` has probability EXACTLY zero of landing at
perfect face-on, so a purely smooth function of that angle cannot represent
"this host galaxy genuinely has negligible disk dust" -- a real,
independent-of-inclination population fact (some hosts, e.g. gas-poor
early-type disks, are just dust-poor). That is why `av_faceon` (this
module's other new parameter) uses `ZeroInflated` rather than a plain
`LogUniform`/`Uniform`: even a face-on view of a genuinely dusty host disk
still shows a nonzero (minimum) reddening under the path-length law below --
exact zero only ever comes from a dust-free `av_faceon` draw, not from
inclination alone.

`transmission()` uses `dust.curve.transmission_with_floor`, NOT the bare
`transmission` -- this curve is not applied below the Lyman limit (~912A),
a real physical domain boundary (dust-grain UV/optical extinction vs.
photoelectric/Compton X-ray absorption, a distinct, deliberately
not-yet-built mechanism). `theta1_slope`'s own registered range (below)
was ALSO narrowed, from an unchecked `Uniform(0,2)` inherited wholesale
from main's general-reach convention to `Uniform(0,1.3)`, anchored on
Prevot et al. (1984, A&A 132, 389)'s real measured SMC-bar far-UV
power-law index (n~1.2) -- the steepest well-established Local Group
extinction curve. That range was the actual root cause of a real bug found
via visual verification of the blended composite (2026-09-11): a draw near
the OLD range's top (1.897 -- steeper than the steepest real curve ever
measured) extrapolated as a bare power law all the way to the Lyman limit
already reached k~5/T~0.01 there, a ~99% jump. With the corrected range, a
floor at the Lyman limit produces a modest, physically defensible
discontinuity instead -- see `dust.curve`'s module docstring for the two
wrong fixes (floor alone; a smooth but ungrounded saturation constant)
tried and discarded before this one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..dust.curve import k_lambda, transmission_with_floor
from ..parameters.samplers import ParameterSampler, PriorSampler

_COSI_DISK = "quasar_continuum.host_disk_reddening.cosi_disk"
_AV_FACEON = "quasar_continuum.host_disk_reddening.av_faceon"
_THETA1_SLOPE = "quasar_continuum.host_disk_reddening.theta1_slope"

# (Y) MAGIC: cap on the secant path-length factor -- prevents the naive
# sec(i) law from diverging as cosi_disk -> 0 (exactly edge-on). Loosely
# brackets the SATURATING (non-divergent) behavior real radiative-transfer
# inclination-attenuation models show (Tuffs et al. 2004, A&A 419, 821)
# without claiming to fit their tables -- the qualitative shape is cited,
# the exact cap value is not.
PATH_LENGTH_CAP = 5.0

# Numerical floor to avoid a literal division by zero at cosi_disk == 0.0
# (which the registered Uniform(0.0, 1.0) prior can draw exactly, unlike
# real continuous angles); PATH_LENGTH_CAP already dominates well before
# this floor would otherwise matter.
_COSI_DISK_FLOOR = 1.0e-6


def _path_length_factor(cosi_disk: float) -> float:
    return min(1.0 / max(cosi_disk, _COSI_DISK_FLOOR), PATH_LENGTH_CAP)


@dataclass(frozen=True)
class HostDiskReddeningResult:
    cosi_disk: float
    av_faceon: float
    theta1_slope: float

    @property
    def av(self) -> float:
        """The actual applied amplitude: `av_faceon` scaled by this draw's
        inclination path-length factor. Exactly 0.0 iff `av_faceon == 0.0`
        (a dust-free host), for any `cosi_disk`."""
        return self.av_faceon * _path_length_factor(self.cosi_disk)

    def transmission(self, wave: np.ndarray) -> np.ndarray:
        """T(lambda) for this draw."""
        wave = np.asarray(wave, dtype=float)
        if self.av_faceon == 0.0:
            return np.ones_like(wave)
        k = k_lambda(wave, self.av, self.theta1_slope, theta2=0.0, theta3=0.0)
        return transmission_with_floor(wave, k)


def draw_host_disk_reddening(
    rng: np.random.Generator,
    *,
    sampler: Optional[ParameterSampler] = None,
) -> HostDiskReddeningResult:
    """Draws `cosi_disk`, `av_faceon`, and `theta1_slope` -- all three,
    unconditionally, every call (no branch skips a draw), so the RNG stream
    stays reproducible/desync-free regardless of which branch a given mock's
    `av_faceon` draw happens to land in (same RNG-hygiene discipline as
    `torus_reddening.draw_torus_reddening`)."""
    if sampler is None:
        sampler = PriorSampler()
    draws = sampler.sample([_COSI_DISK, _AV_FACEON, _THETA1_SLOPE], rng=rng)
    return HostDiskReddeningResult(
        cosi_disk=float(draws[_COSI_DISK]),
        av_faceon=float(draws[_AV_FACEON]),
        theta1_slope=float(draws[_THETA1_SLOPE]),
    )
