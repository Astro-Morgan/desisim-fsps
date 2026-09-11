"""
Shared multiplicative attenuation-curve machinery (dust extinction/
reddening), used by quasar_continuum's torus-local and host-disk-local
reddening channels (torus_reddening.py, host_disk_reddening.py).

Base functional form -- re-derived from the pre-refactor reference
implementation's own family (see `main:py/desisim/dust.py`), not ported
verbatim. Only the base 4-parameter family is built here, since neither
current caller needs the FUV-curvature/NIR-break/scattered-light opt-in
extensions that module also built; those get added later only when a real
channel demonstrably needs one (this project's standing incremental-
buildout policy), not speculatively:

    k(lambda; theta) = theta0 * (lambda/lambda_v)^(-theta1)
                        + theta2 * D(lambda; lambda_bump, bump_width)
                        + theta3,          theta0, theta2, theta3 >= 0

where D is the standard Drude profile used for the UV dust bump
(Fitzpatrick & Massa 1986), normalized so D(lambda_bump) = 1.

Additive-flux-deficit convention (matching `main`'s own dust.py, and this
project's settled 3-bucket continuum/emission/absorption decomposition --
dust attenuation is a member of the "absorption" bucket): the transmission
T(lambda) = 10^(-0.4*k(lambda)) multiplies the intrinsic flux; a caller that
needs the ground-truth deficit as its own additive channel uses
`flux_in * (T - 1)`, which is <= 0 everywhere for k >= 0.

Domain of application (2026-09-11, revised three times this same day --
read this in full before touching VALIDITY_FLOOR_AA, k_lambda_with_floor,
or either caller's theta1 range again; the earlier attempts below are kept
as a record, not because any of their reasoning still applies):

This is a UV/optical dust extinction curve. `quasar_continuum`'s own
native grid extends far into the EUV/X-ray (AGNSED's energy grid reaches
~0.06A), where the physically relevant absorption mechanism is
photoelectric/Compton opacity acting on gas column density, or (shortward
of Lyman-alpha specifically) intervening intergalactic HI (the Lyman-alpha
forest / Gunn-Peterson trough) -- both distinct from UV/optical dust-grain
absorption/scattering, and neither modeled here (deliberately not yet
built -- see quasar_continuum design docs). A naive power-law k(lambda)
with theta1 > 0 also mathematically DIVERGES as lambda -> 0, compounding
the domain mismatch with a real numerical blow-up if evaluated carelessly.

Three fixes were tried, in order, each replacing the last -- all found via
visual verification of the blended composite, each correcting a real,
specific flaw in the one before it:

1. A hard wavelength floor at 912A (the Lyman limit): transmission clamped
   to exactly 1.0 below it. Stopped the divergence, but for a real,
   in-registered-range theta1 draw (host_disk_reddening's
   theta1_slope=1.897, then drawn from a Uniform(0,2) prior), the curve's
   own RAW value already reached k~5 (T~0.01) by 912A -- so transmission
   jumped from exactly 1.0 to ~0.01 within one wavelength grid step: a
   real, visually obvious, unphysical-looking step, because the curve had
   already run away well before the floor engaged.
2. Smoothly saturating k(lambda) at a fixed ceiling (`K_SATURATION`,
   nominally T~1e-4 "already fully dark"): removed the discontinuity, but
   introduced exactly the kind of ungrounded MAGIC constant this project's
   citation discipline exists to avoid -- correctly flagged as such (PI
   direction) rather than accepted.
3. Restoring the hard 912A floor, but with host_disk_reddening's
   theta1_slope range narrowed to a real, citable ceiling (Prevot et al.
   1984's SMC-bar measurement, n~1.2 -- see registry.py): this made the
   discontinuity's SIZE realistic (a few-fold dimming for a typical draw,
   not ~100x) instead of absurd. Still flagged (PI direction) as looking
   "totally bogus" -- correctly: a hard step is not how ANY real physical
   mechanism behaves at a boundary the curve itself has no special feature
   at. The real Lyman limit IS a genuine physical discontinuity, but that
   discontinuity belongs to neutral-hydrogen photoionization (an atomic gas
   effect), not to dust-grain physics -- a UV/optical dust curve has no
   reason to jump at exactly 912A, so hanging the cutoff there and calling
   it physically motivated was importing a real edge from the wrong
   mechanism to justify what was actually just "where the model stops."

Actual fix: `k_lambda_with_floor` clips the WAVELENGTH fed into `k_lambda`
at `VALIDITY_FLOOR_AA` from below, rather than clipping the resulting
transmission. Below the floor, this simply repeats k's own already-real,
already-calibrated value AT the floor, rather than either extrapolating
further (mathematically diverges) or resetting to full transparency (an
equally arbitrary discontinuity in the other direction -- dust does not
suddenly vanish at a specific wavelength). This is perfectly continuous by
construction (transmission at lambda=911.99A exactly equals transmission
at lambda=912.01A -- verified in tests, not merely argued) and needs no
new constant at all: the "floor value" is just this curve's own prediction,
evaluated at a wavelength it is still nominally responsible for, held flat
past the point where it has no further information. It is also more
physically defensible than resetting to transparency: real dust does not
become transparent past some wavelength, if anything grain
absorption/scattering efficiency continues to rise into the UV, so holding
the last real value is the more honest "we don't know, so we don't claim
it gets better" default, not the more dramatic "we don't know, so we
pretend it's zero" default.

`theta1_slope`'s registered ranges (torus: Gaskell et al. 2004-anchored at
Uniform(0,0.8); host-disk: Prevot et al. 1984-anchored at Uniform(0,1.3),
narrowed 2026-09-11 from an unchecked Uniform(0,2)) remain as corrected in
fix 3 above -- a realistic curve is still worth having independent of how
the domain boundary itself is handled, and this is why the same real
citation and Tier-2 status stay in place.
"""
from __future__ import annotations

import numpy as np

# (Y) MAGIC: V-band reference wavelength [Angstrom], matching main's own
# LAMBDA_V convention (Calzetti 2000; Noll et al. 2009) -- kept identical so
# theta0 retains the same "amplitude near V-band" meaning as that reference.
LAMBDA_V = 5500.0

# Below this wavelength [Angstrom], k(lambda) is held at its own value AT
# this floor rather than evaluated further -- see module docstring's
# "Domain of application" note. Anchored on the Lyman limit as a
# recognizable, real UV boundary (not because dust-grain physics has any
# special feature there -- it doesn't; the real Lyman-limit discontinuity
# belongs to neutral-hydrogen photoionization, a different mechanism this
# curve does not model).
VALIDITY_FLOOR_AA = 912.0


def drude(wave: np.ndarray, lambda0: float, gamma: float) -> np.ndarray:
    """Standard Drude profile, normalized so D(lambda0) = 1 (the UV
    dust-bump shape, Fitzpatrick & Massa 1986)."""
    wave = np.asarray(wave, dtype=float)
    return (wave**2 * gamma**2) / ((wave**2 - lambda0**2) ** 2 + wave**2 * gamma**2)


def k_lambda(
    wave: np.ndarray,
    theta0: float,
    theta1: float,
    theta2: float = 0.0,
    theta3: float = 0.0,
    *,
    lambda_v: float = LAMBDA_V,
    lambda_bump: float = 2175.0,
    bump_width: float = 350.0,
) -> np.ndarray:
    """k(lambda; theta) -- see module docstring for the closed form.
    `theta2`/`theta3` default to 0.0 (bump-free, no grey floor) so a caller
    that only ever wants the power-law term doesn't need to pass every
    argument explicitly. Unclamped -- diverges as `wave -> 0` for
    `theta1 > 0`; real callers use `k_lambda_with_floor` instead."""
    wave = np.asarray(wave, dtype=float)
    term1 = theta0 * (wave / lambda_v) ** (-theta1)
    term2 = theta2 * drude(wave, lambda_bump, bump_width)
    return term1 + term2 + theta3


def k_lambda_with_floor(
    wave: np.ndarray,
    theta0: float,
    theta1: float,
    theta2: float = 0.0,
    theta3: float = 0.0,
    *,
    lambda_v: float = LAMBDA_V,
    lambda_bump: float = 2175.0,
    bump_width: float = 350.0,
) -> np.ndarray:
    """k(lambda; theta), with `wave` clipped to `VALIDITY_FLOOR_AA` from
    below before evaluating the curve -- see module docstring's "Actual
    fix" for why this (not clamping the resulting transmission) is what
    real callers should use."""
    wave = np.asarray(wave, dtype=float)
    clipped = np.maximum(wave, VALIDITY_FLOOR_AA)
    return k_lambda(clipped, theta0, theta1, theta2, theta3, lambda_v=lambda_v, lambda_bump=lambda_bump, bump_width=bump_width)


def transmission(k: np.ndarray) -> np.ndarray:
    """T(lambda) = 10^(-0.4*k(lambda)), in (0, 1] for k >= 0."""
    return 10.0 ** (-0.4 * np.asarray(k, dtype=float))


def transmission_with_floor(
    wave: np.ndarray,
    theta0: float,
    theta1: float,
    theta2: float = 0.0,
    theta3: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """T(lambda) = 10^(-0.4 * k_lambda_with_floor(...)) -- this is what
    every real caller (torus_reddening.py, host_disk_reddening.py) uses;
    the bare `k_lambda`/`transmission` above are kept separate for direct
    testing of the unclamped functional form."""
    return transmission(k_lambda_with_floor(wave, theta0, theta1, theta2, theta3, **kwargs))
