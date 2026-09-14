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

Domain of application (2026-09-11/2026-09-13 -- read this in full before
touching VALIDITY_FLOOR_AA, k_lambda_with_floor, or either caller's
theta1/curvature range again; the earlier attempts below are kept as a
record, not because any of their reasoning still applies):

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

Four fixes were tried, in order, each replacing the last -- all found via
visual verification of the blended composite, each correcting a real,
specific flaw in the one before it:

1. A hard wavelength floor at 912A (the Lyman limit): transmission clamped
   to exactly 1.0 below it. Stopped the divergence, but for a real,
   in-registered-range theta1 draw, the curve's own RAW value already
   reached k~5 (T~0.01) by 912A -- so transmission jumped from exactly 1.0
   to ~0.01 within one wavelength grid step: a real, visually obvious,
   unphysical-looking step, because the curve had already run away well
   before the floor engaged.
2. Smoothly saturating k(lambda) at a fixed ceiling (`K_SATURATION`,
   nominally T~1e-4 "already fully dark"): removed the discontinuity, but
   introduced exactly the kind of ungrounded MAGIC constant this project's
   citation discipline exists to avoid -- correctly flagged as such (PI
   direction) rather than accepted.
3. Restoring the hard 912A floor, but with host_disk_reddening's
   theta1_slope range narrowed to a real, citable ceiling (Prevot et al.
   1984's SMC-bar measurement, n~1.2 -- see registry.py): made the
   discontinuity's SIZE realistic instead of absurd, but a hard step is
   still not how any real physical mechanism behaves at a boundary the
   curve itself has no special feature at -- correctly flagged (PI
   direction) as still "looking totally bogus." The real Lyman limit IS a
   genuine physical discontinuity, but it belongs to neutral-hydrogen
   photoionization (an atomic gas effect), not dust-grain physics --
   hanging a dust-curve cutoff there and calling it physically motivated
   was importing a real edge from the wrong mechanism.
4. Holding k(lambda) flat below the floor, at its own already-calibrated
   value AT 912A: perfectly continuous (verified numerically), no new
   constant. Real progress, but still a fixed, deterministic RULE with no
   free parameter -- inconsistent with this project's own standing
   convention that genuinely unconstrained physics gets represented as a
   real Tier-3 NPE-parameter with a prior (to be calibrated later by a
   trained NPE), not baked in as a hardcoded extrapolation policy (PI
   direction, 2026-09-13: since there's no real physical treatment below
   912A, "the actual implementation should be a tier 3 parametric curve
   with forced continuity/smoothness" with the real Tier-2 curve above).

Actual fix: below `VALIDITY_FLOOR_AA`, k(lambda) is a QUADRATIC in
ln(lambda) (`k_lambda_with_floor`), anchored so its value AND slope match
the real Tier-2 curve's value and slope exactly at the floor, for ANY value
of a new free parameter `euv_curvature` -- no constraint-solving needed at
runtime, this is a property of the quadratic form itself (verified
numerically in tests: value and derivative match at the boundary to
float precision, for curvature spanning its whole registered range).
`euv_curvature=0` exactly reproduces continuing the original (diverging)
power law -- deliberately not excluded, since nothing rules it out;
`euv_curvature<0` makes the curve turn over and asymptote back toward
transparency at short wavelengths (the more negative, the sooner);
`euv_curvature>0` makes it diverge even faster than the plain power law.
`quasar_continuum.torus_reddening.euv_curvature`/
`quasar_continuum.host_disk_reddening.euv_curvature` (registry.py) are
each independently-drawn Tier-3 parameters, `Uniform(-2.0, 0.1)` (PI
direction) -- genuinely MAGIC (no informed prior exists for what happens
in this regime), but now a real per-mock random variable like every other
unconstrained quantity in this project, rather than a fixed rule.

`theta1_slope`'s registered ranges (torus: Gaskell et al. 2004-anchored at
Uniform(0,0.8); host-disk: Prevot et al. 1984-anchored at Uniform(0,1.3),
narrowed 2026-09-11 from an unchecked Uniform(0,2)) remain as corrected in
fix 3 above -- a realistic curve above the floor is still worth having
independent of how the extension below it is handled.
"""
from __future__ import annotations

import numpy as np

# (Y) MAGIC: V-band reference wavelength [Angstrom], matching main's own
# LAMBDA_V convention (Calzetti 2000; Noll et al. 2009) -- kept identical so
# theta0 retains the same "amplitude near V-band" meaning as that reference.
LAMBDA_V = 5500.0

# Below this wavelength [Angstrom], k(lambda) switches from the real,
# citation-anchored power-law family to the free (Tier-3) quadratic
# extension in `k_lambda_with_floor` -- see module docstring's "Domain of
# application" note. Anchored on the Lyman limit as a recognizable, real UV
# boundary (not because dust-grain physics has any special feature there --
# it doesn't; the real Lyman-limit discontinuity belongs to
# neutral-hydrogen photoionization, a different mechanism this curve does
# not model).
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
    euv_curvature: float,
    theta2: float = 0.0,
    theta3: float = 0.0,
    *,
    lambda_v: float = LAMBDA_V,
    lambda_bump: float = 2175.0,
    bump_width: float = 350.0,
) -> np.ndarray:
    """k(lambda; theta), with a genuinely free (Tier-3) extension below
    `VALIDITY_FLOOR_AA` in place of any fixed rule -- see module docstring's
    "Actual fix". Above the floor: identical to `k_lambda`. Below it: a
    quadratic in ln(k) vs ln(lambda), anchored to match `k_lambda`'s own
    value AND slope exactly at the floor for ANY `euv_curvature` (this
    holds by construction, not by solving anything at runtime -- see
    tests). `euv_curvature` is the only remaining freedom: 0 exactly
    continues the plain power law; negative values turn the curve over
    toward transparency at short wavelengths; positive values diverge
    faster.

    Requires `theta2 == theta3 == 0` (the only case either real caller
    uses) -- the slope-matching here assumes `k_lambda`'s log-log slope at
    the floor is exactly `-theta1`, true only when the bump/grey-floor
    terms are off.
    """
    if theta2 != 0.0 or theta3 != 0.0:
        raise NotImplementedError(
            "k_lambda_with_floor's below-floor extension assumes theta2 == theta3 == 0 "
            "(the log-log slope at the floor is only exactly -theta1 in that case); "
            "neither current caller (torus_reddening, host_disk_reddening) uses a "
            "nonzero bump/grey-floor term."
        )
    wave = np.asarray(wave, dtype=float)
    if theta0 == 0.0:
        # k_lambda is identically 0 everywhere in this case (theta2=theta3=0
        # already asserted above) -- the log-space construction below would
        # need ln(0), so short-circuit rather than route a legitimate,
        # already-correct answer through it.
        return k_lambda(wave, theta0, theta1, lambda_v=lambda_v, lambda_bump=lambda_bump, bump_width=bump_width)
    above_floor = wave >= VALIDITY_FLOOR_AA
    k_above = k_lambda(wave, theta0, theta1, lambda_v=lambda_v, lambda_bump=lambda_bump, bump_width=bump_width)

    t = np.log(wave)
    t0 = np.log(VALIDITY_FLOOR_AA)
    y0 = np.log(theta0) - theta1 * (t0 - np.log(lambda_v))  # ln(k_lambda(floor)) when theta2=theta3=0
    s0 = -theta1  # dln(k)/dln(lambda) at the floor, for the plain power law
    k_below = np.exp(y0 + s0 * (t - t0) + euv_curvature * (t - t0) ** 2)

    return np.where(above_floor, k_above, k_below)


def transmission(k: np.ndarray) -> np.ndarray:
    """T(lambda) = 10^(-0.4*k(lambda)), in (0, 1] for k >= 0."""
    return 10.0 ** (-0.4 * np.asarray(k, dtype=float))


def transmission_with_floor(
    wave: np.ndarray,
    theta0: float,
    theta1: float,
    euv_curvature: float,
    theta2: float = 0.0,
    theta3: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """T(lambda) = 10^(-0.4 * k_lambda_with_floor(...)) -- this is what
    every real caller (torus_reddening.py, host_disk_reddening.py) uses;
    the bare `k_lambda`/`transmission` above are kept separate for direct
    testing of the unclamped functional form."""
    return transmission(k_lambda_with_floor(wave, theta0, theta1, euv_curvature, theta2, theta3, **kwargs))
