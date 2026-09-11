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

Domain of application (2026-09-11, revised twice this same day -- read this
before touching VALIDITY_FLOOR_AA or either caller's theta1 range again):
this is a UV/optical dust extinction curve. `quasar_continuum`'s own
native grid extends far into the EUV/X-ray (AGNSED's energy grid reaches
~0.06A), where the physically relevant absorption mechanism is
photoelectric/Compton opacity acting on gas column density -- a distinct
process from UV/optical dust-grain absorption/scattering, not something
this curve claims to model (X-ray absorption is a separate, deliberately
not-yet-built mechanism -- see quasar_continuum design docs). A naive
power-law k(lambda) with theta1 > 0 also mathematically DIVERGES as
lambda -> 0, which compounds the domain-mismatch with a real numerical
blow-up if evaluated carelessly.

Two wrong fixes were tried and discarded before this one, both found via
visual verification of the blended composite -- worth knowing so a future
change doesn't repeat either mistake:
1. A hard wavelength floor at 912A (the Lyman limit) with NO other change:
   mathematically stops the divergence, but for a real, in-registered-range
   theta1 draw (host_disk_reddening's theta1_slope=1.897, drawn from what
   was then a Uniform(0,2) prior), the curve's own RAW value already
   reached k~5 (T~0.01, ~99% attenuated) by 912A -- a factor of only ~6 in
   wavelength from lambda_v. The floor didn't prevent a discontinuity, it
   relocated it: transmission jumped from exactly 1.0 to ~0.01 within one
   wavelength grid step, a real, visually-obvious, unphysical-looking step
   in the composite spectrum -- because the underlying curve had already
   run away well before the floor engaged, not because of where the floor
   itself sat.
2. Smoothly saturating k(lambda) at a fixed ceiling (`K_SATURATION`,
   nominally chosen so T~1e-4 "already fully dark"): removes the
   discontinuity, but introduces exactly the kind of ungrounded MAGIC
   constant this project's citation discipline exists to avoid -- correctly
   flagged as such (PI direction, 2026-09-11) rather than accepted.

Real fix, addressing the actual root cause identified above: the registered
theta1_slope range this curve was being evaluated with was itself
unrealistic, not merely "extrapolated." host_disk_reddening's Uniform(0,2)
was inherited wholesale from main's own general-reach "fit any real curve"
convention -- appropriate for main's own use case (fitting empirical
curves over their own measured range), but never checked against what
happens when the SAME slope is extrapolated as a bare power law all the
way to the Lyman limit. Prevot, Lequeux, Maurice, Prevot & Rocca-Volmerange
(1984, A&A 132, 389) measure the steepest well-established Local Group
extinction curve (the SMC bar) to have a far-UV power-law index of
n~1.2 -- host_disk_reddening's theta1_slope is now bounded at that real,
citable ceiling (Uniform(0, 1.3), Tier 2, replacing the former Tier-3
MAGIC range) rather than main's unchecked, generic 2.0. With that
realistic ceiling, VALIDITY_FLOOR_AA (restored, a genuine physical
boundary -- dust-grain physics does not apply past the Lyman limit,
whatever the mechanism below it turns out to be) produces a modest,
physically defensible discontinuity at 912A (at most a few-fold dimming
for a typical draw, not ~100x) -- qualitatively similar to how real
Lyman-limit systems/DLAs genuinely do show a sharp, real spectral feature
at this exact wavelength, not an artifact to be smoothed away.
torus_reddening's own theta1_slope range (Uniform(0, 0.8), already
citation-anchored on Gaskell et al. 2004's flatter-than-SMC finding) was
already well inside this bound and needed no change.
"""
from __future__ import annotations

import numpy as np

# (Y) MAGIC: V-band reference wavelength [Angstrom], matching main's own
# LAMBDA_V convention (Calzetti 2000; Noll et al. 2009) -- kept identical so
# theta0 retains the same "amplitude near V-band" meaning as that reference.
LAMBDA_V = 5500.0

# Below this wavelength [Angstrom], this curve is not applied at all
# (transmission fixed at 1.0) -- see module docstring's "Domain of
# application" note. Anchored on the Lyman limit: a real physical boundary
# (photoionizing photons couple to gas via photoelectric/Compton opacity,
# not dust-grain absorption/scattering), not a numerically-convenient
# cutoff -- the discontinuity this produces is a real, deliberate domain
# boundary, not something to be smoothed over.
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
    argument explicitly."""
    wave = np.asarray(wave, dtype=float)
    term1 = theta0 * (wave / lambda_v) ** (-theta1)
    term2 = theta2 * drude(wave, lambda_bump, bump_width)
    return term1 + term2 + theta3


def transmission(k: np.ndarray) -> np.ndarray:
    """T(lambda) = 10^(-0.4*k(lambda)), in (0, 1] for k >= 0."""
    return 10.0 ** (-0.4 * np.asarray(k, dtype=float))


def transmission_with_floor(wave: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Like `transmission`, but clamped to 1.0 (no reddening applied) below
    `VALIDITY_FLOOR_AA` -- see module docstring's "Domain of application"
    note. This is what every real caller (torus_reddening.py,
    host_disk_reddening.py) uses; the bare `transmission` above is kept for
    direct testing of the unclamped functional form itself."""
    wave = np.asarray(wave, dtype=float)
    return np.where(wave >= VALIDITY_FLOOR_AA, transmission(k), 1.0)
