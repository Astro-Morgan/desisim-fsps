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
"""
from __future__ import annotations

import numpy as np

# (Y) MAGIC: V-band reference wavelength [Angstrom], matching main's own
# LAMBDA_V convention (Calzetti 2000; Noll et al. 2009) -- kept identical so
# theta0 retains the same "amplitude near V-band" meaning as that reference.
LAMBDA_V = 5500.0


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
