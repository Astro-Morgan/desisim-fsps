"""
Thermal Comptonization spectral shape: solves the Kompaneets equation for
photons upscattered by a thermal electron population, following
\\citet{ZdziarskiJohnsonMagdziarz1996} as extended by
\\citet{ZyckiDoneSmith1999} -- the `nthcomp` model, called `donthcomp` in the
Fortran, that both `agnsed.f` and `qsosed.f` call for the warm-Comptonization
and hot-corona spectral shapes (Kubota & Done 2018, hereafter KD18).

HEASARC does not distribute `donthcomp.f` standalone -- it is a core XSPEC
built-in (`nthcomp`), not a community "local model" like `agnsed`/`qsosed`
themselves, so there is no direct official source for this specific piece to
diff against the way `geometry.py` was validated against the compiled
`agnsed.f`/`qsosed.f`. This module is instead a light cleanup of a
previously-published, independently-vetted Python translation of the same
Fortran subroutine: originally adapted by A.D. Thomas for
\\citet{Thomas2016}'s OXAF AGN photoionization model (their supplementary
code, `oxaf.py`), kept algorithmically identical here (same tridiagonal
Kompaneets solve, same variable names as the original Fortran) rather than
reimplemented from scratch, since correctness here rests on that prior
publication rather than on this session's own validation work. Output
convention: a per-bin photon COUNT (arbitrary overall normalization) --
callers renormalize by a target luminosity, exactly as agnsed.f/qsosed.f do
(see `spectrum.py`).
"""
from __future__ import annotations

import numpy as np

M_E_C2_KEV = 511.0  # electron rest energy


def donthcomp(ear: np.ndarray, gamma: float, kte_kev: float, kt_seed_kev: float) -> np.ndarray:
    """Per-bin photon count (arbitrary overall normalization) for a thermally
    Comptonized blackbody-seeded spectrum.

    Args:
        ear: energy bin EDGES [keV], length `ne + 1`.
        gamma: asymptotic power-law photon index.
        kte_kev: electron (plasma) temperature [keV].
        kt_seed_kev: seed blackbody temperature [keV].

    Returns:
        Array of length `ne`, one photon count per bin.
    """
    xth, nth, spt = _thcompton(kt_seed_kev / M_E_C2_KEV, kte_kev / M_E_C2_KEV, gamma)

    xninv = M_E_C2_KEV
    ih = 1
    xx = 1.0 / xninv
    while ih < nth and xx > xth[ih]:
        ih += 1
    il = ih - 1
    spp = spt[il] + (spt[ih] - spt[il]) * (xx - xth[il]) / (xth[ih] - xth[il])
    normfac = 1.0 / spp

    ne = ear.size - 1
    prim = np.zeros(ear.size)
    j = 0
    for i in range(ear.size):
        while j <= nth and M_E_C2_KEV * xth[j] < ear[i]:
            j += 1
        if j <= nth:
            if j > 0:
                jl = j - 1
                prim[i] = spt[jl] + (ear[i] / M_E_C2_KEV - xth[jl]) * (spt[jl + 1] - spt[jl]) / (
                    xth[jl + 1] - xth[jl]
                )
            else:
                prim[i] = spt[0]

    photar = np.zeros(ne)
    for i in range(1, ear.size):
        photar[i - 1] = 0.5 * (prim[i] / ear[i] ** 2 + prim[i - 1] / ear[i - 1] ** 2) * (ear[i] - ear[i - 1]) * normfac
    return photar


def _thcompton(tempbb, theta, gamma):
    """Sets up the photon-production and escape-probability arrays and
    dispatches to the tridiagonal Kompaneets solve (`_thermlc`)."""
    tautom = np.sqrt(2.25 + 3.0 / (theta * ((gamma + 0.5) ** 2 - 2.25))) - 1.5

    dphdot = np.zeros(900)
    rel = np.zeros(900)
    c2 = np.zeros(900)
    sptot = np.zeros(900)
    bet = np.zeros(900)
    x = np.zeros(900)

    delta = 0.02
    deltal = delta * np.log(10.0)
    xmin = 1e-4 * tempbb
    xmax = 40.0 * theta
    jmax = min(899, int(np.log10(xmax / xmin) / delta) + 1)

    x[: jmax + 1] = xmin * 10.0 ** (np.arange(jmax + 1) * delta)

    for j in range(jmax):
        w = x[j]
        w1 = np.sqrt(x[j] * x[j + 1])
        c2[j] = w1 ** 4 / (1.0 + 4.6 * w1 + 1.1 * w1 * w1)
        if w <= 0.05:
            rel[j] = 1.0 - 2.0 * w + 26.0 * w * w * 0.2
        else:
            z1 = (1.0 + w) / w ** 3
            z2 = 1.0 + 2.0 * w
            z3 = np.log(z2)
            z4 = 2.0 * w * (1.0 + w) / z2
            z5 = z3 / 2.0 / w
            z6 = (1.0 + 3.0 * w) / z2 / z2
            rel[j] = 0.75 * (z1 * (z4 - z3) + z5 - z6)

    jmaxth = min(900, int(np.log10(50 * tempbb / xmin) / delta))
    jmaxth = min(jmaxth, jmax)
    planck = 15.0 / (np.pi * tempbb) ** 4
    dphdot[:jmaxth] = planck * x[:jmaxth] ** 2 / (np.exp(x[:jmaxth] / tempbb) - 1)

    jnr = min(int(np.log10(0.10 / xmin) / delta + 1), jmax - 1)
    jrel = min(int(np.log10(1 / xmin) / delta + 1), jmax)
    xnr = x[jnr - 1]
    xr = x[jrel - 1]
    for j in range(jnr - 1):
        taukn = tautom * rel[j]
        bet[j] = 1.0 / tautom / (1.0 + taukn / 3.0)
    for j in range(jnr - 1, jrel):
        taukn = tautom * rel[j]
        arg = (x[j] - xnr) / (xr - xnr)
        flz = 1 - arg
        bet[j] = 1.0 / tautom / (1.0 + taukn / 3.0 * flz)
    for j in range(jrel, jmax):
        bet[j] = 1.0 / tautom

    dphesc = _thermlc(tautom, theta, deltal, x, jmax, dphdot, bet, c2)

    for j in range(jmax - 1):
        sptot[j] = dphesc[j] * x[j] ** 2

    return x, jmax, sptot


def _thermlc(tautom, theta, deltal, x, jmax, dphdot, bet, c2):
    """Tridiagonal-matrix solve of the (nonrelativistic, Klein-Nishina-
    corrected) Kompaneets equation for the escaping photon occupation
    number, following \\citet{ZdziarskiJohnsonMagdziarz1996}'s Sec. 2."""
    dphesc = np.zeros(900)
    a = np.zeros(900)
    b = np.zeros(900)
    c = np.zeros(900)
    d = np.zeros(900)
    alp = np.zeros(900)
    u = np.zeros(900)
    g = np.zeros(900)
    gam = np.zeros(900)

    c20 = tautom / deltal

    for j in range(1, jmax - 1):
        w1 = np.sqrt(x[j] * x[j + 1])
        w2 = np.sqrt(x[j - 1] * x[j])
        a[j] = -c20 * c2[j] * (theta / deltal / w1 + 0.5)
        t1 = -c20 * c2[j] * (0.5 - theta / deltal / w1)
        t2 = c20 * c2[j - 1] * (theta / deltal / w2 + 0.5)
        t3 = x[j] ** 3 * (tautom * bet[j])
        b[j] = t1 + t2 + t3
        c[j] = c20 * c2[j - 1] * (0.5 - theta / deltal / w2)
        d[j] = x[j] * dphdot[j]

    x32 = np.sqrt(x[0] * x[1])
    aa = (theta / deltal / x32 + 0.5) / (theta / deltal / x32 - 0.5)

    u[jmax - 1] = 0.0

    alp[1] = b[1] + c[1] * aa
    gam[1] = a[1] / alp[1]
    for j in range(2, jmax - 1):
        alp[j] = b[j] - c[j] * gam[j - 1]
        gam[j] = a[j] / alp[j]
    g[1] = d[1] / alp[1]
    for j in range(2, jmax - 2):
        g[j] = (d[j] - c[j] * g[j - 1]) / alp[j]
    g[jmax - 2] = (d[jmax - 2] - a[jmax - 2] * u[jmax - 1] - c[jmax - 2] * g[jmax - 3]) / alp[jmax - 2]
    u[jmax - 2] = g[jmax - 2]
    for j in range(2, jmax + 1):
        jj = jmax - j
        u[jj] = g[jj] - gam[jj] * u[jj + 1]
    u[0] = aa * u[1]

    dphesc[:jmax] = x[:jmax] * x[:jmax] * u[:jmax] * bet[:jmax] * tautom
    return dphesc
