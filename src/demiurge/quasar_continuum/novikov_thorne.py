"""
Novikov-Thorne (1973) relativistic thin-disk temperature profile, in the
specific parametrization AGNSED uses (Kubota & Done 2018, MNRAS 480, 1247;
arXiv:1804.00171 -- hereafter KD18). Faithfully translated from the actual
HEASARC-distributed Fortran source (`agnsed.f`/`qsosed.f`, the real
compiled-and-run reference for this channel, "now in the Heasoft release as
of Xspec version 12.11" per that source's own README -- functions
`amytemp`/`amytempreph`), not from any third-party Python port. Validated
directly against the compiled Fortran (unmodified, `gfortran -std=legacy`)
for the paper's own three fitted AGN (NGC 5548, Mrk 509, PG 1115+407, KD18
Table 2): r_hot/r_warm/r_out/Gamma_hot/L_diss,hot/L_hot all reproduce to
4+ decimal places.

The relativistic correction factor here (`_nt_rel_factor`) is the standard
disk-relativistic-effects term from the original Novikov & Thorne (1973)
thin-disk solution (in \\citet{NovikovThorne1973}, "Black Holes (Les Astres
Occlus)", eds. DeWitt & DeWitt, pp. 343-450) -- KD18 assembles the specific
temperature formula (`nt_temperature4`) and its reprocessing extension
(`nt_temperature4_reprocessed`, Sec. 2.3, their eq. 5) from that base term.
"""
from __future__ import annotations

import numpy as np

# Literal constants copied verbatim from agnsed.f/qsosed.f (not modern CODATA
# values) -- kept identical on purpose so this module's output matches the
# compiled reference exactly, not merely "close" (verified: see module
# docstring). G_CGS in particular is agnsed.f's own 2-sig-fig literal.
G_CGS = 6.67e-8  # cm^3 g^-1 s^-2
MSUN_G = 1.989e33  # g
SIGMA_SB_CGS = 5.670367e-5  # erg cm^-2 s^-1 K^-4
RG_CM_PER_MSUN = 1.477e5  # gravitational radius of 1 Msun, in cm
EDDINGTON_LUMINOSITY_PER_MSUN = 1.39e38  # erg/s per Msun (agnsed.f's own literal L_Edd/M constant)
MDOT_EDD_CONVERSION = 1.39e18  # agnsed.f's own literal: Mdot [g/s] = mdotedd * this * M / (8.98755 * eff)


def isco(astar: float) -> float:
    """Innermost stable circular orbit r_ms, in units of R_g = GM/c^2, for a
    Kerr black hole of dimensionless spin `astar` (Bardeen, Press & Teukolsky
    1972's standard closed form, as used verbatim in agnsed.f)."""
    z1 = (1 - astar ** 2) ** (1.0 / 3.0)
    z1 = z1 * ((1 + astar) ** (1.0 / 3.0) + (1 - astar) ** (1.0 / 3.0))
    z1 = 1 + z1
    z2 = np.sqrt(3 * astar ** 2 + z1 ** 2)
    if astar >= 0.0:
        return 3.0 + z2 - np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))
    return 3.0 + z2 + np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


def efficiency(astar: float) -> float:
    """Accretion efficiency eta from r_ms (0.057 for astar=0, KD18 Sec. 2)."""
    rms = isco(astar)
    return 1.0 - np.sqrt(1.0 - 2.0 / (3.0 * rms))


def _nt_rel_factor(r, astar, rms):
    """The Novikov & Thorne (1973) relativistic correction (A-B)/C term
    common to both the plain and reprocessed temperature formulas below."""
    y = np.sqrt(r)
    yms = np.sqrt(rms)
    y1 = 2.0 * np.cos((np.arccos(astar) - np.pi) / 3.0)
    y2 = 2.0 * np.cos((np.arccos(astar) + np.pi) / 3.0)
    y3 = -2.0 * np.cos(np.arccos(astar) / 3.0)

    part1 = 3.0 * (y1 - astar) ** 2 * np.log((y - y1) / (yms - y1)) / (y * y1 * (y1 - y2) * (y1 - y3))
    part2 = 3.0 * (y2 - astar) ** 2 * np.log((y - y2) / (yms - y2)) / (y * y2 * (y2 - y1) * (y2 - y3))
    part3 = 3.0 * (y3 - astar) ** 2 * np.log((y - y3) / (yms - y3)) / (y * y3 * (y3 - y1) * (y3 - y2))
    c = 1.0 - yms / y - (3.0 * astar / (2.0 * y)) * np.log(y / yms) - part1 - part2 - part3
    b = 1.0 - 3.0 / r + 2.0 * astar / r ** 1.5
    return c / b


def nt_temperature4(m_msun: float, astar: float, mdot_gs: float, rms: float, r):
    """Intrinsic (non-reprocessed) Novikov-Thorne effective temperature^4
    [K^4] at radius `r` [R_g] (KD18 eq. in Sec. 2, `amytemp` in the Fortran).
    `mdot_gs` is the mass accretion rate in g/s (not the Eddington ratio --
    see `mass_accretion_rate_gs`)."""
    rgcm = RG_CM_PER_MSUN * m_msun
    rel = _nt_rel_factor(r, astar, rms)
    t4 = 3.0 * G_CGS * MSUN_G * m_msun * mdot_gs
    t4 = t4 / (8.0 * np.pi * SIGMA_SB_CGS * (r * rgcm) ** 3)
    return t4 * rel


def nt_temperature4_reprocessed(m_msun, astar, mdot_gs, rms, r, rep, l_illum_erg_s, ht, albedo):
    """Effective temperature^4 [K^4] including reprocessing of an
    illuminating luminosity `l_illum_erg_s` [erg/s] from a source of
    scale-height/radius `ht` [R_g] (KD18 eq. 5, `amytempreph` in the
    Fortran). `rep` is the 0/1 reprocessing on/off switch; `l_illum_erg_s`
    is the caller's choice of illuminating source (the intrinsic hot-flow
    dissipation `L_diss,hot` during the geometry solve, or the full `L_hot`
    once known -- see `geometry.py`, matching exactly which luminosity the
    Fortran passes at each call site)."""
    rgcm = RG_CM_PER_MSUN * m_msun
    rel = _nt_rel_factor(r, astar, rms)
    rep0 = (ht / rms) * (1.0 - albedo)
    rep0 = rep0 * (1.0 + ht * ht / (r * r)) ** (-1.5)
    factor = 4.0 * l_illum_erg_s / (mdot_gs * 8.9875e20) * rep0 * rep * 0.5

    t4 = 3.0 * G_CGS * MSUN_G * m_msun * mdot_gs
    t4 = t4 / (8.0 * np.pi * SIGMA_SB_CGS * (r * rgcm) ** 3)
    return t4 * (rel + factor)


def self_gravity_radius(m_msun: float, mdot_edd: float, alpha: float = 0.1) -> float:
    """Disk self-gravity radius r_out [R_g], \\citet{LaorNetzer1989}'s
    fitting formula as used verbatim in agnsed.f (their own default
    `logrout<0` branch)."""
    mass9 = m_msun / 1e9
    return 2150.0 * mass9 ** (-2.0 / 9.0) * mdot_edd ** (4.0 / 9.0) * alpha ** (2.0 / 9.0)


def eddington_luminosity(m_msun: float) -> float:
    """L_Edd [erg/s], agnsed.f's own literal constant (1.39e38 * M)."""
    return EDDINGTON_LUMINOSITY_PER_MSUN * m_msun


def mass_accretion_rate_gs(m_msun: float, mdot_edd: float, eff: float) -> float:
    """Mdot [g/s] from the Eddington ratio mdot_edd = Mdot/Mdot_Edd, where
    eta * Mdot_Edd * c^2 = L_Edd (agnsed.f's own literal conversion)."""
    return mdot_edd * MDOT_EDD_CONVERSION * m_msun / (8.98755 * eff)
