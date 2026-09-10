"""
AGNSED's radial disk/warm-Comptonization/hot-corona structure solve (Kubota
& Done 2018, MNRAS 480, 1247; arXiv:1804.00171 -- KD18). Faithfully
translated from the real HEASARC-distributed `agnsed.f`/`qsosed.f` (not a
third-party Python port), and validated directly against the compiled
Fortran (unmodified, `gfortran -std=legacy`):

- Geometry quantities (r_hot, r_warm, r_out, Gamma_hot, L_diss,hot, L_hot)
  reproduce the compiled reference to 4+ decimal places across every case
  tested, including a Tier-2 `hard_xray_luminosity_fraction` sweep the
  official model itself never runs (general `agnsed.f` never derives r_hot
  from a target luminosity at all -- only `qsosed.f` does, hardcoded to
  0.02; see `solve_geometry`'s docstring).
- Along the way, a real bug was found and precisely diagnosed in the
  official Fortran's OWN diagnostic print statements (not in its physics):
  `lumipl` is correctly divided by 4*pi*d^2 late in `amydiskf`/`qsoamydiskf`
  (needed to flux-normalize the corona's Comptonized shape for the final
  output spectrum), then that same now-overwritten variable is reused a few
  lines later in the T(R_hot)/T(R_warm) print statements, which expect a
  true luminosity, not a flux -- confirmed by exactly reproducing the
  Fortran's printed (slightly wrong) number when this variable-reuse is
  deliberately replicated. KD18's own published Table 2 T(R_hot)/T(R_warm)
  values likely inherited this same print-only artifact; it does not affect
  the actual spectrum, and this module always uses the true, undivided
  luminosity (matching what the real per-annulus spectral synthesis in
  `spectrum.py` uses).
- The spectral assembly built on top of this geometry (see `continuum.py`)
  independently reproduces KD18's own Figure 9 (M=1e8 Msun, mdot=0.05 and
  0.5, E^2 N(E) at 100 Mpc, i=45deg) to ~1% once compared on equal footing.
- With `reprocess=True` (the default), the disc+warm+hot spectrum's total
  integrated luminosity exceeds mdot*L_Edd by ~9-13% (2026-09-09 staged
  diagnostic, `scripts/benchmark_quasar_continuum_resolution.py`'s own
  findings section has the numbers) -- confirmed NOT a numerical bug: each
  zone's synthesis independently reproduces its own true local
  Novikov-Thorne-emissivity integral to <0.02%, and the excess is
  reproduced almost exactly (1.088 measured vs. 1.088 predicted) just by
  summing each zone's TRUE reprocessed target directly, with no
  Comptonization/binning machinery involved at all. The real mechanism:
  `L_hot` (`lumipl_erg_s`) already includes seed photons intercepted
  *from* the disc/warm zones (KD18 eq. 1); with reprocessing on, that same
  `L_hot` then illuminates *back* onto the disc/warm zones (KD18 eq. 5,
  `Frep`), boosting their own local luminosity a second time with energy
  that was already counted once. This is a real feature of how KD18
  define each zone's "local luminosity" once reprocessing redistributes
  energy between zones, faithfully reproduced from the official Fortran
  (which does the identical calculation) -- not an error introduced by
  this reimplementation.

  Two distinct, separately-confirmed mechanisms contribute, not one:
  (1) `L_seed` (the corona intercepting ambient disc/warm photons as
  Comptonization seed photons) is *always* added to `L_hot`'s tally
  regardless of `reprocess`, without ever being subtracted from the disc/
  warm zones' own emitted luminosity -- confirmed by disabling
  reprocessing entirely and still finding a real, non-zero, resolution-
  independent excess (~3-4% in a fiducial M=1e8/mdot=0.10 case, matching
  `L_seed`/target almost exactly). (2) The `Frep` illumination boost
  itself, only active when `reprocess=True`, is the dominant remaining
  contribution to the full ~9-13% figure on top of mechanism (1)'s
  baseline. Neither is a numerical bug -- both are inherent to how the
  official model defines "local luminosity" once any cross-zone energy
  exchange (interception or illumination) is included; the model's actual
  energy-conservation guarantee lives at the level of the intrinsic
  Novikov-Thorne dissipation integral alone (confirmed to sum to
  mdot*L_Edd to <0.1%, limited only by the radial grid's own Riemann-sum
  precision), not at the level of each zone's fully-dressed emitted
  luminosity once seed-photon interception and reprocessing are folded in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import novikov_thorne as nt

IMAX = 2000  # matches agnsed.f/qsosed.f exactly -- see module docstring

# Fixed at the values agnsed.f itself hardwires / KD18's own Table 3 "full
# model" defaults use. Neither is exposed as a free parameter even by
# general AGNSED's own 15-parameter interface (albedo is not among
# agnsed.f's Table A1 parameters at all -- "In the public version of
# agnsed, albedo is fixed at a=0.3" per the model's own README); Htmax is
# described in KD18 Sec. 2.3 itself as "the simplest possible assumption"
# and "a tuning parameter for alternative geometries," not a measured
# physical quantity -- neither belongs in the Tier 2/3 NPE-parameter
# registry as a result (HANDOFF3 Sec. 5.2's "genuine property of the
# generative model, not the astrophysical source" carve-out).
ALBEDO = 0.30
HT_MAX = 100.0


def _log_r_grid(r_lo, r_hi, n):
    """Log-spaced, bin-midpoint radial grid -- same construction as the
    Fortran's own `r = 10**(log10(r_lo) + (i-1)*dlogr + dlogr/2)` loop."""
    dlogr = np.log10(r_hi / r_lo) / n
    i = np.arange(1, n + 1)
    logr = np.log10(r_lo) + (i - 1) * dlogr + dlogr / 2.0
    r = 10.0 ** logr
    dr = 10.0 ** (logr + dlogr / 2.0) - 10.0 ** (logr - dlogr / 2.0)
    return r, dr


@dataclass(frozen=True)
class AGNSEDGeometry:
    """The solved radial structure and derived energetics for one AGNSED
    draw -- everything the spectral synthesis in `spectrum.py` needs that
    does not itself depend on kTe_hot/kTe_warm/Gamma_warm (those only shape
    the Comptonization *spectra*, not the energetics/radii)."""

    m_msun: float
    astar: float
    mdot_edd: float
    rms: float
    eff: float
    mdot_gs: float
    ledd: float
    r_hot: float
    r_warm: float
    r_out: float
    displ_erg_s: float  # L_diss,hot: intrinsic dissipation from ISCO to r_hot (KD18 eq. 2)
    lumipl_erg_s: float  # L_hot = L_diss,hot + L_seed (KD18 eq. 1)
    gamma_hot: float
    t_hot_k: float
    t_warm_k: float
    t_out_k: float


def solve_geometry(
    m_msun: float,
    astar: float,
    mdot_edd: float,
    hard_xray_luminosity_fraction: float,
    *,
    r_warm_over_r_hot: float = 2.0,
    alpha: float = 0.1,
    reprocess: bool = True,
) -> AGNSEDGeometry:
    """Solve for r_hot given a TARGET L_diss,hot/L_Edd, then derive
    r_warm/r_out/the characteristic temperatures/Gamma_hot exactly as
    KD18's own Fortran does.

    `hard_xray_luminosity_fraction` generalizes what only `qsoamydiskf`
    (QSOSED) does in the official model -- general `agnsed.f` takes r_hot
    directly as a free parameter (par10) and never performs this inversion
    at all. QSOSED hardcodes the target at 0.02; here it is a real Tier-2
    NPE-parameter (registry: `quasar_continuum.agnsed.hard_xray_luminosity_fraction`),
    citing KD18 Sec. 4.2's own measured 0.02-0.04 L_Edd range across their
    three fitted AGN and \\citet{JinWardDone2012a}'s 50-object sample
    (factor 2-3 scatter, referenced in KD18 Sec. 4.2).

    The inversion itself is mechanically simple and copied directly from
    `qsoamydiskf`: walk outward from the ISCO in `IMAX` log-spaced steps,
    accumulate the disk-luminosity integral (KD18 eq. 2) as you go, and
    freeze r_hot the moment the running total first reaches the target --
    a forward accumulation with a stopping rule, not a root-finder.
    """
    rms = nt.isco(astar)
    eff = nt.efficiency(astar)
    mdot_gs = nt.mass_accretion_rate_gs(m_msun, mdot_edd, eff)
    ledd = nt.eddington_luminosity(m_msun)
    r_out = nt.self_gravity_radius(m_msun, mdot_edd, alpha)
    rgcm = nt.RG_CM_PER_MSUN * m_msun

    target_l_diss_hot = hard_xray_luminosity_fraction * ledd

    # --- pass 1: forward-accumulate disk luminosity from the ISCO outward,
    # freeze r_hot the moment the running total first reaches the target
    # (KD18 eq. 2; `qsoamydiskf`'s own r_hot-search loop). ---
    r_grid, dr_grid = _log_r_grid(rms, r_out, IMAX)
    t4 = nt.nt_temperature4(m_msun, astar, mdot_gs, rms, r_grid)
    per_annulus_l = 2 * 2 * np.pi * r_grid * dr_grid * rgcm ** 2 * nt.SIGMA_SB_CGS * t4
    cum_l = np.cumsum(per_annulus_l)
    idx = min(int(np.searchsorted(cum_l, target_l_diss_hot)), IMAX - 1)
    r_hot = r_grid[idx]
    displ = cum_l[idx]  # L_diss,hot actually realized at this discretization

    r_hot = min(max(r_hot, rms), r_out)
    r_warm = min(r_warm_over_r_hot * r_hot, r_out)
    ht = min(r_hot, HT_MAX)

    # --- pass 2: seed-photon luminosity intercepted by the hot flow from
    # r_hot outward, reprocessing sourced from the FROZEN `displ` above
    # (matches qsoamydiskf: lumipl is not yet defined at this point). ---
    rep_flag = 1.0 if reprocess else 0.0
    mask_outer = r_grid >= r_hot
    if reprocess:
        trepdis4 = nt.nt_temperature4_reprocessed(
            m_msun, astar, mdot_gs, rms, r_grid[mask_outer], rep_flag, displ, ht, ALBEDO
        )
    else:
        trepdis4 = nt.nt_temperature4(m_msun, astar, mdot_gs, rms, r_grid[mask_outer])
    theta0 = np.arcsin(np.clip(ht / r_grid[mask_outer], -1.0, 1.0))
    covering = (theta0 - 0.5 * np.sin(2 * theta0)) / np.pi
    seeddis = np.sum(
        2 * 2 * np.pi * r_grid[mask_outer] * dr_grid[mask_outer] * rgcm ** 2 * nt.SIGMA_SB_CGS * trepdis4 * covering
    )

    lumipl = displ + seeddis  # L_hot, KD18 eq. 1

    # --- Gamma_hot, KD18 eq. 6 (Beloborodov 1999) ---
    gamma_hot = 7.0 / 3.0 * ((displ + seeddis) / seeddis - 1.0) ** (-0.10)

    # --- characteristic temperatures, using the FINAL (undivided) lumipl --
    # see module docstring re: the diagnostic-print-only bug in the
    # official Fortran that this deliberately does NOT replicate. ---
    t_hot4 = nt.nt_temperature4_reprocessed(m_msun, astar, mdot_gs, rms, r_hot, rep_flag, lumipl, ht, ALBEDO)
    t_warm4 = nt.nt_temperature4_reprocessed(m_msun, astar, mdot_gs, rms, r_warm, rep_flag, lumipl, ht, ALBEDO)
    t_out = nt.nt_temperature4(m_msun, astar, mdot_gs, rms, r_out) ** 0.25

    return AGNSEDGeometry(
        m_msun=m_msun, astar=astar, mdot_edd=mdot_edd, rms=rms, eff=eff, mdot_gs=mdot_gs, ledd=ledd,
        r_hot=r_hot, r_warm=r_warm, r_out=r_out,
        displ_erg_s=displ, lumipl_erg_s=lumipl, gamma_hot=gamma_hot,
        t_hot_k=t_hot4 ** 0.25, t_warm_k=t_warm4 ** 0.25, t_out_k=t_out,
    )
