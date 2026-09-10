"""
QuasarContinuum -- the AGN accretion-disk continuum for the quasar mock path
(the `continuum` bucket's AGN contribution, additive alongside
`galaxy_continuum.GalaxyContinuum`'s stellar contribution; blending the two
into one composite continuum is a separate, not-yet-built channel).

Full AGNSED spectral assembly (Kubota & Done 2018, MNRAS 480, 1247;
arXiv:1804.00171 -- KD18): disk blackbody (r_warm..r_out) + per-annulus warm
Comptonization (r_hot..r_warm, each annulus individually shape-matched to
its own local disk luminosity, exactly as agnsed.f/qsosed.f do) + a single
hot-corona Comptonization (renormalized to L_hot). Faithful to the official
Fortran's structure (`geometry.py`'s module docstring has the full
validation summary: geometry exact to 4+ decimal places against the
compiled reference, each zone's own synthesis exact to <0.02% against its
own true local Novikov-Thorne target, and the full spectral shape/
normalization within ~1-2% of KD18's own published Figure 9 once compared
on equal footing -- same M/mdot, same i=45deg inclination, same 100 Mpc
fiducial distance).

One further, deliberate departure from AGNSED's own Fortran (2026-09-09,
see `geometry.py`'s module docstring for the full diagnosis): the disc/warm
synthesis loops below reduce each annulus's emitted luminosity by
`corona_covering_fraction()` -- the same fraction `geometry.py` already
treats as intercepted by the corona for `L_seed` -- to fix a real
energy-bookkeeping error in the official Fortran's own output spectrum
(that same intercepted energy was being counted twice: once as
directly-escaping disc/warm light, again via the corona's `L_hot`). This
forfeits exact bit-for-bit parity with the compiled Fortran's spectrum in
exchange for a total luminosity that self-consistently tracks mdot*L_Edd.

One deliberate departure from the reference Fortran: everything here is
built in pure specific-luminosity space (no assumed distance, no 4*pi*d^2
division anywhere) so the output matches `galaxy_continuum.GalaxyContinuum`'s
own convention directly -- L_nu [Lsun/Hz] by default, L_lambda [Lsun/Angstrom]
via `peraa=True` (same conversion formula/constant as
`galaxy_continuum.continuum._fnu_to_flambda`), rest-frame, no distance
dependence -- rather than going through a flux-at-distance detour. That
detour is exactly what caused a real, confirmed normalization bug in a
third-party Python port of QSOSED investigated (and rejected as a
dependency) before this module was written: it silently dropped the
`cosi` inclination parameter and mixed up a further disk-specific
geometric factor. See project memory / HANDOFF4 for that investigation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import comptonization as comp
from . import novikov_thorne as nt
from .geometry import ALBEDO, HT_MAX, AGNSEDGeometry, corona_covering_fraction, solve_geometry
from ..parameters.samplers import ParameterSampler, PriorSampler

H_ERG_S = 6.62617e-27  # erg s, agnsed.f's own literal (not modern CODATA) -- kept for exact parity
KKEV = 1.16048e7  # K per keV
KEVHZ = 2.417965e17  # Hz per keV, agnsed.f's own literal -- reused for the final Hz/Angstrom display
# conversion too (rather than mixing in a separately-sourced "modern precise" constant), so the
# energy grid this module actually computed on stays internally consistent end to end.
KEV_TO_ERG = 1.602176634e-9  # exact (2019 SI redefinition: 1 eV = 1.602176634e-19 J)
L_SUN_ERG_S = 3.828e33  # IAU 2015 nominal solar luminosity (B3 resolution)
_C_ANGSTROM_PER_S = 2.99792458e18  # identical to galaxy_continuum.continuum's own constant -- no astropy dependency

_REGISTRY_PARAMETER_NAMES = {
    "m_msun": "quasar_continuum.agnsed.black_hole_mass",
    "mdot_edd": "quasar_continuum.agnsed.eddington_ratio",
    "astar": "quasar_continuum.agnsed.spin",
    "cosi": "quasar_continuum.agnsed.cosi",
    "hard_xray_luminosity_fraction": "quasar_continuum.agnsed.hard_xray_luminosity_fraction",
    "kte_hot_kev": "quasar_continuum.agnsed.kte_hot",
    "kte_warm_kev": "quasar_continuum.agnsed.kte_warm",
    "gamma_warm": "quasar_continuum.agnsed.gamma_warm",
    "r_warm_over_r_hot": "quasar_continuum.agnsed.r_warm_over_r_hot",
}


def _log_r_grid_two_zone(r_lo, r_hi, n, r_lo2=None, n2=None):
    """Log-spaced, bin-midpoint radial grid; a two-zone version (icor points
    from r_lo to r_lo2, iout points from r_lo2 to r_hi) when r_lo2/n2 are
    given, matching agnsed.f's own icor+iout structure."""
    if r_lo2 is None:
        dlogr = np.log10(r_hi / r_lo) / n
        i = np.arange(1, n + 1)
        logr = np.log10(r_lo) + (i - 1) * dlogr + dlogr / 2.0
        r = 10.0 ** logr
        dr = 10.0 ** (logr + dlogr / 2.0) - 10.0 ** (logr - dlogr / 2.0)
        return r, dr

    dlogr1 = np.log10(r_lo2 / r_lo) / n
    i1 = np.arange(1, n + 1)
    logr1 = np.log10(r_lo) + (i1 - 1) * dlogr1 + dlogr1 / 2.0
    r1 = 10.0 ** logr1
    dr1 = 10.0 ** (logr1 + dlogr1 / 2.0) - 10.0 ** (logr1 - dlogr1 / 2.0)

    dlogr2 = np.log10(r_hi / r_lo2) / n2
    i2 = np.arange(1, n2 + 1)
    logr2 = np.log10(r_lo2) + (i2 - 1) * dlogr2 + dlogr2 / 2.0
    r2 = 10.0 ** logr2
    dr2 = 10.0 ** (logr2 + dlogr2 / 2.0) - 10.0 ** (logr2 - dlogr2 / 2.0)

    return np.concatenate([r1, r2]), np.concatenate([dr1, dr2])


def _synthesize_photon_rates(geom: AGNSEDGeometry, ear_kev, kte_hot_kev, kte_warm_kev, gamma_warm, icor, iout, reprocess=True):
    """Returns (disk_rate, warm_rate, hot_rate), each a per-bin photon rate
    [photons/s] (luminosity-domain, no distance, `cosi` NOT yet applied --
    the caller applies `cosi/0.5` to disk+warm, none to hot, per KD18's
    Lambertian-disc-vs-isotropic-corona geometry). `reprocess` must match
    whatever was passed to `solve_geometry` for this `geom` -- otherwise
    the disc/warm temperature profile here (reprocessed or not) would be
    inconsistent with `geom.lumipl_erg_s`/`geom.t_hot_k` (computed under
    the OTHER setting), a real bug this project's own energy-conservation
    diagnostic (2026-09-09) caught: this function used to hardcode
    rep_flag=1.0 regardless of what `reprocess` value `from_parameters`
    was actually given."""
    ne = ear_kev.size - 1
    rgcm = nt.RG_CM_PER_MSUN * geom.m_msun
    rep_flag = 1.0 if reprocess else 0.0
    en_mid = np.sqrt(ear_kev[:-1] * ear_kev[1:])
    bin_width_hz = (ear_kev[1:] - ear_kev[:-1]) * KEVHZ

    # ---- disk blackbody, r_warm .. r_out ----
    r_disk, dr_disk = _log_r_grid_two_zone(geom.r_warm, geom.r_out, iout)
    t_disk4 = nt.nt_temperature4_reprocessed(
        geom.m_msun, geom.astar, geom.mdot_gs, geom.rms, r_disk, rep_flag, geom.lumipl_erg_s,
        min(geom.r_hot, HT_MAX), ALBEDO
    )
    t_disk_kev = t_disk4 ** 0.25 / KKEV
    disk_escaping = 1.0 - corona_covering_fraction(r_disk, geom.r_hot)

    disk_rate = np.zeros(ne)
    for r, dr, tkev, escaping in zip(r_disk, dr_disk, t_disk_kev, disk_escaping):
        area_term = 4.0 * np.pi * r * dr * rgcm ** 2 * escaping
        with np.errstate(over="ignore"):
            occ = 1.0 / (np.exp(np.clip(en_mid / tkev, None, 700)) - 1.0)
        dflux = np.pi * 2.0 * H_ERG_S * (en_mid * KEVHZ) ** 3 / 8.98755e20 * area_term * occ
        dflux = np.where(en_mid < 30.0 * tkev, dflux, 0.0)
        disk_rate += dflux / (H_ERG_S * en_mid * KEVHZ) * bin_width_hz

    # ---- warm Comptonization, r_hot .. r_warm, per-annulus shape+renormalize ----
    r_warm_grid, dr_warm_grid = _log_r_grid_two_zone(geom.r_hot, geom.r_warm, icor)
    t_warm4 = nt.nt_temperature4_reprocessed(
        geom.m_msun, geom.astar, geom.mdot_gs, geom.rms, r_warm_grid, rep_flag, geom.lumipl_erg_s,
        min(geom.r_hot, HT_MAX), ALBEDO
    )
    t_warm_kev_local = t_warm4 ** 0.25 / KKEV
    warm_escaping = 1.0 - corona_covering_fraction(r_warm_grid, geom.r_hot)

    warm_rate = np.zeros(ne)
    for r, dr, tkev, escaping in zip(r_warm_grid, dr_warm_grid, t_warm_kev_local, warm_escaping):
        shape = comp.donthcomp(ear_kev, gamma_warm, kte_warm_kev, tkev)
        dllth = np.sum(shape * en_mid * KEVHZ * H_ERG_S)
        d_l_annulus = 2 * 2 * np.pi * r * dr * rgcm ** 2 * nt.SIGMA_SB_CGS * (tkev * KKEV) ** 4 * escaping
        if dllth > 0:
            warm_rate += shape * (d_l_annulus / dllth)

    # ---- hot corona, single Comptonization call, renormalized to L_hot ----
    t0_kev = geom.t_hot_k / KKEV
    hot_shape = comp.donthcomp(ear_kev, geom.gamma_hot, kte_hot_kev, t0_kev)
    pow_ = np.sum(hot_shape * en_mid * KEVHZ * H_ERG_S)
    hot_rate = hot_shape * (geom.lumipl_erg_s / pow_) if pow_ > 0 else np.zeros(ne)

    return disk_rate, warm_rate, hot_rate, en_mid


@dataclass(frozen=True)
class QuasarContinuum:
    wave: np.ndarray
    flux: np.ndarray
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_parameters(
        cls,
        m_msun: float,
        astar: float,
        mdot_edd: float,
        cosi: float,
        hard_xray_luminosity_fraction: float,
        kte_hot_kev: float,
        kte_warm_kev: float,
        gamma_warm: float,
        *,
        r_warm_over_r_hot: float = 2.0,
        n_energy_bins: int = 200,
        icor: int = 20,
        iout: int = 400,
        reprocess: bool = True,
        peraa: bool = False,
    ) -> "QuasarContinuum":
        """Directly-specified construction path (mirrors
        `GalaxyContinuum.from_single_population`/`from_arrays`'s role) --
        every physical input given explicitly, no NPE sampling. This is the
        path the test suite's regression checks against the compiled
        `agnsed.f`/`qsosed.f` reference use.
        """
        ear_kev = np.geomspace(1e-4, 200.0, n_energy_bins + 1)
        geom = solve_geometry(
            m_msun, astar, mdot_edd, hard_xray_luminosity_fraction,
            r_warm_over_r_hot=r_warm_over_r_hot, reprocess=reprocess,
        )
        disk_rate, warm_rate, hot_rate, en_mid = _synthesize_photon_rates(
            geom, ear_kev, kte_hot_kev, kte_warm_kev, gamma_warm, icor, iout, reprocess=reprocess
        )

        cosi_scale = cosi / 0.5  # Lambertian disc/warm-region geometry; hot corona is isotropic (KD18 Sec. 2.1/2.2)
        total_rate = disk_rate * cosi_scale + warm_rate * cosi_scale + hot_rate

        energy_rate_erg_s = total_rate * (en_mid * KEV_TO_ERG)
        freq_edges_hz = ear_kev * KEVHZ
        freq_bin_width_hz = np.diff(freq_edges_hz)
        l_nu_lsun_hz = (energy_rate_erg_s / freq_bin_width_hz) / L_SUN_ERG_S

        wave_aa = _C_ANGSTROM_PER_S / (en_mid * KEVHZ)
        order = np.argsort(wave_aa)
        wave = wave_aa[order]
        flux_nu = l_nu_lsun_hz[order]
        flux = flux_nu * _C_ANGSTROM_PER_S / wave ** 2 if peraa else flux_nu

        return cls(
            wave=wave,
            flux=flux,
            meta=dict(
                geometry=geom,
                disk_rate_unscaled=disk_rate,
                warm_rate_unscaled=warm_rate,
                hot_rate=hot_rate,
                energy_kev=en_mid,
                cosi=cosi,
                kte_hot_kev=kte_hot_kev,
                kte_warm_kev=kte_warm_kev,
                gamma_warm=gamma_warm,
                r_warm_over_r_hot=r_warm_over_r_hot,
            ),
        )

    @classmethod
    def from_agnsed(
        cls,
        rng: np.random.Generator,
        *,
        sampler: Optional[ParameterSampler] = None,
        n_energy_bins: int = 600,
        icor: int = 20,
        iout: int = 400,
        reprocess: bool = True,
        peraa: bool = False,
    ) -> "QuasarContinuum":
        """The Tier-2/3 NPE-parameter draw path (mirrors
        `GalaxyContinuum.from_dense_basis`'s role): draws every registered
        `quasar_continuum.agnsed.*` parameter from its default prior (or a
        trained NPE, once one exists) and synthesizes the resulting AGNSED
        continuum.
        """
        if sampler is None:
            sampler = PriorSampler()

        drawn = sampler.sample(list(_REGISTRY_PARAMETER_NAMES.values()), rng=rng)
        kwargs = {local_name: drawn[reg_name] for local_name, reg_name in _REGISTRY_PARAMETER_NAMES.items()}

        result = cls.from_parameters(
            **kwargs,
            n_energy_bins=n_energy_bins, icor=icor, iout=iout, reprocess=reprocess, peraa=peraa,
        )
        result.meta["drawn_parameters"] = drawn
        return result
