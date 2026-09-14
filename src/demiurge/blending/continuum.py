"""
Composes `galaxy_continuum.GalaxyContinuum` and `quasar_continuum.QuasarContinuum`
into one composite continuum, weighted by `blending.quasar_frac` -- the
fraction of BOLOMETRIC luminosity associated with the quasar's accretion
physics/SED (each channel's own native, full-spectrum integral), not a
reference-wavelength flux ratio.

This generalizes an earlier, simpler proposal (`quasar_frac*quasar +
(1-quasar_frac)*galaxy`, applied directly to each channel's raw output) that
implicitly assumed both channels were already on a common flux scale (PI
direction, 2026-09-10). They are not: each is independently normalized from
its own real physical parameters with no imposed relation between the two
(galaxy: `galaxy_continuum.sfh.total_stellar_mass`; quasar:
`quasar_continuum.agnsed.black_hole_mass` x `eddington_ratio`, tied to
mdot*L_Edd) -- summing raw outputs would give whatever ratio those two
independent draws happen to produce, not a user-dialable fraction.

Mechanism -- symmetric total-preserving rescale:

    L_bol_gal = integral of galaxy.flux over galaxy's own native grid
    L_bol_qso = integral of quasar.flux (BEFORE torus/host-disk reddening)
                over quasar's own native grid
    L_bol_total = L_bol_gal + L_bol_qso
    c_gal = (1 - quasar_frac) * L_bol_total / L_bol_gal
    c_qso = quasar_frac * L_bol_total / L_bol_qso
    flux_out(wave) = c_gal * flux_galaxy(wave)
                      + c_qso * [flux_quasar(wave) * T_torus(wave) * T_hostdisk(wave)]

`L_bol_qso` uses the INTRINSIC (pre-reddening) quasar flux deliberately:
`quasar_frac` describes the underlying accretion-power budget, not how
obscured this particular mock's view of it happens to be. Torus-local and
host-disk-local reddening (both optional) are applied AFTER the rescale,
only to how much of the (already-rescaled) quasar light actually reaches
the observer -- reddening never perturbs the `quasar_frac` accounting
itself.

Both bolometric integrals are self-contained (each over its own channel's
own native wavelength grid) -- no cross-channel grid alignment needed for
that part. Grid alignment is only needed for the final composite flux
array: the output `wave` is galaxy's own native grid, clipped to its
overlap with quasar's own native range (both channels' output is only
physically meaningful within their own actually-synthesized domain --
extrapolating either channel past its own native range would not be).

By construction (verified, not just asserted -- see
`tests/blending/test_continuum.py`), the composite's ACTUAL bolometric
fraction equals `quasar_frac` exactly at every value in [0, 1], not only at
the edges -- `c_gal`/`c_qso` are solved directly from that requirement.
This is well-posed (no singularity) across the full range, given both
channels have positive native bolometric luminosity (guaranteed for any
physically valid mock). One real, deliberately-flagged consequence of this
choice, worth knowing before reading too much into either edge case:
`quasar_frac=0` gives EXACTLY ZERO quasar contribution (`c_qso=0`), but the
galaxy component is rescaled to `L_bol_total` (= `L_bol_gal + L_bol_qso`),
NOT bit-for-bit its own untouched native output -- symmetrically,
`quasar_frac=1` gives exactly zero galaxy contribution but rescales the
quasar component the same way. No scheme can give bit-for-bit untouched
single-channel output at BOTH edges while also staying singularity-free
across the whole range and keeping `quasar_frac` numerically accurate at
every intermediate value (holding one channel fixed to its own native
scale, instead, moves the singularity to the OTHER edge -- verified by
direct construction, not assumed). Absolute brightness is not the point
here regardless (per PI direction: the composite is expected to be
renormalized for magnitude invariance downstream), only composition is.

Full provenance is kept, not silently discarded: `meta` carries both
channels' own untouched `meta` (drawn parameters, native flux), the
bolometric luminosities and rescale factors, and the torus/host-disk
reddening draws (if given) -- so a consumer needing the intrinsic
per-channel SEDs and sampled geometry (e.g. a future ionization emulator)
can recover them without re-generating anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..galaxy_continuum.continuum import GalaxyContinuum
from ..quasar_continuum.continuum import QuasarContinuum
from ..quasar_continuum.host_disk_reddening import HostDiskReddeningResult
from ..quasar_continuum.torus_reddening import TorusReddeningResult

# Identical convention/constant used throughout this project (galaxy_continuum
# and quasar_continuum's own continuum.py modules each define this locally
# rather than depending on astropy -- see their module docstrings).
_C_ANGSTROM_PER_S = 2.99792458e18


def _bolometric_luminosity(wave: np.ndarray, flux_fnu: np.ndarray) -> float:
    """L_bol = integral of L_nu dnu = integral of L_lambda dlambda (the same
    physical quantity, whichever variable is integrated over) -- computed
    via the L_lambda form since `wave` (ascending) is already the natural
    integration variable for both channels' own native grids, and both
    channels' default output convention is L_nu [Lsun/Hz]."""
    flux_flambda = flux_fnu * _C_ANGSTROM_PER_S / wave**2
    return float(np.trapezoid(flux_flambda, wave))


@dataclass(frozen=True)
class CompositeContinuum:
    wave: np.ndarray
    flux: np.ndarray
    meta: dict = field(default_factory=dict)


def blend_continua(
    galaxy: GalaxyContinuum,
    quasar: QuasarContinuum,
    quasar_frac: float,
    *,
    torus: Optional[TorusReddeningResult] = None,
    host_disk: Optional[HostDiskReddeningResult] = None,
) -> CompositeContinuum:
    """Explicit, deterministic composition path -- no sampling, every input
    already generated (mirrors `GalaxyContinuum.from_arrays`/
    `QuasarContinuum.from_parameters`'s explicit-inputs convention rather
    than a sampler-driven one). `torus`/`host_disk` are optional: omitting
    either applies unit transmission for that screen (no reddening), not an
    error -- a caller may legitimately want only one, or neither, applied.

    Both `galaxy.flux` and `quasar.flux` must be in the same convention
    (each channel's shared default, L_nu [Lsun/Hz] -- i.e. both built with
    `peraa=False`, the default for both). This is a precondition, not
    checked here, matching this project's convention of trusting internal
    callers rather than validating at non-boundary call sites.
    """
    if not (0.0 <= quasar_frac <= 1.0):
        raise ValueError(f"quasar_frac must be in [0, 1], got {quasar_frac}")

    l_bol_gal = _bolometric_luminosity(galaxy.wave, galaxy.flux)
    l_bol_qso = _bolometric_luminosity(quasar.wave, quasar.flux)
    if l_bol_gal <= 0.0 or l_bol_qso <= 0.0:
        raise ValueError(
            "blend_continua requires a positive bolometric luminosity from both channels, "
            f"got L_bol_gal={l_bol_gal}, L_bol_qso={l_bol_qso}"
        )
    l_bol_total = l_bol_gal + l_bol_qso

    c_gal = (1.0 - quasar_frac) * l_bol_total / l_bol_gal
    c_qso = quasar_frac * l_bol_total / l_bol_qso

    lo = max(galaxy.wave.min(), quasar.wave.min())
    hi = min(galaxy.wave.max(), quasar.wave.max())
    mask = (galaxy.wave >= lo) & (galaxy.wave <= hi)
    wave = galaxy.wave[mask]

    flux_quasar_reddened = quasar.flux
    if torus is not None:
        flux_quasar_reddened = flux_quasar_reddened * torus.transmission(quasar.wave)
    if host_disk is not None:
        flux_quasar_reddened = flux_quasar_reddened * host_disk.transmission(quasar.wave)

    flux_galaxy_on_grid = galaxy.flux[mask]
    flux_quasar_on_grid = np.interp(wave, quasar.wave, flux_quasar_reddened)

    flux_out = c_gal * flux_galaxy_on_grid + c_qso * flux_quasar_on_grid

    meta = dict(
        quasar_frac=quasar_frac,
        l_bol_galaxy=l_bol_gal,
        l_bol_quasar=l_bol_qso,
        l_bol_total=l_bol_total,
        scale_galaxy=c_gal,
        scale_quasar=c_qso,
        galaxy_meta=galaxy.meta,
        quasar_meta=quasar.meta,
        torus_reddening=torus,
        host_disk_reddening=host_disk,
    )
    return CompositeContinuum(wave=wave, flux=flux_out, meta=meta)
