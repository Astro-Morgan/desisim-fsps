"""
desisim.balmer_continuum
==========================

Task #32: the hydrogen recombination cascade's Balmer continuum
(free-bound n=2 recombination) and, for the first time in this fork, a
genuine broad (AGN/BLR-like) component for the H recombination line
series itself (Lyalpha + Balmer n=3 through n=50).

--------------------------------------------------------------------------
Why this exists
--------------------------------------------------------------------------
When a free electron recombines directly onto hydrogen's n=2 level, the
emitted photon carries the electron's (continuous, thermal) kinetic
energy plus n=2's fixed binding energy -- since the kinetic-energy term
is continuous, so is the resulting photon-energy distribution. That is
the Balmer continuum: a genuine physical continuum (unlike Fe II, which
only *looks* continuous because of line crowding), with a sharp edge at
3646A (the series limit) where the opacity/emissivity jumps. This sits in
the same ~2000-4000A "small blue bump" region as Fe II and is routinely
fit jointly with it in the AGN literature (Grandi 1982; Wills, Netzer &
Wills 1985; Dietrich et al. 2002, 2003; Kovacevic, Popovic & Dojcinovic
2014). AGNPowerLawContinuum had none of this physics before this module.

Separately, review of EMSpectrum found that the entire H recombination
cascade table (recombination_lines.ecsv: Lyalpha, Balmer Halpha-H18,
Paschen, Brackett) is narrow-only -- there was no broad (BLR) component
for Hbeta/Halpha/etc. at all, only for the seven unrelated Sec 1.3 lines
([NeIII]/[OIII]4363/HeII4686/[NII]5755/[SII]4068,4076). Real AGN broad-
line spectra have the *entire* recombination cascade in emission at
broad (BLR) widths, not just those seven lines. Per PI direction, this
gap is folded into this task rather than #33 (broad-line coverage)
because the two pieces below share the same underlying physics (the same
n=2 recombination) and, critically, the same (T_e, log n_e) physical
state -- whereas #33's remaining lines (CIV, CIII], SiIV, MgII) are
non-recombination resonance/collisional lines with no analogous
shared-physics shortcut, and stay on the existing independent-per-line-
prior mechanism (NEW_LINE_NAMES/NEW_LINE_PRIORS).

This module does NOT touch EMSpectrum, recombination_lines.ecsv, or any
other production-sensitive GALAXY/QSO code -- it is a standalone,
additive module wired in via mock_spectrum.py/decompose.py, same
precedent as feii_continuum.py (task #31), chosen specifically for
minimal-refactor compatibility with the planned normalizing-flow-driven
redesign (existing narrow-only Halpha/Hbeta from EMSpectrum are
unaffected; this module adds a second, independent broad contribution on
top).

--------------------------------------------------------------------------
Two pieces, one shared physical state
--------------------------------------------------------------------------
(A) Free-bound edge (Grandi 1982), blueward of the 3646A series limit:

        F_lambda^BaC = F_BE * B_lambda(T_e) * [1 - exp(-tau_BE*(lambda/3646)^3)]

    B_lambda(T_e) is the Planck function; tau_BE is the optical depth at
    the edge (its own free parameter -- not derivable from n_e in this
    formalism). F_BE (edge_norm below) sets the absolute scale.

(B) Bound-bound merged high-order line series: Lyalpha plus every Balmer
    line n=3->2 through n=50->2 (n=19-50 did not exist as individually
    tabulated ratios anywhere in this fork before), broadened at the same
    BLR-like velocity width as the edge, so the discrete lines visually
    merge into a smooth tail that connects to the free-bound edge right
    at the series limit -- no separate analytic patching is needed for
    that merge; it falls out of the broadening once nearby line spacing
    drops below the drawn sigma_kms.

Per PI direction, (A) and (B) share one drawn (T_e, log n_e) -- physically
the same recombining gas -- rather than being independent free parameters
the way Fe II's UV/optical bands are (Fe II's independence was a response
to a *specific known bias* in that grid's cross-band coupling; no
analogous bias is documented here). tau_BE, edge_norm, and line_norm
remain independent of each other and of (T_e, log n_e).

--------------------------------------------------------------------------
Data source for the line ratios
--------------------------------------------------------------------------
Storey & Hummer (1995, MNRAS 272, 41; VizieR catalog VI/64) case-B
hydrogen recombination emissivities, extracted here via PyNeb's bundled
`h_i_rec_SH95.hdf5` (build-time tool only -- NOT a runtime dependency of
this module or of desisim; see data/balmer_case_b_grid.npz, produced by
a one-off extraction script, not tracked in this repo). Emissivities for
Lyalpha and Balmer n=3..50 are converted to ratios relative to Hbeta on
the table's native (log n_e in [2,14], T_e in {500...30000K}) grid, the
same "ratio relative to Hbeta" convention already used by
recombination_lines.ecsv. Wavelengths for n=19..50 (not present in that
file) are computed from the hydrogenic Rydberg formula; n=3..18 use the
same lab (vacuum) wavelengths already in recombination_lines.ecsv --
Rydberg agrees with those to within ~0.08A (sub-fine-structure-level,
irrelevant at any velocity broadening used here; verified at build time).

--------------------------------------------------------------------------
Interpolation
--------------------------------------------------------------------------
Manual bilinear interpolation over (log n_e, T_e), boundary-clipped
(never extrapolated) -- same _bracket() convention as feii_continuum.py,
duplicated here (not imported) for module self-containment.

--------------------------------------------------------------------------
Bucket
--------------------------------------------------------------------------
Both pieces are real emitted photons, not line-of-sight removal, despite
"continuum" appearing in the name of piece (A) -- same reasoning as
feii_flux (see decompose.py). Both land in combine_into_channels'
"emission" bucket via the single `balmer_flux` array this module returns.

--------------------------------------------------------------------------
Task #42 empirical-backtest audit: disclosed limitation
--------------------------------------------------------------------------
Unlike feii_continuum.py's R_FEII_OPTICAL_BROAD_HBETA_PRIOR (task #42),
which was backtested against a real measured population distribution
(Shen & Ho 2014; Panda et al. 2020's cleaned SDSS DR7 subsample -- see
test_feii_continuum.py's TestEmpiricalBacktest), no equivalent
independent, precisely-measured POPULATION distribution for either the
Balmer-edge-to-line-series continuity condition (edge_norm's default,
task #41) or the broad/narrow Hbeta ratio (STANDALONE_BROAD_NARROW_HBETA_RATIO
/ mock_spectrum.HBETA_BROAD_NARROW_RATIO_RANGE) was found in the
literature search backing tasks #41/#42. The continuity condition itself
IS verified against real physics (Kovacevic, Popovic & Kollatschny 2015;
see test_balmer_continuum.py's test_default_edge_norm_is_continuous_with_line_series_at_the_edge),
and TE_RANGE/TAU_BE_RANGE are anchored on real fiducial fitting values
(Dietrich et al. 2002, 2003; Kurk et al. 2007) -- but there is currently
no real per-object catalog this module's ENSEMBLE output (e.g. a
population of drawn Balmer-edge-to-broad-Hbeta flux ratios) has been
checked against. This is honestly disclosed here rather than fabricating
a population backtest without real reference data; a natural target for
a future task if/when a suitable public catalog (e.g. a large
Balmer-continuum-strength measurement sample) is identified.
"""

import numpy as np

from desisim.templates import C_LIGHT, _use_torch_backend, _lines_to_spectrum_numpy, _lines_to_spectrum_torch


_GRID_CACHE = None


def _load_grid():
    """Lazily load and cache the packaged Storey & Hummer case-B grid."""
    global _GRID_CACHE
    if _GRID_CACHE is None:
        from importlib import resources
        path = str(resources.files('desisim').joinpath('data', 'balmer_case_b_grid.npz'))
        with np.load(path) as d:
            _GRID_CACHE = dict(
                wave=d['wave'].astype(np.float64),
                names=[str(s) for s in d['names']],
                dens_grid=d['dens_grid'].astype(np.float64),
                temp_grid=d['temp_grid'].astype(np.float64),
                ratio=d['ratio'].astype(np.float64),
            )
    return _GRID_CACHE


def _bracket(grid_axis, value):
    """Clip `value` to [grid_axis.min(), grid_axis.max()] then return
    (i0, i1, w0, w1) such that value ~= w0*grid_axis[i0] + w1*grid_axis[i1]
    (i0==i1, w0=1, w1=0 at either exact boundary -- no extrapolation ever
    occurs)."""
    value = np.clip(value, grid_axis[0], grid_axis[-1])
    i1 = int(np.searchsorted(grid_axis, value, side='right'))
    i1 = min(max(i1, 1), len(grid_axis) - 1)
    i0 = i1 - 1
    lo, hi = grid_axis[i0], grid_axis[i1]
    if hi == lo:
        return i0, i1, 1.0, 0.0
    w1 = (value - lo) / (hi - lo)
    return i0, i1, 1.0 - w1, w1


def _bilinear_ratio(ratio_grid, dens_grid, temp_grid, log_ne, T_e):
    """ratio_grid: [nline, ndens, ntemp] Hbeta-relative case-B ratios.
    Returns [nline] bilinearly interpolated at (log_ne, T_e), clipped to
    the grid boundary (see _bracket)."""
    di0, di1, dw0, dw1 = _bracket(dens_grid, log_ne)
    ti0, ti1, tw0, tw1 = _bracket(temp_grid, T_e)
    out = np.zeros(ratio_grid.shape[0])
    for di, dw in ((di0, dw0), (di1, dw1)):
        if dw == 0.0:
            continue
        for ti, tw in ((ti0, tw0), (ti1, tw1)):
            if tw == 0.0:
                continue
            out += dw * tw * ratio_grid[:, di, ti]
    return out


def _planck_lambda(wave_ang, T_e):
    """Planck function B_lambda(T_e); arbitrary overall units (only the
    *shape* is used -- this module peak-normalizes before applying its own
    free edge_norm scale, same convention as feii_continuum.py's bands and
    AGNPowerLawContinuum.spectrum()'s flux_norm).

    wave_ang : ndarray [A] (Angstrom)
    T_e : float [K]
    """
    h = 6.62607015e-27   # erg s
    c = 2.99792458e10    # cm/s
    kB = 1.380649e-16    # erg/K
    wave_cm = wave_ang * 1.0e-8
    x = np.clip(h * c / (wave_cm * kB * T_e), None, 700.0)  # avoid exp overflow
    return (2.0 * h * c ** 2 / wave_cm ** 5) / np.expm1(x)


class BalmerContinuum(object):
    """Additive Balmer continuum: free-bound recombination edge (Grandi
    1982) plus a broadened Lyalpha+Balmer(n=3..50) line series sharing one
    physical (T_e, log n_e) state. See module docstring for full
    rationale.
    """

    LAMBDA_BE = 3646.0  # Balmer series limit / free-bound edge [A, vacuum]

    # ⚠ MAGIC: electron temperature draw range [K]. Anchored on the
    # Dietrich et al. (2002, 2003) grid (10000-20000K, with 15000K as
    # their fiducial single-value fit) -- not a fit to any dataset.
    TE_RANGE = (10000.0, 20000.0)

    # ⚠ MAGIC: log10(electron density) draw range [cm^-3]. Per PI
    # direction, spans the full Storey & Hummer (1995) tabulated range
    # rather than being restricted to a narrower BLR-typical subset (cf.
    # Fe II's LOGNH_RANGE=(9,14), which mirrors CLOUDY-grid density
    # coverage for a different, denser line-forming region) -- onus is on
    # the downstream NPE to learn which part of this wider range is
    # physically realized.
    LOGNE_RANGE = (5.0, 14.0)

    # ⚠ MAGIC: free-bound optical depth at the edge. Anchored on the
    # Dietrich et al. (2002, 2003) grid (0.1-2, fiducial tau_BE=1) -- not
    # a fit to any dataset.
    TAU_BE_RANGE = (0.1, 2.0)

    # ⚠ MAGIC: same broad-line-region kinematic scale as
    # EMSpectrum.BROADSIGMA_RANGE_KMS/BROADSHIFT_KMS_RANGE, independently
    # owned by this module (no cross-module coupling) since this is the
    # same physical BLR-like gas producing both.
    SIGMA_KMS_RANGE = (425.0, 4250.0)
    VELSHIFT_KMS_RANGE = (-1000.0, 200.0)

    # --------------------------------------------------------------------
    # Task #41 (retroactive audit follow-up to task #32): literature-
    # anchored default for line_norm, replacing the previous arbitrary
    # "effective Hbeta flux of 1.0" default -- see feii_continuum.py's
    # STANDALONE_BROAD_NARROW_HBETA_RATIO (task #40) for the identical
    # derivation; duplicated here rather than imported for the same
    # module-self-containment reasons _bracket() is duplicated rather
    # than imported (see this module's own docstring). This is this
    # module's own broad-Hbeta-equivalent line's flux, in this pipeline's
    # native narrow-Hbeta=1 unit (see EMSpectrum.spectrum()'s
    # hbetaflux=None convention) -- i.e.
    # line_norm=STANDALONE_BROAD_NARROW_HBETA_RATIO means "this cascade's
    # broad Hbeta is that ratio times EMSpectrum's narrow Hbeta," matching
    # this module's own docstring description of line_norm as "an
    # effective Hbeta flux."
    #
    # Task #46: mock_spectrum.py's generate_qso_component() does NOT rely
    # on this fixed fallback -- it draws its own shared
    # hbeta_broad_narrow_ratio once per mock (see its own
    # HBETA_BROAD_NARROW_RATIO_RANGE) and passes it explicitly as
    # line_norm, kept consistent with the SAME draw used for
    # feii_kwargs['optical_flux_hbeta']/['uv_flux_hbeta']. This constant
    # is used ONLY as this module's own zero-argument fallback when
    # spectrum() is called directly, outside the orchestrator (e.g. in
    # this module's own unit tests) -- see feii_continuum.py's identical
    # constant for the full history of why a plain fixed constant is not
    # used by the orchestrator.
    STANDALONE_BROAD_NARROW_HBETA_RATIO = 5.0  # ⚠ MAGIC (order-of-magnitude only; see mock_spectrum.HBETA_BROAD_NARROW_RATIO_RANGE for the real per-mock draw)

    def __init__(self, minwave=1000.0, maxwave=10000.0, cdelt_kms=20.0, log10wave=None):
        """
        Args:
            minwave, maxwave (float): rest-frame output grid bounds [A].
                Only used if log10wave is not provided.
            cdelt_kms (float): output-grid pixel size [km/s] (log-uniform
                grid, same convention as EMSpectrum/AbsorptionSpectrum/
                FeIIPseudoContinuum).
            log10wave (ndarray, optional): explicit output log10(wave)
                grid, to match an external continuum/emission grid.
        """
        if log10wave is None:
            cdelt_loglam = cdelt_kms / C_LIGHT / np.log(10)
            log10wave = np.arange(np.log10(minwave), np.log10(maxwave), cdelt_loglam)
        self.log10wave = log10wave
        self._grid = _load_grid()

    def spectrum(self, T_e=None, log_ne=None, tau_BE=None, edge_norm=None, line_norm=None,
                 sigma_kms=None, velshift_kms=None, zshift=0.0, seed=None,
                 backend='auto', device=None, dtype=None):
        """Build the additive Balmer-continuum-plus-cascade flux array.

        Args:
            T_e (float, optional): electron temperature [K], shared by
                both pieces. Default None: uniform draw from TE_RANGE.
            log_ne (float, optional): log10(electron density) [cm^-3],
                shared by both pieces. Default None: uniform draw from
                LOGNE_RANGE. Out-of-range explicit values are clipped to
                the grid boundary when looking up line ratios (see
                _bracket), never extrapolated.
            tau_BE (float, optional): free-bound optical depth at the
                edge. Default None: log-uniform draw from TAU_BE_RANGE.
            edge_norm (float, optional): multiplicative scale applied to
                the free-bound edge's peak-normalized shape (i.e. piece
                (A) peaks at exactly this value), in the caller's flux
                units. Default None (task #41): rather than an arbitrary
                placeholder, resolved so that piece (A) is continuous
                with piece (B) at the Balmer edge itself (lambda=3646A) --
                i.e. the free-bound edge flux exactly matches the summed
                flux of the high-order line series at that wavelength,
                using whatever line_norm/line_scale is in effect. This is
                not a free literature ratio but a physical continuity
                condition: Kovacevic, Popovic & Kollatschny (2015,
                arXiv:1311.6653) show that the sum of high-order
                (n up to 400) Balmer lines at the edge reproduces the
                Grandi (1982) free-bound edge intensity there (their
                Eq. 5/7), for the same (T_e, tau_BE)=(15000K, 1) fiducial
                values this module's own TE_RANGE/TAU_BE_RANGE already
                bracket -- so this default holds regardless of whether
                line_norm was itself explicitly passed or left at its own
                literature default (see line_norm below). An explicit
                edge_norm always overrides this and is applied exactly as
                before task #41 (byte-for-byte back-compatible escape
                hatch).
            line_norm (float, optional): multiplicative scale applied to
                the Hbeta-relative case-B ratios (i.e. an
                "effective Hbeta flux" for this cascade -- a line with
                ratio=1 would have flux exactly line_norm). Default None
                (task #41): resolves to STANDALONE_BROAD_NARROW_HBETA_RATIO
                (see its own class-level comment for the derivation and
                caveats; same order-of-magnitude anchor as
                feii_continuum.FeIIPseudoContinuum.STANDALONE_BROAD_NARROW_HBETA_RATIO
                from task #40) when this module is used standalone,
                replacing the previous uncalibrated default of 1.0 (task
                #39's audit finding: 1.0 implied broad Hbeta = narrow
                Hbeta, understating real Type 1 quasars' broad-line
                dominance). Task #46: mock_spectrum.py's
                generate_qso_component() overrides this default with a
                properly drawn, cross-module-shared value instead of
                relying on this fixed fallback -- see its own
                HBETA_BROAD_NARROW_RATIO_RANGE. Pre-task-#41 callers that
                never passed edge_norm/line_norm will see their absolute
                output values change; callers that explicitly passed them
                are completely unaffected.
            sigma_kms (float, optional): shared macroscopic BLR velocity
                width [km/s] applied to both pieces. Default None:
                log-uniform draw from SIGMA_KMS_RANGE.
            velshift_kms (float, optional): shared bulk velocity shift
                [km/s], independent of zshift. Default None: uniform draw
                from VELSHIFT_KMS_RANGE.
            zshift (float): redshift applied on top of velshift_kms (same
                convention as EMSpectrum/AbsorptionSpectrum/
                FeIIPseudoContinuum).
            seed (int, optional): RNG seed for reproducibility.
            backend, device, dtype: passed through to the line-profile
                construction step (same torch/CUDA backend convention as
                EMSpectrum.spectrum(); see _use_torch_backend).

        Returns:
            Tuple of (balmer_flux, wave, params): balmer_flux is an array
            [npix] (sum of pieces (A) and (B), resampled onto this
            object's output grid); wave is 10**self.log10wave; params is
            a dict with the fully resolved T_e, log_ne, tau_BE, edge_norm,
            line_norm, sigma_kms, velshift_kms.
        """
        rand = np.random.RandomState(seed)

        if T_e is None:
            T_e = rand.uniform(*self.TE_RANGE)
        if log_ne is None:
            log_ne = rand.uniform(*self.LOGNE_RANGE)
        if tau_BE is None:
            tau_BE = 10 ** rand.uniform(np.log10(self.TAU_BE_RANGE[0]), np.log10(self.TAU_BE_RANGE[1]))
        if sigma_kms is None:
            sigma_kms = 10 ** rand.uniform(np.log10(self.SIGMA_KMS_RANGE[0]),
                                            np.log10(self.SIGMA_KMS_RANGE[1]))
        if velshift_kms is None:
            velshift_kms = rand.uniform(*self.VELSHIFT_KMS_RANGE)

        # Task #41: line_norm resolved first (before edge_norm), since
        # edge_norm's own new default depends on the fully-resolved line
        # series (see edge_norm's docstring above for the physical
        # continuity condition this implements).
        line_scale = self.STANDALONE_BROAD_NARROW_HBETA_RATIO if line_norm is None else line_norm

        wave_out = 10 ** self.log10wave

        # --- (B) bound-bound merged line series (computed first; see above) ---
        g = self._grid
        ratios = _bilinear_ratio(g['ratio'], g['dens_grid'], g['temp_grid'], log_ne, T_e)
        amp = line_scale * ratios

        log10sigma = sigma_kms / C_LIGHT / np.log(10)
        linecenters = (np.log10(g['wave'] * (1.0 + zshift))
                       + velshift_kms / C_LIGHT / np.log(10))
        norm = amp / (np.sqrt(2.0 * np.pi) * log10sigma)

        if _use_torch_backend(backend):
            line_flux = _lines_to_spectrum_torch(self.log10wave, linecenters, norm, log10sigma,
                                                  device=device, dtype=dtype)
        else:
            line_flux = _lines_to_spectrum_numpy(self.log10wave, linecenters, norm, log10sigma)

        # --- (A) free-bound edge (Grandi 1982), built directly on the
        # output grid, then shifted (interpolated) by velshift_kms+zshift.
        edge_shape = np.where(
            wave_out <= self.LAMBDA_BE,
            _planck_lambda(wave_out, T_e) * (1.0 - np.exp(-tau_BE * (wave_out / self.LAMBDA_BE) ** 3)),
            0.0)
        edge_peak = edge_shape.max()
        edge_shape = edge_shape / edge_peak if edge_peak > 0 else edge_shape
        edge_native_wave = wave_out * (1.0 + zshift) * (1.0 + velshift_kms / C_LIGHT)

        if edge_norm is None:
            # Task #41: continuity condition (Kovacevic, Popovic &
            # Kollatschny 2015, arXiv:1311.6653) -- the free-bound edge's
            # unit-normalized shape at lambda=LAMBDA_BE exactly (where
            # (lambda/LAMBDA_BE)**3=1, computed analytically, no
            # interpolation needed) must, after scaling by edge_scale,
            # equal the already-scaled line series' value at that same
            # physical (shifted) wavelength.
            edge_raw_at_edge = _planck_lambda(self.LAMBDA_BE, T_e) * (1.0 - np.exp(-tau_BE))
            edge_unit_at_edge = edge_raw_at_edge / edge_peak if edge_peak > 0 else 0.0
            edge_wave_shifted = self.LAMBDA_BE * (1.0 + zshift) * (1.0 + velshift_kms / C_LIGHT)
            line_flux_at_edge = np.interp(edge_wave_shifted, wave_out, line_flux)
            edge_scale = line_flux_at_edge / edge_unit_at_edge if edge_unit_at_edge > 0 else 0.0
        else:
            edge_scale = edge_norm

        edge_flux = edge_scale * np.interp(wave_out, edge_native_wave, edge_shape, left=0.0, right=0.0)

        balmer_flux = edge_flux + line_flux

        params = dict(T_e=T_e, log_ne=log_ne, tau_BE=tau_BE, edge_norm=edge_scale,
                      line_norm=line_scale, sigma_kms=sigma_kms, velshift_kms=velshift_kms)
        return balmer_flux, wave_out, params
