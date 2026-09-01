"""
desisim.feii_continuum
========================

Task #31: the Fe II UV/optical pseudo-continuum, as an independent,
additive emission channel.

--------------------------------------------------------------------------
Why this exists
--------------------------------------------------------------------------
Fe+'s dense energy-level structure produces tens of thousands of
overlapping permitted/semi-forbidden transitions in two AGN spectral
windows -- roughly 2000-3500A (blended with Mg II 2800 and the Balmer
continuum) and roughly 4400-4700/5150-5850A (blended with Hbeta and
[OIII]). At real broad-line-region velocity widths the individual lines
are not resolvable; the blend is observationally indistinguishable from a
genuine quasi-continuum, hence "pseudo-continuum." No choice of parameters
in this fork's existing discrete-Gaussian line-list architecture
(EMSpectrum) can reproduce this without literally tens of thousands of
independently-known line strengths, which don't exist as reliable inputs
anyway -- this is exactly why the AGN literature uses templates/grids
instead of line lists for this component. See the project's 2026-08-25
codebase review for the full physical writeup.

--------------------------------------------------------------------------
Data source
--------------------------------------------------------------------------
Ashwani-88/Fe2_template (CC-BY-4.0), the public grid accompanying "New
Theoretical Fe II Templates for Bright Quasars" (2025, ApJS 277, 36,
arXiv:2401.18052) -- a CLOUDY C23.0 photoionization grid using the Smyth
et al. (2019) Fe II atomic dataset. No CLOUDY installation is required;
the published grid outputs are used directly (unlike FSPS, which needs a
compiled native backend not installed in this fork's dev sandbox).

The grid physical parameters are: H-ionizing photon flux log(Phi_H)
[cm^-2 s^-1] in [17, 22], gas density log(n_H) [cm^-3] in [9, 14],
microturbulent velocity in {0, 20, 50, 100} km/s, and two assumed AGN
continuum SED shapes ("AGN_SED" ~ Mathews & Ferland 1987-like; and
"Intermediate_SED", Jin et al. 2012). Fixed column density 1e24 cm^-2,
solar abundance. This module's packaged data
(data/feii_templates_grid.npz) is a coarse boundary+interior subsample of
that public grid at step 1.0 dex on both (log_phi, log_nH), covering the
full documented range on all axes (so every draw here interpolates,
never extrapolates, unless an explicit out-of-range value is passed) --
271 of the 288 (sed, turb, phi, nH) grid cells are genuine CLOUDY runs;
the remaining 17 (all at the same known no-Fe-II-emission corner the
paper's own README documents: high flux, low density) are backfilled
from their nearest measured neighbor and flagged in the packaged
`measured` boolean array for provenance -- see build history for the
exact backfill list.

--------------------------------------------------------------------------
Why UV and optical are drawn independently (the actual design decision)
--------------------------------------------------------------------------
Even the newest CLOUDY grids, with the best available atomic data,
systematically overpredict the UV-to-optical Fe II flux ratio relative to
real AGN spectra, and the 2025 paper's own stated conclusion is that no
single set of physical parameters (in this single-zone, single-density,
one-sided-illumination framework) reproduces the observed UV/optical
balance. The leading suspected causes are structural, not atomic-data
related: the constant-density single-zone assumption (real BLR gas is
plausibly a superposition of multiple zones/densities) and uncertainty in
the assumed heating mechanism for the Fe II-emitting gas -- not errors in
the relative line-ratio *shape* within each band.

Given that diagnosis, trusting the grid's own absolute UV-to-optical
normalization would bake in a known, unresolved bias. Per explicit PI
direction, this module instead draws the UV piece's physical parameters
(log_phi, log_nH, turb, sed) and the optical piece's physical parameters
completely independently -- potentially from different, uncorrelated grid
cells -- and exposes the relative strength between the two bands as its
own free parameter (uv_norm / optical_norm below) rather than inheriting
whatever ratio the grid happens to predict for a single shared cell. This
keeps the within-band *shape* variation (which is not the part of the
model under suspicion) while explicitly declining to trust the
cross-band absolute normalization (which is). The onus of learning the
real population's UV/optical covariance falls on the downstream NPE, per
this fork's standing convention (e.g. QSO/galaxy blend fraction, dust
priors) of exposing maximal independently-tunable degrees of freedom
rather than hardcoding a specific (and in this case known-questionable)
physical correlation.

--------------------------------------------------------------------------
Interpolation
--------------------------------------------------------------------------
Manual trilinear interpolation over the three continuous grid axes
(log_phi, log_nH, turb), implemented directly in numpy (not
scipy.interpolate.RegularGridInterpolator, to avoid any dependence on
scipy-version-specific vector-valued-grid support) -- see
_trilinear_interp_shape(). Requested parameter values are clipped to the
grid's own [min, max] range on each axis before interpolating, so a draw
can never extrapolate past the boundary of the parameter space (per the
PI's explicit "pull uniformly from the interior, bounded by the boundary
files" framing) -- it is instead pinned to the nearest edge. sed_shape is
a discrete (categorical) choice, not interpolated, since there is no
physically meaningful continuum between two different assumed ionizing
SED shapes.

--------------------------------------------------------------------------
Normalization convention
--------------------------------------------------------------------------
The packaged grid's own absolute flux scale is tied to CLOUDY's assumed
covering factor and a specific real calibration object (RM 102, per the
source repo's README) -- not meaningful as an absolute scale for an
arbitrary synthetic object. Each interpolated per-band shape is therefore
renormalized to its own peak value before uv_norm/optical_norm (each a
plain multiplicative scale in the caller's flux units) is applied -- the
same "arbitrary absolute scale unless told otherwise" convention already
used by AGNPowerLawContinuum.spectrum()'s flux_norm.

--------------------------------------------------------------------------
Macroscopic broadening
--------------------------------------------------------------------------
The grid's own microturbulence parameter is a small-scale (0-100 km/s),
line-formation-physics quantity internal to CLOUDY's radiative transfer,
NOT the macroscopic bulk velocity width of the whole Fe II-emitting
region as seen by a distant observer (hundreds to thousands of km/s,
comparable to other broad-line-region kinematics). This module applies
that macroscopic broadening (sigma_kms) as a separate Gaussian
convolution in log10-wavelength space via scipy.ndimage.gaussian_filter1d
(the same backend GALAXY.make_galaxy_templates() already uses for
velocity-dispersion blurring), plus an independent bulk velocity_shift
(velshift_kms). Both draw randomly by default (no prior "off" behavior to
preserve -- this is a brand-new module, same convention as
AbsorptionSpectrum's sigma_kms/tau0).
"""

import numpy as np

from desisim.templates import C_LIGHT


_GRID_CACHE = None


def _load_grid():
    """Lazily load and cache the packaged Fe II template grid."""
    global _GRID_CACHE
    if _GRID_CACHE is None:
        from importlib import resources
        path = str(resources.files('desisim').joinpath('data', 'feii_templates_grid.npz'))
        with np.load(path) as d:
            _GRID_CACHE = dict(
                wave=d['wave'].astype(np.float64),
                flux=d['flux'].astype(np.float64),
                measured=d['measured'],
                phi_grid=d['phi_grid'].astype(np.float64),
                nH_grid=d['nH_grid'].astype(np.float64),
                turb_grid=d['turb_grid'].astype(np.float64),
                sed_labels=[str(s) for s in d['sed_labels']],
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


def _trilinear_interp_shape(flux_sed_slice, phi_grid, nH_grid, turb_grid, phi0, nH0, turb0):
    """flux_sed_slice: ndarray [n_turb, n_phi, n_nH, n_wave] for one fixed
    sed_shape. Returns the interpolated [n_wave] flux vector at
    (phi0, nH0, turb0), clipped to the grid boundary on each axis (see
    _bracket)."""
    ti0, ti1, tw0, tw1 = _bracket(turb_grid, turb0)
    pi0, pi1, pw0, pw1 = _bracket(phi_grid, phi0)
    ni0, ni1, nw0, nw1 = _bracket(nH_grid, nH0)

    out = np.zeros(flux_sed_slice.shape[-1])
    for ti, tw in ((ti0, tw0), (ti1, tw1)):
        if tw == 0.0:
            continue
        for pi, pw in ((pi0, pw0), (pi1, pw1)):
            if pw == 0.0:
                continue
            for ni, nw in ((ni0, nw0), (ni1, nw1)):
                if nw == 0.0:
                    continue
                out += tw * pw * nw * flux_sed_slice[ti, pi, ni, :]
    return out


class FeIIPseudoContinuum(object):
    """Independent, additive Fe II UV+optical pseudo-continuum, built from
    an independently-interpolated UV piece and optical piece of a public
    CLOUDY grid (see module docstring for the full rationale and the
    UV/optical decoupling design decision).
    """

    PHI_RANGE = (17.0, 22.0)      # log10(ionizing photon flux) [cm^-2 s^-1]
    LOGNH_RANGE = (9.0, 14.0)     # log10(gas density) [cm^-3]
    TURB_VALUES = (0.0, 20.0, 50.0, 100.0)  # microturbulence grid values [km/s]
    SED_SHAPES = ('AGN_SED', 'Intermediate_SED')

    # Standard UV/optical Fe II divide used throughout the literature
    # (e.g. Tsuzuki et al. 2006's own UV window ends at 3500A). Not
    # MAGIC in the "unjustified constant" sense -- this is a documented
    # convention, not a free parameter.
    UV_MAXWAVE = 3500.0

    # ⚠ MAGIC: macroscopic bulk velocity broadening applied to each band's
    # interpolated shape, log-uniform draw range [km/s]. Anchored on the
    # same broad-line-region kinematic scale as EMSpectrum's
    # BROADSIGMA_RANGE_KMS (425-4250 km/s) -- Fe II is understood to arise
    # from BLR-like gas, not the narrower NLR. Not a fit to any dataset.
    SIGMA_KMS_RANGE = (425.0, 4250.0)

    # ⚠ MAGIC: independent bulk velocity shift, uniform draw range [km/s].
    # Fe II velocity shifts distinct from other broad lines have been
    # measured in the literature (Kovacevic, Popovic & Dimitrijevic 2010
    # report a nonzero Fe II shift relative to Hbeta, motivating an
    # "intermediate line region" origin) -- this range is a broad
    # superset of that scale, not a fit to it.
    VELSHIFT_KMS_RANGE = (-500.0, 500.0)

    # --------------------------------------------------------------------
    # Task #40 (retroactive audit follow-up to task #31): literature-
    # anchored default relative strengths, replacing the previous
    # arbitrary "peak-normalized to 1.0" default norm.
    # --------------------------------------------------------------------
    # This pipeline's shared flux unit convention (set by EMSpectrum, see
    # templates.py's spectrum() with hbetaflux=oiiflux=None) is
    # "narrow Hbeta flux = 1.0". Every real literature Fe II/Hbeta ratio
    # below (R_FeII) is conventionally measured against BROAD Hbeta (Fe II
    # cannot be deblended from a narrow-only reference in real spectra),
    # which in THIS fork is supplied separately by
    # balmer_continuum.BalmerContinuum's broad H-recombination cascade
    # (task #32), not by EMSpectrum's narrow-only Hbeta row. Converting a
    # broad-Hbeta-relative literature ratio into this pipeline's native
    # narrow-Hbeta=1 unit therefore requires one extra factor: the typical
    # broad/narrow Hbeta flux ratio in luminous, broad-line-dominated
    # ("Type 1"/Seyfert-1.0-like) quasars -- the population this fork's
    # QSO pipeline actually targets.
    #
    # Task #46: this factor is NOT a fixed constant -- mock_spectrum.py's
    # generate_qso_component() draws it ONCE per mock (see its own
    # HBETA_BROAD_NARROW_RATIO_RANGE) and passes the resulting
    # optical_flux_hbeta/uv_flux_hbeta into feii.spectrum() explicitly,
    # so that this module and BalmerContinuum's line_norm always agree on
    # the same broad-Hbeta reference for a given mock. An earlier version
    # of this task used a plain fixed constant here, which the PI flagged
    # as inconsistent with treating an uncertain physical quantity as a
    # real (eventually NPE-calibrated) free parameter, and as a latent
    # cross-module desync risk. STANDALONE_BROAD_NARROW_HBETA_RATIO below
    # is used ONLY as this module's own zero-argument fallback when
    # spectrum() is called directly, outside the orchestrator (e.g. in
    # this module's own unit tests) -- it has no bearing on
    # generate_qso_component()'s actual behavior.
    STANDALONE_BROAD_NARROW_HBETA_RATIO = 5.0  # ⚠ MAGIC (order-of-magnitude only; see mock_spectrum.HBETA_BROAD_NARROW_RATIO_RANGE for the real per-mock draw)

    # R_FeII = EW(FeII 4434-4684)/EW(Hbeta,broad), i.e. the same quantity
    # as this module's `optical` band relative to broad Hbeta. Real
    # quasars show a broad, right-skewed distribution of this quantity,
    # NOT a single fixed value -- Eigenvector 1 (Boroson & Green 1992,
    # ApJS 80, 109) is fundamentally a statement about R_FeII's spread
    # across the quasar population, and Shen & Ho (2014, Nature 513, 210,
    # arXiv:1409.2887) report a peak/mode of ~0.6 in the SDSS Eigenvector-
    # 1 plane, while Marziani et al.-type cleaned SDSS DR7 subsamples
    # (Panda et al. 2020, arXiv:2001.08765, using the Shen et al. 2011
    # DR7 quasar catalog) report values ranging up to R_FeII=6.56 in the
    # same population. Task #42 (retroactive audit follow-up to task
    # #40): this is therefore drawn per mock (lognormal in R_FeII, i.e.
    # log10(R_FeII) ~ Normal(mean, sigma)) rather than fixed at its
    # median -- see mock_spectrum.py's generate_qso_component(), which
    # owns this draw (same "orchestrator draws, module consumes the
    # resolved flux target" split as HBETA_BROAD_NARROW_RATIO_RANGE,
    # task #46). mean is the real measured median; sigma is a ⚠ MAGIC
    # width tuned (by direct numerical check, not an analytic formula --
    # a single lognormal cannot match both real constraints exactly) to
    # jointly give a reasonable simultaneous match to two independent
    # real statistics from the SAME Panda et al. (2020) cleaned DR7
    # subsample: (a) ~17% (468/2734) of real quasars have R_FeII>=1
    # ("extreme Fe II emitters"), and (b) the maximum R_FeII observed in
    # that same n=2734 subsample is 6.56 (expected around the 99.96th
    # percentile of a sample that size). sigma=0.24 alone matches (a)
    # almost exactly but undershoots (b) by ~2x; sigma=0.31 alone matches
    # (b) but overshoots (a) to ~24%; sigma=0.28 (adopted here) gives
    # ~22% for (a) and ~5.0 for (b) -- both within a factor of ~1.3-1.5
    # of the real values, a reasonable compromise given a single
    # lognormal is only an approximate model of what is likely a
    # somewhat heavier-tailed real distribution. Deferred to NPE
    # calibration (a real non-lognormal shape, learned directly from
    # data) like every other ⚠ MAGIC width in this fork.
    R_FEII_OPTICAL_BROAD_HBETA_PRIOR = dict(mean=np.log10(0.6), sigma=0.28)  # ⚠ MAGIC sigma; real mean

    # FeII(lambda4570)/FeII(UV) = 10**(-0.8 +/- 0.2 dex): mean AND sigma
    # are BOTH precisely measured population statistics from 884 SDSS
    # quasars (Sameshima, Kawara, Matsuoka, Oyabu, Asami & Ienaka 2010,
    # MNRAS 410, 1018, arXiv:1008.2405, Figure 7 / Section 4.1 -- "the
    # average of log FeII(4570)/FeII(UV) is -0.8 and the standard
    # deviation is 0.2 dex"), where FeII(UV) is integrated over
    # 2200-3000A and FeII(4570) over 4435-4685A -- the same UV/optical
    # bands (to within the 3500A divide documented at UV_MAXWAVE above)
    # that this module's uv/optical pieces represent. Unlike every other
    # sigma in this module, this one is NOT a MAGIC guess.
    LOG_FEII_OPTICAL_TO_UV_PRIOR = dict(mean=-0.8, sigma=0.2)  # real measured mean AND sigma

    # Point-value fallbacks for this module's OWN zero-argument default
    # (spectrum() called directly, outside the orchestrator -- see
    # STANDALONE_BROAD_NARROW_HBETA_RATIO's comment above for the
    # identical convention). Uses each prior's mean (median), i.e. no
    # scatter -- the orchestrator is what actually draws these per mock.
    DEFAULT_OPTICAL_FLUX_HBETA = (10.0 ** R_FEII_OPTICAL_BROAD_HBETA_PRIOR['mean']
                                   * STANDALONE_BROAD_NARROW_HBETA_RATIO)
    DEFAULT_UV_FLUX_HBETA = (10.0 ** R_FEII_OPTICAL_BROAD_HBETA_PRIOR['mean']
                              / 10.0 ** LOG_FEII_OPTICAL_TO_UV_PRIOR['mean']
                              * STANDALONE_BROAD_NARROW_HBETA_RATIO)

    def __init__(self, minwave=1000.0, maxwave=10000.0, cdelt_kms=20.0, log10wave=None):
        """
        Args:
            minwave, maxwave (float): rest-frame output grid bounds [A].
                Only used if log10wave is not provided.
            cdelt_kms (float): output-grid pixel size [km/s] (log-uniform
                grid, same convention as EMSpectrum/AbsorptionSpectrum).
            log10wave (ndarray, optional): explicit output log10(wave)
                grid, to match an external continuum/emission grid.
        """
        if log10wave is None:
            cdelt_loglam = cdelt_kms / C_LIGHT / np.log(10)
            log10wave = np.arange(np.log10(minwave), np.log10(maxwave), cdelt_loglam)
        self.log10wave = log10wave
        self._grid = _load_grid()

    def _resolve_band_params(self, rand, params):
        """params: None (draw) or an explicit dict with any subset of
        log_phi/log_nH/turb/sed; unspecified keys are drawn."""
        params = dict(params) if params else {}
        log_phi = params.get('log_phi')
        if log_phi is None:
            log_phi = rand.uniform(*self.PHI_RANGE)
        log_nH = params.get('log_nH')
        if log_nH is None:
            log_nH = rand.uniform(*self.LOGNH_RANGE)
        turb = params.get('turb')
        if turb is None:
            turb = self.TURB_VALUES[rand.randint(len(self.TURB_VALUES))]
        sed = params.get('sed')
        if sed is None:
            sed = self.SED_SHAPES[rand.randint(len(self.SED_SHAPES))]
        elif sed not in self.SED_SHAPES:
            raise ValueError('Unknown sed {!r}; must be one of {}'.format(sed, self.SED_SHAPES))
        return dict(log_phi=log_phi, log_nH=log_nH, turb=turb, sed=sed)

    def _band_shape(self, resolved):
        g = self._grid
        sed_idx = g['sed_labels'].index(resolved['sed'])
        flux_sed_slice = g['flux'][sed_idx]  # [n_turb, n_phi, n_nH, n_wave]
        shape = _trilinear_interp_shape(flux_sed_slice, g['phi_grid'], g['nH_grid'], g['turb_grid'],
                                         resolved['log_phi'], resolved['log_nH'], resolved['turb'])
        return shape

    def spectrum(self, uv_params=None, optical_params=None, uv_norm=None, optical_norm=None,
                 uv_flux_hbeta=None, optical_flux_hbeta=None,
                 sigma_kms=None, velshift_kms=None, zshift=0.0, seed=None):
        """Build the additive Fe II pseudo-continuum flux array.

        Args:
            uv_params, optical_params (dict, optional): explicit physical
                grid position for each band -- any subset of log_phi
                (PHI_RANGE), log_nH (LOGNH_RANGE), turb (one of
                TURB_VALUES), sed (one of SED_SHAPES). Unspecified keys
                are drawn independently and uniformly (log_phi, log_nH:
                continuous uniform; turb, sed: discrete uniform). Out-of-
                range explicit values are clipped to the grid boundary,
                never extrapolated (see _bracket).
            uv_norm, optical_norm (float, optional): multiplicative scale
                applied to that band's peak-normalized shape (i.e. the
                returned per-band contribution peaks at exactly this
                value), in the caller's flux units. Default None: resolved
                from uv_flux_hbeta/optical_flux_hbeta instead (see below)
                -- an explicit uv_norm/optical_norm always overrides that
                resolution and is applied exactly as before task #40
                (byte-for-byte back-compatible escape hatch).
            uv_flux_hbeta, optical_flux_hbeta (float, optional): desired
                *integrated* band flux, as a multiple of this pipeline's
                narrow-Hbeta=1.0 flux unit (see EMSpectrum.spectrum()'s
                hbetaflux=None convention). Only used when the
                corresponding uv_norm/optical_norm is None. Internally,
                this method integrates the band's own unit-peak-
                normalized shape (which depends on the drawn/explicit
                uv_params/optical_params) and solves for the norm that
                gives it the requested integrated flux -- so, unlike
                uv_norm/optical_norm, this value means the same thing
                regardless of the drawn grid position or broadening.
                Default None: resolves to DEFAULT_UV_FLUX_HBETA /
                DEFAULT_OPTICAL_FLUX_HBETA, this class's literature-
                anchored defaults (task #40; see their own class-level
                comments for the full derivation and citations:
                Shen & Ho 2014; Boroson & Green 1992;
                Sameshima et al. 2010, MNRAS 410, 1018). Pre-task-#40
                callers that never passed uv_norm/optical_norm will see
                their absolute output values change (the old default was
                an uncalibrated peak=1.0 placeholder, not a deliberately
                validated behavior -- see task #39's audit); callers that
                explicitly passed uv_norm/optical_norm are completely
                unaffected.
            sigma_kms (float, optional): shared macroscopic bulk velocity
                broadening [km/s] applied to both bands via Gaussian
                convolution. Default None: log-uniform draw from
                SIGMA_KMS_RANGE.
            velshift_kms (float, optional): shared bulk velocity shift
                [km/s], independent of zshift. Default None: uniform draw
                from VELSHIFT_KMS_RANGE.
            zshift (float): redshift applied on top of velshift_kms
                (matches EMSpectrum/AbsorptionSpectrum's zshift
                convention).
            seed (int, optional): RNG seed for reproducibility.

        Returns:
            Tuple of (feii_flux, wave, params), where feii_flux is an
            array [npix] (sum of the independently-drawn/scaled UV and
            optical pieces, resampled onto this object's output grid);
            wave is 10**self.log10wave; params is a dict with the fully
            resolved uv/optical physical parameters, norms, sigma_kms,
            velshift_kms, and each band's `measured` flag (True if every
            grid corner contributing to its interpolation was a genuine
            CLOUDY run, False if at least one corner was a backfilled
              value -- see module docstring).
        """
        from scipy.ndimage import gaussian_filter1d

        rand = np.random.RandomState(seed)

        uv_resolved = self._resolve_band_params(rand, uv_params)
        optical_resolved = self._resolve_band_params(rand, optical_params)

        if sigma_kms is None:
            sigma_kms = 10 ** rand.uniform(np.log10(self.SIGMA_KMS_RANGE[0]),
                                            np.log10(self.SIGMA_KMS_RANGE[1]))
        if velshift_kms is None:
            velshift_kms = rand.uniform(*self.VELSHIFT_KMS_RANGE)

        g = self._grid
        native_wave = g['wave']
        uv_mask = native_wave <= self.UV_MAXWAVE
        opt_mask = ~uv_mask

        uv_shape_native = self._band_shape(uv_resolved)
        opt_shape_native = self._band_shape(optical_resolved)

        uv_band = np.where(uv_mask, uv_shape_native, 0.0)
        opt_band = np.where(opt_mask, opt_shape_native, 0.0)

        uv_peak = uv_band.max()
        opt_peak = opt_band.max()
        uv_band = uv_band / uv_peak if uv_peak > 0 else uv_band
        opt_band = opt_band / opt_peak if opt_peak > 0 else opt_band

        # Task #40: resolve uv_norm/optical_norm from the literature-
        # anchored flux-ratio parameters when not given explicitly. Each
        # band's unit-peak-normalized shape is integrated (trapezoidal,
        # over the full native grid -- exactly 0 outside the band's own
        # mask, so integrating over the whole grid is equivalent to
        # integrating over just the band) to convert the requested
        # *integrated flux* into the peak-amplitude norm this method has
        # always internally applied.
        if uv_norm is None:
            target_uv_flux = self.DEFAULT_UV_FLUX_HBETA if uv_flux_hbeta is None else uv_flux_hbeta
            uv_unit_integral = np.trapezoid(uv_band, native_wave)
            uv_scale = target_uv_flux / uv_unit_integral if uv_unit_integral > 0 else 0.0
        else:
            uv_scale = uv_norm
        if optical_norm is None:
            target_opt_flux = self.DEFAULT_OPTICAL_FLUX_HBETA if optical_flux_hbeta is None else optical_flux_hbeta
            opt_unit_integral = np.trapezoid(opt_band, native_wave)
            opt_scale = target_opt_flux / opt_unit_integral if opt_unit_integral > 0 else 0.0
        else:
            opt_scale = optical_norm
        combined_native = uv_scale * uv_band + opt_scale * opt_band

        # Macroscopic broadening in log10-wavelength space (native grid is
        # linear in wavelength with fixed 2A binning; approximate the
        # local pixel scale in km/s at each wavelength for the Gaussian
        # sigma -- equivalent to a constant fractional (log-uniform)
        # width to the same precision the rest of this fork already uses
        # for the log-uniform output grid).
        cdelt_ang = native_wave[1] - native_wave[0]
        # sigma in Angstrom at each pixel's own wavelength (velocity width
        # -> wavelength width: dlambda = lambda * sigma_kms / C_LIGHT).
        sigma_ang = native_wave * sigma_kms / C_LIGHT
        # gaussian_filter1d requires one scalar sigma (in pixels); use the
        # array's median wavelength as a representative pixel scale,
        # consistent with this being a broad, smooth pseudo-continuum
        # rather than a sharp line needing per-pixel-exact sigma.
        sigma_pix = max(np.median(sigma_ang) / cdelt_ang, 1e-6)
        broadened_native = gaussian_filter1d(combined_native, sigma=sigma_pix, mode='constant', cval=0.0)

        shifted_wave = native_wave * (1.0 + zshift) * (1.0 + velshift_kms / C_LIGHT)

        wave_out = 10 ** self.log10wave
        feii_flux = np.interp(wave_out, shifted_wave, broadened_native, left=0.0, right=0.0)

        def _band_measured(resolved):
            gm = g['measured']
            sed_idx = g['sed_labels'].index(resolved['sed'])
            pi0, pi1, _, _ = _bracket(g['phi_grid'], resolved['log_phi'])
            ni0, ni1, _, _ = _bracket(g['nH_grid'], resolved['log_nH'])
            ti0, ti1, _, _ = _bracket(g['turb_grid'], resolved['turb'])
            corners = gm[sed_idx][np.ix_([ti0, ti1], [pi0, pi1], [ni0, ni1])]
            return bool(np.all(corners))

        params = dict(
            uv=dict(uv_resolved, norm=uv_scale, measured=_band_measured(uv_resolved)),
            optical=dict(optical_resolved, norm=opt_scale, measured=_band_measured(optical_resolved)),
            sigma_kms=sigma_kms,
            velshift_kms=velshift_kms,
        )
        return feii_flux, wave_out, params
