"""
desisim.bal_trough
====================

Task #36: an additive, self-contained, physically-parametrized broad
absorption line (BAL) trough model for QSO spectra, replacing the
previously-descoped plan to reuse bal.py's real-template mechanism.

--------------------------------------------------------------------------
Why this exists instead of reusing bal.py
--------------------------------------------------------------------------
bal.py's BAL.insert_bals() multiplies a real, observed BAL transmission
template (drawn from a library of ~1500 real SDSS DR14 BAL quasars
curated by Niu 2020, built during Guo & Martini 2019's CNN-classifier
work) onto QSO flux. That template library is read via
$DESI_BASIS_TEMPLATES, DESI's internal collaboration data tree -- not
publicly obtainable, and confirmed (2026-08-27 review) unavailable in
this project's dev environment. More fundamentally, a real-template draw
only gives an NPE-recoverable ground truth of "which template ID was
picked" -- a categorical index, not a continuous physical quantity, unlike
every other channel in this project (Balmer's T_e/log_ne, Fe II's norms,
IGM's zqso-conditioned transmission). This module instead builds a new,
from-scratch parametric trough family with continuous, physically
motivated parameters, following this project's established practice
(mirrors associated_absorption.py's precedent of building a new stochastic
absorption feature from scratch, rather than task #35's precedent of
reusing an already self-contained, already-parametric legacy module).

--------------------------------------------------------------------------
Physical picture: partial covering + smoothed velocity-space trough
--------------------------------------------------------------------------
BAL troughs are conventionally characterized by the Weymann et al. (1991,
ApJ, 373, 23) balnicity index,

    BI = - integral_{-25000}^{-3000 km/s} [1 - f(v)/0.9] * C(v) dv

where f(v) is the continuum-normalized transmission as a function of
velocity blueward of the line center, and C(v) = 1 only where f(v) has
been continuously below 0.9 for at least 2000 km/s (0 elsewhere) -- i.e.
narrow, weak dips do not count as BAL troughs by definition. BI_V_LO/
BI_V_HI/MIN_TROUGH_WIDTH_KMS below are exactly this definition's window
and threshold, not free/MAGIC parameters.

Partial-covering outflow models (e.g. Arav, Hamann, de Kool and
collaborators' work on quasar-wind absorbers) commonly parametrize the
trough as

    T(v) = 1 - C_f * (1 - exp(-tau(v)))

with a covering fraction C_f in [0,1] and a velocity-dependent optical
depth tau(v). This module collapses C_f and tau(v)'s saturation into a
single scalar "depth" in [0,1] (the maximum fractional flux removed at
the trough's core) -- a simplification appropriate here since this
project's ground truth is the trough's SHAPE and STRENGTH for NPE
training, not a first-principles radiative-transfer decomposition into
C_f and tau(v) separately (which real BAL studies cannot uniquely
separate from a single ionic transition's profile alone either, without
a doublet ratio measurement this module does not attempt to model).

The trough's edges are smoothed with an error-function ramp (real
troughs do not have infinitely sharp edges in velocity space; a smoothing
scale of a few hundred km/s is a reasonable resolution-driven choice, not
a literature-measured quantity -- MAGIC-flagged below).

--------------------------------------------------------------------------
Empirical backtest against the real DR14Q BAL population (PI-directed,
2026-08-27: "confirm it can recreate the empirical BALs")
--------------------------------------------------------------------------
The real, literal Niu (2020) template curves live only in DESI's private
collaboration data tree and are not obtainable in this environment (see
above). As a public proxy that validates against the SAME criterion Niu
(2020) used to build those private templates in the first place --
"selected to have the same AI and BI distributions as the complete DR14
BAL catalog" -- this module's free parameters (STRENGTH_SCALE,
WIDTH_SCALE, DEPTH_SCALE, and the width/depth draw shapes below) were
calibrated by comparing this model's OWN synthetic Weymann-BI distribution
against the real per-object BI(CIV) column of the public SDSS DR14Q quasar
catalog (Paris et al. 2018, VizieR catalog VII/286, table
"VII/286/dr14q", column "BI(CIV)"), queried live via the VizieR TAP
service (https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync) on
2026-08-27 -- n=21870 real BALQSOs with BI(CIV)>0, mean=1852.5 km/s,
P(BI<500)=34.6%, P(BI<1000)=48.9%, P(BI<2000)=67.0%, P(BI<5000)=91.1%,
P(BI<10000)=99.1%, max~23800 km/s. A random search over this module's
prior hyperparameters (see weymann_bi() and the calibration script logic
reproduced in test_bal_trough.py's TestEmpiricalBacktest) found the
values hardcoded below reproduce that real distribution to within a few
percentage points on every bin (synthetic: mean=1558, P(<500)=36.0%,
P(<1000)=53.6%, P(<2000)=73.8%, P(<5000)=93.8%, P(<10000)=99.4%,
max~16900 -- the extreme tail above ~17000 km/s is somewhat underpopulated
relative to the real 8000-sample draw used for calibration, a known,
disclosed limitation rather than a hidden one). This is a genuine
recoverability check against real population statistics, not merely an
internal self-consistency test -- and per PI direction (2026-08-27) this
kind of empirical backtest is now standard practice for future physics
modules in this project (tracked follow-up: task #39, auditing modules
#31-#35 for the same).

--------------------------------------------------------------------------
Scope: CIV-only by default
--------------------------------------------------------------------------
Real BAL troughs are conventionally measured on C IV 1549 (Weymann's own
BI definition is CIV-specific), and Niu (2020)'s own templates apply the
SAME velocity-space profile to other resonance transitions (Si IV, N V,
Lyalpha) as an outflow-covers-multiple-species simplification. This
module supports that extension via the `lines` argument, but defaults to
CIV-only, matching the scope explicitly agreed with the PI (2026-08-27)
and keeping the empirical backtest's target quantity (BI, a CIV-specific
statistic) directly interpretable against what was actually validated.

--------------------------------------------------------------------------
Rest-frame, no zqso needed (unlike igm_absorption.py)
--------------------------------------------------------------------------
Unlike Lyman-alpha forest/DLA absorption (task #35), which is intrinsically
tied to the QSO's actual redshift along the line of sight, a BAL trough is
an INTRINSIC/associated absorption feature at a velocity offset from the
QSO's own systemic redshift -- exactly the same rest-frame, velocity-space
convention already used by associated_absorption.py. No zqso input is
needed; this module fits cleanly into mock_spectrum.py's "rest-frame only"
design without the architectural exception task #35 required.

--------------------------------------------------------------------------
Multiplicative -> additive (same pattern as dust.py/igm_absorption.py)
--------------------------------------------------------------------------
Transmission is multiplicative by definition, so the additive deficit is
exact: bal_flux(wave) = flux_to_absorb(wave) * (T(wave) - 1). Per the same
reasoning already used for igm_absorption.py (a real BAL trough measurably
eats into the QSO's own blue emission-line flux, not just its continuum),
flux_to_absorb should be the fuller pre-absorption QSO flux, not
continuum_agn alone -- left to the caller (mock_spectrum.py) to assemble,
mirroring igm_absorption.py's own calling convention exactly.
"""

import numpy as np
from scipy.special import erf

from desisim.templates import C_LIGHT

# Weymann et al. (1991) balnicity-index definition -- literal literature
# constants, not MAGIC/tunable.
BI_V_LO_KMS = -25000.0
BI_V_HI_KMS = -3000.0
MIN_TROUGH_WIDTH_KMS = 2000.0

# Rest-frame vacuum wavelengths [A] of the resonance transitions this
# module can apply a trough to. CIV matches forbidden_lines.ecsv's
# CIV_1549 entry (task #33); the others are standard BAL-relevant lines
# (Hall et al. 2002; Niu 2020) provided for optional multi-line extension,
# not enabled by default (see module docstring, "Scope").
LINE_WAVES = {
    'CIV': 1549.06,
    'SiIV': 1396.76,
    'NV': 1240.81,
    'Lya': 1215.67,
    'MgII': 2798.75,
}


def weymann_bi(v_kms, transmission, v_lo=BI_V_LO_KMS, v_hi=BI_V_HI_KMS,
                min_width_kms=MIN_TROUGH_WIDTH_KMS):
    """Literal Weymann et al. (1991) balnicity index of a transmission
    curve. Exposed at module level (not just used internally) so this
    exact function is what both the production code and the empirical
    backtest test use -- one implementation, not two independently-
    maintained copies that could silently diverge.

    Args:
        v_kms (ndarray): velocity grid [km/s], blueshift negative.
        transmission (ndarray): continuum-normalized transmission at
            each v_kms (1 = no absorption).
        v_lo, v_hi (float): integration window [km/s] (default: Weymann's
            own -25000 to -3000 km/s).
        min_width_kms (float): minimum contiguous below-threshold run
            length [km/s] for it to count at all (default: Weymann's own
            2000 km/s).

    Returns:
        float: BI [km/s], always >= 0.
    """
    v_kms = np.asarray(v_kms)
    transmission = np.asarray(transmission)
    mask = (v_kms >= v_lo) & (v_kms <= v_hi)
    v = v_kms[mask]
    f = transmission[mask]
    if v.size < 2:
        return 0.0
    order = np.argsort(v)
    v = v[order]
    f = f[order]
    below = f < 0.9
    C = np.zeros_like(f)
    i = 0
    n = len(f)
    while i < n:
        if below[i]:
            j = i
            while j < n and below[j]:
                j += 1
            width = v[j - 1] - v[i] if j > i + 1 else 0.0
            if width >= min_width_kms:
                C[i:j] = 1.0
            i = j
        else:
            i += 1
    integrand = (1.0 - f / 0.9) * C
    return float(abs(np.trapezoid(integrand, v)))


class BALTrough(object):
    """Additive, parametric broad absorption line (BAL) trough channel.
    See module docstring for the full physical/empirical-validation
    rationale.
    """

    # Real-population incidence: quickquasars' own documented default
    # (`--balprob`), itself grounded in the same Niu (2020) DR14-based
    # template lineage this module replaces -- not independently MAGIC,
    # but still a population-level average that varies with selection
    # (Hewett & Foltz 2003 report an intrinsic fraction of order 15-22%
    # at z=1.5-3.0; this is the single number this project's mocks use,
    # not a claim that BAL incidence is redshift/luminosity-independent).
    BALPROB = 0.16

    # Calibrated (2026-08-27, see module docstring) against the real
    # public SDSS DR14Q BI(CIV) distribution -- MAGIC in the sense that
    # these specific numeric values are backtest-fitted, not first-
    # principles physical constants, but NOT arbitrary: they were chosen
    # to reproduce a real, cited empirical distribution, and that fit is
    # itself checked by a standing regression test (see
    # test_bal_trough.py's TestEmpiricalBacktest).
    STRENGTH_SHAPE = 1.0
    STRENGTH_SCALE = 0.041  # Gamma(shape, scale) latent "BAL strength"
    WIDTH_SCALE = 0.288
    DEPTH_SCALE = 0.110
    WIDTH_JITTER_RANGE = (0.7, 1.3)
    DEPTH_JITTER_RANGE = (0.6, 1.15)
    V_MAX_KMS_RANGE = (-3500.0, -3000.0)  # low-velocity (least blueshifted) trough edge
    SMOOTH_KMS_RANGE = (300.0, 1200.0)    # ⚠ MAGIC: edge-smoothing scale, resolution-motivated guess, not literature-measured

    def __init__(self, minwave=1200.0, maxwave=1700.0, cdelt_kms=20.0, log10wave=None):
        """
        Args:
            minwave, maxwave (float): rest-frame output grid bounds [A].
                Only used if log10wave is not provided. Defaults bracket
                CIV 1549 with enough blueward room for the full Weymann
                window (CIV*(1+BI_V_LO/c) ~ 1420A).
            cdelt_kms (float): output-grid pixel size [km/s] (log-uniform
                grid, matching this project's universal convention).
            log10wave (ndarray, optional): explicit output log10(wave)
                grid, to match an external continuum/emission grid.
        """
        if log10wave is None:
            cdelt_loglam = cdelt_kms / C_LIGHT / np.log(10)
            log10wave = np.arange(np.log10(minwave), np.log10(maxwave), cdelt_loglam)
        self.log10wave = log10wave

    def _trough_transmission(self, line_name, v_min_kms, v_max_kms, depth, smooth_kms):
        """Smoothed partial-covering transmission for one line, on this
        object's own wavelength grid. 1.0 everywhere outside the line's
        own velocity window (within numerical precision of the erf ramp).
        """
        lambda0 = LINE_WAVES[line_name]
        v_kms = C_LIGHT * np.log(10) * (self.log10wave - np.log10(lambda0))
        ramp_lo = 0.5 * (1.0 + erf((v_kms - v_min_kms) / (smooth_kms / 1.665)))
        ramp_hi = 0.5 * (1.0 + erf((v_max_kms - v_kms) / (smooth_kms / 1.665)))
        shape = ramp_lo * ramp_hi
        return 1.0 - depth * shape, v_kms

    def transmission(self, hasbal=None, balprob=None, strength=None,
                      v_min_kms=None, v_max_kms=None, depth=None, smooth_kms=None,
                      lines=None, seed=None):
        """Combined trough transmission across the requested line(s).

        Args:
            hasbal (bool, optional): force BAL presence/absence. Default
                None draws Bernoulli(balprob) via seed.
            balprob (float, optional): BAL incidence probability. Default
                None uses self.BALPROB.
            strength (float, optional): latent "BAL strength" S >= 0
                (Gamma-distributed by default) driving correlated
                width/depth -- see module docstring. Default None draws
                Gamma(STRENGTH_SHAPE, STRENGTH_SCALE).
            v_min_kms, v_max_kms (float, optional): trough velocity
                window edges [km/s], v_min more blueshifted (more
                negative). Default None derives v_max from
                V_MAX_KMS_RANGE and v_min from strength-derived width.
            depth (float, optional): trough core depth in [0,1] (fraction
                of flux removed at line center). Default None derives
                from strength.
            smooth_kms (float, optional): edge-smoothing scale [km/s].
                Default None draws Uniform(*SMOOTH_KMS_RANGE).
            lines (list of str, optional): which LINE_WAVES entries to
                apply the (shared-shape) trough to. Default ['CIV'] (see
                module docstring, "Scope").
            seed (int, optional): RNG seed.

        Returns:
            Tuple of (T, wave, params): T is the combined transmission
            array [npix] in [0,1] (product over requested lines); wave is
            10**self.log10wave; params is a dict with hasbal, strength,
            v_min_kms, v_max_kms, depth, smooth_kms, lines, and bi (the
            literal Weymann BI of the CIV trough if CIV is among `lines`
            and hasbal is True, else None) for provenance.
        """
        rand = np.random.RandomState(seed)
        wave_rest = 10 ** self.log10wave

        if balprob is None:
            balprob = self.BALPROB
        if hasbal is None:
            hasbal = bool(rand.uniform() < balprob)

        if lines is None:
            lines = ['CIV']

        if not hasbal:
            params = dict(hasbal=False, strength=None, v_min_kms=None, v_max_kms=None,
                           depth=None, smooth_kms=None, lines=lines, bi=None)
            return np.ones_like(wave_rest), wave_rest, params

        if strength is None:
            strength = rand.gamma(shape=self.STRENGTH_SHAPE, scale=self.STRENGTH_SCALE)

        if v_max_kms is None:
            v_max_kms = rand.uniform(*self.V_MAX_KMS_RANGE)

        if v_min_kms is None:
            jitter = rand.uniform(*self.WIDTH_JITTER_RANGE)
            width = MIN_TROUGH_WIDTH_KMS + (1.0 - np.exp(-strength / self.WIDTH_SCALE)) * \
                (BI_V_HI_KMS - BI_V_LO_KMS) * jitter
            width = np.clip(width, MIN_TROUGH_WIDTH_KMS, BI_V_HI_KMS - BI_V_LO_KMS)
            v_min_kms = max(v_max_kms - width, BI_V_LO_KMS)

        if depth is None:
            jitter = rand.uniform(*self.DEPTH_JITTER_RANGE)
            depth = np.clip((1.0 - np.exp(-strength / self.DEPTH_SCALE)) * jitter, 0.0, 1.0)

        if smooth_kms is None:
            smooth_kms = rand.uniform(*self.SMOOTH_KMS_RANGE)

        T_total = np.ones_like(wave_rest)
        bi = None
        for name in lines:
            T_line, v_kms = self._trough_transmission(name, v_min_kms, v_max_kms, depth, smooth_kms)
            T_total = T_total * T_line
            if name == 'CIV':
                bi = weymann_bi(v_kms, T_line)

        params = dict(hasbal=True, strength=float(strength), v_min_kms=float(v_min_kms),
                      v_max_kms=float(v_max_kms), depth=float(depth), smooth_kms=float(smooth_kms),
                      lines=lines, bi=bi)
        return T_total, wave_rest, params

    def spectrum(self, flux_to_absorb, hasbal=None, balprob=None, strength=None,
                 v_min_kms=None, v_max_kms=None, depth=None, smooth_kms=None,
                 lines=None, seed=None):
        """Additive BAL trough deficit flux.

        Args:
            flux_to_absorb (ndarray): flux array [npix, same grid as this
                object's wave] the trough acts on -- per PI-approved
                design (mirroring igm_absorption.py) this should be the
                full pre-absorption QSO flux (continuum_agn +
                broad_emission + feii_flux + balmer_flux), not
                continuum_agn alone, since a real BAL trough measurably
                eats into the QSO's own blue emission-line flux.
            hasbal, balprob, strength, v_min_kms, v_max_kms, depth,
            smooth_kms, lines, seed: see transmission().

        Returns:
            Tuple of (bal_flux, wave, params): bal_flux is the additive
            deficit flux_to_absorb*(T-1) [npix], always <= 0 (same
            convention as ism_absorption/associated_absorption_flux/
            dust_flux/igm_flux); wave and params as returned by
            transmission().
        """
        T, wave, params = self.transmission(hasbal=hasbal, balprob=balprob, strength=strength,
                                             v_min_kms=v_min_kms, v_max_kms=v_max_kms, depth=depth,
                                             smooth_kms=smooth_kms, lines=lines, seed=seed)
        flux_to_absorb = np.asarray(flux_to_absorb)
        bal_flux = flux_to_absorb * (T - 1.0)
        return bal_flux, wave, params
