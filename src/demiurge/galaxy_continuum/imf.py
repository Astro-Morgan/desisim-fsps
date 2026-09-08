"""
IMF(t) for the galaxy continuum channel -- a deliberately simplified first
pass, not the full IGIMF machinery.

The full Integrated Galaxy-wide IMF (IGIMF) theory (Kroupa & Weidner 2003;
modern form Jerabkova, Hasani Zonoozi, Kroupa et al. 2018, A&A 620, A39;
reference implementation: `galIMF`, Yan, Jerabkova & Kroupa,
https://github.com/Azeret/galIMF, based on Yan, Jerabkova & Kroupa 2017,
A&A 607, A126 and Yan, Jerabkova, Kroupa & Vazdekis 2019, A&A) derives a
galaxy-wide IMF by integrating a per-embedded-cluster stellar IMF against an
embedded-cluster-mass-function whose slope depends on the instantaneous SFR,
with the maximum stellar mass per cluster set by solving a transcendental
mass-normalization equation. That's a real numerical subsystem (root-finding
+ nested integration), not a closed-form relation.

Per project direction (2026-09-04): a simpler method now, so the channel is
visually testable sooner, real IGIMF machinery deferred (same treatment as
the Sharda & Krumholz gas-thermodynamics/dust/radiation-field mechanism and
the AGN-radiation coupling already flagged as future work for this channel).

What IS implemented here, verified directly against `galIMF`'s own source
(`galimf.py`, `function_alpha_1_change`/`function_alpha_2_change`,
`alpha1_model=alpha2_model=1`): the metallicity-dependent shift of the
canonical Kroupa (2001) IMF's low-mass (0.08-0.5 Msun) and intermediate-mass
(0.5-1 Msun) power-law slopes,

    alpha1(Z) = alpha1_canonical + 0.5 * [M/H]
    alpha2(Z) = alpha2_canonical + 0.5 * [M/H]

with [M/H] = log10(Z / Z_sun) -- the same quantity as FSPS's own `logzsol`
parameter, so no separate conversion is needed once Z is expressed relative
to FSPS's assumed solar metallicity (Z_sun = 0.0142, Asplund et al. 2009,
matching the MIST isochrones this project uses). alpha1_canonical=1.3,
alpha2_canonical=2.3 (Kroupa 2001).

NOT implemented (fixed at the canonical Kroupa 2001 value, 2.3, regardless
of Z or SFR): alpha3, the high-mass (>1 Msun) slope -- galIMF's own
metallicity/density-dependent alpha3 relations require the embedded-cluster
density, which requires the cluster-mass-function integration this pass is
explicitly skipping. This means the SFR-driven "top-heavy at high SFR"
behavior IGIMF theory predicts is NOT captured here -- only the
metallicity-driven low/intermediate-mass shift is. Flagged, not hidden.

This module only computes IMF slopes for a *given* metallicity -- splitting
a continuous SFH into per-metallicity (and, in dynamic-IMF mode, per-IMF)
segments lives in `continuum.py`, not here, because that binning turned out
to be needed for metallicity too: `python-fsps` 0.5.0's multi-metallicity
tabulated-SFH mode (`zcontinuous=3`) hits an unconditional
`assert self._zcontinuous < 2, "Cannot use MDF with afe enhancement"` in its
own `_compute_csp()` regardless of whether alpha-enhancement is actually in
use (confirmed against the installed package and the current `dfm/python-
fsps` source on GitHub) -- there is no supported way to use `zcontinuous=3`
in this version. `continuum.py` works around this by binning Z the same way
this module's IMF slopes are applied per-bin: single representative
metallicity per FSPS call (`zcontinuous=1`), summed across bins.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Z_SUN = 0.0142  # Asplund et al. (2009); matches the MIST isochrones this project uses.

ALPHA1_CANONICAL = 1.3
ALPHA2_CANONICAL = 2.3
ALPHA3_CANONICAL = 2.3


@dataclass(frozen=True)
class IMFSlopes:
    imf1: float
    imf2: float
    imf3: float


def canonical_slopes() -> IMFSlopes:
    """The fixed, Z/SFR-independent Kroupa (2001) IMF -- used for
    `imf_mode="shared"` and as the fallback alpha3 in dynamic mode."""
    return IMFSlopes(ALPHA1_CANONICAL, ALPHA2_CANONICAL, ALPHA3_CANONICAL)


def metallicity_dependent_slopes(z_absolute: float) -> IMFSlopes:
    """alpha1/alpha2 shifted per the cited galIMF relation; alpha3 held
    canonical (see module docstring for what's NOT implemented here).
    `z_absolute` is metallicity in absolute units (matching
    `galaxy_continuum.metallicity`'s Z(t), not log or solar-relative).
    """
    if z_absolute <= 0.0:
        raise ValueError(f"z_absolute must be > 0, got {z_absolute}")
    m_over_h = np.log10(z_absolute / Z_SUN)
    return IMFSlopes(
        imf1=ALPHA1_CANONICAL + 0.5 * m_over_h,
        imf2=ALPHA2_CANONICAL + 0.5 * m_over_h,
        imf3=ALPHA3_CANONICAL,
    )
