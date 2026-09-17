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

alpha3, the HIGH-mass (>1 Msun) slope -- the one that actually controls how
many massive, metal-producing stars form -- is ALSO now metallicity-
dependent (added 2026-09-16, PI direction), via Recchi, Calura, Gibson &
Kroupa (2014, MNRAS 437, 994) eq. 6 ("mild" model, galIMF's own `'R14'`
option, independently confirmed against galIMF's source):

    alpha3(Z) = alpha3_canonical + 0.0572 * [Fe/H]

This is a genuine metallicity-ONLY simplification of galIMF's full
density+metallicity joint relation (Marks, Kroupa, Dabringhausen &
Pawlowski 2012, MNRAS 422, 2246, their eq. 15) -- Recchi et al. (2014)
derive it by fixing the embedded-cluster density at a constant fiducial
value inside that joint relation, avoiding the cluster-mass-function
integration this project still doesn't implement. [Fe/H] is approximated
here by the same [M/H]=log10(Z/Z_sun) quantity alpha1/alpha2 already use
(consistent with this module's existing convention, not a new one).
Previously alpha3 was held fixed at the canonical value regardless of Z --
this NOT-implemented gap, and the "SFR-driven top-heavy at high SFR"
behavior IGIMF theory separately predicts (still not captured -- that
piece needs the SFR-dependent term of the Marks et al. relation, which R14
does not simplify), are both worth remembering as the actual limitations,
now that the metallicity-only piece is real.

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
ALPHA3_R14_SLOPE = 0.0572  # Recchi, Calura, Gibson & Kroupa (2014, MNRAS 437, 994) eq. 6, "mild" model


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
    """alpha1/alpha2/alpha3 all shifted per the cited relations (see module
    docstring for what's still NOT implemented -- the SFR-dependent term).
    `z_absolute` is metallicity in absolute units (matching
    `galaxy_continuum.metallicity`'s Z(t), not log or solar-relative).
    """
    if z_absolute <= 0.0:
        raise ValueError(f"z_absolute must be > 0, got {z_absolute}")
    m_over_h = np.log10(z_absolute / Z_SUN)
    return IMFSlopes(
        imf1=ALPHA1_CANONICAL + 0.5 * m_over_h,
        imf2=ALPHA2_CANONICAL + 0.5 * m_over_h,
        imf3=ALPHA3_CANONICAL + ALPHA3_R14_SLOPE * m_over_h,
    )
