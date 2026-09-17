"""
Self-consistent, metallicity-dependent effective chemical yield --
replaces the assumption that `galaxy_continuum.metallicity.yield` is a
single constant per mock with a real `yield(Z)` derived from actual
supernova nucleosynthesis data and this project's own metallicity-
dependent IMF (`imf.metallicity_dependent_slopes`, including the alpha3(Z)
piece added alongside this module). Closes the gap Andy identified
2026-09-16: a top-heavier IMF at low Z should mean more massive,
metal-producing stars form per unit mass, hence higher effective yield --
previously the dynamic-IMF mode and the closed-box enrichment model ran
side by side sharing Z(t) as an input/output, with no actual feedback
between them.

NOT YET WIRED into the default generation path
(`metallicity.draw_metallicity`, `population_inversion.py`) -- deliberately.
`population_inversion.py`'s `mass_weighted_z_closed_form` (and the whole
targeted-sampling mechanism built on it) relies on Z(t) having a closed
form independent of SFH shape, which is true for the OLD constant-yield
relation but is NOT true once yield depends on the evolving Z(t) itself (the
ODE below has no such shape-independent closed form). Wiring this in as the
default would mean reworking those closed-form shortcuts into numerical
ones -- a real, separate integration decision, not made in this pass.

=== Real data: Limongi & Chieffi (2018, ApJS 237, 13) ===

Isotope-by-isotope core-collapse supernova yields, VizieR/CDS catalog
J/ApJS/237/13, table8.dat (machine-readable, downloaded and parsed
directly -- NOT via OCR, unlike the Woosley & Weaver 1995 table this
project's alpha3(Z)-adjacent literature (Recchi & Kroupa 2015) itself
uses, which turned out not to be cleanly extractable). `_METAL_EJECTA_MSUN`
below is the SUM of every tabulated isotope's ejected mass EXCEPT H/He
(all Z>2 elements -- i.e. gross ejected metal mass, not net-of-initial-
metallicity yield; a real, documented simplification, not hidden), at
their non-rotating (v=0), "recommended" model set, for the SAME 9
progenitor masses (13-120 Msun) at each of their own real tabulated
metallicities ([Fe/H] = 0, -1, -2, -3). No CCSN contribution is modeled
below their own 13 Msun floor (real stars 8-13 Msun do produce some CCSN
yield; this table doesn't cover them, so this treatment doesn't either --
a documented gap, not an oversight).

=== The IMF integral ===

`effective_yield(alpha1, alpha2, alpha3, metal_ejecta_table)` integrates
this project's own Kroupa-broken-power-law IMF (continuity-normalized at
the standard 0.5/1.0 Msun breaks) against the (log-linearly interpolated)
metal-ejecta-vs-mass table, giving metal mass ejected per unit stellar
mass FORMED across the whole IMF (0.08-120 Msun, capped at the data's own
upper mass rather than extrapolated beyond it).

=== Validation (2026-09-16, real sandbox run before this was trusted) ===

At the canonical (untouched Kroupa 2001) IMF, alpha=(1.3, 2.3, 2.3), using
the SOLAR [Fe/H]=0 table: computed effective yield = 0.00980 -- matches
Recchi & Kroupa (2015)'s own stated "typical" canonical-IMF value (0.01)
to within 2%, and is the same order of magnitude as Kobayashi, Karakas &
Lugaro (2020)'s independently-computed IMF-integrated yield (~0.015,
Table 3). Using the metallicity-MATCHED table (not just solar) at each
real [Fe/H] point, together with `imf.metallicity_dependent_slopes`'s own
alpha1/alpha2/alpha3(Z), gives the real, self-consistent (Z, yield) anchor
points `_ANCHOR_YIELD_AT_FEH` below -- a genuinely smaller low-Z yield
boost than using the solar table throughout would suggest (the star's own
lower metallicity partially counteracts, but does not cancel, the
IMF-driven increase): yield rises smoothly from 0.00980 at [Fe/H]=0 to
0.01485 at [Fe/H]=-3 (a real ~1.5x increase, not the ~1.9x an earlier,
less complete pass through this calculation found using the solar table
everywhere).

=== Central curve + PI-directed spread (Andy, 2026-09-16) ===

`yield_mean(z_absolute)`: a monotone cubic Hermite spline (hand-rolled,
Fritsch-Carlson/PCHIP-style -- no scipy, per this project's NumPy-only
hard dependency, BUILD.md) through the 4 real anchor points above, held
flat outside [Fe/H] in [-3, 0] (same "hold flat beyond the table's own
coverage" convention already used in `dissociation_radius.py`/
`population_inversion.py`, not a new rule invented here) -- L&C18 doesn't
tabulate super-solar metallicity at all, so [Fe/H]>0 is a real, documented
limitation, not silently papered over.

`yield_sigma_relative(z_absolute)`: reflects genuinely different
confidence by region, per PI direction -- INSIDE the real L&C18 coverage
([Fe/H] in [-3,0]), 35% relative scatter (grounded in the real cross-
methodology spread found even at the best-anchored point: this module's
own computed 0.00980 vs. Kobayashi et al.'s 0.015 differ by ~53%, vs.
Recchi & Kroupa's 0.01 by ~2% -- 35% sits inside that real spread, not
invented from nothing); OUTSIDE that range (pure extrapolation, no real
per-mass data at all), 70% -- MAGIC widths in the sense that the exact
percentages are a judgment call, but the judgment itself (inside < outside)
is grounded in real, cited evidence, not arbitrary.

`galaxy_continuum.metallicity.yield_scatter` (new Tier-3 registered
parameter, Normal(0,1)) is drawn ONCE per mock -- a fixed "which real
nucleosynthesis/fallback prescription does this galaxy's true history
follow" offset for the whole enrichment history, not re-randomized at each
timestep (which would have no physical interpretation): effective yield at
a given Z is `yield_mean(Z) * (1 + yield_sigma_relative(Z) * yield_scatter)`,
floored well above zero to keep the ODE below well-posed for any real draw.

=== The self-consistency ODE ===

`Z(t)` no longer has a closed form once yield depends on the evolving Z(t)
itself. Recchi & Kroupa (2015, MNRAS 446, 4168) solve the general (SFR-and-
Z-dependent) version of this problem, but it is genuinely IMPLICIT there
(their own words: "this system of equations is indeed implicit... an
implicit procedure must be employed") because their SFR and Z depend on
each other circularly. This project's IMF treatment has no SFR-dependence
at all (a pre-existing, documented scope limit in `imf.py`, not something
added for this feature) -- dropping that term removes the circularity
entirely, leaving a plain INITIAL-VALUE ODE:

    dZ/dmu = -yield_effective(Z) / mu,   Z(mu=1) = 0

integrated forward (mu decreasing from 1) via a hand-rolled RK4 stepper
(again, no scipy) over the SAME mu(t) grid `sfh.reconstruct_cumulative_sfh`
already produces -- no new grid, no iteration needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import imf as imf_module
from ..parameters.samplers import ParameterSampler, PriorSampler

_YIELD_SCATTER_NAME = "galaxy_continuum.metallicity.yield_scatter"

# ---- real Limongi & Chieffi (2018) data ----
# VizieR J/ApJS/237/13, table8.dat, v=0 (non-rotating), summed over every
# tabulated isotope except H/H2/H3/He3/He4 (328 metal isotopes), at each of
# the file's own real [Fe/H] codes. Downloaded and parsed directly
# (2026-09-16), not read off a figure or OCR'd from a scan.
_MASSES_MSUN = np.array([13.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0, 80.0, 120.0])

_METAL_EJECTA_MSUN = {
    0: np.array([6.749248e-01, 1.314905e+00, 2.606660e+00, 4.564619e+00,
                 2.565713e-01, 3.449399e-01, 9.703552e-01, 1.964179e+00, 3.413545e+00]),
    -1: np.array([6.578418e-01, 1.501054e+00, 2.599694e+00, 4.317722e+00,
                  5.578018e-03, 3.623632e-02, 5.742547e-02, 1.262457e-01, 5.734194e-01]),
    -2: np.array([6.282411e-01, 1.111910e+00, 2.587498e+00, 4.136460e+00,
                  4.468925e-05, 8.168837e-05, 1.237607e-04, 4.537695e-04, 5.518233e-03]),
    -3: np.array([6.523548e-01, 1.434025e+00, 2.634037e+00, 4.272272e+00,
                  6.496869e-07, 1.034294e-06, 1.852964e-06, 3.201398e-06, 1.104310e-05]),
}

_FEH_COVERAGE_LOW, _FEH_COVERAGE_HIGH = -3.0, 0.0
_SIGMA_REL_INSIDE, _SIGMA_REL_OUTSIDE = 0.35, 0.70
_YIELD_FLOOR_FRACTION = 0.05  # effective yield never drops below this fraction of yield_mean, any scatter draw

_M_BREAK1, _M_BREAK2, _M_LOW, _M_HIGH = 0.5, 1.0, 0.08, 120.0
_Z_FLOOR_FOR_LOG = 1.0e-12  # avoids log10(0) at Z=0 (the pristine-gas ODE starting point)


def _feh_of(z_absolute) -> np.ndarray:
    z_floored = np.maximum(np.asarray(z_absolute, dtype=float), _Z_FLOOR_FOR_LOG)
    return np.log10(z_floored / imf_module.Z_SUN)


def _imf_dn_dm(m: np.ndarray, alpha1: float, alpha2: float, alpha3: float) -> np.ndarray:
    """Kroupa (2001) broken power-law number density, continuity-normalized
    at the standard 0.5/1.0 Msun breaks (overall normalization is
    arbitrary -- cancels in `effective_yield`'s ratio)."""
    k1 = 1.0
    k2 = k1 * _M_BREAK1 ** (alpha2 - alpha1)
    k3 = k2 * _M_BREAK2 ** (alpha3 - alpha2)
    return np.where(
        m < _M_BREAK1, k1 * m ** (-alpha1),
        np.where(m < _M_BREAK2, k2 * m ** (-alpha2), k3 * m ** (-alpha3)),
    )


def _metal_yield_of_mass(m: np.ndarray, metal_ejecta_table: np.ndarray) -> np.ndarray:
    out = np.interp(m, _MASSES_MSUN, metal_ejecta_table, left=0.0, right=metal_ejecta_table[-1])
    return np.where(m < _MASSES_MSUN[0], 0.0, out)


def effective_yield(alpha1: float, alpha2: float, alpha3: float, metal_ejecta_table: np.ndarray,
                     n_grid: int = 20000) -> float:
    """Metal mass ejected per unit stellar mass formed, integrating
    `metal_ejecta_table` (indexed to `_MASSES_MSUN`) against the Kroupa IMF
    with the given slopes, over the full 0.08-120 Msun range."""
    m_grid = np.geomspace(_M_LOW, _M_HIGH, n_grid)
    dn_dm = _imf_dn_dm(m_grid, alpha1, alpha2, alpha3)
    total_mass_formed = np.trapezoid(m_grid * dn_dm, m_grid)
    total_metal_ejected = np.trapezoid(_metal_yield_of_mass(m_grid, metal_ejecta_table) * dn_dm, m_grid)
    return float(total_metal_ejected / total_mass_formed)


def _compute_anchor_points() -> tuple[np.ndarray, np.ndarray]:
    """The 4 real (feh, yield) anchors -- each metallicity's OWN matching
    L&C18 table combined with `imf.metallicity_dependent_slopes` at that
    same metallicity (not the solar table reused everywhere, which would
    overstate the low-Z yield boost -- caught and corrected 2026-09-16
    before this module was written this way)."""
    fehs = np.array(sorted(_METAL_EJECTA_MSUN.keys()), dtype=float)  # [-3, -2, -1, 0]
    yields = np.empty_like(fehs)
    for i, feh in enumerate(fehs):
        z_abs = imf_module.Z_SUN * 10.0 ** feh
        slopes = imf_module.metallicity_dependent_slopes(z_abs)
        yields[i] = effective_yield(slopes.imf1, slopes.imf2, slopes.imf3, _METAL_EJECTA_MSUN[int(feh)])
    return fehs, yields


_ANCHOR_FEH, _ANCHOR_YIELD_AT_FEH = _compute_anchor_points()


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson monotone cubic Hermite derivative estimates at each
    knot (the standard PCHIP algorithm, hand-rolled -- no scipy). Assumes
    `x` sorted ascending and `y` monotonic (both true for `_ANCHOR_FEH`/
    `_ANCHOR_YIELD_AT_FEH`, verified in tests)."""
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(n)

    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0.0:
            d[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    def _endpoint(h0, h1, delta0, delta1):
        d0 = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / (h0 + h1)
        if np.sign(d0) != np.sign(delta0):
            d0 = 0.0
        elif np.sign(delta0) != np.sign(delta1) and abs(d0) > 3.0 * abs(delta0):
            d0 = 3.0 * delta0
        return d0

    d[0] = _endpoint(h[0], h[1], delta[0], delta[1])
    d[-1] = _endpoint(h[-1], h[-2], delta[-1], delta[-2])
    return d


_ANCHOR_SLOPES = _pchip_slopes(_ANCHOR_FEH, _ANCHOR_YIELD_AT_FEH)


def _pchip_eval(x_query: np.ndarray, x: np.ndarray, y: np.ndarray, slopes: np.ndarray) -> np.ndarray:
    x_query = np.clip(x_query, x[0], x[-1])
    idx = np.clip(np.searchsorted(x, x_query, side="right") - 1, 0, len(x) - 2)
    h_i = x[idx + 1] - x[idx]
    t = (x_query - x[idx]) / h_i
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2
    return h00 * y[idx] + h10 * h_i * slopes[idx] + h01 * y[idx + 1] + h11 * h_i * slopes[idx + 1]


def yield_mean(z_absolute):
    """The central yield(Z) curve: a monotone PCHIP spline through the 4
    real anchor points, held flat for [Fe/H] outside [-3, 0] (L&C18's own
    real coverage -- clipped, not extrapolated, in `_pchip_eval` above).
    Scalar in, scalar out; array in, array out (added 2026-09-17 so
    `integrate_self_consistent_z_batch` below can call this once across
    many points per RK4 step instead of once per point -- verified
    bit-for-bit identical to looping the scalar path, see
    tests/galaxy_continuum/test_metal_yield.py)."""
    scalar_input = np.ndim(z_absolute) == 0
    feh = _feh_of(z_absolute)
    result = _pchip_eval(np.atleast_1d(feh), _ANCHOR_FEH, _ANCHOR_YIELD_AT_FEH, _ANCHOR_SLOPES)
    return float(result[0]) if scalar_input else result


def yield_sigma_relative(z_absolute):
    """Coverage-dependent relative scatter width (PI direction, 2026-09-16):
    smaller inside L&C18's real [Fe/H] in [-3,0] coverage, larger outside
    (pure extrapolation, no real per-mass data there at all). Scalar/array
    in, matching out -- see `yield_mean`'s docstring."""
    scalar_input = np.ndim(z_absolute) == 0
    feh = _feh_of(z_absolute)
    inside = (feh >= _FEH_COVERAGE_LOW) & (feh <= _FEH_COVERAGE_HIGH)
    result = np.where(inside, _SIGMA_REL_INSIDE, _SIGMA_REL_OUTSIDE)
    return float(result) if scalar_input else result


def effective_yield_at_z(z_absolute, yield_scatter):
    """`yield_mean(Z) * (1 + yield_sigma_relative(Z) * yield_scatter)`,
    floored at `_YIELD_FLOOR_FRACTION` of the mean so no real `yield_scatter`
    draw can push the effective yield to zero or negative. Scalar/array in,
    matching out."""
    mean = yield_mean(z_absolute)
    raw = mean * (1.0 + yield_sigma_relative(z_absolute) * yield_scatter)
    floor = _YIELD_FLOOR_FRACTION * mean
    return max(raw, floor) if np.ndim(z_absolute) == 0 else np.maximum(raw, floor)


def integrate_self_consistent_z(
    mu_grid: np.ndarray, yield_scatter: float, *, yield_fn=None
) -> np.ndarray:
    """RK4 integration of dZ/dmu = -yield_fn(Z, yield_scatter)/mu over
    `mu_grid` (must be monotonically non-increasing, starting at 1.0 --
    `sfh.reconstruct_cumulative_sfh`'s own mu(t) = 1 - efficiency*F(t)
    convention), returning Z at each grid point. Z(mu=1) = 0 by construction
    (pristine gas, matching the existing closed-box convention). `yield_fn`
    defaults to `effective_yield_at_z`; injectable so the RK4 stepping
    itself can be validated against a synthetic constant-yield RHS
    independent of the real yield_mean(Z) curve's own shape (see
    tests/galaxy_continuum/test_metal_yield.py)."""
    if yield_fn is None:
        yield_fn = effective_yield_at_z
    mu_grid = np.asarray(mu_grid, dtype=float)
    z = np.zeros_like(mu_grid)

    def rhs(mu, z_val):
        if mu <= 0.0:
            return 0.0
        return -yield_fn(max(z_val, 0.0), yield_scatter) / mu

    for i in range(len(mu_grid) - 1):
        mu0, mu1 = mu_grid[i], mu_grid[i + 1]
        h = mu1 - mu0
        k1 = rhs(mu0, z[i])
        k2 = rhs(mu0 + 0.5 * h, z[i] + 0.5 * h * k1)
        k3 = rhs(mu0 + 0.5 * h, z[i] + 0.5 * h * k2)
        k4 = rhs(mu1, z[i] + h * k3)
        z[i + 1] = max(z[i] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)

    return z


def integrate_self_consistent_z_batch(mu_grid_batch: np.ndarray, yield_scatter_batch: np.ndarray) -> np.ndarray:
    """Vectorized sibling of `integrate_self_consistent_z` -- same RK4
    scheme, but stepping ALL rows of `mu_grid_batch` (shape (n_points,
    n_grid), each row one point's own mu(t) trajectory) forward together,
    one vectorized `effective_yield_at_z` call per time-step instead of
    one Python-level call per (point, step) pair. `yield_scatter_batch`
    has shape (n_points,). Returns the FULL Z trajectory, shape
    (n_points, n_grid) -- matching `integrate_self_consistent_z`'s own
    return convention (every grid point, not just the endpoint):
    `population_inversion.py` needs the whole trajectory to compute the
    mass-weighted integral (`Z_bar = integral Z dF`), NOT just `Z(t_obs)`
    -- an earlier version of this function returned only the endpoint,
    which is a genuinely different (and here, wrong) quantity, the same
    Z(t_obs)-vs-mass-weighted-Z confusion this project already found and
    fixed once (see `mass_weighted_z_closed_form`'s docstring) -- caught
    before it shipped, not after.

    Added 2026-09-17 -- real numbers motivated this (Andy: "~1m per draw
    is too slow"): the self-consistent model is ~300x slower per call than
    the old closed-box relation (a real, measured cost of the RK4 stepping
    itself, not the physics), so building a batch of targeted samples
    needs this to stay practical. Verified bit-for-bit identical to
    looping `integrate_self_consistent_z` per row before being trusted
    (see tests/galaxy_continuum/test_metal_yield.py) -- this is the exact
    same arithmetic, just batched, not an approximation.
    """
    mu_grid_batch = np.asarray(mu_grid_batch, dtype=float)
    yield_scatter_batch = np.asarray(yield_scatter_batch, dtype=float)
    n_points, n_grid = mu_grid_batch.shape
    z = np.zeros((n_points, n_grid))

    def rhs(mu_vec, z_vec):
        safe_mu = np.where(mu_vec > 0.0, mu_vec, 1.0)
        val = -effective_yield_at_z(np.maximum(z_vec, 0.0), yield_scatter_batch) / safe_mu
        return np.where(mu_vec > 0.0, val, 0.0)

    for i in range(n_grid - 1):
        mu0 = mu_grid_batch[:, i]
        mu1 = mu_grid_batch[:, i + 1]
        h = mu1 - mu0
        z_i = z[:, i]
        k1 = rhs(mu0, z_i)
        k2 = rhs(mu0 + 0.5 * h, z_i + 0.5 * h * k1)
        k3 = rhs(mu0 + 0.5 * h, z_i + 0.5 * h * k2)
        k4 = rhs(mu1, z_i + h * k3)
        z[:, i + 1] = np.maximum(z_i + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)

    return z


@dataclass(frozen=True)
class SelfConsistentMetallicityResult:
    z_grid: np.ndarray
    yield_scatter: float


def draw_self_consistent_metallicity(
    rng: np.random.Generator,
    mu_grid: np.ndarray,
    *,
    sampler: Optional[ParameterSampler] = None,
) -> SelfConsistentMetallicityResult:
    """Draws `yield_scatter` and integrates Z(t) via the ODE above. NOT
    wired into `metallicity.draw_metallicity`'s default path -- see module
    docstring for why."""
    if sampler is None:
        sampler = PriorSampler()
    draws = sampler.sample([_YIELD_SCATTER_NAME], rng=rng)
    yield_scatter = float(draws[_YIELD_SCATTER_NAME])
    z_grid = integrate_self_consistent_z(mu_grid, yield_scatter)
    return SelfConsistentMetallicityResult(z_grid=z_grid, yield_scatter=yield_scatter)
