"""
Gas-phase metal depletion onto dust grains -- the mechanism this project's
ionization-emulator work needs to keep an emission-line calculation's
gas-phase + dust-locked metal budget self-consistent with whatever total
metallicity a volume element's own stellar population implies (see project
memory: `project-ionization-emulator-design`'s "beyond Cue" item 5), not yet
consumed by anything since the emulator itself doesn't exist yet.

Gutkin, Charlot & Bruzual (2016, MNRAS 462, 1757, "GCB16") parameterize
depletion with a single free dust-to-metal MASS ratio, `xi_d` (0<=xi_d<=1,
solar baseline `xi_d,sun=0.36` -- i.e. 36% of solar heavy-element mass in the
solid phase): every element's gas-phase depletion factor is a piecewise-
linear function of `xi_d`, anchored at three points -- `f_dpl=0` at `xi_d=0`
(no depletion, gas-phase abundance equals the total), GCB16's own real
measured Table 1 value at `xi_d=xi_d,sun=0.36` (the ONLY independently
measured anchor -- the endpoints are boundary conditions, not separate
measurements), and `f_dpl=1` at `xi_d=1` (fully depleted). This is a
genuinely stronger self-consistency mechanism than a fixed set of depletion
factors (e.g. Cue's own Dopita et al. 2000 constants, [[project-ionization-
emulator-design]]) -- one Tier-2 scalar per volume element ties gas-phase
depletion directly to whatever total (gas+dust) metal budget that element's
own stellar population implies, rather than an independently-fixed rule with
no connection to the actual enrichment history.

`GCB16_TABLE1_AT_SOLAR` below is GCB16's own Table 1, `f_dpl^i` AT
`xi_d=0.36` specifically (not at `xi_d=1`, an easy misreading this project's
own planning session made once before correcting it against the real paper).
Only the elements this project currently has a use for are included --
extend when a real need arises, not speculatively.

`xi_d^NLR` extends the identical mechanism to AGN narrow-line-region gas
(Vidal-Garcia, Plat, Curtis-Lake, Feltre, Hirschmann, Chevallard & Charlot
2024, MNRAS 527, 7217, "beagle-agn I", built on Feltre, Charlot & Gutkin
2016, MNRAS 456, 3354's own AGN-NLR CLOUDY grid) -- same `depletion_factor`
mechanism, its own independently-registered draw
(`dust.depletion.xi_d_nlr`). Both parameters are plain registered scalars
(drawn via `PriorSampler` like `blending.quasar_frac`, no dedicated `draw_*`
wrapper needed here -- neither draw depends on any other module's output the
way `quasar_continuum.dissociation_radius` needed AGNSED's own luminosity).
"""
from __future__ import annotations

# GCB16 Table 1: f_dpl^i at xi_d = XI_D_SOLAR (0.36), the one independently
# measured anchor point -- verified against the actual paper 2026-09-15, not
# assumed to be at xi_d=1.
XI_D_SOLAR = 0.36

GCB16_TABLE1_AT_SOLAR = {
    "C": 0.50,
    "N": 0.0,  # non-refractory (GCB16's own Table 1 footnote, verified 2026-09-15:
    # "the non-refractory elements He, N, Ne, S and Ar have f_dpl^i = 0" -- held at
    # 0 for EVERY xi_d, not interpolated like the refractory elements below.
    "O": 0.30,
    "Mg": 0.80,
    "Si": 0.90,
    "Fe": 0.99,
}


def depletion_factor(element: str, xi_d: float) -> float:
    """f_dpl(element, xi_d).

    For a REFRACTORY element (GCB16's own term -- everything in
    `GCB16_TABLE1_AT_SOLAR` with a nonzero table value), this is
    piecewise-linear through (0, 0), (XI_D_SOLAR,
    GCB16_TABLE1_AT_SOLAR[element]), (1, 1) -- GCB16's own stated boundary
    condition ("this must satisfy f_dpl^i = 0 and 1 for xi_d = 0 and 1,
    respectively"), holding by construction at the anchor (verified in
    tests, not just asserted).

    NON-refractory elements (nitrogen here -- GCB16's own Table 1 footnote
    also lists He/Ne/S/Ar, not yet needed by this project) do NOT
    interpolate at all: GCB16 state explicitly that non-refractory elements
    have `f_dpl=0` for EVERY xi_d, not just at xi_d=XI_D_SOLAR -- an earlier
    version of this function wrongly ran nitrogen through the same (0,1)
    interpolation as the refractory elements (caught by
    `test_nitrogen_is_undepleted_at_every_xi_d`, verified against the real
    paper text before fixing rather than just patched to pass).
    """
    if not (0.0 <= xi_d <= 1.0):
        raise ValueError(f"xi_d must be in [0, 1], got {xi_d}")
    try:
        f_solar = GCB16_TABLE1_AT_SOLAR[element]
    except KeyError:
        raise KeyError(
            f"{element!r} has no registered GCB16 Table 1 depletion factor -- "
            f"known elements: {sorted(GCB16_TABLE1_AT_SOLAR)}"
        ) from None

    if f_solar == 0.0:
        return 0.0  # non-refractory: f_dpl=0 for every xi_d, not just at XI_D_SOLAR
    if xi_d <= XI_D_SOLAR:
        return f_solar * (xi_d / XI_D_SOLAR)
    return f_solar + (1.0 - f_solar) * (xi_d - XI_D_SOLAR) / (1.0 - XI_D_SOLAR)


def gas_phase_fraction(element: str, xi_d: float) -> float:
    """1 - depletion_factor(element, xi_d) -- the fraction of that element's
    TOTAL (gas+dust) abundance remaining in the gas phase."""
    return 1.0 - depletion_factor(element, xi_d)
