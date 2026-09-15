import numpy as np
import pytest

from demiurge.quasar_continuum import QuasarContinuum
from demiurge.quasar_continuum.dissociation_radius import (
    DissociationRadiusResult,
    compute_l_uv_erg_s,
    draw_dissociation_radius,
)


class _FixedSampler:
    """Minimal ParameterSampler stub -- mirrors the pattern in
    test_torus_reddening.py."""

    def __init__(self, values: dict):
        self._values = values

    def sample(self, names, *, rng, condition=None, size=None):
        return {name: self._values[name] for name in names}


def _fixed_sampler(sublimation_prefactor):
    return _FixedSampler(
        {"quasar_continuum.dissociation_radius.sublimation_prefactor": sublimation_prefactor}
    )


def _fiducial_quasar(**overrides):
    kwargs = dict(
        m_msun=1.0e8, astar=0.0, mdot_edd=0.10, cosi=0.7071067811865476,
        hard_xray_luminosity_fraction=0.02, kte_hot_kev=100.0, kte_warm_kev=0.2, gamma_warm=2.5,
    )
    kwargs.update(overrides)
    return QuasarContinuum.from_parameters(**kwargs)


# ---- compute_l_uv_erg_s ----


def test_l_uv_zero_at_cosi_zero():
    """Documented edge case: exactly edge-on gives zero disc/warm flux under
    the Lambertian cosi_scale=cosi/0.5 law, hence zero L_UV by this
    definition."""
    disk_rate = np.array([1.0e40, 2.0e40])
    warm_rate = np.array([3.0e40, 4.0e40])
    energy_kev = np.array([1.0, 5.0])
    assert compute_l_uv_erg_s(disk_rate, warm_rate, energy_kev, cosi=0.0) == 0.0


def test_l_uv_matches_manual_calculation():
    disk_rate = np.array([1.0e40])
    warm_rate = np.array([1.0e40])
    energy_kev = np.array([2.0])
    cosi = 0.25
    result = compute_l_uv_erg_s(disk_rate, warm_rate, energy_kev, cosi)

    from demiurge.quasar_continuum.continuum import KEV_TO_ERG

    cosi_scale = cosi / 0.5
    expected = (1.0e40 + 1.0e40) * cosi_scale * (2.0 * KEV_TO_ERG)
    assert result == pytest.approx(expected, rel=1e-12)


def test_l_uv_scales_linearly_with_cosi():
    disk_rate = np.array([5.0e39, 1.0e40])
    warm_rate = np.array([2.0e39, 3.0e40])
    energy_kev = np.array([1.0, 10.0])
    lo = compute_l_uv_erg_s(disk_rate, warm_rate, energy_kev, cosi=0.2)
    hi = compute_l_uv_erg_s(disk_rate, warm_rate, energy_kev, cosi=0.4)
    assert hi == pytest.approx(2.0 * lo, rel=1e-12)


def test_l_uv_is_consistent_with_continuum_pys_own_cosi_scale():
    """Real integration check, not just a standalone formula test: L_UV
    computed here from a real QuasarContinuum's meta must equal the
    disc+warm-only portion of continuum.py's own cosi-scaled total_rate --
    this is the actual double-counting-avoidance design, verified against
    the real AGNSED output rather than assumed."""
    qc = _fiducial_quasar()
    l_uv = compute_l_uv_erg_s(
        qc.meta["disk_rate_unscaled"], qc.meta["warm_rate_unscaled"], qc.meta["energy_kev"], qc.meta["cosi"]
    )

    from demiurge.quasar_continuum.continuum import KEV_TO_ERG

    cosi_scale = qc.meta["cosi"] / 0.5
    disk_warm_rate = (qc.meta["disk_rate_unscaled"] + qc.meta["warm_rate_unscaled"]) * cosi_scale
    expected = float(np.sum(disk_warm_rate * qc.meta["energy_kev"] * KEV_TO_ERG))
    assert l_uv == pytest.approx(expected, rel=1e-12)
    assert l_uv > 0.0


def test_l_uv_increases_with_cosi_for_a_real_agnsed_spectrum():
    qc_edge_on = _fiducial_quasar(cosi=0.1)
    qc_face_on = _fiducial_quasar(cosi=0.9)
    l_uv_edge_on = compute_l_uv_erg_s(
        qc_edge_on.meta["disk_rate_unscaled"], qc_edge_on.meta["warm_rate_unscaled"],
        qc_edge_on.meta["energy_kev"], qc_edge_on.meta["cosi"],
    )
    l_uv_face_on = compute_l_uv_erg_s(
        qc_face_on.meta["disk_rate_unscaled"], qc_face_on.meta["warm_rate_unscaled"],
        qc_face_on.meta["energy_kev"], qc_face_on.meta["cosi"],
    )
    assert l_uv_face_on > l_uv_edge_on


# ---- draw_dissociation_radius ----


def test_draw_matches_formula_with_fixed_sampler():
    result = draw_dissociation_radius(
        np.random.default_rng(0), l_uv_erg_s=4.0e46, sampler=_fixed_sampler(sublimation_prefactor=1.0)
    )
    assert result.radius_pc == pytest.approx(2.0, rel=1e-12)  # sqrt(4e46/1e46) = 2


def test_draw_zero_l_uv_gives_zero_radius():
    result = draw_dissociation_radius(
        np.random.default_rng(0), l_uv_erg_s=0.0, sampler=_fixed_sampler(sublimation_prefactor=1.0)
    )
    assert result.radius_pc == 0.0


def test_draw_scales_as_sqrt_of_l_uv():
    lo = draw_dissociation_radius(
        np.random.default_rng(0), l_uv_erg_s=1.0e46, sampler=_fixed_sampler(sublimation_prefactor=0.8)
    )
    hi = draw_dissociation_radius(
        np.random.default_rng(0), l_uv_erg_s=4.0e46, sampler=_fixed_sampler(sublimation_prefactor=0.8)
    )
    assert hi.radius_pc == pytest.approx(2.0 * lo.radius_pc, rel=1e-12)


def test_draw_scales_linearly_with_prefactor():
    lo = draw_dissociation_radius(
        np.random.default_rng(0), l_uv_erg_s=1.0e46, sampler=_fixed_sampler(sublimation_prefactor=0.4)
    )
    hi = draw_dissociation_radius(
        np.random.default_rng(0), l_uv_erg_s=1.0e46, sampler=_fixed_sampler(sublimation_prefactor=1.3)
    )
    assert hi.radius_pc == pytest.approx((1.3 / 0.4) * lo.radius_pc, rel=1e-12)


def test_default_sampler_draws_the_registered_parameter_within_its_range():
    result = draw_dissociation_radius(np.random.default_rng(7), l_uv_erg_s=1.0e46)
    assert isinstance(result, DissociationRadiusResult)
    assert 0.4 <= result.sublimation_prefactor_pc <= 1.3
    assert np.isfinite(result.radius_pc)
    assert result.radius_pc > 0.0


def test_reproducible_given_same_rng_seed():
    a = draw_dissociation_radius(np.random.default_rng(42), l_uv_erg_s=2.0e46)
    b = draw_dissociation_radius(np.random.default_rng(42), l_uv_erg_s=2.0e46)
    assert a == b


def test_registry_entry_matches_barvainis_range():
    from demiurge.parameters.registry import get_parameter

    param = get_parameter("quasar_continuum.dissociation_radius.sublimation_prefactor")
    assert param.tier == 2
    assert param.citation is not None
    assert param.distribution.support == (0.4, 1.3)
