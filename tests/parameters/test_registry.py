import pytest

from demiurge.parameters.distributions import Uniform
from demiurge.parameters.registry import REGISTRY, NPEParameter, get_parameter, list_parameters


def test_registry_is_empty_scaffolding():
    """The registry is deliberately empty until real channel modules are
    built alongside real entries (ground-up rebuild, not a bulk port from
    main -- see registry.py's module docstring). This test exists so an
    accidental bulk-population regression is caught immediately."""
    assert REGISTRY == {}


def test_construction_rejects_tier_1():
    with pytest.raises(ValueError):
        NPEParameter(
            name="bogus.tier1",
            owner="bogus",
            tier=1,
            physical=True,
            distribution=Uniform(0.0, 1.0),
            description="d",
            rationale="r",
        )


def test_construction_rejects_tier_2_without_citation():
    with pytest.raises(ValueError):
        NPEParameter(
            name="bogus.tier2_no_citation",
            owner="bogus",
            tier=2,
            physical=True,
            distribution=Uniform(0.0, 1.0),
            description="d",
            rationale="r",
            citation=None,
        )


def test_construction_rejects_missing_rationale():
    with pytest.raises(ValueError):
        NPEParameter(
            name="bogus.no_rationale",
            owner="bogus",
            tier=3,
            physical=True,
            distribution=Uniform(0.0, 1.0),
            description="d",
            rationale="",
        )


def test_construction_accepts_valid_tier3_without_citation():
    param = NPEParameter(
        name="bogus.valid_tier3",
        owner="bogus",
        tier=3,
        physical=True,
        distribution=Uniform(0.0, 1.0),
        description="d",
        rationale="uncited placeholder",
    )
    assert param.citation is None


def test_construction_accepts_valid_tier2_with_citation():
    param = NPEParameter(
        name="bogus.valid_tier2",
        owner="bogus",
        tier=2,
        physical=True,
        distribution=Uniform(0.0, 1.0),
        description="d",
        rationale="r",
        citation="Someone et al. (Year) measured this.",
    )
    assert param.citation is not None


def test_get_parameter_returns_the_right_entry(fake_registry):
    name = next(iter(fake_registry))
    assert get_parameter(name) is fake_registry[name]


def test_get_parameter_raises_keyerror_for_unknown_name(fake_registry):
    with pytest.raises(KeyError):
        get_parameter("not_a_real_parameter")


def test_list_parameters_filters_by_tier(fake_registry):
    tier2 = list_parameters(tier=2)
    assert all(p.tier == 2 for p in tier2)
    tier3 = list_parameters(tier=3)
    assert all(p.tier == 3 for p in tier3)
    assert len(tier2) + len(tier3) == len(fake_registry)


def test_list_parameters_filters_by_physical(fake_registry):
    physical = list_parameters(physical=True)
    assert all(p.physical for p in physical)
    nonphysical = list_parameters(physical=False)
    assert all(not p.physical for p in nonphysical)
    expected_nonphysical = {name for name, p in fake_registry.items() if not p.physical}
    assert {p.name for p in nonphysical} == expected_nonphysical
    assert len(expected_nonphysical) > 0


def test_list_parameters_filters_by_owner(fake_registry):
    owned = list_parameters(owner="fake_channel")
    assert len(owned) == len(fake_registry)
    assert list_parameters(owner="nonexistent_channel") == []


def test_list_parameters_combines_filters(fake_registry):
    combined = list_parameters(tier=3, physical=True)
    expected = {name for name, p in fake_registry.items() if p.tier == 3 and p.physical}
    assert {p.name for p in combined} == expected
    assert len(expected) > 0


def test_every_distribution_object_is_actually_drawable(fake_registry):
    import numpy as np

    rng = np.random.default_rng(0)
    for name, param in fake_registry.items():
        value = param.distribution.draw(rng)
        assert value is not None, f"{name}'s distribution produced None"
