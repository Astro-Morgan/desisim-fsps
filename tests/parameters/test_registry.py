import pytest

from demiurge.parameters.distributions import Uniform
from demiurge.parameters.registry import NPEParameter, REGISTRY, get_parameter, list_parameters


def test_registry_is_nonempty():
    assert len(REGISTRY) > 0


def test_every_entry_is_tier_2_or_3():
    for name, param in REGISTRY.items():
        assert param.tier in (2, 3), f"{name} has invalid tier {param.tier}"


def test_every_tier_2_entry_has_a_citation():
    for name, param in REGISTRY.items():
        if param.tier == 2:
            assert param.citation, f"{name} is Tier 2 but has no citation"


def test_every_entry_has_a_rationale():
    for name, param in REGISTRY.items():
        assert param.rationale, f"{name} has no rationale"


def test_names_are_unique_by_construction():
    # REGISTRY is a dict keyed by name, so this is really testing _build_registry's
    # duplicate check didn't silently swallow a collision -- every param's own
    # .name must match the key it's filed under.
    for key, param in REGISTRY.items():
        assert key == param.name


def test_get_parameter_returns_the_right_entry():
    name = next(iter(REGISTRY))
    assert get_parameter(name) is REGISTRY[name]


def test_get_parameter_raises_keyerror_for_unknown_name():
    with pytest.raises(KeyError):
        get_parameter("not_a_real_parameter")


def test_list_parameters_filters_by_tier():
    tier2 = list_parameters(tier=2)
    assert all(p.tier == 2 for p in tier2)
    assert len(tier2) < len(REGISTRY)


def test_list_parameters_filters_by_physical():
    physical = list_parameters(physical=True)
    assert all(p.physical for p in physical)


def test_list_parameters_filters_by_owner():
    owner = next(iter(REGISTRY.values())).owner
    owned = list_parameters(owner=owner)
    assert all(p.owner == owner for p in owned)
    assert len(owned) > 0


def test_list_parameters_combines_filters():
    combined = list_parameters(tier=2, physical=True)
    assert all(p.tier == 2 and p.physical for p in combined)


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


def test_known_bug_context_dla_boost_is_tier2_physical():
    """Spot-check one parameter with a well-documented provenance (HANDOFF3
    Sec. 6.4 names dla_boost as the worked Tier 2 example) to catch a wholesale
    mis-transcription of the registry."""
    param = get_parameter("igm_absorption.dla_boost")
    assert param.tier == 2
    assert param.physical is True
    assert param.owner == "igm_absorption"


def test_every_distribution_object_is_actually_drawable():
    import numpy as np

    rng = np.random.default_rng(0)
    for name, param in REGISTRY.items():
        value = param.distribution.draw(rng)
        assert value is not None, f"{name}'s distribution produced None"
