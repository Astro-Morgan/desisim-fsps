import pytest

from demiurge.parameters import registry as registry_module
from demiurge.parameters.distributions import Uniform
from demiurge.parameters.registry import NPEParameter

FAKE_TIER2_PHYSICAL = "fake.tier2_physical"
FAKE_TIER3_PHYSICAL = "fake.tier3_physical"
FAKE_TIER3_NONPHYSICAL = "fake.tier3_nonphysical"


@pytest.fixture
def fake_registry(monkeypatch):
    """A small, self-contained registry (one Tier 2 physical, one Tier 3
    physical, one Tier 3 non-physical entry) swapped in for the real
    registry -- which is deliberately empty right now, see registry.py's
    module docstring -- so tests exercise lookup/filter/conditioning
    mechanics without depending on real channel content existing yet.
    """
    params = {
        FAKE_TIER2_PHYSICAL: NPEParameter(
            name=FAKE_TIER2_PHYSICAL,
            owner="fake_channel",
            tier=2,
            physical=True,
            distribution=Uniform(0.0, 1.0),
            description="fixture parameter",
            rationale="fixture parameter",
            citation="fixture citation",
        ),
        FAKE_TIER3_PHYSICAL: NPEParameter(
            name=FAKE_TIER3_PHYSICAL,
            owner="fake_channel",
            tier=3,
            physical=True,
            distribution=Uniform(0.0, 1.0),
            description="fixture parameter",
            rationale="fixture parameter",
        ),
        FAKE_TIER3_NONPHYSICAL: NPEParameter(
            name=FAKE_TIER3_NONPHYSICAL,
            owner="fake_channel",
            tier=3,
            physical=False,
            distribution=Uniform(0.0, 1.0),
            description="fixture parameter",
            rationale="fixture parameter",
        ),
    }
    monkeypatch.setattr(registry_module, "REGISTRY", params)
    return params
