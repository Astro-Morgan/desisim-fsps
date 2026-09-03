import numpy as np
import pytest

from demiurge.parameters.registry import REGISTRY
from demiurge.parameters.samplers import PriorSampler

SOME_TIER2_PHYSICAL = "igm_absorption.dla_boost"
SOME_TIER3_PHYSICAL = "dust.theta0_amplitude"


def test_sample_returns_one_value_per_name():
    sampler = PriorSampler()
    names = [SOME_TIER2_PHYSICAL, SOME_TIER3_PHYSICAL]
    out = sampler.sample(names, rng=np.random.default_rng(0))
    assert set(out.keys()) == set(names)


def test_sample_is_reproducible_given_identical_generator_state():
    sampler = PriorSampler()
    names = list(REGISTRY.keys())[:10]
    out1 = sampler.sample(names, rng=np.random.default_rng(123))
    out2 = sampler.sample(names, rng=np.random.default_rng(123))
    for name in names:
        assert out1[name] == out2[name] or np.array_equal(out1[name], out2[name])


def test_condition_fixes_value_instead_of_drawing():
    sampler = PriorSampler()
    out = sampler.sample(
        [SOME_TIER2_PHYSICAL], rng=np.random.default_rng(0), condition={SOME_TIER2_PHYSICAL: 1.5}
    )
    assert out[SOME_TIER2_PHYSICAL] == 1.5


def test_condition_on_unregistered_name_raises_keyerror():
    sampler = PriorSampler()
    with pytest.raises(KeyError):
        sampler.sample(
            [SOME_TIER2_PHYSICAL], rng=np.random.default_rng(0), condition={"not_a_real_parameter": 1.0}
        )


def test_all_registered_parameters_are_physical_so_no_non_physical_rejection_case_exists_yet():
    """Every current registry entry is `physical=True` (HANDOFF3 Sec 5.2's
    non-physical example, camera calibration, isn't ported yet) -- this test
    exists to make that fact explicit and force a real non-physical-rejection
    test to be added the moment the first non-physical parameter lands."""
    assert all(p.physical for p in REGISTRY.values())


def test_vectorized_sample_shape():
    sampler = PriorSampler()
    out = sampler.sample([SOME_TIER3_PHYSICAL], rng=np.random.default_rng(0), size=50)
    assert out[SOME_TIER3_PHYSICAL].shape == (50,)


def test_vectorized_condition_broadcasts_fixed_value():
    sampler = PriorSampler()
    out = sampler.sample(
        [SOME_TIER2_PHYSICAL],
        rng=np.random.default_rng(0),
        condition={SOME_TIER2_PHYSICAL: 1.5},
        size=20,
    )
    arr = out[SOME_TIER2_PHYSICAL]
    assert arr.shape == (20,)
    assert np.all(arr == 1.5)


def test_sampling_every_registered_parameter_at_once_does_not_raise():
    sampler = PriorSampler()
    names = list(REGISTRY.keys())
    out = sampler.sample(names, rng=np.random.default_rng(0))
    assert len(out) == len(names)
