import numpy as np
import pytest

from demiurge.parameters.samplers import PriorSampler


def _names_by(fake_registry, **filters):
    return [
        name
        for name, p in fake_registry.items()
        if all(getattr(p, k) == v for k, v in filters.items())
    ]


def test_sample_returns_one_value_per_name(fake_registry):
    sampler = PriorSampler()
    names = list(fake_registry.keys())
    out = sampler.sample(names, rng=np.random.default_rng(0))
    assert set(out.keys()) == set(names)


def test_sample_is_reproducible_given_identical_generator_state(fake_registry):
    sampler = PriorSampler()
    names = list(fake_registry.keys())
    out1 = sampler.sample(names, rng=np.random.default_rng(123))
    out2 = sampler.sample(names, rng=np.random.default_rng(123))
    for name in names:
        assert out1[name] == out2[name] or np.array_equal(out1[name], out2[name])


def test_condition_fixes_value_instead_of_drawing(fake_registry):
    sampler = PriorSampler()
    physical_name = _names_by(fake_registry, physical=True)[0]
    out = sampler.sample([physical_name], rng=np.random.default_rng(0), condition={physical_name: 1.5})
    assert out[physical_name] == 1.5


def test_condition_on_unregistered_name_raises_keyerror(fake_registry):
    sampler = PriorSampler()
    physical_name = _names_by(fake_registry, physical=True)[0]
    with pytest.raises(KeyError):
        sampler.sample([physical_name], rng=np.random.default_rng(0), condition={"not_a_real_parameter": 1.0})


def test_condition_on_nonphysical_parameter_is_rejected(fake_registry):
    """Enforces HANDOFF3 Sec. 5.2's rule: only physical NPE-parameters may be
    supplied as a conditioning input. This is the first real test of that
    rejection path -- the production registry had no non-physical entries
    yet when the sampler was first built, so this could only be asserted as
    vacuously true before; the fixture registry gives it a real case."""
    sampler = PriorSampler()
    nonphysical_name = _names_by(fake_registry, physical=False)[0]
    with pytest.raises(ValueError):
        sampler.sample(
            [nonphysical_name], rng=np.random.default_rng(0), condition={nonphysical_name: 1.0}
        )


def test_vectorized_sample_shape(fake_registry):
    sampler = PriorSampler()
    name = next(iter(fake_registry))
    out = sampler.sample([name], rng=np.random.default_rng(0), size=50)
    assert out[name].shape == (50,)


def test_vectorized_condition_broadcasts_fixed_value(fake_registry):
    sampler = PriorSampler()
    physical_name = _names_by(fake_registry, physical=True)[0]
    out = sampler.sample(
        [physical_name], rng=np.random.default_rng(0), condition={physical_name: 1.5}, size=20
    )
    arr = out[physical_name]
    assert arr.shape == (20,)
    assert np.all(arr == 1.5)


def test_sampling_every_fixture_parameter_at_once_does_not_raise(fake_registry):
    sampler = PriorSampler()
    names = list(fake_registry.keys())
    out = sampler.sample(names, rng=np.random.default_rng(0))
    assert len(out) == len(names)
