import numpy as np
import pytest

from demiurge.rng import child_generator, child_generators


def test_reproducible_for_same_seed():
    a = child_generators(123, 4)
    b = child_generators(123, 4)
    for ga, gb in zip(a, b):
        assert ga.uniform() == gb.uniform()


def test_different_seeds_diverge():
    a = child_generators(1, 1)[0]
    b = child_generators(2, 1)[0]
    assert a.uniform() != b.uniform()


def test_children_are_mutually_independent_streams():
    streams = child_generators(7, 3)
    draws = [g.uniform(size=10) for g in streams]
    assert not np.array_equal(draws[0], draws[1])
    assert not np.array_equal(draws[1], draws[2])


def test_spawn_prefix_stability():
    """Requesting the first N children and later requesting more must reproduce
    the same first N -- this is the property HANDOFF2 Sec. 4's narrow/broad
    desync bug violated (consuming extra draws from one stream shifted a
    sibling's own results)."""
    first_two = child_generators(99, 2)
    first_three = child_generators(99, 3)
    for g1, g2 in zip(first_two, first_three):
        assert g1.uniform() == g2.uniform()


def test_child_generator_singular_matches_plural():
    single = child_generator(5)
    plural = child_generators(5, 1)[0]
    assert single.uniform() == plural.uniform()


def test_rejects_zero_or_negative_n():
    with pytest.raises(ValueError):
        child_generators(1, 0)


def test_none_seed_is_non_reproducible():
    a = child_generators(None, 1)[0]
    b = child_generators(None, 1)[0]
    assert a.uniform() != b.uniform()
