import numpy as np
import pytest

from demiurge.parameters.distributions import (
    Bernoulli,
    DiscreteUniform,
    Gamma,
    LogNormal,
    LogUniform,
    MaxwellBoltzmann,
    Normal,
    Poisson,
    Uniform,
)

RNG = np.random.default_rng(0)


@pytest.mark.parametrize(
    "dist,support",
    [
        (Uniform(1.0, 5.0), (1.0, 5.0)),
        (LogUniform(0.1, 100.0), (0.1, 100.0)),
        (Gamma(1.0, 0.041), (0.0, np.inf)),
        (LogNormal(0.0, 0.5), (0.0, np.inf)),
        (Poisson(2.5), (0.0, np.inf)),
        (MaxwellBoltzmann(50.0), (0.0, np.inf)),
    ],
)
def test_scalar_draw_within_support(dist, support):
    lo, hi = support
    for _ in range(200):
        x = dist.draw(RNG)
        assert lo <= x <= hi


def test_uniform_rejects_low_ge_high():
    with pytest.raises(ValueError):
        Uniform(5.0, 5.0)
    with pytest.raises(ValueError):
        Uniform(5.0, 1.0)


def test_loguniform_rejects_nonpositive_low():
    with pytest.raises(ValueError):
        LogUniform(0.0, 10.0)
    with pytest.raises(ValueError):
        LogUniform(-1.0, 10.0)


def test_normal_and_lognormal_reject_nonpositive_sigma():
    with pytest.raises(ValueError):
        Normal(0.0, 0.0)
    with pytest.raises(ValueError):
        LogNormal(0.0, -1.0)


def test_gamma_rejects_nonpositive_params():
    with pytest.raises(ValueError):
        Gamma(0.0, 1.0)
    with pytest.raises(ValueError):
        Gamma(1.0, 0.0)


def test_bernoulli_rejects_out_of_range_p():
    with pytest.raises(ValueError):
        Bernoulli(1.5)
    with pytest.raises(ValueError):
        Bernoulli(-0.1)


def test_poisson_rejects_negative_mean():
    with pytest.raises(ValueError):
        Poisson(-1.0)


def test_discrete_uniform_only_returns_listed_values():
    values = (0.0, 20.0, 50.0, 100.0)
    dist = DiscreteUniform(values)
    draws = [dist.draw(RNG) for _ in range(200)]
    assert set(draws) <= set(values)
    assert dist.support == values


def test_discrete_uniform_rejects_empty():
    with pytest.raises(ValueError):
        DiscreteUniform(())


def test_bernoulli_extremes_are_deterministic():
    assert Bernoulli(0.0).draw(RNG) == False  # noqa: E712
    assert Bernoulli(1.0).draw(RNG) == True  # noqa: E712


def test_vectorized_draw_shape():
    dist = Uniform(0.0, 1.0)
    arr = dist.draw(RNG, size=1000)
    assert arr.shape == (1000,)
    assert np.all((arr >= 0.0) & (arr <= 1.0))


def test_reproducible_given_same_generator_state():
    d = LogNormal(0.5, 0.2)
    a = d.draw(np.random.default_rng(42), size=10)
    b = d.draw(np.random.default_rng(42), size=10)
    np.testing.assert_array_equal(a, b)


def test_lognormal_matches_manual_transform():
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(1)
    dist = LogNormal(mean=1.0, sigma=0.3)
    drawn = dist.draw(rng1, size=100)
    manual = 10.0 ** rng2.normal(1.0, 0.3, size=100)
    np.testing.assert_array_equal(drawn, manual)


def test_loguniform_matches_manual_transform():
    rng1 = np.random.default_rng(2)
    rng2 = np.random.default_rng(2)
    dist = LogUniform(low=2.0, high=20.0)
    drawn = dist.draw(rng1, size=100)
    manual = 10.0 ** rng2.uniform(np.log10(2.0), np.log10(20.0), size=100)
    np.testing.assert_array_equal(drawn, manual)


def test_maxwell_boltzmann_mean_matches_theory():
    # Analytic mean of a Maxwell-Boltzmann speed distribution is scale * 2*sqrt(2/pi)
    dist = MaxwellBoltzmann(50.0)
    draws = dist.draw(np.random.default_rng(3), size=200_000)
    expected_mean = 50.0 * 2.0 * np.sqrt(2.0 / np.pi)
    assert draws.mean() == pytest.approx(expected_mean, rel=0.02)
