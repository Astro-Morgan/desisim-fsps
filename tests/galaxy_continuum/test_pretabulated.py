"""
Tests for demiurge.galaxy_continuum.pretabulated's interpolation math, using
small synthetic grids -- these don't need the real shipped SSP data, so they
run fast and don't require python-fsps. Fidelity-against-fsps_direct tests
(using the real shipped grid) live in test_continuum.py alongside the other
`slow`, FSPS-dependent tests.
"""
import numpy as np
import pytest

from demiurge.galaxy_continuum.pretabulated import (
    SSPGrid,
    _bilinear_indices_weights,
    _reconstruct_numpy,
    synthesize,
)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _make_synthetic_grid(n_z=3, n_age=5, n_wave=4) -> SSPGrid:
    """data[z, a, w] = 10*z_idx + a_idx (broadcast identically across wave)
    -- simple enough to hand-verify interpolated values."""
    z_grid = np.logspace(-3, -1, n_z)  # e.g. [0.001, 0.01, 0.1] for n_z=3
    data = np.zeros((n_z, n_age, n_wave), dtype=np.float32)
    for zi in range(n_z):
        for ai in range(n_age):
            data[zi, ai, :] = 10 * zi + ai
    return SSPGrid(
        imf_mode="test",
        z_grid=z_grid,
        wave=np.linspace(1000.0, 2000.0, n_wave),
        age_logyr_min=6.0,
        age_logyr_max=10.0,
        n_age=n_age,
        data=data,
    )


# ---------------------------------------------------------------------------
# _bilinear_indices_weights
# ---------------------------------------------------------------------------
def test_bilinear_weights_at_grid_min():
    i0, i1, w = _bilinear_indices_weights(np.array([0.0]), 0.0, 4.0, 5)
    assert i0[0] == 0 and i1[0] == 1
    assert w[0] == pytest.approx(0.0, abs=1e-9)


def test_bilinear_weights_at_grid_max():
    i0, i1, w = _bilinear_indices_weights(np.array([4.0]), 0.0, 4.0, 5)
    assert i0[0] == 3 and i1[0] == 4
    assert w[0] == pytest.approx(1.0, abs=1e-6)


def test_bilinear_weights_at_midpoint():
    i0, i1, w = _bilinear_indices_weights(np.array([2.0]), 0.0, 4.0, 5)
    assert i0[0] == 2 and i1[0] == 3
    assert w[0] == pytest.approx(0.0, abs=1e-9)  # 2.0 lands exactly on grid index 2


def test_bilinear_weights_quarter_point():
    # grid points at 0,1,2,3,4 (n=5); coord=0.5 -> between index 0 and 1, w=0.5
    i0, i1, w = _bilinear_indices_weights(np.array([0.5]), 0.0, 4.0, 5)
    assert i0[0] == 0 and i1[0] == 1
    assert w[0] == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# synthesize() end-to-end on the synthetic grid
# ---------------------------------------------------------------------------
def test_synthesize_single_timestep_recovers_exact_grid_point_numpy():
    """A single-timestep 'SFH' at exactly a grid (age, Z) point should
    recover that grid cell's value exactly (times the formed mass)."""
    grid = _make_synthetic_grid()
    # t_obs - t = 10**6 / 1e9 Gyr -> logage=6.0 exactly (grid's age_logyr_min, index 0)
    t_obs = 10.0**6 / 1.0e9
    t_grid = np.array([0.0, t_obs])  # two points so np.gradient has something to work with
    sfr = np.array([0.0, 0.0])
    sfr[-1] = 1.0  # all "mass" at the last step... simpler: just test the interpolation directly instead

    # Direct check via internal reconstruction with a single explicit step
    logage_per_step = np.array([6.0])  # exactly grid_min -> a0=0,a1=1,wa=0
    logz_per_step = np.array([np.log10(grid.z_grid[0])])  # exactly z_grid[0] -> z0=0,z1=1,wz=0
    mass_per_step = np.array([1.0])
    flux = _reconstruct_numpy(grid, logage_per_step, logz_per_step, mass_per_step)
    # data[z=0, a=0, :] = 10*0 + 0 = 0
    np.testing.assert_allclose(flux, np.zeros(4), atol=1e-6)


def test_synthesize_interpolates_between_grid_points():
    grid = _make_synthetic_grid()
    # halfway between age index 0 (logyr=6.0) and 1 (logyr=7.0) -> logyr=6.5
    logage_per_step = np.array([6.5])
    logz_per_step = np.array([np.log10(grid.z_grid[0])])  # z0=0 exactly
    mass_per_step = np.array([1.0])
    flux = _reconstruct_numpy(grid, logage_per_step, logz_per_step, mass_per_step)
    # data[z=0,a=0]=0, data[z=0,a=1]=1 -> halfway = 0.5
    np.testing.assert_allclose(flux, np.full(4, 0.5), atol=1e-6)


def test_synthesize_mass_weighting_scales_linearly():
    grid = _make_synthetic_grid()
    logage_per_step = np.array([6.0])
    logz_per_step = np.array([np.log10(grid.z_grid[1])])  # z index 1 -> data=10
    mass_per_step = np.array([3.0])
    flux = _reconstruct_numpy(grid, logage_per_step, logz_per_step, mass_per_step)
    np.testing.assert_allclose(flux, np.full(4, 30.0), atol=1e-5)


def test_synthesize_clips_out_of_range_z_and_flags_it(monkeypatch):
    grid = _make_synthetic_grid()
    monkeypatch.setattr("demiurge.galaxy_continuum.pretabulated.load_grid", lambda imf_mode: grid)

    t_grid = np.linspace(0.0, 1.0, 5)
    sfr = np.ones_like(t_grid)
    z_way_too_high = np.full_like(t_grid, 10.0)  # far beyond grid.z_grid[-1]

    result = synthesize(t_grid, sfr, z_way_too_high, t_obs_gyr=1.0, imf_mode="test", backend="numpy")
    assert result.n_clipped_steps == len(t_grid)
    assert np.all(np.isfinite(result.flux))


def test_synthesize_no_clipping_when_z_within_range(monkeypatch):
    grid = _make_synthetic_grid()
    monkeypatch.setattr("demiurge.galaxy_continuum.pretabulated.load_grid", lambda imf_mode: grid)

    t_grid = np.linspace(0.0, 1.0, 5)
    sfr = np.ones_like(t_grid)
    z_in_range = np.full_like(t_grid, grid.z_grid[1])

    result = synthesize(t_grid, sfr, z_in_range, t_obs_gyr=1.0, imf_mode="test", backend="numpy")
    assert result.n_clipped_steps == 0


def test_synthesize_rejects_invalid_imf_mode(monkeypatch):
    def fake_load_grid(imf_mode):
        if imf_mode != "test":
            raise ValueError(f"No pretabulated grid for imf_mode={imf_mode!r}; available: ['test']")
        return _make_synthetic_grid()

    monkeypatch.setattr("demiurge.galaxy_continuum.pretabulated.load_grid", fake_load_grid)
    with pytest.raises(ValueError):
        synthesize([0, 1], [1, 1], [0.01, 0.01], t_obs_gyr=1.0, imf_mode="bogus")


def test_synthesize_rejects_invalid_backend(monkeypatch):
    grid = _make_synthetic_grid()
    monkeypatch.setattr("demiurge.galaxy_continuum.pretabulated.load_grid", lambda imf_mode: grid)
    with pytest.raises(ValueError):
        synthesize([0, 1], [1, 1], [0.01, 0.01], t_obs_gyr=1.0, imf_mode="test", backend="bogus")


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_torch_and_numpy_backends_agree(monkeypatch):
    grid = _make_synthetic_grid(n_z=5, n_age=6, n_wave=8)
    monkeypatch.setattr("demiurge.galaxy_continuum.pretabulated.load_grid", lambda imf_mode: grid)

    rng = np.random.default_rng(0)
    t_grid = np.sort(rng.uniform(0.0, 1.0, 20))
    sfr = rng.uniform(0.1, 1.0, 20)
    z_grid_vals = np.exp(rng.uniform(np.log(grid.z_grid[0]), np.log(grid.z_grid[-1]), 20))

    result_numpy = synthesize(t_grid, sfr, z_grid_vals, t_obs_gyr=1.0, imf_mode="test", backend="numpy")
    result_torch = synthesize(t_grid, sfr, z_grid_vals, t_obs_gyr=1.0, imf_mode="test", backend="torch")

    np.testing.assert_allclose(result_numpy.flux, result_torch.flux, rtol=1e-4, atol=1e-6)


def test_torch_backend_raises_when_torch_unavailable(monkeypatch):
    monkeypatch.setattr("demiurge.galaxy_continuum.pretabulated.torch", None)
    grid = _make_synthetic_grid()
    monkeypatch.setattr("demiurge.galaxy_continuum.pretabulated.load_grid", lambda imf_mode: grid)
    with pytest.raises(ImportError):
        synthesize([0, 1], [1, 1], [0.01, 0.01], t_obs_gyr=1.0, imf_mode="test", backend="torch")
