"""
Fast, pretabulated GalaxyContinuum synthesis backend.

`continuum.py`'s `fsps_direct` backend pays FSPS's own expensive isochrone-
to-spectrum computation (~15-25s, confirmed empirically 2026-09-04) once per
unique (Z, IMF) setting -- but a *fixed* (Z, IMF) setting's response across
*all ages* is essentially free once computed (FSPS caches its own full
native age grid internally). This module exploits exactly that: rather than
calling FSPS once per SFH time-bin, it uses a small, precomputed grid of
FSPS's full age-response at a modest number of Z points (built once, offline,
by `scripts/build_galaxy_continuum_ssp_grid.py`, shipped as package data
under `data/`) and reconstructs any SFH(t)/Z(t)-weighted composite spectrum
via pure numpy/torch bilinear interpolation in (log-age, log-Z) -- no FSPS
calls at all at generation time. Reconstruction cost is ~2-30ms regardless
of the grid's Z-resolution (confirmed empirically), several orders of
magnitude faster than `fsps_direct`'s per-bin FSPS calls.

Two grids ship (one per `imf_mode`, matching `continuum.py`'s own
"shared"/"dynamic" split): `ssp_grid_shared.npy` (fixed canonical Kroupa
2001 IMF at every Z) and `ssp_grid_dynamic.npy` (this project's own
simplified Z-dependent IMF -- see `imf.py`'s module docstring for exactly
what that does and doesn't capture). Each is `(n_z, n_age, n_wave)` float32,
Z log-spaced from `Z_FLOOR` to `Z_MASTER_MAX` (see the build script's own
docstring for the exact bounds/resolution rationale -- summary: wide enough
to cover the registered metallicity priors' full theoretical range, dense
enough (log-spaced, not linear -- ~4.4x better fidelity per point,
benchmarked directly) to keep reconstruction error small, sized to fit
under GitHub's 100MB per-file limit).

**Out-of-grid Z is a safety net, not a primary mechanism**: a Z(t) draw
that exceeds the grid's built bounds (a genuine tail case given how wide
they are) is clipped to the nearest edge rather than extrapolated or
erroring -- `SynthesisResult.n_clipped_steps` reports how many timesteps
needed clipping, so a caller can notice if this is happening more than
expected (which would suggest the grid needs widening, not that clipping
itself is wrong).

Every quantity here is validated against `fsps_direct` (the ground truth)
in `tests/galaxy_continuum/test_pretabulated.py` -- this module is not a
substitute for understanding whether `fsps_direct` itself is correct, only
a fast reconstruction of what it would have produced.

**Batched generation** (`synthesize_batch`): a single `synthesize()` call is
already ~2-30ms, but generating a real NPE-training set (thousands to
millions of mocks) at that per-call cost still means looping in Python,
paying per-call dispatch/transfer overhead N times. `synthesize_batch`
instead interpolates many mocks' worth of (SFR(t), Z(t)) draws in one
vectorized tensor op -- the same bilinear-interpolation math, with an added
leading batch axis. Requires all mocks in a batch to share the same
time-grid length (true by construction for `sfh.draw_sfh`'s fixed `n_grid`)
-- `t_obs_gyr` may still vary per mock. Measured speedup at the whole-
`GalaxyContinuum.from_dense_basis_batch` level (2026-09-08, N=50 mocks):
~1.7x, not orders of magnitude -- because only the synthesis step is
batched, not the SFH/metallicity *draw* step (still a Python loop of cheap
per-mock NumPy calls), which now dominates the batched path's total time
since synthesis itself is already so fast. Batching the draw step too is
the natural next optimization if throughput at real training-set scale
(not yet attempted) turns out to need it -- not done here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover -- exercised only in torch-less environments
    torch = None

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class SSPGrid:
    imf_mode: str
    z_grid: np.ndarray  # (n_z,) absolute Z, log-spaced, ascending
    wave: np.ndarray  # (n_wave,)
    age_logyr_min: float
    age_logyr_max: float
    n_age: int
    data: np.ndarray  # memmapped (n_z, n_age, n_wave) float32

    @property
    def logz_grid(self) -> np.ndarray:
        return np.log10(self.z_grid)


_GRID_CACHE: dict = {}


def load_grid(imf_mode: str) -> SSPGrid:
    """Loads (memmapped, not read fully into RAM) the shipped pretabulated
    grid for the given IMF mode. Cached per-process -- repeated calls are free.
    """
    if imf_mode in _GRID_CACHE:
        return _GRID_CACHE[imf_mode]

    metadata_path = DATA_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No pretabulated grid metadata at {metadata_path} -- run "
            f"scripts/build_galaxy_continuum_ssp_grid.py first."
        )
    metadata = json.loads(metadata_path.read_text())
    if imf_mode not in metadata:
        raise ValueError(f"No pretabulated grid for imf_mode={imf_mode!r}; available: {list(metadata)}")
    info = metadata[imf_mode]

    grid = SSPGrid(
        imf_mode=imf_mode,
        z_grid=np.asarray(info["z_grid"], dtype=np.float64),
        wave=np.load(DATA_DIR / "wave_grid.npy"),
        age_logyr_min=info["age_logyr_min"],
        age_logyr_max=info["age_logyr_max"],
        n_age=info["n_age"],
        data=np.load(DATA_DIR / f"ssp_grid_{imf_mode}.npy", mmap_mode="r"),
    )
    _GRID_CACHE[imf_mode] = grid
    return grid


def _resolve_backend(backend: str) -> str:
    if backend not in ("auto", "torch", "numpy"):
        raise ValueError(f"backend must be 'auto', 'torch', or 'numpy', got {backend!r}")
    if backend == "numpy":
        return "numpy"
    if backend == "torch" and torch is None:
        raise ImportError("backend='torch' requested but torch is not installed.")
    if backend == "auto" and torch is None:
        return "numpy"
    return "torch"


def _torch_device() -> "torch.device":
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass(frozen=True)
class SynthesisResult:
    wave: np.ndarray
    flux: np.ndarray
    n_clipped_steps: int
    backend: str


def synthesize(
    t_grid_gyr,
    sfr_msun_per_yr,
    z_grid_absolute,
    t_obs_gyr: float,
    *,
    imf_mode: str = "dynamic",
    backend: str = "auto",
) -> SynthesisResult:
    """SFH(t)-weighted composite spectrum via bilinear (log-age, log-Z)
    interpolation of the pretabulated grid -- no FSPS calls.

    `backend`: "auto" (torch-cuda if available, else torch-cpu, else numpy),
    "torch" (forces torch, raises if unavailable), or "numpy" (forces the
    numpy reference implementation).
    """
    grid = load_grid(imf_mode)
    resolved = _resolve_backend(backend)

    t_grid_gyr = np.asarray(t_grid_gyr, dtype=np.float64)
    sfr_msun_per_yr = np.asarray(sfr_msun_per_yr, dtype=np.float64)
    z_grid_absolute = np.asarray(z_grid_absolute, dtype=np.float64)

    age_floor_gyr = 10.0**grid.age_logyr_min / 1.0e9
    age_gyr_per_step = np.clip(t_obs_gyr - t_grid_gyr, age_floor_gyr, None)
    logage_per_step = np.log10(age_gyr_per_step * 1.0e9)

    z_clipped = np.clip(z_grid_absolute, grid.z_grid[0], grid.z_grid[-1])
    n_clipped_steps = int(np.sum(z_clipped != z_grid_absolute))
    logz_per_step = np.log10(z_clipped)

    mass_per_step = sfr_msun_per_yr * np.gradient(t_grid_gyr) * 1.0e9  # Msun/yr * Gyr(as yr) = Msun formed

    if resolved == "torch":
        flux = _reconstruct_torch(grid, logage_per_step, logz_per_step, mass_per_step)
    else:
        flux = _reconstruct_numpy(grid, logage_per_step, logz_per_step, mass_per_step)

    return SynthesisResult(wave=grid.wave, flux=flux, n_clipped_steps=n_clipped_steps, backend=resolved)


def _bilinear_indices_weights(coord: np.ndarray, grid_min: float, grid_max: float, n_grid: int):
    frac_idx = (coord - grid_min) / (grid_max - grid_min) * (n_grid - 1)
    frac_idx = np.clip(frac_idx, 0.0, n_grid - 1 - 1.0e-9)
    i0 = np.floor(frac_idx).astype(np.int64)
    i1 = i0 + 1
    w = frac_idx - i0
    return i0, i1, w


def _reconstruct_numpy(grid: SSPGrid, logage_per_step, logz_per_step, mass_per_step) -> np.ndarray:
    a0, a1, wa = _bilinear_indices_weights(logage_per_step, grid.age_logyr_min, grid.age_logyr_max, grid.n_age)
    z0, z1, wz = _bilinear_indices_weights(
        logz_per_step, grid.logz_grid[0], grid.logz_grid[-1], len(grid.z_grid)
    )

    data = np.asarray(grid.data)  # materialize the memmap slice actually touched
    f_z0a0 = data[z0, a0]
    f_z0a1 = data[z0, a1]
    f_z1a0 = data[z1, a0]
    f_z1a1 = data[z1, a1]

    wa_col = wa[:, None]
    wz_col = wz[:, None]
    per_step_flux = (
        (1 - wz_col) * (1 - wa_col) * f_z0a0
        + (1 - wz_col) * wa_col * f_z0a1
        + wz_col * (1 - wa_col) * f_z1a0
        + wz_col * wa_col * f_z1a1
    )
    return np.sum(per_step_flux * mass_per_step[:, None], axis=0)


def _reconstruct_torch(grid: SSPGrid, logage_per_step, logz_per_step, mass_per_step) -> np.ndarray:
    """Selects the (small) subset of grid points actually touched via numpy
    fancy indexing first (which -- unlike basic slicing -- always copies,
    so the result is writable and cheap to move to a device), rather than
    transferring the full ~90MB memmapped grid to the device on every call.
    """
    device = _torch_device()
    a0, a1, wa = _bilinear_indices_weights(logage_per_step, grid.age_logyr_min, grid.age_logyr_max, grid.n_age)
    z0, z1, wz = _bilinear_indices_weights(
        logz_per_step, grid.logz_grid[0], grid.logz_grid[-1], len(grid.z_grid)
    )

    data = grid.data  # memmapped (n_z, n_age, n_wave); fancy-indexing below copies just what's needed
    f_z0a0_np = data[z0, a0]
    f_z0a1_np = data[z0, a1]
    f_z1a0_np = data[z1, a0]
    f_z1a1_np = data[z1, a1]

    f_z0a0 = torch.as_tensor(f_z0a0_np, dtype=torch.float32, device=device)
    f_z0a1 = torch.as_tensor(f_z0a1_np, dtype=torch.float32, device=device)
    f_z1a0 = torch.as_tensor(f_z1a0_np, dtype=torch.float32, device=device)
    f_z1a1 = torch.as_tensor(f_z1a1_np, dtype=torch.float32, device=device)
    wa_t = torch.as_tensor(wa, dtype=torch.float32, device=device).unsqueeze(-1)
    wz_t = torch.as_tensor(wz, dtype=torch.float32, device=device).unsqueeze(-1)
    mass_t = torch.as_tensor(mass_per_step, dtype=torch.float32, device=device).unsqueeze(-1)

    per_step_flux = (
        (1 - wz_t) * (1 - wa_t) * f_z0a0
        + (1 - wz_t) * wa_t * f_z0a1
        + wz_t * (1 - wa_t) * f_z1a0
        + wz_t * wa_t * f_z1a1
    )
    composite = torch.sum(per_step_flux * mass_t, dim=0)
    return composite.detach().cpu().numpy().astype(np.float64)


@dataclass(frozen=True)
class BatchSynthesisResult:
    wave: np.ndarray  # (n_wave,)
    flux: np.ndarray  # (n_mocks, n_wave)
    n_clipped_steps: np.ndarray  # (n_mocks,)
    backend: str


def synthesize_batch(
    t_grid_gyr_batch,
    sfr_msun_per_yr_batch,
    z_grid_absolute_batch,
    t_obs_gyr_batch,
    *,
    imf_mode: str = "dynamic",
    backend: str = "auto",
) -> BatchSynthesisResult:
    """Batched version of `synthesize`: interpolates many mocks' worth of
    (SFR(t), Z(t)) draws in one vectorized call rather than looping. All
    three `*_batch` arrays must be shape (n_mocks, n_steps) -- same time-grid
    *length* across the batch (true by construction for `sfh.draw_sfh`'s
    fixed `n_grid`), though each mock's own `t_grid`/`t_obs` values may
    differ. `t_obs_gyr_batch` is shape (n_mocks,).
    """
    grid = load_grid(imf_mode)
    resolved = _resolve_backend(backend)

    t_grid_gyr_batch = np.asarray(t_grid_gyr_batch, dtype=np.float64)
    sfr_msun_per_yr_batch = np.asarray(sfr_msun_per_yr_batch, dtype=np.float64)
    z_grid_absolute_batch = np.asarray(z_grid_absolute_batch, dtype=np.float64)
    t_obs_gyr_batch = np.asarray(t_obs_gyr_batch, dtype=np.float64)

    age_floor_gyr = 10.0**grid.age_logyr_min / 1.0e9
    age_gyr_per_step = np.clip(t_obs_gyr_batch[:, None] - t_grid_gyr_batch, age_floor_gyr, None)
    logage_per_step = np.log10(age_gyr_per_step * 1.0e9)

    z_clipped = np.clip(z_grid_absolute_batch, grid.z_grid[0], grid.z_grid[-1])
    n_clipped_steps = np.sum(z_clipped != z_grid_absolute_batch, axis=1)
    logz_per_step = np.log10(z_clipped)

    mass_per_step = sfr_msun_per_yr_batch * np.gradient(t_grid_gyr_batch, axis=1) * 1.0e9

    if resolved == "torch":
        flux = _reconstruct_torch_batch(grid, logage_per_step, logz_per_step, mass_per_step)
    else:
        flux = _reconstruct_numpy_batch(grid, logage_per_step, logz_per_step, mass_per_step)

    return BatchSynthesisResult(wave=grid.wave, flux=flux, n_clipped_steps=n_clipped_steps, backend=resolved)


def _reconstruct_numpy_batch(grid: SSPGrid, logage_batch, logz_batch, mass_batch) -> np.ndarray:
    """Same math as `_reconstruct_numpy`, with a leading (n_mocks,) batch
    axis on every per-step quantity; sums over the steps axis (1), not the
    mocks axis (0)."""
    a0, a1, wa = _bilinear_indices_weights(logage_batch, grid.age_logyr_min, grid.age_logyr_max, grid.n_age)
    z0, z1, wz = _bilinear_indices_weights(logz_batch, grid.logz_grid[0], grid.logz_grid[-1], len(grid.z_grid))

    data = np.asarray(grid.data)
    f_z0a0 = data[z0, a0]  # (n_mocks, n_steps, n_wave)
    f_z0a1 = data[z0, a1]
    f_z1a0 = data[z1, a0]
    f_z1a1 = data[z1, a1]

    wa_ = wa[..., None]
    wz_ = wz[..., None]
    per_step_flux = (
        (1 - wz_) * (1 - wa_) * f_z0a0
        + (1 - wz_) * wa_ * f_z0a1
        + wz_ * (1 - wa_) * f_z1a0
        + wz_ * wa_ * f_z1a1
    )
    return np.sum(per_step_flux * mass_batch[..., None], axis=1)


def _reconstruct_torch_batch(grid: SSPGrid, logage_batch, logz_batch, mass_batch) -> np.ndarray:
    device = _torch_device()
    a0, a1, wa = _bilinear_indices_weights(logage_batch, grid.age_logyr_min, grid.age_logyr_max, grid.n_age)
    z0, z1, wz = _bilinear_indices_weights(logz_batch, grid.logz_grid[0], grid.logz_grid[-1], len(grid.z_grid))

    data = grid.data
    f_z0a0 = torch.as_tensor(data[z0, a0], dtype=torch.float32, device=device)
    f_z0a1 = torch.as_tensor(data[z0, a1], dtype=torch.float32, device=device)
    f_z1a0 = torch.as_tensor(data[z1, a0], dtype=torch.float32, device=device)
    f_z1a1 = torch.as_tensor(data[z1, a1], dtype=torch.float32, device=device)
    wa_t = torch.as_tensor(wa, dtype=torch.float32, device=device).unsqueeze(-1)
    wz_t = torch.as_tensor(wz, dtype=torch.float32, device=device).unsqueeze(-1)
    mass_t = torch.as_tensor(mass_batch, dtype=torch.float32, device=device).unsqueeze(-1)

    per_step_flux = (
        (1 - wz_t) * (1 - wa_t) * f_z0a0
        + (1 - wz_t) * wa_t * f_z0a1
        + wz_t * (1 - wa_t) * f_z1a0
        + wz_t * wa_t * f_z1a1
    )
    composite = torch.sum(per_step_flux * mass_t, dim=1)
    return composite.detach().cpu().numpy().astype(np.float64)
