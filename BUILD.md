# Building `demiurge`

This is the `refactor` branch's build/environment doc, replacing the old
`SETUP.md` (which described the legacy desisim-based fork and lives only on
`main` now). Describes current, real state only -- update it in the same
commit as whatever change makes it stale.

## Install

```bash
pip install -e .
```

Hard dependency: NumPy only (`requirements.txt` is the actual enforcement
mechanism for this -- see charter Sec. 3). Python >=3.10.

## Optional: PyTorch

```bash
pip install -e .[torch]
```

Per the charter's dependency-reduction target (Sec. 3.1): every
numerically-heavy operation auto-detects torch + CUDA at runtime and falls
back to NumPy when torch isn't installed, with NumPy as the reference
implementation both backends are tested against. First real example:
`demiurge.galaxy_continuum.pretabulated`'s SFH-weighted interpolation
(`backend="auto"|"torch"|"numpy"`, tested for numpy/torch agreement).

## Optional: python-fsps (only needed for the `fsps_direct` GalaxyContinuum
backend, or to rebuild the pretabulated grid)

**Most users do not need this.** `GalaxyContinuum`'s default
`backend="pretabulated"` reconstructs spectra via fast numpy/torch
interpolation of a precomputed SSP grid that ships as package data
(`src/demiurge/galaxy_continuum/data/`) -- no python-fsps, no Fortran
compiler, no WSL. python-fsps is only required for: `GalaxyContinuum.
from_single_population` (the exact-baseline-recovery path),
`backend="fsps_direct"` (the slower per-time-bin validation reference the
pretabulated grid is tested against), or rebuilding the pretabulated grid
itself (`scripts/build_galaxy_continuum_ssp_grid.py`, see that script's own
docstring). This resolves the "how do downstream users get this dependency"
question flagged in an earlier revision of this doc -- the pretabulated
path was built specifically to make python-fsps optional for ordinary use.

If you do need it: **requires WSL2 (Ubuntu) on Windows -- do not attempt a
native-Windows build.** `python-fsps`'s own CI only tests Ubuntu and macOS;
Windows is untested upstream and not worth the risk of silent breakage. On
Linux/macOS directly, skip the WSL framing below and just follow the same
steps in your native shell.

```bash
sudo apt-get install -y gfortran gcc make python3-dev   # if not already present
export SPS_HOME=/path/to/fsps_data                       # pick a location, ~5GB
git clone https://github.com/cconroy20/fsps.git "$SPS_HOME"
# persist SPS_HOME for this venv (e.g. append the export line above to
# <venv>/bin/activate) so it's set every time the venv is activated
FFLAGS="-DC3K_LR=0 -DC3K_HR=1" pip install fsps --no-binary fsps
```

`SPS_HOME` must be set *before* installing -- the build compiles FSPS's own
Fortran source found there, and reads its data files (isochrones, spectral
libraries) at runtime too. C3K_HR (high-resolution) replaces the default
C3K_LR spectral library; MIST isochrones are already FSPS's default, no
flag needed. Verify with:

```python
import fsps
sp = fsps.StellarPopulation(zcontinuous=1)
sp.libraries  # -> (b'mist', b'c3k_hr')
```

## What's NOT established yet

Mirroring the old `SETUP.md`'s own honesty about not guessing at
environment specifics: nothing about NERSC, simqso, or any other
DESI-specific data product or environment is documented here, because no
physics-generation module in this refactor currently depends on any of them.
These sections get written for real once a module that actually needs them
is ported (charter Sec. 3.2/3.5), not guessed at in advance.

## `scripts/`

Auxiliary one-off/maintenance scripts that aren't part of the installed
package (not imported at runtime, not shipped) -- each documents its own
purpose and usage in its own header docstring. Currently:
`build_galaxy_continuum_ssp_grid.py` (precomputes the pretabulated galaxy-
continuum SSP grid shipped as package data; needs python-fsps, see above).

## Testing

```bash
pip install -e ".[dev]"
pytest                  # fast tests only (excludes `slow`)
pytest -m slow           # the FSPS-dependent tests too (~15-25s per real fsps call)
```

`dev` pulls in `pytest` and `matplotlib` (test/visual-verification tools,
not runtime dependencies -- kept out of `requirements.txt`). Tests live
under `tests/`, mirroring the `src/demiurge/` package layout. The charter's
per-push testing gate (Sec. 4) applies: every push runs the fast suite at
minimum; anything touching `galaxy_continuum` should also run `-m slow`
before pushing.

Current coverage: `demiurge.rng` (reproducible/independent child RNG
streams); `demiurge.parameters` (the NPE-parameter registry, its
distribution families, and `PriorSampler`); `demiurge.galaxy_continuum`
(SFH/metallicity/IMF generation, the pretabulated numpy/torch interpolation
backend, and `GalaxyContinuum` itself -- including exact bit-for-bit
baseline recovery against raw `fsps.StellarPopulation`, `slow`-marked since
most of these exercise real FSPS calls).
