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
numerically-heavy operation should auto-detect torch + CUDA at runtime and
fall back to NumPy when torch isn't installed, with NumPy as the reference
implementation both backends are tested against. **Not implemented yet** --
no torch-backed code exists in this skeleton. This section gets filled in
once the first such module lands.

## Optional: python-fsps (stellar population synthesis, C3K_HR + MIST)

Needed for the galaxy continuum channel. **Requires WSL2 (Ubuntu) on
Windows -- do not attempt a native-Windows build.** `python-fsps`'s own CI
only tests Ubuntu and macOS; Windows is untested upstream and not worth
the risk of silent breakage. On Linux/macOS directly, skip the WSL framing
below and just follow the same steps in your native shell.

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

Per the charter's dependency-reduction goal (Sec. 3.3): `python-fsps` is
being kept as a real dependency for now, not vendored away -- that's
explicitly allowed as a fallback outcome if the exercised FSPS code path
turns out not to be cleanly separable, and is being treated as the current
state rather than pre-judged. How this affects downstream `pip install
demiurge` users (who won't have WSL/a Fortran toolchain set up by default)
is an open design question, not yet resolved -- see project discussion
before assuming an answer.

## What's NOT established yet

Mirroring the old `SETUP.md`'s own honesty about not guessing at
environment specifics: nothing about NERSC, simqso, or any other
DESI-specific data product or environment is documented here, because no
physics-generation module in this refactor currently depends on any of them.
These sections get written for real once a module that actually needs them
is ported (charter Sec. 3.2/3.5), not guessed at in advance.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

`dev` pulls in `pytest` (test-only, not a runtime dependency -- kept out of
`requirements.txt`). Tests live under `tests/`, mirroring the `src/demiurge/`
package layout. The charter's per-push testing gate (Sec. 4) applies: every
push runs this first.

Current coverage: `demiurge.rng` (reproducible/independent child RNG streams)
and `demiurge.parameters` (the NPE-parameter registry, its distribution
families, and `PriorSampler`).
