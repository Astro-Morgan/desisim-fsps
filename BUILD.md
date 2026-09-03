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

## What's NOT established yet

Mirroring the old `SETUP.md`'s own honesty about not guessing at
environment specifics: nothing about NERSC, FSPS, simqso, or any other
DESI-specific data product or environment is documented here, because no
module in this refactor currently depends on any of them -- this is a
bootstrap skeleton (`src/demiurge/__init__.py` only). These sections get
written for real once a module that actually needs them is ported (charter
Sec. 3.2/3.3/3.5), not guessed at in advance.

## Testing

No test suite exists yet. Once physics modules land, this section
documents how to run it, and the charter's per-push testing gate (Sec. 4)
applies from the first real module onward.
