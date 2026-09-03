# demiurge

Ground-up simulator + NPE-parameter sampling framework for synthetic
DESI-like spectra, decomposed by construction into named additive channels
(continuum / emission / absorption) with known ground truth -- the training
signal a downstream neural decomposer needs, since that decomposition isn't
observable for real spectra. Part of the DESI-Flow framework.

**Current state: bootstrap skeleton.** No physics modules have been ported
yet -- see the `refactor` branch's own history for progress. This
description will be extended as real modules land; it does not describe
anything aspirational.

## Relationship to `main`

`main` holds the prior `desisim-fsps` fork -- a working reference
implementation this package is being re-derived from, not incrementally
patched from. See `main`'s own `SETUP.md`/module docstrings for that
implementation's design rationale and validated citations/math; nothing on
`main` is being changed by this effort.

## Install

See [BUILD.md](BUILD.md).

## Design docs

Design decisions, physics/citation rationale, and the NPE-parameter
architecture this package implements are tracked in project documentation
outside this repository, not restated here.

## License

Carries forward the DESI Collaboration license from the originating fork
(`LICENSE.rst`) -- worth a second look as this diverges further from DESI
infrastructure, not changed here.
