# demiurge

Ground-up simulator + NPE-parameter sampling framework for synthetic
DESI-like spectra, decomposed by construction into named additive channels
(continuum / emission / absorption) with known ground truth -- the training
signal a downstream neural decomposer needs, since that decomposition isn't
observable for real spectra. Part of the DESI-Flow framework.

**Current state: NPE-parameter registry scaffolding.** No physics-generation
channels have been built yet. `demiurge.parameters` provides the schema
(`NPEParameter`: tier, citation, physical/non-physical tag, default prior
distribution) and `PriorSampler`, but the registry itself is deliberately
empty -- entries are added one channel at a time, alongside that channel's
real code, not bulk-ported ahead of it (this is a ground-up rebuild; `main`
is a reference to consult per-feature, not a source to copy from). `demiurge.rng`
provides reproducible, independent RNG streams for the pipeline. See the
`refactor` branch's own history for progress. This description will be
extended as real modules land; it does not describe anything aspirational.

Citations and physics rationale accumulate alongside the code in
[`docs/paper/methods.tex`](docs/paper/methods.tex) +
[`docs/paper/refs.bib`](docs/paper/refs.bib) (AASTeX v7, ApJ-targeted) --
also currently just scaffolding.

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
