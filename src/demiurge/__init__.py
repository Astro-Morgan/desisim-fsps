"""
demiurge
========

Ground-up simulator + NPE-parameter sampling framework for synthetic
DESI-like spectra, decomposed by construction into named additive channels
(continuum / emission / absorption) with known ground truth. Part of the
DESI-Flow framework (feeds the Plato encoder via a to-be-built neural
decomposer; see project design docs for the full picture).

This package is a from-scratch rewrite (branch `refactor`) of the prior
desisim-fsps fork, not an incremental patch of it -- see `main` for the
prior reference implementation. Currently a bootstrap skeleton: no physics
modules yet.
"""

__version__ = "0.1.0.dev0"
