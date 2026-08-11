# Tau Core Rank-7 Parent Packet: Technical Paper VII-B

**Repository kind:** technical paper
**Status:** conditional internal closure; unrestricted physical realization open

This repository owns the rank-seven P3 parent packet previously embedded in
Foundation Paper VII. It proves the support--port reconstruction, Tau-local
composition law, pointed exterior carrier, operator-square selector and a
class-minimal enriched realization inside an explicit finite completion class.

The paper does not claim that the unrestricted physical base--seed parent
realizes this class, that the packet exhausts the ambient parent, or that a
Tau-specific empirical signal has been observed.

The finite-local occupation criterion now makes the boundary exact: unloaded
positive split irreps vanish, while stable loaded or cross-coupled irreps are
generated support. The enriched rank-seven law supplies occupation only
inside its declared class.

## Reproduce

```bash
python3 scripts/reproduce.py
```

This regenerates the deterministic ledgers, compiles the PDF, creates the
arXiv source archive and runs the test suite.

It is connected to the central Tau Core AI Knowledge Base through `.tau-core-kb`.
Use that layer for current state, claim boundaries, blocker status, and context
packs before loading large source documents.

## Claim Boundary

``Class-minimal'' means minimal only in the declared finite,
source-irredundant one-copy completion class. It is not a claim of a unique or
absolutely minimal law of Nature.

## AI Startup

Read `.tau-core-kb/COMPACT_CORE.md`, then generate the focused routing pack.

```bash
python .tau-core-kb/scripts/tc_kb.py status
python .tau-core-kb/scripts/tc_kb.py pack big_picture
```
