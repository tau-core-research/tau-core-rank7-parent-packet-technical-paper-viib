# P3 federated Gram-completion audit

## Frozen question

Can the 14 individually incomplete public candidate packets be
combined into one identifying common-source P3 packet?

## Result

**FEDERATED_MARGINALS_DO_NOT_IDENTIFY_CROSS_ROLE_GRAM**

The current union contains 14 quantum-capable
candidates, 7 morphology/stress-capable
candidates, and 12 candidates with some
same-source indexing.  It contains **0**
eligible Q--M cross edge with both stacked rank two and an informationally
complete symmetrized-product record.

Even under the optimistic assumption that both normalized marginal Gram
entries are known, the design rank is
2/3 and its nullity is
1.  The unobserved
direction is the cross term `G_QM`.  Consequently the positive-semidefinite
family

```text
G(rho) = [[1, rho], [rho, 1]],   -1 <= rho <= 1
```

contains the Tau Gram candidate (`rho = 0`) and aligned or anti-aligned
non-Tau completions (`rho = -1` or `rho = +1`).  Marginals cannot select
among them.

A positive control that adds one valid common-source cross edge has rank 3/3.
This closes the scalar two-role Gram only; it does not replace process or
sequential tomography of the operator anticommutator.

The operator control makes this limitation explicit.  The anticommuting pair
`(Z, X)` and the commuting pair `(Z tensor I, I tensor Z)` both have the
identity normalized-trace Gram, while their normalized anticommutator norms
are respectively
0.0 and
2.0.  Even a
closed scalar Gram therefore cannot distinguish the Tau operator law from the
equal-Gram commuting wrong family.

## Seven-role control

An unconstrained real symmetric seven-role Gram has
28 entries.  Its diagonals have rank
7 and leave nullity
21.  Adding a six-edge spanning tree gives
rank 13 and still leaves nullity
15.  Complete pair coverage
gives rank 28 and nullity
0.

Therefore overlap-graph connectivity is not enough.  A smaller measurement
set can close the Gram only after additional algebraic or low-rank identities
have been independently derived rather than fitted to the same terminal data.

## Transport boundary

Outputs from different experiments may enter one Gram only after independently
certified source-frozen transports to a common carrier representation and a
common action unit.  None of the current candidate records contains that
cross-experiment certificate.  The audit therefore forbids constructing a
synthetic Nature packet by treating unrelated marginals as if they were
jointly measured coordinates.

## Reproducibility

Checks: **13/13**
passed.

```bash
python3 scripts/audit_federated_gram_completion.py
```
