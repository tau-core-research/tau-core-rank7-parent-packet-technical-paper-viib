#!/usr/bin/env python3
"""Audit whether incomplete public packets jointly identify the P3 source Gram.

This is an identifiability audit, not a Tau score.  It asks whether separately
observed marginal and cross-role functionals determine one common whitened
Gram matrix after a source-frozen transport to a common representation.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "data/derived/minimal_pair_nature_test.json"
OUT_JSON = ROOT / "data/derived/federated_gram_completion_audit.json"
OUT_REPORT = ROOT / "data/derived/federated_gram_completion_audit_report.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def design_stats(rows: list[list[float]], parameter_count: int) -> dict[str, object]:
    matrix = np.asarray(rows, dtype=float)
    if matrix.size == 0:
        matrix = np.zeros((0, parameter_count), dtype=float)
    else:
        matrix = matrix.reshape((-1, parameter_count))
    rank = int(np.linalg.matrix_rank(matrix))
    return {
        "rows": matrix.tolist(),
        "shape": list(matrix.shape),
        "rank": rank,
        "nullity": parameter_count - rank,
    }


def pair_gram(rho: float) -> np.ndarray:
    return np.array([[1.0, rho], [rho, 1.0]], dtype=float)


def is_psd(matrix: np.ndarray, tolerance: float = 1e-12) -> bool:
    return bool(np.linalg.eigvalsh(matrix).min() >= -tolerance)


def coordinate_rows(
    size: int, observed_pairs: list[tuple[int, int]]
) -> tuple[list[list[float]], list[tuple[int, int]]]:
    parameters = list(itertools.combinations_with_replacement(range(size), 2))
    parameter_index = {pair: index for index, pair in enumerate(parameters)}
    rows: list[list[float]] = []
    for left, right in observed_pairs:
        pair = (min(left, right), max(left, right))
        row = [0.0] * len(parameters)
        row[parameter_index[pair]] = 1.0
        rows.append(row)
    return rows, parameters


def normalized_trace_gram(operators: list[np.ndarray]) -> np.ndarray:
    dimension = operators[0].shape[0]
    return np.array(
        [
            [
                np.trace(left.conj().T @ right).real / dimension
                for right in operators
            ]
            for left in operators
        ],
        dtype=float,
    )


def normalized_anticommutator_norm(left: np.ndarray, right: np.ndarray) -> float:
    anticommutator = left @ right + right @ left
    return float(np.linalg.norm(anticommutator, ord="fro") / np.sqrt(left.shape[0]))


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    candidates = source["candidates"]

    quantum_rows = [
        row for row in candidates if row["gates"]["quantum_effect_or_state_map"]
    ]
    morphology_rows = [
        row
        for row in candidates
        if row["gates"]["metric_stress_or_morphology_map"]
    ]
    same_source_rows = [
        row for row in candidates if row["gates"]["same_source_joint_indexing"]
    ]
    eligible_cross_rows = [
        row
        for row in candidates
        if row["gates"]["same_source_joint_indexing"]
        and row["gates"]["stacked_pair_rank_two"]
        and row["gates"]["informationally_complete_symmetrized_product"]
    ]

    # Pair parameter order is (G_QQ, G_MM, G_QM).  The first two rows are an
    # intentionally optimistic fusion: even if both normalized marginals are
    # granted, they do not observe the cross-role parameter.
    optimistic_pair = design_stats(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], parameter_count=3
    )
    current_rows: list[list[float]] = []
    if quantum_rows:
        current_rows.append([1.0, 0.0, 0.0])
    if morphology_rows:
        current_rows.append([0.0, 1.0, 0.0])
    if eligible_cross_rows:
        current_rows.append([0.0, 0.0, 1.0])
    current_pair = design_stats(current_rows, parameter_count=3)
    positive_control = design_stats(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        parameter_count=3,
    )

    identity = np.eye(2, dtype=complex)
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    tau_operators = [pauli_z, pauli_x]
    commuting_operators = [np.kron(pauli_z, identity), np.kron(identity, pauli_z)]
    operator_control = {
        "tau_anticommuting_gram": normalized_trace_gram(tau_operators).tolist(),
        "tau_anticommutator_norm": normalized_anticommutator_norm(*tau_operators),
        "commuting_wrong_family_gram": normalized_trace_gram(
            commuting_operators
        ).tolist(),
        "commuting_wrong_family_anticommutator_norm": (
            normalized_anticommutator_norm(*commuting_operators)
        ),
        "interpretation": (
            "The scalar normalized-trace Gram is identical, while the operator "
            "anticommutator differs. Gram completion is necessary but not sufficient."
        ),
    }

    representative_rho = (-1.0, 0.0, 1.0)
    representative_grams = []
    for rho in representative_rho:
        gram = pair_gram(rho)
        representative_grams.append(
            {
                "rho": rho,
                "gram": gram.tolist(),
                "eigenvalues": np.linalg.eigvalsh(gram).tolist(),
                "positive_semidefinite": is_psd(gram),
                "same_observed_diagonal": np.allclose(np.diag(gram), [1.0, 1.0]),
            }
        )

    scan_controls = []
    for resolution in (101, 1001, 10001):
        grid = np.linspace(-1.25, 1.25, resolution)
        feasible = np.array([is_psd(pair_gram(rho)) for rho in grid])
        feasible_rho = grid[feasible]
        analytic_error = max(
            float(
                np.max(
                    np.abs(
                        np.linalg.eigvalsh(pair_gram(rho))
                        - np.array([1.0 - abs(rho), 1.0 + abs(rho)])
                    )
                )
            )
            for rho in grid
        )
        scan_controls.append(
            {
                "resolution": resolution,
                "feasible_min": float(feasible_rho.min()),
                "feasible_max": float(feasible_rho.max()),
                "analytic_eigenvalue_max_abs_error": analytic_error,
            }
        )

    role_count = 7
    diagonal_pairs = [(index, index) for index in range(role_count)]
    tree_pairs = [(index, index + 1) for index in range(role_count - 1)]
    complete_cross_pairs = list(itertools.combinations(range(role_count), 2))
    diagonal_rows, parameter_pairs = coordinate_rows(role_count, diagonal_pairs)
    tree_rows, _ = coordinate_rows(role_count, diagonal_pairs + tree_pairs)
    complete_rows, _ = coordinate_rows(
        role_count, diagonal_pairs + complete_cross_pairs
    )
    seven_diagonal = design_stats(diagonal_rows, len(parameter_pairs))
    seven_tree = design_stats(tree_rows, len(parameter_pairs))
    seven_complete = design_stats(complete_rows, len(parameter_pairs))

    transport_fields = {
        "common_carrier_id",
        "source_frozen_transport",
        "common_action_unit",
        "intertwiner",
    }
    candidates_with_transport_certificate = [
        row["id"] for row in candidates if transport_fields.issubset(row.keys())
    ]

    checks = {
        "input_is_pair_audit": source.get("test_id") == "P3-NATURE-PAIR-1",
        "no_current_eligible_cross_edge": len(eligible_cross_rows) == 0,
        "optimistic_pair_has_one_hidden_direction": (
            optimistic_pair["rank"] == 2 and optimistic_pair["nullity"] == 1
        ),
        "cross_direction_is_unobserved": all(
            row[2] == 0.0 for row in optimistic_pair["rows"]
        ),
        "tau_and_wrong_family_grams_share_marginals": all(
            item["positive_semidefinite"] and item["same_observed_diagonal"]
            for item in representative_grams
        ),
        "one_valid_cross_edge_closes_pair_gram": (
            positive_control["rank"] == 3 and positive_control["nullity"] == 0
        ),
        "equal_gram_commuting_operator_countermodel": (
            np.allclose(
                operator_control["tau_anticommuting_gram"],
                operator_control["commuting_wrong_family_gram"],
            )
            and operator_control["tau_anticommutator_norm"] < 1e-12
            and operator_control["commuting_wrong_family_anticommutator_norm"]
            > 1.0
        ),
        "seven_diagonals_leave_21_cross_terms": seven_diagonal["nullity"] == 21,
        "seven_role_spanning_tree_is_insufficient": (
            seven_tree["rank"] == 13 and seven_tree["nullity"] == 15
        ),
        "complete_pair_coverage_closes_unconstrained_gram": (
            seven_complete["rank"] == 28 and seven_complete["nullity"] == 0
        ),
        "no_cross_experiment_transport_certificate": (
            len(candidates_with_transport_certificate) == 0
        ),
        "resolution_control_recovers_psd_interval": all(
            abs(item["feasible_min"] + 1.0) <= 0.025
            and abs(item["feasible_max"] - 1.0) <= 0.025
            for item in scan_controls
        ),
        "analytic_numeric_eigenvalues_agree": all(
            item["analytic_eigenvalue_max_abs_error"] < 1e-12
            for item in scan_controls
        ),
    }

    result = {
        "schema_version": "1.0",
        "audit_id": "P3-FEDERATED-GRAM-1",
        "freeze_date": "2026-08-10",
        "source": {
            "path": SOURCE.name,
            "sha256": sha256(SOURCE),
            "candidate_count": len(candidates),
        },
        "question": (
            "Can individually incomplete public experiments be assembled into "
            "one identifying common-source P3 packet?"
        ),
        "fusion_rule": (
            "A datum may constrain one common Gram only after a source-frozen "
            "transport/intertwiner, common action unit and compatible carrier "
            "representation have been independently established."
        ),
        "pair_parameterization": {
            "parameters": ["G_QQ", "G_MM", "G_QM"],
            "current_design": current_pair,
            "optimistic_both_marginals_design": optimistic_pair,
            "positive_cross_edge_control": positive_control,
            "operator_level_countermodel": operator_control,
            "quantum_candidate_count": len(quantum_rows),
            "morphology_candidate_count": len(morphology_rows),
            "same_source_candidate_count": len(same_source_rows),
            "eligible_cross_edge_count": len(eligible_cross_rows),
            "representative_psd_completions": representative_grams,
            "psd_scan_controls": scan_controls,
        },
        "seven_role_parameterization": {
            "role_count": role_count,
            "symmetric_gram_parameter_count": len(parameter_pairs),
            "diagonals_only": seven_diagonal,
            "diagonals_plus_spanning_tree": seven_tree,
            "complete_pair_coverage": seven_complete,
            "interpretation": (
                "Graph connectivity alone does not identify an unconstrained "
                "rank-7 symmetric Gram. Fewer cross edges suffice only if extra "
                "algebraic or rank identities are independently derived."
            ),
        },
        "transport_certificate": {
            "required_fields": sorted(transport_fields),
            "certified_candidates": candidates_with_transport_certificate,
        },
        "verdict": "FEDERATED_MARGINALS_DO_NOT_IDENTIFY_CROSS_ROLE_GRAM",
        "claim_boundary": {
            "proved_here": (
                "The present marginal union leaves the Q--M cross Gram free; "
                "even a connected seven-role overlap tree leaves 15 unconstrained "
                "symmetric-Gram directions."
            ),
            "not_proved_here": (
                "A completed scalar Gram would not by itself prove the full "
                "operator anticommutator, complete typed incidence, or physical "
                "base--seed occupation."
            ),
            "nature_score_authorized": False,
        },
        "checks": checks,
        "check_summary": {
            "passed": sum(checks.values()),
            "total": len(checks),
            "all_passed": all(checks.values()),
        },
    }

    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    pair = result["pair_parameterization"]
    seven = result["seven_role_parameterization"]
    report = f"""# P3 federated Gram-completion audit

## Frozen question

Can the {len(candidates)} individually incomplete public candidate packets be
combined into one identifying common-source P3 packet?

## Result

**{result['verdict']}**

The current union contains {pair['quantum_candidate_count']} quantum-capable
candidates, {pair['morphology_candidate_count']} morphology/stress-capable
candidates, and {pair['same_source_candidate_count']} candidates with some
same-source indexing.  It contains **{pair['eligible_cross_edge_count']}**
eligible Q--M cross edge with both stacked rank two and an informationally
complete symmetrized-product record.

Even under the optimistic assumption that both normalized marginal Gram
entries are known, the design rank is
{pair['optimistic_both_marginals_design']['rank']}/3 and its nullity is
{pair['optimistic_both_marginals_design']['nullity']}.  The unobserved
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
{operator_control['tau_anticommutator_norm']:.1f} and
{operator_control['commuting_wrong_family_anticommutator_norm']:.1f}.  Even a
closed scalar Gram therefore cannot distinguish the Tau operator law from the
equal-Gram commuting wrong family.

## Seven-role control

An unconstrained real symmetric seven-role Gram has
{seven['symmetric_gram_parameter_count']} entries.  Its diagonals have rank
{seven['diagonals_only']['rank']} and leave nullity
{seven['diagonals_only']['nullity']}.  Adding a six-edge spanning tree gives
rank {seven['diagonals_plus_spanning_tree']['rank']} and still leaves nullity
{seven['diagonals_plus_spanning_tree']['nullity']}.  Complete pair coverage
gives rank {seven['complete_pair_coverage']['rank']} and nullity
{seven['complete_pair_coverage']['nullity']}.

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

Checks: **{result['check_summary']['passed']}/{result['check_summary']['total']}**
passed.

```bash
python3 scripts/audit_federated_gram_completion.py
```
"""
    OUT_REPORT.write_text(report, encoding="utf-8")

    print(
        f"P3 federated Gram audit: {result['check_summary']['passed']}/"
        f"{result['check_summary']['total']} checks passed; "
        f"verdict={result['verdict']}"
    )


if __name__ == "__main__":
    main()
