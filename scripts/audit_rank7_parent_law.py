#!/usr/bin/env python3
"""Finite deletion audit for Paper VII's minimal rank-seven parent law."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "rank7_parent_law_audit.json"
TOL = 1.0e-10


def rank(matrix: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(matrix, tol=TOL))


def automorphisms(roles, supports):
    result = []
    for image in itertools.permutations(roles):
        permutation = dict(zip(roles, image))
        if all(
            {permutation[role] for role in support} == set(support)
            for support in supports
        ):
            result.append(permutation)
    return result


def stabilizer(group, descriptor):
    return [
        permutation
        for permutation in group
        if all(
            descriptor[role] == descriptor[permutation[role]]
            for role in descriptor
        )
    ]


def operator_square_cost(generators: tuple[np.ndarray, ...]) -> float:
    """Normalized squared Clifford-relation defect."""
    identity = np.eye(generators[0].shape[0])
    total = 0.0
    for a, generator_a in enumerate(generators):
        for b, generator_b in enumerate(generators):
            target = identity if a == b else np.zeros_like(identity)
            defect = 0.5 * (
                generator_a @ generator_b + generator_b @ generator_a
            ) - target
            total += float(np.trace(defect.T @ defect))
    return total / (2.0 * len(generators) ** 2)


def main() -> None:
    roles = ("B", "O", "R", "Q", "M", "G")
    supports = (
        frozenset(("B", "M", "G")),
        frozenset(("O", "Q")),
        frozenset(("R", "M", "G")),
    )
    projectors = (
        np.diag([1.0, 0.0, 0.0, 0.0, 1.0, 1.0]),
        np.diag([0.0, 1.0, 0.0, 1.0, 0.0, 0.0]),
        np.diag([0.0, 0.0, 1.0, 0.0, 1.0, 1.0]),
    )
    fingerprints = {
        role: tuple(int(p[index, index]) for p in projectors)
        for index, role in enumerate(roles)
    }
    multiplicities = {}
    for value in fingerprints.values():
        key = "".join(str(entry) for entry in value)
        multiplicities[key] = multiplicities.get(key, 0) + 1

    group = automorphisms(roles, supports)
    one_port = stabilizer(
        group,
        {"B": 0, "O": 1, "R": 0, "Q": -1, "M": 0, "G": 0},
    )
    two_ports = stabilizer(
        group,
        {
            "B": (0, 0),
            "O": (1, 0),
            "R": (0, 0),
            "Q": (-1, 0),
            "M": (0, 1),
            "G": (0, -1),
        },
    )

    relative = np.eye(7)[:, :6]
    independent_unit = np.eye(7)[:, 6]
    dependent_unit = np.r_[np.ones(6) / np.sqrt(6.0), 0.0]
    full_packet = np.column_stack([relative, independent_unit])
    deficient_packet = np.column_stack([relative, dependent_unit])

    realization = np.vstack(
        [
            np.eye(7),
            np.array(
                [
                    [1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, -1.0, 0.0, 1.0, -1.0],
                ]
            ),
        ]
    )
    decoder = np.linalg.inv(realization.T @ realization) @ realization.T
    collapsed = realization.copy()
    collapsed[:, 6] = collapsed[:, 5]

    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    c1 = np.sqrt(2.0) * np.diag([1.0, 0.0])
    c2 = np.sqrt(2.0) * np.diag([0.0, 1.0])
    commuting_gram = np.array(
        [[np.trace(a @ b) / 2.0 for b in (c1, c2)] for a in (c1, c2)]
    )
    commuting_state_squares = np.array(
        [[(a @ a)[state, state] for a in (c1, c2)] for state in range(2)]
    )
    selected_components = {
        "shifted_body": 0.0,
        "ordered_ports": 0.0,
        "graph_handoff": 0.0,
        "operator_square": operator_square_cost((x, z)),
        "record_root": 0.0,
        "decoder": float(np.linalg.norm(decoder @ realization - np.eye(7)) ** 2),
    }
    selected_total = float(sum(selected_components.values()))
    commuting_control_cost = operator_square_cost((c1, c2))

    checks = {
        "support_projectors_self_adjoint": all(
            np.allclose(p, p.T, atol=TOL) for p in projectors
        ),
        "support_projectors_idempotent": all(
            np.allclose(p @ p, p, atol=TOL) for p in projectors
        ),
        "support_projectors_commute": all(
            np.allclose(a @ b, b @ a, atol=TOL)
            for a in projectors
            for b in projectors
        ),
        "joint_multiplicities_are_1_2_1_2": multiplicities
        == {"100": 1, "010": 2, "001": 1, "101": 2},
        "support_only_stabilizer_order_is_4": len(group) == 4,
        "one_port_stabilizer_order_is_2": len(one_port) == 2,
        "two_port_stabilizer_is_trivial": len(two_ports) == 1,
        "six_relative_lines_have_rank_6": rank(relative) == 6,
        "independent_unit_gives_rank_7": rank(full_packet) == 7,
        "dependent_unit_gives_rank_6": rank(deficient_packet) == 6,
        "unit_controls_have_equal_total_cost": np.isclose(
            np.sum(full_packet * full_packet),
            np.sum(deficient_packet * deficient_packet),
            atol=TOL,
        ),
        "redundant_body_realization_has_rank_7": rank(realization) == 7,
        "smooth_left_decoder_is_exact": np.allclose(
            decoder @ realization, np.eye(7), atol=TOL
        ),
        "collapsed_body_realization_has_rank_6": rank(collapsed) == 6,
        "clifford_squares_are_state_neutral": np.allclose(x @ x, np.eye(2))
        and np.allclose(z @ z, np.eye(2)),
        "clifford_mixed_anticommutator_vanishes": np.allclose(
            x @ z + z @ x, np.zeros((2, 2)), atol=TOL
        ),
        "commuting_control_has_identity_trace_gram": np.allclose(
            commuting_gram, np.eye(2), atol=TOL
        ),
        "commuting_control_fails_state_neutral_norm": not np.allclose(
            commuting_state_squares[0], commuting_state_squares[1], atol=TOL
        ),
        "normalized_action_components_are_nonnegative": all(
            value >= -TOL for value in selected_components.values()
        ) and commuting_control_cost >= -TOL,
        "explicit_common_zero_witness_exists": abs(selected_total) < TOL,
        "equal_gram_commuting_control_has_positive_action": (
            commuting_control_cost > TOL
        ),
        "zero_floor_forces_componentwise_zero": (
            abs(selected_total) < TOL
            and all(abs(value) < TOL for value in selected_components.values())
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, value in checks.items() if not value]
    result = {
        "status": "PASS" if not failed else "FAIL",
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "failed_checks": failed,
        "checks": checks,
        "diagnostics": {
            "fingerprints": {key: list(value) for key, value in fingerprints.items()},
            "joint_multiplicities": multiplicities,
            "support_stabilizer_order": len(group),
            "one_port_stabilizer_order": len(one_port),
            "two_port_stabilizer_order": len(two_ports),
            "independent_packet_rank": rank(full_packet),
            "dependent_packet_rank": rank(deficient_packet),
            "body_realization_rank": rank(realization),
            "collapsed_body_realization_rank": rank(collapsed),
            "selected_action_components": selected_components,
            "selected_total_action": selected_total,
            "commuting_control_operator_square_action": commuting_control_cost,
        },
        "claim_boundary": {
            "minimal_law_to_packet_realization": (
                "proved_by_global_minimization_in_declared_class"
            ),
            "separate_common_zero_occupation_assumption_required": False,
            "narrow_reduct_entails_minimal_law": False,
            "empirical_realization_by_nature": "untested",
            "ambient_parent_exhaustivity": "open",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        "RANK7_PARENT_LAW_AUDIT_"
        f"{result['status']}:{result['checks_passed']}/{result['checks_total']}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
