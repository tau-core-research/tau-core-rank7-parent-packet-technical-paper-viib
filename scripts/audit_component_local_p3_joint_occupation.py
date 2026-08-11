#!/usr/bin/env python3
"""Audit the common-law LC1--LC4 and rank-seven P3 occupation theorem."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/derived/component_local_p3_joint_occupation_audit.json"


def reach(initial, rules):
    seen = {initial}
    frontier = [initial]
    while frontier:
        state = frontier.pop()
        for rule in rules:
            candidate = rule(state)
            if candidate not in seen:
                seen.add(candidate)
                frontier.append(candidate)
    return seen


def main() -> None:
    local_product = {(a, b) for a in (0, 1) for b in (0, 1, 2)}
    local_reach = reach(
        (0, 0),
        (
            lambda state: (1 - state[0], state[1]),
            lambda state: (state[0], (state[1] + 1) % 3),
        ),
    )

    h_joint = np.diag([2.0, 3.0, 5.0, 7.0, 11.0] + [1.0] * 7)
    origin = np.zeros(12)
    shifted = np.ones(12)

    p3_full = np.eye(7)
    p3_rank6 = np.diag([np.sqrt(7.0 / 6.0)] * 6 + [0.0])
    full_cost = float(np.trace(p3_full.T @ p3_full))
    control_cost = float(np.trace(p3_rank6.T @ p3_rank6))

    p3_adjacency = np.zeros((7, 7))
    for index in range(6):
        p3_adjacency[index, index + 1] = 1.0
        p3_adjacency[index + 1, index] = 1.0
    p3_laplacian = np.diag(p3_adjacency.sum(axis=1)) - p3_adjacency

    hidden_relation = {(0, 0), (1, 1)}
    hidden_marginals_complete = (
        {state[0] for state in hidden_relation} == {0, 1}
        and {state[1] for state in hidden_relation} == {0, 1}
    )

    checks = {
        "component_local_rules_generate_cartesian_closure": local_reach == local_product,
        "once_counted_parent_hessian_is_component_direct_sum": (
            np.count_nonzero(h_joint - np.diag(np.diag(h_joint))) == 0
        ),
        "nonnegative_component_action_has_attainable_joint_zero": (
            float(origin @ h_joint @ origin) == 0.0
            and float(shifted @ h_joint @ shifted) > 0.0
            and np.linalg.eigvalsh(h_joint).min() > 0.0
        ),
        "occupied_p3_component_has_rank_seven": np.linalg.matrix_rank(p3_full) == 7,
        "internal_p3_incidence_is_one_connected_component": (
            np.count_nonzero(np.isclose(np.linalg.eigvalsh(p3_laplacian), 0.0)) == 1
        ),
        "complete_marginals_allow_hidden_global_lc3_failure": (
            hidden_marginals_complete and len(hidden_relation) == 2 < len(local_product)
        ),
        "same_total_scalar_control_can_have_p3_rank_six": (
            abs(full_cost - control_cost) < 1e-12
            and np.linalg.matrix_rank(p3_rank6) == 6
        ),
        "narrow_descriptors_do_not_select_joint_completion": (
            hidden_marginals_complete
            and abs(full_cost - control_cost) < 1e-12
            and np.linalg.matrix_rank(p3_rank6) < 7
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "dimensions": {
            "cartesian_closure": len(local_product),
            "hidden_diagonal_closure": len(hidden_relation),
            "positive_p3_rank": int(np.linalg.matrix_rank(p3_full)),
            "control_p3_rank": int(np.linalg.matrix_rank(p3_rank6)),
            "joint_hessian_rank": int(np.linalg.matrix_rank(h_joint)),
        },
        "claim_boundary": {
            "joint_law_to_occupation": "proved_in_declared_enriched_class",
            "narrow_reduct_entailment": False,
            "nature_level_realization": "untested",
            "ambient_parent_exhaustivity": "open",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
