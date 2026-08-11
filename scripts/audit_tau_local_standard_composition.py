#!/usr/bin/env python3
"""Finite witnesses for Paper VII's Tau-local composition theorem."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/derived/tau_local_standard_composition_audit.json"


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


def main():
    first = (0, 1)
    second = (0, 1, 2)
    product = tuple((a, b) for a in first for b in second)
    local_rules = (
        lambda state: (1 - state[0], state[1]),
        lambda state: (state[0], (state[1] + 1) % 3),
    )
    local_reach = reach((0, 0), local_rules)

    tensor_map = np.eye(len(product))
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    cycle = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    local_first = np.kron(swap, np.eye(3))
    local_second = np.kron(np.eye(2), cycle)

    action_first = np.array([0.2, 0.7])
    action_second = np.array([0.1, 0.4, 0.9])
    joint_action = np.array(
        [action_first[a] + action_second[b] for a, b in product]
    )
    kronecker_sum = (
        np.kron(action_first, np.ones(3))
        + np.kron(np.ones(2), action_second)
    )

    hidden_diagonal = {(0, 0), (1, 1)}
    crossing_rules = (lambda state: (1 - state[0], 1 - state[1]),)
    crossing_reach = reach((0, 0), crossing_rules)

    canonical_metric = np.eye(len(product))
    rescaled_metric = 2.0 * canonical_metric
    canonical_error = np.linalg.norm(
        tensor_map.T @ canonical_metric @ tensor_map - np.eye(len(product))
    )
    rescaled_error = np.linalg.norm(
        tensor_map.T @ rescaled_metric @ tensor_map - np.eye(len(product))
    )

    checks = {
        "component_local_rules_generate_cartesian_closure": (
            local_reach == set(product)
        ),
        "delta_linearization_has_tensor_rank": (
            np.linalg.matrix_rank(tensor_map) == len(first) * len(second)
        ),
        "independent_local_operators_commute": (
            np.linalg.norm(
                local_first @ local_second - local_second @ local_first
            ) < 1.0e-12
        ),
        "relation_free_action_is_kronecker_sum": (
            np.linalg.norm(joint_action - kronecker_sum) < 1.0e-12
        ),
        "canonical_metric_is_unrescaled_isometry": canonical_error < 1.0e-12,
        "complete_marginals_allow_hidden_global_relation": (
            {state[0] for state in hidden_diagonal} == {0, 1}
            and {state[1] for state in hidden_diagonal} == {0, 1}
            and len(hidden_diagonal) == 2 < 4
        ),
        "crossing_rule_destroys_cartesian_factorization": (
            crossing_reach == hidden_diagonal
        ),
        "abstract_tensor_shape_does_not_fix_metric_scale": (
            rescaled_error > 1.0e-6
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "dimensions": {
            "first_closure": len(first),
            "second_closure": len(second),
            "cartesian_closure": len(product),
            "hidden_diagonal_closure": len(hidden_diagonal),
        },
        "errors": {
            "local_commutator": float(
                np.linalg.norm(
                    local_first @ local_second - local_second @ local_first
                )
            ),
            "kronecker_sum": float(np.linalg.norm(joint_action - kronecker_sum)),
            "canonical_isometry": float(canonical_error),
            "rescaled_isometry": float(rescaled_error),
        },
        "claim_boundary": {
            "tau_local_composition": (
                "proved_inside_component_local_no_bridge_closure_class"
            ),
            "old_bare_entailment": False,
            "physical_base_seed_occupation": "open",
            "interaction_scope": (
                "crossing_incidence_invalidates_relation_free_premise"
            ),
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
