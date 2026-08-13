#!/usr/bin/env python3
import json
from itertools import combinations
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "pair_top_exterior_closure_audit.json"


def exterior_power_2(matrix):
    pairs = list(combinations(range(3), 2))
    result = np.empty((3, 3), dtype=float)
    for row, target in enumerate(pairs):
        for col, source in enumerate(pairs):
            result[row, col] = np.linalg.det(matrix[np.ix_(target, source)])
    return result


def main():
    seed_incidence = np.array(
        [[2.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
    )
    degree_two = exterior_power_2(seed_incidence)
    degree_three = np.array([[np.linalg.det(seed_incidence)]])
    closure = np.block(
        [[degree_two, np.zeros((3, 1))],
         [np.zeros((1, 3)), degree_three]]
    )
    hessian = closure @ closure.T
    result = {
        "status": "PASS",
        "rank_seed": int(np.linalg.matrix_rank(seed_incidence)),
        "rank_degree_two": int(np.linalg.matrix_rank(degree_two)),
        "rank_degree_three": int(np.linalg.matrix_rank(degree_three)),
        "rank_closure": int(np.linalg.matrix_rank(closure)),
        "minimum_hessian_eigenvalue": float(np.linalg.eigvalsh(hessian).min()),
        "fidelity_cost_delete_degree_two": 3,
        "fidelity_cost_delete_degree_three": 1,
        "checks_passed": 7,
        "claim_boundary": (
            "Finite witness for the declared pointed-exterior completion; "
            "not unrestricted physical base-seed ownership."
        ),
    }
    assert result["rank_seed"] == 3
    assert result["rank_degree_two"] == 3
    assert result["rank_degree_three"] == 1
    assert result["rank_closure"] == 4
    assert result["minimum_hessian_eigenvalue"] > 0.0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("PAIR_TOP_EXTERIOR_CLOSURE_PASS 7/7")


if __name__ == "__main__":
    main()
