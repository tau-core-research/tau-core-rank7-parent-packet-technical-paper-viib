#!/usr/bin/env python3
"""Audit the minimal two-role falsification packet for the P3 Clifford law."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "data/derived/public_source_eligibility_audit.json"
OUT_JSON = ROOT / "data/derived/minimal_pair_nature_test.json"
OUT_REPORT = ROOT / "data/derived/minimal_pair_nature_test_report.md"


PAIR_CANDIDATES = {
    "hoj_room_temperature_quantum_optomechanics": (
        "Mechanical and Gaussian-state candidates factor through the same "
        "squared displacement; stacked novelty is rank one."
    ),
    "qhe_dissipation_engineered_transmon": (
        "Heat/work and state legs are reconstructed from the same population record."
    ),
    "hbn_spin_rf_sensing": (
        "The RF field is a control and the spin response its sensor terminal, "
        "not an independent M/G source jet."
    ),
    "labranca_self_calibrated_wqed_photon": (
        "Energy and density-matrix estimates reduce the same calibrated field moments."
    ),
    "chen_bolometric_microwave_tomography": (
        "Both candidate legs are functionals of one bolometric histogram family."
    ),
    "najafabadi_intensity_wigner": (
        "The g2 leg is constrained by the standard Wigner-moment identity, "
        "not an independent stress/action jet."
    ),
    "zhang_single_photon_vlbi": (
        "Counting and homodyne branches lack a common shot key and cross-product record."
    ),
    "lualdi_energy_entangled_interferometry": (
        "Sample morphology and source tomography are not paired to one endpoint instrument."
    ),
}


ADDITIONAL_NEAR_CANDIDATES = (
    {
        "id": "antesberger_higher_order_quantum_switch",
        "url": "https://doi.org/10.5281/zenodo.7974799",
        "gates": {
            "quantum_effect_or_state_map": True,
            "metric_stress_or_morphology_map": False,
            "same_source_joint_indexing": True,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "Higher-order process tomography is quantum-sector complete, but "
            "does not supply an independently sourced M/G typed perturbation."
        ),
    },
    {
        "id": "stemp_donor_spin_gate_set_tomography",
        "url": "https://doi.org/10.5061/dryad.w3r2280zm",
        "gates": {
            "quantum_effect_or_state_map": True,
            "metric_stress_or_morphology_map": False,
            "same_source_joint_indexing": True,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "Gate-set tomography reconstructs quantum operations; SET current "
            "is the detector record, not an independent M/G source map."
        ),
    },
    {
        "id": "van_thiel_piezo_optomechanical_qubit_readout",
        "url": "https://doi.org/10.5281/zenodo.14293253",
        "gates": {
            "quantum_effect_or_state_map": True,
            "metric_stress_or_morphology_map": False,
            "same_source_joint_indexing": False,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "The public figure data join qubit readout and pump-dependent "
            "T1/T2* characterization in one device, but provide no independent "
            "mechanical/stress state map and no shared record-level cross product."
        ),
        "public_file_sha256": {
            "Figure1.xlsx": "16b4cb6142791eafde275974645dc9faf4ca7ab5b12037c94cf6dcd73cd0943e",
            "Figure2.xlsx": "962662ab68e73274b0c76d2914b6b0226470eb46a430d82763703f54c2e64fef",
            "Figure3.xlsx": "4e476c24e29a5d95e7b70da922b9745c9efab63826cadb5815edfb50a5c3b93c",
            "Figure4a.xlsx": "5e0ad99b5c728f732bd8a390770492a77372f205ecfb843f3d820d83a074a05d",
            "Figure4b.xlsx": "3f2d4b0b992be5d0183a075cabab19ab2586cd15e1c6f73402e77633260dcb8a",
        },
    },
    {
        "id": "fluehmann_trapped_ion_mechanical_grid_qubit",
        "url": "https://tiqi.ethz.ch/publications-and-awards/public-datasets.html",
        "gates": {
            "quantum_effect_or_state_map": True,
            "metric_stress_or_morphology_map": False,
            "same_source_joint_indexing": True,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "The public characteristic-function, Wigner, Pauli and process-"
            "tomography tables richly reconstruct a motional qubit, but the "
            "mechanical oscillator is itself the Q carrier. Its displacement "
            "and state maps do not provide an independent M/G typed output."
        ),
        "public_file_sha256": {
            "data.zip": "1c4000816503d5abfc376b4879a409dd63e327b4efbe5930dc5ea0bd4bb927eb",
        },
    },
    {
        "id": "burd_quantum_amplification_mechanical_motion",
        "url": "https://doi.org/10.18434/M32051",
        "gates": {
            "quantum_effect_or_state_map": True,
            "metric_stress_or_morphology_map": False,
            "same_source_joint_indexing": True,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "The public same-run tables join calibrated displacement, squeezing "
            "and spin response, but displacement is a controlled source value and "
            "the oscillator output is inferred through the spin/Q readout. No "
            "independent mechanical/stress output map or Q--M product is present."
        ),
        "public_file_sha256": {
            "Figure_2.zip": "2c2cca834f5bbd9935f5e107b8aa086b8266d28665e3aca4bd2c5901f7615c3b",
            "Figure_3.zip": "98dffe636292226b5b14526bf5d00bb2ef05944375145b36b4ef9790e487bd4b",
            "Figure_S3.zip": "abe8e6cbedc26dbe10fb9f344b8b431f728fd6af2a8ad8f541a6c5925ee2cadc",
        },
    },
    {
        "id": "thomas_macroscopic_mechanical_spin_entanglement",
        "url": "https://doi.org/10.1038/s41567-020-1031-5",
        "gates": {
            "quantum_effect_or_state_map": True,
            "metric_stress_or_morphology_map": False,
            "same_source_joint_indexing": True,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "The experiment couples distinct membrane-mechanical and atomic-spin "
            "systems and reconstructs a joint Gaussian EPR witness, but the public "
            "files expose one combined optical record. The separate mechanics, "
            "atoms and backaction spectra are fitted model components rather than "
            "independently measured output maps, so typed rank two and the required "
            "cross-product cannot be certified."
        ),
        "public_file_sha256": {
            "Source Data Fig. 1.xlsx": "b846bff964f951514a60dbbee2c7182bf00473b683ac4978ab7400d281bb51f0",
            "Source Data Fig. 3.xlsx": "c7830b64c2583417f92142bcbcf8c875fd2b63e1008e6466b40c039db3d79d37",
            "Source Data Fig. 4.xlsx": "39fe35970b9076d506c642e8a86fe0e6c9e14afec3dac1190cca274abb7a998f",
        },
    },
)


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    indexed = {row["id"]: row for row in source["candidates"]}
    rows = []
    for candidate_id, reason in PAIR_CANDIDATES.items():
        original = indexed[candidate_id]
        gates = original["gates"]
        pair_gates = {
            "quantum_effect_or_state_map": gates["quantum_state_coherence_or_phase"],
            "metric_stress_or_morphology_map": gates[
                "independent_faraday_or_em_morphology"
            ],
            "same_source_joint_indexing": gates["same_system_joint_indexing"],
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": gates["complete_standard_comparator"],
        }
        rows.append(
            {
                "id": candidate_id,
                "gates": pair_gates,
                "passed": sum(pair_gates.values()),
                "eligible": all(pair_gates.values()),
                "reason": reason,
            }
        )
    for candidate in ADDITIONAL_NEAR_CANDIDATES:
        gates = candidate["gates"]
        rows.append(
            {
                "id": candidate["id"],
                "url": candidate["url"],
                "gates": gates,
                "passed": sum(gates.values()),
                "eligible": all(gates.values()),
                "reason": candidate["reason"],
                **(
                    {"public_file_sha256": candidate["public_file_sha256"]}
                    if "public_file_sha256" in candidate
                    else {}
                ),
            }
        )
    # ROOT--TT is a valuable common-preparation identity but not an independent
    # pair test.  For t = z c with nonzero complex z, the real TT Jacobian is
    # an invertible 2x2 multiple of the coherence Jacobian.  Stacking both
    # terminals therefore adds no differential rank beyond either terminal.
    z = 1.3 + 0.7j
    j_root = np.eye(2)
    j_tt = np.array([[z.real, -z.imag], [z.imag, z.real]])
    root_rank = int(np.linalg.matrix_rank(j_root))
    tt_rank = int(np.linalg.matrix_rank(j_tt))
    stacked_rank = int(np.linalg.matrix_rank(np.vstack([j_root, j_tt])))
    root_tt_novelty_rank = stacked_rank - max(root_rank, tt_rank)

    result = {
        "schema_version": "1.0",
        "test_id": "P3-NATURE-PAIR-1",
        "freeze_date": "2026-08-10",
        "prediction": (
            "For every predeclared distinct typed pair: "
            "C_a^2=C_b^2=I and {C_a,C_b}=0 after source whitening."
        ),
        "falsification": (
            "A statistically resolved nonzero anticommutator on an eligible pair "
            "rejects the minimal rank-7 completion."
        ),
        "non_confirmation": (
            "One passing pair does not establish the other roles or full rank seven."
        ),
        "candidate_count": len(rows),
        "eligible_count": sum(row["eligible"] for row in rows),
        "candidates": rows,
        "root_tt_factorization_control": {
            "law": "T_TT = N_O K rho_ROOT,01",
            "root_rank": root_rank,
            "tt_rank": tt_rank,
            "stacked_rank": stacked_rank,
            "novelty_rank": root_tt_novelty_rank,
            "verdict": "SAME_INFORMATION_FACTOR_NOT_AN_INDEPENDENT_PAIR",
        },
        "verdict": "NO_ELIGIBLE_TWO_ROLE_PACKET",
        "minimum_reopening_packet": [
            "one predeclared O/Q-side and one M/G-side map on the same source",
            "positive two-direction source Gram and frozen whitening",
            "informationally complete measurement of the symmetrized product",
            "shared run keys and a complete standard/noise comparator",
        ],
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "non_claim": (
            "Failure of pair eligibility is not a measured violation or confirmation."
        ),
    }
    checks = {
        "all_candidates_present": set(PAIR_CANDIDATES).issubset(indexed),
        "no_pair_scored_without_rank_two": all(
            not row["eligible"] or row["gates"]["stacked_pair_rank_two"]
            for row in rows
        ),
        "no_pair_scored_without_product_tomography": all(
            not row["eligible"]
            or row["gates"]["informationally_complete_symmetrized_product"]
            for row in rows
        ),
        "zero_current_eligible_pairs": result["eligible_count"] == 0,
        "root_tt_adds_no_independent_rank": root_tt_novelty_rank == 0,
        "near_candidates_not_mistyped_as_cross_role": all(
            not row["gates"]["stacked_pair_rank_two"]
            for row in rows
            if row["id"] in {
                "antesberger_higher_order_quantum_switch",
                "stemp_donor_spin_gate_set_tomography",
                "van_thiel_piezo_optomechanical_qubit_readout",
                "fluehmann_trapped_ion_mechanical_grid_qubit",
                "burd_quantum_amplification_mechanical_motion",
                "thomas_macroscopic_mechanical_spin_entanglement",
            }
        ),
    }
    result["checks"] = checks
    result["checks_passed"] = sum(checks.values())
    result["checks_total"] = len(checks)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Tau Core minimal P3 pair Nature-test audit",
        "",
        "**Status:** no eligible public two-role packet",
        "",
        "```math",
        "C_a^2=C_b^2=I,\\qquad \\{C_a,C_b\\}=0.",
        "```",
        "",
        "A single eligible pair can falsify the full completion, but cannot confirm it.",
        "",
        "| Candidate | Gates | Pair verdict |",
        "| --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(f"| `{row['id']}` | {row['passed']}/6 | {row['reason']} |")
    lines.extend(
        [
            "",
        f"Eligible pair packets: **{result['eligible_count']}/{len(rows)}**.",
        "",
        "The common failure is the absence of two independent typed source",
        "directions together with an informationally complete symmetrized-product record.",
        "",
        "The existing ROOT--TT common-preparation identity is not an escape:",
        "its stacked differential has rank 2 while each complex terminal already",
        "has rank 2, so its cross-terminal novelty rank is 0.",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise SystemExit(f"P3_PAIR_NATURE_TEST_FAIL failed={failed}")
    print(
        f"P3_PAIR_NATURE_TEST_PASS checks={result['checks_passed']}/"
        f"{result['checks_total']} eligible={result['eligible_count']}/{len(rows)}"
    )


if __name__ == "__main__":
    main()
