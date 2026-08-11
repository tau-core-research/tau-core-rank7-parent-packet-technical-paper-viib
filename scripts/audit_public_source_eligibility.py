#!/usr/bin/env python3
"""Finite public-source eligibility audit for a Faraday--quantum Tau score."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "public_source_eligibility_audit.json"

GATES = (
    "independent_faraday_or_em_morphology",
    "quantum_state_coherence_or_phase",
    "same_system_joint_indexing",
    "independent_positive_body_action",
    "complete_standard_comparator",
    "candidate_specific_operational_tau_map",
)

CANDIDATES = [
    {
        "id": "ar11158_faraday",
        "source": "https://doi.org/10.5281/zenodo.1034404",
        "gates": [True, False, False, False, True, False],
        "verdict": "Faraday certificate only; no independent quantum tomography.",
    },
    {
        "id": "wen_single_photon_paths",
        "source": "https://doi.org/10.5061/dryad.x0k6djj14",
        "gates": [False, True, True, False, True, False],
        "verdict": "Quantum path phase is public, but no independent spacetime Faraday field or body action is present.",
    },
    {
        "id": "vaartjes_spin_qudit",
        "source": "https://doi.org/10.5061/dryad.547d7wmj0",
        "paper": "https://arxiv.org/abs/2410.07641",
        "gates": [False, True, True, False, True, False],
        "verdict": "Tomography and single-shot spin data are public; the magnetic field is a Hamiltonian control, not an independently reconstructed morphological action.",
    },
    {
        "id": "hbn_spin_rf_sensing",
        "source": "https://zenodo.org/records/8135158",
        "paper": "https://doi.org/10.1038/s41467-023-40473-w",
        "gates": [False, True, True, False, True, False],
        "verdict": "RF field and spin response are public, but the response is the standard sensor terminal and does not independently measure the Tau body action.",
    },
    {
        "id": "lualdi_energy_entangled_interferometry",
        "source": "https://doi.org/10.5061/dryad.1rn8pk154",
        "paper": "https://doi.org/10.1126/sciadv.adw4938",
        "gates": [True, True, False, False, True, False],
        "verdict": "AFM, ellipsometry, classical/quantum sample scans and source tomography are public, but the tomography is not endpoint-paired after the sample and no independent positive body-action separation is supplied.",
    },
    {
        "id": "hoj_room_temperature_quantum_optomechanics",
        "source": "https://doi.org/10.5281/zenodo.10040032",
        "paper": "https://doi.org/10.1038/s41586-023-06997-3",
        "gates": [True, True, True, False, True, False],
        "provisional": False,
        "verdict": "Raw mechanical response and Gaussian-state reconstruction are public, but MOPR-HOJIND1 shows that the candidate action and Uhlmann legs both factor through the same squared phase-space displacement; no independent endpoint action is present.",
    },
    {
        "id": "gunyho_thermal_qubit_readout",
        "source": "https://doi.org/10.5281/zenodo.7773981",
        "paper": "https://arxiv.org/abs/2303.03668",
        "gates": [True, False, True, False, True, False],
        "verdict": "Single-shot thermal readout, Rabi controls and instrument snapshots are public, but the bolometer signal is the qubit-state readout itself; no full state tomography or independent endpoint action leg is supplied.",
    },
    {
        "id": "qhe_dissipation_engineered_transmon",
        "source": "https://doi.org/10.5281/zenodo.14935889",
        "paper": "https://doi.org/10.1038/s41467-026-72651-x",
        "gates": [True, True, True, False, True, False],
        "verdict": "Cycle-resolved single-shot populations, controls and calibrations are public, but heat and work are reconstructed from those same populations and controlled level spacings rather than measured by an independent endpoint action channel.",
    },
    {
        "id": "zeptojoule_calorimetry",
        "source": "https://doi.org/10.5281/zenodo.16038651",
        "paper": "https://doi.org/10.1038/s41928-026-01615-2",
        "gates": [True, False, False, True, True, False],
        "verdict": "Raw single-shot calorimetric pulse-energy data and complete reconstruction code are public, but the source is calibrated microwave test pulses rather than a jointly indexed qubit state/tomography record.",
    },
    {
        "id": "cottet_quantum_maxwell_demon",
        "paper": "https://doi.org/10.1073/pnas.1704827114",
        "gates": [True, True, False, True, True, False],
        "verdict": "The same superconducting circuit supports full cavity-state tomography and a genuinely direct output-power work measurement. The published endpoints are ensemble measurements rather than a reusable jointly indexed record table, and no public raw-data repository was located.",
    },
    {
        "id": "dassonneville_single_qubit_engine",
        "paper": "https://doi.org/10.1103/rygc-bc3c",
        "gates": [True, True, False, True, True, False],
        "verdict": "The experiment directly compares microwave-output work with work inferred from full qubit tomography, but states explicitly that the two measurements were not simultaneous. Its data are available only from the authors on request, so no public common-index scoring packet exists.",
    },
    {
        "id": "zhang_single_ion_information_engine",
        "paper": "https://doi.org/10.1103/g45c-ssfx",
        "gates": [True, True, True, False, True, False],
        "verdict": "A separate motional quantum battery stores work and phonon-number-resolved detection reconstructs its state, but stored energy and quantum-state information are both computed from that same phonon distribution. No public raw-data repository was located.",
    },
    {
        "id": "lindenfels_spin_engine_flywheel",
        "paper": "https://doi.org/10.1103/PhysRevLett.123.080602",
        "gates": [True, True, True, False, True, False],
        "verdict": "The trapped-ion oscillator is a genuine work repository and its Husimi-Q function is reconstructed, but deposited energy and ergotropy are inferred from the same reconstructed flywheel state rather than an independent action detector. No public raw-data packet was located.",
    },
    {
        "id": "zhang_cyclic_engine_quantum_battery",
        "paper": "https://doi.org/10.1038/s41467-025-60179-5",
        "gates": [True, False, True, True, True, False],
        "verdict": "Battery energy is measured after controlled cycle counts and exhibits a quantum-coherence witness, but the engine is deliberately not measured and no independent full state tomography is supplied. Data are available only on request.",
    },
    {
        "id": "kim_superradiant_photonic_engine",
        "paper": "https://doi.org/10.1038/s41566-022-01039-2",
        "gates": [True, False, True, True, True, False],
        "verdict": "The cavity engine has directly observed output power and second-order optical correlations, while atomic coherence is a prepared source control rather than independent endpoint tomography. No reusable public raw packet was located.",
    },
    {
        "id": "mallik_quantum_anomalous_heat_flow",
        "source": "https://doi.org/10.5281/zenodo.14275515",
        "paper": "https://doi.org/10.1002/qute.202500328",
        "gates": [False, True, True, False, True, False],
        "verdict": "Seventy-nine public circuit/data files resolve quantum and TPM heat on IBM devices, but both heat legs are expectation-value functionals of related circuit-outcome probabilities. No independent calorimetric or generalized-force endpoint is present.",
    },
    {
        "id": "zanin_correlated_photonic_demon",
        "source": "https://doi.org/10.5281/zenodo.5113016",
        "paper": "https://doi.org/10.22331/Q-2022-09-20-810",
        "gates": [True, False, False, True, True, False],
        "verdict": "The public archive contains one-second detector-count records for demon power and raw time tags for g2 source characterization. The power and correlation files are separate ensembles without a common run key, and no endpoint state tomography is supplied.",
    },
    {
        "id": "labranca_self_calibrated_wqed_photon",
        "source": "https://doi.org/10.17632/vzs49gddsm.2",
        "paper": "https://doi.org/10.1088/1367-2630/ade736",
        "gates": [True, True, True, False, True, False],
        "verdict": "Public frequency- and time-domain records include absolute power calibration, Rabi emission and quadrature distributions for ground, excited and superposition preparations. The emitted energy and density matrix are nevertheless reconstructed from the same calibrated field moments, so no informationally independent positive endpoint action is present.",
    },
    {
        "id": "li_thermal_vapor_quantum_battery",
        "paper": "https://arxiv.org/abs/2604.17518",
        "gates": [True, True, False, True, True, False],
        "verdict": "The same collective-spin battery is evaluated by operational extremal-energy scans and by state tomography, giving a strong two-protocol architecture. The protocols use separate ensembles and no public raw run-level packet or common pairing key was located.",
    },
    {
        "id": "chen_bolometric_microwave_tomography",
        "paper": "https://arxiv.org/abs/2506.20318",
        "gates": [True, True, True, False, True, False],
        "verdict": "A millikelvin bolometer provides amplification-free Wigner tomography through phase-resolved power detection. The action and quantum legs are two functionals of the same bolometric projection histograms, and no separate public endpoint-action packet was located.",
    },
    {
        "id": "zhang_exceptional_point_ion_engine",
        "source": "https://www.scidb.cn/en/anonymous/eTZ2QTNx",
        "paper": "https://doi.org/10.1038/s41467-022-33667-1",
        "gates": [False, True, True, False, True, False],
        "verdict": "Public figure data resolve the single-ion engine across exceptional-point controls, but work, power, efficiency and the coherence signature are all inferred from the same excited-state population record rather than an independent action measurement and full tomography.",
    },
    {
        "id": "bimbard_cold_atom_single_photon",
        "paper": "https://doi.org/10.1103/PhysRevLett.112.033601",
        "gates": [True, True, False, True, True, False],
        "verdict": "The same on-demand cold-atom photon source is characterized by two physically distinct methods: direct photon counting and homodyne state tomography. This is close to the required split-detector architecture, but no reusable public raw-data archive or shared run-level pairing key was located; the published endpoints therefore cannot support source-frozen scoring.",
    },
    {
        "id": "lueders_frequency_resolved_homodyne",
        "source": "https://doi.org/10.5281/zenodo.7528261",
        "paper": "https://doi.org/10.1038/s41598-020-79686-0",
        "gates": [True, False, True, False, True, False],
        "verdict": "A 2.48 GB public archive preserves frequency-resolved homodyne records and photon-number-noise diagnostics. It supplies neither an independent tomographic detector leg nor an independent endpoint-action measurement: the candidate observables are reductions of the same homodyne record.",
    },
    {
        "id": "thekkadath_weak_field_homodyne_tes",
        "paper": "https://doi.org/10.1103/PhysRevA.101.031801",
        "gates": [True, False, False, True, True, False],
        "verdict": "Two transition-edge sensors directly resolve absorbed optical energy while the same joint TES outcomes interpolate between photon-number and quadrature statistics. This is a strong hardware control, but it supplies no independent full endpoint tomography and the candidate action and quantum coordinates are functions of the same detector-event record; no public raw packet was located.",
    },
    {
        "id": "takase_multiphoton_subtraction",
        "source": "https://doi.org/10.6084/m9.figshare.c.6443666",
        "paper": "https://doi.org/10.1364/OE.486270",
        "gates": [True, True, True, False, True, False],
        "verdict": "The experiment combines TES-resolved multi-photon subtraction with phase-locked pulsed homodyne tomography and publishes reconstructed Wigner functions. The TES outcome is a heralding/preparation label rather than an independent endpoint-action measurement of the tomographed output, and the archive contains processed Wigner products rather than a common raw event ledger.",
    },
    {
        "id": "endo_optically_sampled_pnrd",
        "source": "https://doi.org/10.6084/m9.figshare.28321442",
        "paper": "https://doi.org/10.1364/OE.558551",
        "gates": [True, True, True, False, True, False],
        "verdict": "Optically sampled photon-number-resolving heralding is followed by homodyne tomography, with public reconstructed density matrices. The PNR detector prepares/labels the nonclassical state instead of independently measuring the same output action, and the raw homodyne records are available only on request.",
    },
    {
        "id": "cooper_fdp_homodyne_power_calibration",
        "paper": "https://doi.org/10.1038/ncomms5332",
        "gates": [True, True, False, False, True, False],
        "verdict": "Balanced-homodyne data reconstruct one-, two- and three-photon Fock states, while a NIST-traceable power meter independently fixes the amplitudes of the coherent calibration probes. The power leg does not measure the unknown Fock-state endpoint, so it is calibration metadata rather than an independent endpoint action; no public common raw packet was located.",
    },
    {
        "id": "stokowski_integrated_phase_sensor",
        "paper": "https://doi.org/10.1038/s41467-023-38246-6",
        "gates": [True, False, False, False, True, False],
        "verdict": "The integrated sensor records balanced-homodyne squeezing and phase sensitivity together with absolute detector powers, but the power channels monitor and normalize the optical operating point rather than independently measuring the same output's positive body action. The quantum leg is a squeezing/noise witness rather than endpoint state tomography, and the underlying data are available only on request.",
    },
    {
        "id": "kalash_opa_wigner_tomography",
        "paper": "https://doi.org/10.1364/OPTICA.488697",
        "gates": [True, True, False, False, True, False],
        "verdict": "Phase-sensitive optical parametric amplification followed by pulse-resolved direct photon-number detection reconstructs the input Wigner function. Photon number is the tomographic measurement record itself, not an independent endpoint-action leg, and no reusable public raw event archive with a second detector record was located.",
    },
    {
        "id": "najafabadi_intensity_wigner",
        "source": "https://doi.org/10.6084/m9.figshare.25817155.v1",
        "source_10mw": "https://doi.org/10.6084/m9.figshare.25816894.v1",
        "paper": "https://doi.org/10.1098/rsta.2023.0337",
        "gates": [True, True, True, False, True, False],
        "verdict": "This is the strongest public two-detector optical control found: direct SNSPD photon correlations and homodyne Wigner tomography are supplied on the same pump-power/HWP-angle grid. The archives contain processed direct g2 scalars and reconstructed Wigner grids rather than a shot-level joint ledger, and g2 is an independent correlation readout rather than a positive endpoint energy/work/action measurement. The published standard Wigner-to-g2 identity therefore makes the packet a cross-readout comparator, not an A8b action packet.",
    },
    {
        "id": "endo_picosecond_cat_tes_homodyne",
        "source": "https://doi.org/10.5281/zenodo.20640587",
        "paper": "https://arxiv.org/abs/2606.24002",
        "gates": [True, True, True, False, True, False],
        "verdict": "The experiment digitizes TES pulse height and the corresponding pulsed-homodyne outcome simultaneously for each trigger, so the physical protocol has the strongest event-level common ledger located in this search. The public Zenodo record currently contains only a README and explicitly withholds the raw homodyne data because of size, despite the manuscript's broader raw-data statement. Moreover, the TES outcome heralds photon subtraction on the other output arm; it is not yet an independently justified positive endpoint body-action measurement of the tomographed state.",
    },
    {
        "id": "dalbec_constant_ottawa_tes_signals",
        "source": "https://doi.org/10.5281/zenodo.14042152",
        "paper": "https://arxiv.org/abs/2411.05737",
        "gates": [True, False, True, True, True, False],
        "verdict": "The 2 GB public archive supplies raw TES voltage traces, attenuation calibration and OPO-generated optical signals, making it the strongest reusable positive energy/photon-number leg found in this search. It contains no event-paired homodyne quadratures, coherence witness or endpoint state tomography. It therefore cannot be joined to the separate Endo experiment without violating the same-system and source-frozen requirements.",
    },
    {
        "id": "kawasaki_high_rate_pss_tomography",
        "source": "https://doi.org/10.5061/dryad.9p8cz8wqn",
        "paper": "https://doi.org/10.1038/s41467-024-53408-w",
        "gates": [True, True, True, False, True, False],
        "verdict": "The public Dryad packet contains event-level phase-resolved quadrature samples for 1 GHz photon-subtracted-state tomography across pump-power and excess-loss settings, rather than only reconstructed Wigner products. The photon counter heralds state preparation, but its pulse/energy record is absent; pump power and loss are controls rather than positive endpoint body action. The packet is therefore a reusable quantum-tomography leg, not a joint action--quantum endpoint ledger.",
    },
    {
        "id": "zhang_single_photon_vlbi",
        "source": "https://doi.org/10.5061/dryad.fj6q57484",
        "gates": [True, True, True, False, True, False],
        "verdict": "The public Dryad packet combines photon-coincidence/g2 characterization with 17,500 pulse-level homodyne samples for the same heralded single-photon source and includes processing code. The branches are separate ensembles without a shot-level common key, and coincidence/g2 is a photon-statistics comparator rather than an independent positive endpoint energy/work/action measurement. It is therefore a strong two-method control, not an eligible A8b action--quantum packet.",
    },
    {
        "id": "houck_flying_microwave_photon",
        "paper": "https://doi.org/10.1038/nature06126",
        "gates": [True, True, False, True, True, False],
        "verdict": "The experiment combines diode-detected output power, homodyne output voltage and independent source-qubit tomography for an on-demand flying microwave photon. It is a genuine multi-detector action/coherence architecture, but no public raw common-run repository was located. Moreover, the emitted-photon reconstruction uses the measured power and voltage moments themselves, so it does not yet provide an informationally independent endpoint action and full photon-state distance on a reusable ledger.",
    },
]


def main() -> None:
    rows = []
    for candidate in CANDIDATES:
        if len(candidate["gates"]) != len(GATES):
            raise ValueError(f"gate-count mismatch for {candidate['id']}")
        gate_map = dict(zip(GATES, candidate["gates"]))
        passed = all(gate_map.values())
        rows.append(
            {
                **candidate,
                "gates": gate_map,
                "passed_gate_count": sum(gate_map.values()),
                "eligible": passed,
            }
        )

    data_gates = GATES[:-1]
    report = {
        "audit": "MOPR-FQELIG1",
        "audit_freeze": "2026-08-07",
        "search_protocol": "data/public_source_search_protocol.md",
        "external_payload_hashes": "not_recorded_external_data_not_redistributed",
        "claim_level": "finite public-source eligibility audit",
        "eligibility_rule": "all six gates must pass before Tau residual scoring",
        "candidates": rows,
        "eligible_count": sum(row["eligible"] for row in rows),
        "candidate_count": len(rows),
        "gate_pass_counts": {
            gate: sum(bool(row["gates"][gate]) for row in rows) for gate in GATES
        },
        "passed_gate_count_distribution": {
            str(count): sum(row["passed_gate_count"] == count for row in rows)
            for count in range(len(GATES) + 1)
        },
        "nearest_candidate_gate_count": max(row["passed_gate_count"] for row in rows),
        "nearest_candidates": [
            row["id"] for row in rows
            if row["passed_gate_count"] == max(item["passed_gate_count"] for item in rows)
        ],
        "formal_tau_law_frozen": True,
        "data_ready_count_before_operational_map": sum(
            all(row["gates"][gate] for gate in data_gates) for row in rows
        ),
        "maximum_data_gate_count": max(
            sum(bool(row["gates"][gate]) for gate in data_gates) for row in rows
        ),
        "next_required_packet": {
            "same_run_inputs": [
                "independently reconstructed electromagnetic morphology",
                "independent quantum state/coherence tomography",
                "source-side generalized-force or stiffness response defining positive x_A",
                "controls sufficient for a complete standard Hamiltonian/noise comparator",
            ],
            "tau_operationalization": "independent candidate-specific maps from the common source to the frozen gain-free Tau observables, fixed before endpoint scoring",
        },
        "best_pipeline_control": {
            "id": "dassonneville_single_qubit_engine",
            "reason": "It realizes both required physical legs and compares them on one device, but destructive tomography forces separate ensembles and the underlying data are not public.",
            "required_rescue": "public interleaved run blocks with shared control labels, drift monitors and independently reconstructed tomography and output-power endpoints",
        },
        "non_claim": "Failure of source eligibility is not a null Tau signal.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"MOPR_FQ_PUBLIC_ELIGIBILITY {report['eligible_count']}/{report['candidate_count']}")
    for row in rows:
        print(f"{'PASS' if row['eligible'] else 'BLOCK'} {row['id']}")


if __name__ == "__main__":
    main()
