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
        "id": "li_cosmic_ray_qubit_binary_cross_readout",
        "url": "https://doi.org/10.6084/m9.figshare.28815041.v1",
        "gates": {
            "quantum_effect_or_state_map": True,
            "metric_stress_or_morphology_map": True,
            "same_source_joint_indexing": True,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "The public 63-qubit cosmic-ray packet supplies two muon-detector "
            "streams and qubit charge-parity/bit-flip responses on a shared time "
            "base, with binary physical events, controlled quasiparticle injection "
            "and shielding controls. The released Fig3 workbook contains 555148- "
            "and 1000004-row time-series sheets. Its qubit columns are already "
            "smoothed collective response ratios and its detector columns are "
            "continuous voltages. More importantly, the two legs are a commuting "
            "source-monitor/response pair, not independently source-whitened P8 "
            "primitive roles on one carrier. Thresholding them cannot create typed "
            "rank two."
        ),
        "public_file_md5": {
            "Fig3.xlsx": "e50c98fda3253f99d03dfef95e865113",
        },
    },
    {
        "id": "castanet_hybrid_atom_classical_inertial_sensor",
        "url": "https://doi.org/10.5281/zenodo.11543715",
        "gates": {
            "quantum_effect_or_state_map": True,
            "metric_stress_or_morphology_map": True,
            "same_source_joint_indexing": True,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "The public 875.5 MB archive separates atom-interferometer Raman "
            "population/phase records from synchronously acquired classical "
            "accelerometer and fiber-gyro streaming records. This is a genuine "
            "hybrid quantum--classical measurement of one inertial environment "
            "with raw data and an explicit standard phase model. The classical "
            "signals are also used for real-time compensation, however, and the "
            "release does not reconstruct two source-whitened P8 operators or "
            "their informationally complete symmetrized product."
        ),
        "public_file_checksums": {
            "Raw Data.zip": "md5:103d382e42a6f600fd0768c2a3286c40",
        },
    },
    {
        "id": "arnold_all_optical_superconducting_qubit_readout",
        "url": "https://doi.org/10.5281/zenodo.14033026",
        "gates": {
            "quantum_effect_or_state_map": True,
            "metric_stress_or_morphology_map": False,
            "same_source_joint_indexing": True,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "The 32.0 MB public archive contains single-shot qubit IQ data and "
            "matched microwave-to-microwave, microwave-to-optical and all-optical "
            "readout comparisons with analysis code. The physically distinct "
            "detector routes measure the same qubit-cavity observable, so they "
            "are a detector-factorization control rather than a rank-two typed "
            "Q--M/G pair; no P8 symmetrized-product record is present."
        ),
        "public_file_checksums": {
            "AllopticalSCQreadout_data.zip": "md5:d2c73dc589981208cd4444be6adffd26",
        },
    },
    {
        "id": "kono_qubit_accelerometer_common_trigger",
        "url": "https://doi.org/10.5281/zenodo.11034817",
        "gates": {
            "quantum_effect_or_state_map": True,
            "metric_stress_or_morphology_map": True,
            "same_source_joint_indexing": True,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "The public 3.1 GB archive contains synchronously triggered qubit "
            "single-shot quadratures and an independently detected accelerometer "
            "trace. A selective archive audit recovers an 8192-shot, 1 ms qubit "
            "series and a 100000-sample acceleration series aligned by the "
            "published trigger time. This is a genuine common-event dual-detector "
            "packet, but it does not publish independently whitened source "
            "Jacobians or an informationally complete symmetrized operator "
            "product. The accelerometer is an environmental morphology monitor, "
            "not yet a reconstructed P8 mechanical operator."
        ),
        "public_file_checksums": {
            "Zenodo.zip": "md5:24f16c6f981e069d1b915ffb11342dbd",
        },
    },
    {
        "id": "chou_two_resonator_multiphonon_tomography",
        "url": "https://doi.org/10.1038/s41467-025-56454-0",
        "gates": {
            "quantum_effect_or_state_map": True,
            "metric_stress_or_morphology_map": False,
            "same_source_joint_indexing": True,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "Two SAW resonators on separate substrates are each coupled to a "
            "dedicated superconducting qubit, and mechanical Bell and N00N states "
            "are reconstructed tomographically. Public source data reproduce the "
            "reported figure products. Both mechanical endpoints are nevertheless "
            "decoded through their associated qubits, and no independent direct "
            "mechanical/stress detector or P8 typed anticommutator is published."
        ),
    },
    {
        "id": "novikov_hybrid_spin_epr_sensor_network",
        "url": "https://doi.org/10.1038/s41586-025-09224-3",
        "gates": {
            "quantum_effect_or_state_map": True,
            "metric_stress_or_morphology_map": False,
            "same_source_joint_indexing": False,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "The open 2025 atom--light network records two independently "
            "homodyned EPR optical arms and publishes source data and analysis "
            "code. In the realized experiment the signal arm does not traverse "
            "the proposed force sensor, while the spin response is encoded in "
            "the idler optical photocurrent rather than measured by an "
            "independent spin/body terminal. The release provides processed "
            "spectra and figure tables, not immutable common-run typed maps, "
            "source whitening or a P8 symmetrized-product tomography. It is a "
            "strong common-carrier/readout precedent, but not an occupied "
            "cross-role Tau packet."
        ),
    },
    {
        "id": "scigliuzzo_superstrong_quantum_acoustics",
        "url": "https://doi.org/10.5281/zenodo.19737799",
        "gates": {
            "quantum_effect_or_state_map": False,
            "metric_stress_or_morphology_map": True,
            "same_source_joint_indexing": False,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "The 2026 Kerr--SAW platform has dedicated electromagnetic and "
            "mechanical measurement lines and an 8.4 GB public figure/code "
            "archive. It reconstructs spectroscopy, participation ratios and "
            "cross-Kerr shifts rather than a quantum state/effect map, common-"
            "shot typed Jacobians or an informationally complete symmetrized "
            "operator product. It proves hardware separability, not P8 endpoint "
            "eligibility."
        ),
    },
    {
        "id": "manenti_saw_transmon_independent_ports",
        "url": "https://doi.org/10.1038/s41467-017-01063-9",
        "gates": {
            "quantum_effect_or_state_map": False,
            "metric_stress_or_morphology_map": True,
            "same_source_joint_indexing": False,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "The SAW cavity is measured through two IDT ports while the transmon "
            "is read independently through a CPW resonator. Published results are "
            "spectroscopy and Stark shifts, not public common-shot state/process "
            "tomography or a typed anticommutator. The device is a dual-line "
            "hardware precedent but not a scoreable packet."
        ),
    },
    {
        "id": "yang_kladari_mechanical_resonator_quantum_computing",
        "url": "https://doi.org/10.5281/zenodo.18162758",
        "gates": {
            "quantum_effect_or_state_map": True,
            "metric_stress_or_morphology_map": False,
            "same_source_joint_indexing": True,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "The 2026 quantum-acoustic packet publishes timestamped HDF5 runs, "
            "qubit and phonon calibrations, multi-mode state/process tomography "
            "and simulations. The mechanical endpoint is nevertheless decoded "
            "through transmon population readout. No independent mechanical-"
            "stress output map or typed Q--M symmetrized-product record is "
            "published, so two-role source rank is not certified."
        ),
        "public_file_sha256": {
            "MRQC_files.zip": "802768df8387e33b630195e3acaf4e290374ae00f6347360ece2c14359c2cf10",
        },
    },
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
    {
        "id": "bozkurt_mechanical_quantum_memory",
        "url": "https://doi.org/10.5281/zenodo.15069397",
        "gates": {
            "quantum_effect_or_state_map": True,
            "metric_stress_or_morphology_map": False,
            "same_source_joint_indexing": True,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "The public packet supports coherent transmon--nanomechanical state "
            "transfer and tomography, but the mechanical state is decoded through "
            "the transmon. It supplies no independent mechanical/stress output map "
            "or informationally complete cross-terminal symmetrized product."
        ),
    },
    {
        "id": "burkhart_acoustic_phonon_phase_gates",
        "url": "https://doi.org/10.6084/m9.figshare.28589732",
        "gates": {
            "quantum_effect_or_state_map": True,
            "metric_stress_or_morphology_map": False,
            "same_source_joint_indexing": True,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "The public figure packet demonstrates one- and two-phonon phase "
            "control and number-resolving catch, but the acoustic state is decoded "
            "through a transmon qutrit. It is a coherent-degree control rather than "
            "an independently detected Q--M/G pair."
        ),
    },
    {
        "id": "sahu_ista_microwave_optical_tms_event_packet",
        "url": "https://doi.org/10.5281/zenodo.7789418",
        "gates": {
            "quantum_effect_or_state_map": True,
            "metric_stress_or_morphology_map": False,
            "same_source_joint_indexing": True,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "A targeted public processed block preserves 151846 aligned "
            "microwave and optical I/Q events, per-event phase corrections, "
            "30 common frequency bins and scale/noise calibration. The "
            "independent audit reproduces all published variance and Duan/"
            "anti-Duan summary rows to 7.11e-15 and verifies that row pairing "
            "carries genuine cross-modal covariance. The two modes are still "
            "one quantum terminal family rather than independently certified "
            "Tau primitive roles; immutable original run keys, independent "
            "source Gram, informationally complete symmetrized-product "
            "tomography and equal-Gram wrong-family control are absent."
        ),
        "public_member_sha256": {
            "ProcessedData/2021_12_23_1526_TMS_375ms_500mV_23Dec/"
            "IQ_data_502.h5": (
                "9635c34984fb94b5bcba6027fe090a47115f29e78ff7864690c3bc54c85a3b88"
            ),
        },
    },
    {
        "id": "meesala_nonclassical_microwave_optical_pairs",
        "url": "https://doi.org/10.5281/zenodo.10456905",
        "gates": {
            "quantum_effect_or_state_map": True,
            "metric_stress_or_morphology_map": False,
            "same_source_joint_indexing": False,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "One pulsed piezo-optomechanical source feeds an optical SNSPD and "
            "an independent microwave heterodyne chain, and the public aggregate "
            "packet reproduces nonclassical cross-correlations and conditional "
            "microwave antibunching. The release contains figure curves, moment "
            "matrices and bootstrap distributions rather than individual "
            "click--heterodyne pairs or immutable run keys. Its optical and "
            "microwave outputs are not source-certified independent Q--M/G roles, "
            "and no P8 symmetrized-product tomography is published."
        ),
        "public_file_sha256": {
            "Source_Data_Zenodo.zip": (
                "35a9aa2b106f5cd40d73262848866bb8604a1f050347434d74905ae5129c0a3e"
            ),
        },
    },
    {
        "id": "xanadu_gkp_pnr_homodyne_common_event_packet",
        "url": "https://github.com/XanaduAI/xanadu-gkp-data",
        "gates": {
            "quantum_effect_or_state_map": True,
            "metric_stress_or_morphology_map": False,
            "same_source_joint_indexing": True,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "The public GKP archive records three photon-number-resolving "
            "detector outcomes and one homodyne quadrature for every repetition, "
            "together with local-oscillator phase, calibration arrays, code and "
            "reconstructed density matrices. It is therefore a strong public "
            "common-event multidetector packet. The PNR records herald the state "
            "on three modes and the homodyne record tomographs the remaining "
            "mode, however. These are commuting preparation--tomography roles "
            "on different subsystems, not two independently reconstructed P8 "
            "primitive terminal maps on one occupied carrier. Their joint "
            "availability cannot be promoted into typed rank two or a measured "
            "symmetrized product."
        ),
    },
    {
        "id": "murch_thermal_coherent_readout_energetics_archive",
        "url": "https://murch.physics.wustl.edu/docs/Thermal_readout_data_and_script.zip",
        "gates": {
            "quantum_effect_or_state_map": True,
            "metric_stress_or_morphology_map": True,
            "same_source_joint_indexing": False,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "The public archive compares thermal and coherent microwave readout "
            "through calibrated qubit-population, voltage and histogram sweeps. "
            "It is a useful quantum-state--energetics/readout control with four "
            "MATLAB data files and a public analysis script. The release contains "
            "processed aggregate sweeps from separate configurations rather than "
            "one immutable event-level ledger carrying two independently "
            "reconstructed terminal maps. It therefore cannot establish joint "
            "source indexing, typed rank two or the P8 product observable."
        ),
        "public_file_sha256": {
            "Thermal_readout_data_and_script.zip": (
                "087abd71bc638aa6c216746f5cc0d2a8d86091aa2470053710438d134016224a"
            ),
        },
    },
    {
        "id": "makihara_quantum_jumps_of_sound_trajectories",
        "url": "https://doi.org/10.5281/zenodo.20944616",
        "gates": {
            "quantum_effect_or_state_map": True,
            "metric_stress_or_morphology_map": True,
            "same_source_joint_indexing": True,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "The public quantum-acoustic release contains 8447 trajectories "
            "with 294 binary phonon-parity checks each, number-splitting data "
            "and the complete forward--backward reconstruction code. It is an "
            "excellent event-level natural-jump control. The phonon-number "
            "trajectory and the quantum measurement record are both decoded "
            "from the same repeated qubit-parity stream, however. They do not "
            "supply two independently reconstructed terminal maps, so their "
            "stacked source novelty remains rank one and no independent P8 "
            "symmetrized product is measured."
        ),
        "public_file_md5": {
            "post_selected_trajectories.csv": "a58ef64030570b966680747238468ce7",
            "forward_backward.py": "b5275e8026efbbfa30aaa90a35690afd",
        },
    },
    {
        "id": "gumus_quantum_phase_slip_calorimetry",
        "url": "https://doi.org/10.5281/zenodo.6389955",
        "gates": {
            "quantum_effect_or_state_map": False,
            "metric_stress_or_morphology_map": True,
            "same_source_joint_indexing": False,
            "stacked_pair_rank_two": False,
            "informationally_complete_symmetrized_product": False,
            "complete_standard_comparator": True,
        },
        "reason": (
            "The public phase-slip calorimetry packet contains calibrated "
            "resonator spectra, time-resolved absorber electron temperature and "
            "return-to-equilibrium traces. It is a direct energetic/morphological "
            "response control for a quantum process. The release consists of "
            "figure-level measurements from separate settings and does not "
            "contain an independently reconstructed quantum state/effect map or "
            "an immutable event ledger pairing that map with the calorimetric "
            "response. It is therefore not a two-terminal P8 packet."
        ),
        "public_file_md5": {
            "FIG3A_DATA.xlsx": "8b265a7b3902e98ae95b50fa76c0ab66",
            "FIG3B_DATA.xlsx": "7c33354365d2598ddbb1d5448a35afdd",
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
                **(
                    {"public_file_md5": candidate["public_file_md5"]}
                    if "public_file_md5" in candidate
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

    # Equal-summary countermodel: single-role spectra, squares and scalar Gram
    # do not identify the symmetrized product.
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    identity_2 = np.eye(2)
    c_a = np.kron(sigma_z, identity_2)
    c_b_anti = np.kron(sigma_x, identity_2)
    c_b_comm = np.kron(identity_2, sigma_z)
    anti_product = c_a @ c_b_anti + c_b_anti @ c_a
    comm_product = c_a @ c_b_comm + c_b_comm @ c_a
    normalized_gram_anti = float(np.trace(c_a @ c_b_anti).real / 4.0)
    normalized_gram_comm = float(np.trace(c_a @ c_b_comm).real / 4.0)
    marginal_countermodel = {
        "carrier_dimension": 4,
        "a_spectrum": np.linalg.eigvalsh(c_a).tolist(),
        "b_anti_spectrum": np.linalg.eigvalsh(c_b_anti).tolist(),
        "b_comm_spectrum": np.linalg.eigvalsh(c_b_comm).tolist(),
        "normalized_gram_anti": normalized_gram_anti,
        "normalized_gram_comm": normalized_gram_comm,
        "anticommutator_anti_frobenius_norm": float(
            np.linalg.norm(anti_product)
        ),
        "anticommutator_comm_frobenius_norm": float(
            np.linalg.norm(comm_product)
        ),
        "verdict": "MARGINAL_SUMMARIES_DO_NOT_IDENTIFY_THE_JORDAN_PRODUCT",
    }
    rho_plus_plus = np.zeros((4, 4))
    rho_plus_plus[0, 0] = 1.0
    anti_moments = (
        float(np.trace(rho_plus_plus @ c_a).real),
        float(np.trace(rho_plus_plus @ c_b_anti).real),
    )
    comm_moments = (
        float(np.trace(rho_plus_plus @ c_a).real),
        float(np.trace(rho_plus_plus @ c_b_comm).real),
    )
    anti_linear_witness = float((anti_moments[0] + anti_moments[1]) / np.sqrt(2.0))
    comm_linear_witness = float((comm_moments[0] + comm_moments[1]) / np.sqrt(2.0))
    marginal_disk_eligible = [
        row["id"]
        for row in rows
        if row["gates"]["quantum_effect_or_state_map"]
        and row["gates"]["metric_stress_or_morphology_map"]
        and row["gates"]["same_source_joint_indexing"]
        and row["gates"]["stacked_pair_rank_two"]
        and row["gates"]["complete_standard_comparator"]
    ]

    result = {
        "schema_version": "1.0",
        "test_id": "P3-NATURE-PAIR-1",
        "freeze_date": "2026-08-10",
        "census_update_date": "2026-08-27",
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
        "equal_summary_commuting_countermodel": marginal_countermodel,
        "product_free_marginal_disk_falsifier": {
            "law": "<C_a>^2+<C_b>^2 <= 1",
            "anti_control_moments": list(anti_moments),
            "anti_control_disk_value": float(sum(x * x for x in anti_moments)),
            "commuting_control_moments": list(comm_moments),
            "commuting_control_disk_value": float(sum(x * x for x in comm_moments)),
            "eligible_count": len(marginal_disk_eligible),
            "eligible_candidates": marginal_disk_eligible,
            "verdict": "NO_CURRENT_INDEPENDENT_TYPED_RANK_TWO_PACKET",
        },
        "fixed_direction_linear_falsifier": {
            "law": "|<C_a>+<C_b>| <= sqrt(2)",
            "predeclared_direction": "(1,1)/sqrt(2)",
            "anti_control_value": anti_linear_witness,
            "commuting_control_value": comm_linear_witness,
            "eligible_count": len(marginal_disk_eligible),
            "eligible_candidates": marginal_disk_eligible,
            "warning": (
                "Binary coding does not establish a typed involution. Sparse source "
                "events and their response cannot be relabelled as a P8 pair."
            ),
            "verdict": "NO_CURRENT_INDEPENDENT_TYPED_RANK_TWO_PACKET",
        },
        "engineered_pauli_non_discrimination": {
            "chosen_pair": ["sigma_z", "sigma_x"],
            "anticommutator_norm": float(
                np.linalg.norm(sigma_z @ sigma_x + sigma_x @ sigma_z)
            ),
            "verdict": "STANDARD_QM_IDENTITY_NOT_POSITIVE_TAU_EVIDENCE",
            "requirement": (
                "Reconstruct typed terminal maps independently of the target "
                "Clifford orientation before testing their product law."
            ),
        },
        "verdict": "NO_ELIGIBLE_TWO_ROLE_PACKET",
        "minimum_reopening_packet": [
            "one predeclared O/Q-side and one M/G-side map on the same source",
            "positive two-direction source Gram and frozen whitening",
            "informationally complete measurement of the symmetrized product",
            "shared run keys and a complete standard/noise comparator",
        ],
        "product_free_reopening_packet": [
            "one certified preparation family shared by both role measurements",
            "two independently calibrated source-whitened maps of stacked rank two",
            "both expectation values with joint covariance",
            "frozen standard, SPAM, drift and wrong-family controls",
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
                "scigliuzzo_superstrong_quantum_acoustics",
                "manenti_saw_transmon_independent_ports",
                "yang_kladari_mechanical_resonator_quantum_computing",
                "novikov_hybrid_spin_epr_sensor_network",
                "antesberger_higher_order_quantum_switch",
                "stemp_donor_spin_gate_set_tomography",
                "van_thiel_piezo_optomechanical_qubit_readout",
                "fluehmann_trapped_ion_mechanical_grid_qubit",
                "burd_quantum_amplification_mechanical_motion",
                "thomas_macroscopic_mechanical_spin_entanglement",
            }
        ),
        "countermodel_has_equal_single_role_summaries": bool(
            np.allclose(c_a @ c_a, np.eye(4))
            and np.allclose(c_b_anti @ c_b_anti, np.eye(4))
            and np.allclose(c_b_comm @ c_b_comm, np.eye(4))
            and np.allclose(
                np.linalg.eigvalsh(c_b_anti), np.linalg.eigvalsh(c_b_comm)
            )
            and abs(normalized_gram_anti - normalized_gram_comm) < 1e-12
        ),
        "countermodel_separates_joint_products": bool(
            np.linalg.norm(anti_product) < 1e-12
            and np.linalg.norm(comm_product) > 1.0
        ),
        "marginal_disk_theorem_witness": bool(
            sum(x * x for x in anti_moments) <= 1.0 + 1e-12
            and sum(x * x for x in comm_moments) > 1.0
        ),
        "fixed_direction_linear_witness": bool(
            abs(anti_linear_witness) <= 1.0 + 1e-12
            and abs(comm_linear_witness) > 1.0
        ),
        "engineered_pauli_pair_is_standard_identity": bool(
            np.linalg.norm(sigma_z @ sigma_x + sigma_x @ sigma_z) < 1e-12
        ),
        "zero_current_marginal_disk_eligible_pairs": (
            len(marginal_disk_eligible) == 0
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
        "**Prediction freeze:** 2026-08-10; **public-census update:** 2026-08-27",
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
        "",
        "The explicit four-dimensional countermodel has identical single-role",
        "spectra, unit squares and scalar Gram for an anticommuting and a",
        "commuting pair, while their anticommutators have Frobenius norms",
        f"{np.linalg.norm(anti_product):.1f} and {np.linalg.norm(comm_product):.1f}.",
        "Marginal weakening therefore cannot identify the P8-D1 product.",
        "",
        "The product-free necessary law <C_a>^2+<C_b>^2<=1 gives values",
        f"{sum(x * x for x in anti_moments):.1f} and",
        f"{sum(x * x for x in comm_moments):.1f} on the two controls.",
        f"Current marginal-disk eligibility: {len(marginal_disk_eligible)}/{len(rows)}.",
        "The fixed positive-direction witness |<C_a>+<C_b>|<=sqrt(2) gives values",
        f"{anti_linear_witness:.6f} and {comm_linear_witness:.6f} on the controls.",
        "An engineered Pauli pair is a standard-QM protocol control, not positive",
        "Tau evidence; typed terminal maps must be reconstructed independently of",
        "the target Clifford orientation before the product law is tested.",
        "The first missing object is independent typed rank two, not product tomography.",
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
