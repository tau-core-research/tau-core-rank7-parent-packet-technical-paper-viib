# Tau Core minimal P3 pair Nature-test audit

**Status:** no eligible public two-role packet

```math
C_a^2=C_b^2=I,\qquad \{C_a,C_b\}=0.
```

A single eligible pair can falsify the full completion, but cannot confirm it.

| Candidate | Gates | Pair verdict |
| --- | ---: | --- |
| `hoj_room_temperature_quantum_optomechanics` | 4/6 | Mechanical and Gaussian-state candidates factor through the same squared displacement; stacked novelty is rank one. |
| `qhe_dissipation_engineered_transmon` | 4/6 | Heat/work and state legs are reconstructed from the same population record. |
| `hbn_spin_rf_sensing` | 3/6 | The RF field is a control and the spin response its sensor terminal, not an independent M/G source jet. |
| `labranca_self_calibrated_wqed_photon` | 4/6 | Energy and density-matrix estimates reduce the same calibrated field moments. |
| `chen_bolometric_microwave_tomography` | 4/6 | Both candidate legs are functionals of one bolometric histogram family. |
| `najafabadi_intensity_wigner` | 4/6 | The g2 leg is constrained by the standard Wigner-moment identity, not an independent stress/action jet. |
| `zhang_single_photon_vlbi` | 4/6 | Counting and homodyne branches lack a common shot key and cross-product record. |
| `lualdi_energy_entangled_interferometry` | 3/6 | Sample morphology and source tomography are not paired to one endpoint instrument. |
| `antesberger_higher_order_quantum_switch` | 3/6 | Higher-order process tomography is quantum-sector complete, but does not supply an independently sourced M/G typed perturbation. |
| `stemp_donor_spin_gate_set_tomography` | 3/6 | Gate-set tomography reconstructs quantum operations; SET current is the detector record, not an independent M/G source map. |
| `van_thiel_piezo_optomechanical_qubit_readout` | 2/6 | The public figure data join qubit readout and pump-dependent T1/T2* characterization in one device, but provide no independent mechanical/stress state map and no shared record-level cross product. |
| `fluehmann_trapped_ion_mechanical_grid_qubit` | 3/6 | The public characteristic-function, Wigner, Pauli and process-tomography tables richly reconstruct a motional qubit, but the mechanical oscillator is itself the Q carrier. Its displacement and state maps do not provide an independent M/G typed output. |
| `burd_quantum_amplification_mechanical_motion` | 3/6 | The public same-run tables join calibrated displacement, squeezing and spin response, but displacement is a controlled source value and the oscillator output is inferred through the spin/Q readout. No independent mechanical/stress output map or Q--M product is present. |
| `thomas_macroscopic_mechanical_spin_entanglement` | 3/6 | The experiment couples distinct membrane-mechanical and atomic-spin systems and reconstructs a joint Gaussian EPR witness, but the public files expose one combined optical record. The separate mechanics, atoms and backaction spectra are fitted model components rather than independently measured output maps, so typed rank two and the required cross-product cannot be certified. |

Eligible pair packets: **0/14**.

The common failure is the absence of two independent typed source
directions together with an informationally complete symmetrized-product record.

The existing ROOT--TT common-preparation identity is not an escape:
its stacked differential has rank 2 while each complex terminal already
has rank 2, so its cross-terminal novelty rank is 0.
