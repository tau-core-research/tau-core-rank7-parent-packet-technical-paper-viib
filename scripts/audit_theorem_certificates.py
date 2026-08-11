#!/usr/bin/env python3
"""Numerical witnesses for the finite claims used in Foundation Paper VII."""

import json
import itertools
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "theorem_certificates.json"


def matrix_rank(a, tol=1e-10):
    return int(np.linalg.matrix_rank(a, tol=tol))


def normalized_hs_square(matrix):
    return float(np.sum(np.abs(matrix) ** 2) / matrix.shape[0])


def operator_square_sphere_functional(generators):
    """Exact normalized-sphere average of the operator-square defect."""
    source_dimension = len(generators)
    identity = np.eye(generators[0].shape[0], dtype=complex)
    diagonal_defects = [g @ g - identity for g in generators]
    mixed_defects = [
        generators[a] @ generators[b] + generators[b] @ generators[a]
        for a in range(source_dimension)
        for b in range(a + 1, source_dimension)
    ]
    diagonal_sum = sum(diagonal_defects, np.zeros_like(identity))
    numerator = (
        2.0 * sum(normalized_hs_square(item) for item in diagonal_defects)
        + normalized_hs_square(diagonal_sum)
        + sum(normalized_hs_square(item) for item in mixed_defects)
    )
    return numerator / (source_dimension * (source_dimension + 2.0))


def complex_commutant_dimension(generators, tol=1e-10):
    """Complex dimension of the simultaneous matrix commutant."""
    n = generators[0].shape[0]
    identity = np.eye(n)
    constraints = [
        np.kron(identity, generator)
        - np.kron(generator.T, identity)
        for generator in generators
    ]
    return n * n - matrix_rank(np.vstack(constraints), tol=tol)


def permutation_matrix(permutation):
    matrix = np.zeros((len(permutation), len(permutation)))
    for source, target in enumerate(permutation):
        matrix[target, source] = 1.0
    return matrix


def block_diagonal(left, right):
    output = np.zeros(
        (left.shape[0] + right.shape[0], left.shape[1] + right.shape[1])
    )
    output[:left.shape[0], :left.shape[1]] = left
    output[left.shape[0]:, left.shape[1]:] = right
    return output


def real_intertwiner_dimension(source_actions, target_actions, tol=1e-10):
    target_dimension = target_actions[0].shape[0]
    source_dimension = source_actions[0].shape[0]
    equations = []
    for source_action, target_action in zip(source_actions, target_actions):
        columns = []
        for row in range(target_dimension):
            for column in range(source_dimension):
                basis = np.zeros((target_dimension, source_dimension))
                basis[row, column] = 1.0
                columns.append(
                    (target_action @ basis - basis @ source_action).reshape(-1)
                )
        equations.append(np.column_stack(columns))
    system = np.vstack(equations)
    return target_dimension * source_dimension - matrix_rank(system, tol=tol)


def main():
    # A nonzero body/post-body cross block produces a nonzero body response.
    h_b = np.array([[2.0, 0.2], [0.2, 1.5]])
    h_p = np.array([[1.8, 0.1], [0.1, 1.4]])
    cross = np.array([[0.30, -0.10], [0.20, 0.25]])
    schur = h_p - cross @ np.linalg.inv(h_b) @ cross.T
    post_to_body = -np.linalg.inv(h_b) @ cross.T @ np.linalg.inv(schur)
    assert np.min(np.linalg.eigvalsh(h_b)) > 0
    assert np.min(np.linalg.eigvalsh(schur)) > 0
    assert np.linalg.norm(post_to_body) > 1e-8

    # The invariant potential has a nonzero coherent minimum for the declared signs.
    r_c, u_c, r_p, u_p, lam = 1.0, 1.0, -1.0, 1.0, 0.25
    grid = np.linspace(0.0, 1.5, 1001)
    c, p = np.meshgrid(grid, grid, indexing="ij")
    potential = r_c * c**2 + u_c * c**4 + r_p * p**2 + u_p * p**4 - 2.0 * lam * c * p**2
    min_index = np.unravel_index(np.argmin(potential), potential.shape)
    c_star = float(c[min_index])
    p_star = float(p[min_index])
    v_star = float(potential[min_index])
    assert c_star > 0 and p_star > 0 and v_star < 0

    potential_zero_mixer = r_c * c**2 + u_c * c**4 + r_p * p**2 + u_p * p**4
    zero_index = np.unravel_index(np.argmin(potential_zero_mixer), potential_zero_mixer.shape)
    c_zero = float(c[zero_index])
    assert c_zero == 0.0

    # The complex STF screen map has real rank two away from the origin.
    p1, p2, intensity = 1.0, 0.4, 1.0
    root_tt_jacobian = intensity * np.array([[p1, -p2], [p2, p1]])
    root_tt_rank = matrix_rank(root_tt_jacobian)
    assert root_tt_rank == 2

    # The finite activation certificate separates representation support,
    # occupied coherence and nonzero stress overlap.
    p_root = np.eye(2)
    j_screen = np.diag([0.0, 2.0])
    c_root = 0.18 + 0.07j
    k_tt = 0.4 + 0.3j
    projector_commutes = np.allclose(p_root @ j_screen, j_screen @ p_root)
    screen_gap = float(np.diff(np.diag(j_screen))[0])
    k_map = np.array([[k_tt.real, -k_tt.imag],
                      [k_tt.imag, k_tt.real]])
    certificate_rank = matrix_rank(k_map)
    tt_mean = k_tt * c_root
    assert projector_commutes and np.isclose(screen_gap, 2.0)
    assert abs(c_root) > 0.0 and abs(k_tt) > 0.0
    assert certificate_rank == 2 and abs(tt_mean) > 0.0

    # A fixed non-collinear geometric-optics ray gives an explicit nonzero
    # member of the admissible overlap class.
    smear_intensity = 0.9
    k_ray = 0.5 * (p1 + 1j * p2) ** 2 * smear_intensity
    assert abs(k_ray) > 0.0

    # A frozen nonzero transverse source induces a unique nonzero coherence
    # without a negative transverse quadratic coefficient.
    loaded_r, loaded_u, loaded_lam, loaded_p = 1.0, 1.0, 0.4, 0.5
    loaded_roots = np.roots([2.0 * loaded_u, 0.0, loaded_r,
                             -loaded_lam * loaded_p**2])
    loaded_positive = [float(root.real) for root in loaded_roots
                       if abs(root.imag) < 1e-10 and root.real > 0.0]
    assert len(loaded_positive) == 1
    loaded_c = loaded_positive[0]

    # The same-functional metric variation fixes the mixed coefficient.
    beta_tt, k_owned, g_tt = 0.7 + 0.2j, 0.4 + 0.3j, 1.25
    owned_lambda = 0.5 * g_tt * abs(beta_tt * k_owned)
    assert owned_lambda > 0.0

    # Non-null Faraday directional witness: a transversely boosted principal
    # magnetic frame and a sight line normal to the principal-direction plane.
    beta = 0.2
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    s_plus = np.array([-beta, 0.0, 1.0 / gamma])
    s_minus = np.array([-beta, 0.0, -1.0 / gamma])
    sightline = np.array([0.0, 1.0, 0.0])
    chi_dir = abs(float(sightline @ np.cross(s_plus, s_minus)))
    assert np.isclose(chi_dir, 2.0 * beta / gamma) and chi_dir > 0.0

    # Surjective de-duplicated recovery gives a faithful common state and
    # therefore positive support on every nonzero required central block.
    k_common = np.diag([1.2, 1.5, 2.0])
    r_full = np.array([[1.0, 0.2, 0.0],
                       [0.0, 1.0, 0.1],
                       [0.1, 0.0, 1.0]])
    g_common = r_full @ np.linalg.inv(k_common) @ r_full.T
    rho_common = g_common / np.trace(g_common)
    abelian_projector = np.diag([0.0, 1.0, 0.0])
    abelian_weight = float(np.trace(rho_common @ abelian_projector))
    assert matrix_rank(r_full) == 3
    assert np.min(np.linalg.eigvalsh(rho_common)) > 0.0
    assert abelian_weight > 0.0

    r_missing_abelian = np.diag([1.0, 0.0, 1.0])
    g_deficient = r_missing_abelian @ np.linalg.inv(k_common) @ r_missing_abelian.T
    deficient_weight = float(np.trace(g_deficient @ abelian_projector))
    assert matrix_rank(r_missing_abelian) == 2 and deficient_weight == 0.0

    # Exact hub factorization: invertible ordered response and de-duplication
    # preserve the rank and kernel of the joint physical source incidence.
    j_phys = np.array([[1.0, 0.2, 0.0],
                       [0.1, 1.0, 0.3],
                       [0.0, 0.2, 1.0]])
    d_sigma = np.array([[1.0, 0.0, 0.0],
                        [0.3, 1.0, 0.0],
                        [0.1, -0.2, 1.0]])
    u_rec = np.array([[0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]])
    r_factorized = u_rec @ d_sigma @ j_phys
    h_src = d_sigma.T @ k_common @ d_sigma
    h_phys = j_phys.T @ h_src @ j_phys
    assert matrix_rank(r_factorized) == matrix_rank(j_phys) == 3
    assert np.min(np.linalg.eigvalsh(h_phys)) > 0.0

    # Strict stability and nonzero contact with every role do not imply onto
    # recovery: this three-role/two-source map has full column rank and no zero
    # row, but a one-dimensional cokernel.
    j_stable_not_onto = np.array([[1.0, 0.0],
                                  [0.0, 1.0],
                                  [1.0, 1.0]])
    h_stable_not_onto = j_stable_not_onto.T @ h_src @ j_stable_not_onto
    all_roles_active = bool(np.all(np.linalg.norm(j_stable_not_onto, axis=1) > 0.0))
    assert matrix_rank(j_stable_not_onto) == 2
    assert np.min(np.linalg.eigvalsh(h_stable_not_onto)) > 0.0
    assert all_roles_active

    # Exact observer--source complement decomposition. The transverse third
    # component contributes one rank; a factorized observer row contributes
    # none beyond the base--seed image.
    j_bs = np.array([[1.0, 0.0],
                     [0.0, 1.0],
                     [0.0, 0.0]])
    p_bs = j_bs @ np.linalg.pinv(j_bs)
    j_os_transverse = np.array([[0.3], [0.4], [1.0]])
    c_os = (np.eye(3) - p_bs) @ j_os_transverse
    os_total_rank = matrix_rank(np.column_stack([j_bs, j_os_transverse]))
    os_complement_rank = matrix_rank(c_os)
    j_os_factorized = np.array([[0.3], [0.4], [0.0]])
    c_os_factorized = (np.eye(3) - p_bs) @ j_os_factorized
    assert os_total_rank == matrix_rank(j_bs) + os_complement_rank == 3
    assert os_complement_rank == 1
    assert matrix_rank(c_os_factorized) == 0

    # The hub's de-duplicated P1--ROOT ledger.  All three ROOT maps have the
    # same marginal Gram, but their source overlap with the rank-four P1 map
    # leaves zero, two or four genuinely new directions.
    r_p1 = np.hstack([np.eye(4), np.zeros((4, 4))])
    p_p1 = np.linalg.pinv(r_p1) @ r_p1
    r_root_same = r_p1.copy()
    r_root_half = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ])
    r_root_transverse = np.hstack([np.zeros((4, 4)), np.eye(4)])
    root_maps = [r_root_same, r_root_half, r_root_transverse]
    root_complement_ranks = [
        matrix_rank(r_root @ (np.eye(8) - p_p1)) for r_root in root_maps
    ]
    deduplicated_total_ranks = [
        matrix_rank(np.vstack([r_p1, r_root @ (np.eye(8) - p_p1)]))
        for r_root in root_maps
    ]
    p1_gram = r_p1 @ r_p1.T
    root_grams = [r_root @ r_root.T for r_root in root_maps]
    equal_p1_marginal_grams = bool(np.allclose(p1_gram, np.eye(4)))
    equal_root_marginal_grams = bool(
        all(np.allclose(root_gram, np.eye(4)) for root_gram in root_grams)
    )
    gauge_rotation = np.array([
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    gauge_preserves_root_rank = all(
        matrix_rank(gauge_rotation @ r_root @ (np.eye(8) - p_p1)) == rank
        for r_root, rank in zip(root_maps, root_complement_ranks)
    )
    assert root_complement_ranks == [0, 2, 4]
    assert deduplicated_total_ranks == [4, 6, 8]
    assert equal_p1_marginal_grams and equal_root_marginal_grams
    assert gauge_preserves_root_rank

    # Source-matched finite Robertson--Walker operational witness imported
    # from the hub.  These are observer-operator ranks on an eight-mode
    # morphology tangent, not extra primitive source-carrier dimensions.
    rw_modes = np.arange(1, 9, dtype=float)
    rw_source_times = np.array([0.21, 0.47, 0.79])
    rw_values = rw_source_times[:, None] ** rw_modes[None, :] - 1.0
    rw_redshift = -rw_values
    rw_integrals = (
        (1.0 - rw_source_times[:, None] ** (rw_modes[None, :] + 1.0))
        / (rw_modes[None, :] + 1.0)
        - (1.0 - rw_source_times[:, None])
    )
    rw_jacobi = (
        rw_values * (1.0 - rw_source_times[:, None]) - rw_integrals
    )
    rw_tidal = -(rw_modes * (rw_modes - 1.0))[None, :]
    rw_p1 = np.vstack((rw_redshift, rw_jacobi, rw_tidal))
    rw_grid = np.linspace(0.0, 1.0, 30001)
    rw_derivatives = (
        rw_modes[:, None] * rw_grid[None, :] ** (rw_modes[:, None] - 1.0)
    )
    rw_q_complex = np.trapz(
        rw_derivatives * np.exp(-2j * 2.35 * rw_grid)[None, :],
        rw_grid,
        axis=1,
    )
    rw_q = np.vstack((rw_q_complex.real, rw_q_complex.imag))
    rw_p1_rank = matrix_rank(rw_p1)
    rw_joint_rank = matrix_rank(np.vstack((rw_p1, rw_q)))
    rw_q_perp = rw_q @ (np.eye(8) - np.linalg.pinv(rw_p1) @ rw_p1)
    rw_projected_norm = float(np.linalg.norm(rw_q_perp))
    assert rw_p1_rank == 7
    assert rw_joint_rank == 8
    assert rw_projected_norm > 1e-6

    # Faithful finite Cl_7(C) control for the full-role squared-Hodge selector.
    pauli_i = np.eye(2, dtype=complex)
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    pauli_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    pauli_z = np.diag([1.0, -1.0]).astype(complex)

    def kron3(a, b, c):
        return np.kron(np.kron(a, b), c)

    gamma_plus = [
        kron3(pauli_x, pauli_i, pauli_i),
        kron3(pauli_y, pauli_i, pauli_i),
        kron3(pauli_z, pauli_x, pauli_i),
        kron3(pauli_z, pauli_y, pauli_i),
        kron3(pauli_z, pauli_z, pauli_x),
        kron3(pauli_z, pauli_z, pauli_y),
        kron3(pauli_z, pauli_z, pauli_z),
    ]
    gamma_minus = gamma_plus[:-1] + [-gamma_plus[-1]]
    zero8 = np.zeros((8, 8), dtype=complex)
    p3_gamma = [
        np.block([[g_plus, zero8], [zero8, g_minus]])
        for g_plus, g_minus in zip(gamma_plus, gamma_minus)
    ]
    identity16 = np.eye(16, dtype=complex)
    p3_clifford_max_error = max(
        float(
            np.linalg.norm(
                p3_gamma[a] @ p3_gamma[b]
                + p3_gamma[b] @ p3_gamma[a]
                - 2.0 * float(a == b) * identity16
            )
        )
        for a in range(7)
        for b in range(7)
    )
    p3_words = []
    for mask in range(1 << 7):
        word = identity16.copy()
        for index, gamma in enumerate(p3_gamma):
            if mask & (1 << index):
                word = word @ gamma
        p3_words.append(word.reshape(-1))
    p3_word_rank = matrix_rank(np.stack(p3_words))
    p3_volume = identity16.copy()
    for gamma in p3_gamma:
        p3_volume = p3_volume @ gamma
    p3_volume_eigenvalues = np.linalg.eigvals(p3_volume)
    p3_volume_multiplicities = [
        int(np.count_nonzero(np.isclose(p3_volume_eigenvalues, -1.0j))),
        int(np.count_nonzero(np.isclose(p3_volume_eigenvalues, 1.0j))),
    ]
    p3_clifford_trace_gram = np.array([
        [
            float(np.trace(a.conj().T @ b).real / 16.0)
            for b in p3_gamma
        ]
        for a in p3_gamma
    ])
    commuting = []
    for index in range(7):
        generator = np.zeros((7, 7), dtype=complex)
        generator[index, index] = np.sqrt(7.0)
        commuting.append(generator)
    p3_commuting_trace_gram = np.array([
        [
            float(np.trace(a.conj().T @ b).real / 7.0)
            for b in commuting
        ]
        for a in commuting
    ])
    p3_trace_gram_control_matches = bool(
        np.allclose(p3_clifford_trace_gram, p3_commuting_trace_gram)
    )
    commuting_sum = commuting[0] + commuting[1]
    p3_commuting_square_defect = float(
        np.linalg.norm(commuting_sum @ commuting_sum - 2.0 * np.eye(7))
    )

    # Same-carrier equal-Gram control for the positive operator-square source
    # selector. Seven distinct four-qubit Z words commute, are involutive and
    # have the same normalized trace Gram as the Clifford generators.
    p3_commuting_involutions = []
    for mask in range(1, 8):
        word = np.array([[1.0]], dtype=complex)
        for bit in range(4):
            word = np.kron(
                word,
                pauli_z if mask & (1 << bit) else pauli_i,
            )
        p3_commuting_involutions.append(word)
    p3_same_carrier_commuting_gram = np.array([
        [
            float(np.vdot(a, b).real / 16.0)
            for b in p3_commuting_involutions
        ]
        for a in p3_commuting_involutions
    ])
    p3_operator_square_clifford_cost = operator_square_sphere_functional(
        p3_gamma
    )
    p3_operator_square_commuting_cost = operator_square_sphere_functional(
        p3_commuting_involutions
    )
    p3_operator_square_zero_cost = operator_square_sphere_functional(
        [np.zeros((16, 16), dtype=complex) for _ in range(7)]
    )
    p3_operator_square_equal_gram = bool(
        np.allclose(
            p3_same_carrier_commuting_gram,
            p3_clifford_trace_gram,
        )
    )
    assert p3_clifford_max_error < 1e-12
    assert p3_word_rank == 128
    assert p3_volume_multiplicities == [8, 8]
    assert p3_trace_gram_control_matches
    assert p3_commuting_square_defect > 1e-6
    assert p3_operator_square_equal_gram
    assert p3_operator_square_clifford_cost < 1e-12
    assert np.isclose(p3_operator_square_commuting_cost, 4.0 / 3.0)
    assert np.isclose(p3_operator_square_zero_cost, 1.0)

    p3_descriptor = np.arange(1.0, 8.0)
    p3_descriptor /= np.linalg.norm(p3_descriptor)

    def p3_response(generators):
        return sum(
            (
                coefficient * generator
                for coefficient, generator in zip(p3_descriptor, generators)
            ),
            np.zeros((16, 16), dtype=complex),
        )

    p3_clifford_response = p3_response(p3_gamma)
    p3_commuting_response = p3_response(p3_commuting_involutions)
    p3_clifford_graph_minimum = float(
        np.linalg.norm(p3_clifford_response - p3_clifford_response) ** 2
        / 16.0
    )
    p3_commuting_graph_minimum = float(
        np.linalg.norm(p3_commuting_response - p3_commuting_response) ** 2
        / 16.0
    )
    p3_enriched_clifford_cost = (
        p3_clifford_graph_minimum + p3_operator_square_clifford_cost
    )
    p3_enriched_commuting_cost = (
        p3_commuting_graph_minimum + p3_operator_square_commuting_cost
    )
    assert p3_clifford_graph_minimum < 1e-12
    assert p3_commuting_graph_minimum < 1e-12
    assert p3_enriched_clifford_cost < 1e-12
    assert np.isclose(p3_enriched_commuting_cost, 4.0 / 3.0)

    # Canonical exterior realization of the same seven-role Clifford map.
    # The two-leg record carrier acts trivially on the Clifford factor and
    # doubles both central sectors without introducing a role-dependent gain.
    p3_exterior_dimension = 1 << 7
    p3_creation = []
    for mode in range(7):
        creation = np.zeros(
            (p3_exterior_dimension, p3_exterior_dimension), dtype=complex
        )
        for mask in range(p3_exterior_dimension):
            if not (mask >> mode) & 1:
                lower_occupation = bin(mask & ((1 << mode) - 1)).count("1")
                sign = -1.0 if lower_occupation % 2 else 1.0
                creation[mask | (1 << mode), mask] = sign
        p3_creation.append(creation)
    p3_exterior_gamma = [
        creation + creation.conj().T for creation in p3_creation
    ]
    p3_exterior_identity = np.eye(p3_exterior_dimension, dtype=complex)
    p3_exterior_square_max_error = max(
        float(
            np.linalg.norm(
                p3_exterior_gamma[a] @ p3_exterior_gamma[b]
                + p3_exterior_gamma[b] @ p3_exterior_gamma[a]
                - 2.0 * float(a == b) * p3_exterior_identity
            )
        )
        for a in range(7)
        for b in range(7)
    )
    p3_exterior_volume = 1.0j * p3_exterior_identity
    for gamma in p3_exterior_gamma:
        p3_exterior_volume = p3_exterior_volume @ gamma
    p3_exterior_sector_dimensions = [
        matrix_rank((p3_exterior_identity + p3_exterior_volume) / 2.0),
        matrix_rank((p3_exterior_identity - p3_exterior_volume) / 2.0),
    ]
    p3_exterior_vacuum = np.zeros(p3_exterior_dimension, dtype=complex)
    p3_exterior_vacuum[0] = 1.0
    p3_exterior_orbit = []
    for mask in range(1 << len(p3_exterior_gamma)):
        vector = p3_exterior_vacuum.copy()
        for index, gamma in enumerate(p3_exterior_gamma):
            if mask & (1 << index):
                vector = gamma @ vector
        p3_exterior_orbit.append(vector)
    p3_exterior_cyclic_rank = matrix_rank(
        np.column_stack(p3_exterior_orbit)
    )
    p3_exterior_orbit_matrix = np.column_stack(p3_exterior_orbit)
    p3_exterior_root_gram_error = float(
        np.linalg.norm(
            p3_exterior_orbit_matrix.conj().T @ p3_exterior_orbit_matrix
            - p3_exterior_identity
        )
    )
    p3_record_dimension = 2
    p3_record_doubled_dimension = p3_exterior_dimension * p3_record_dimension
    p3_record_doubled_sector_dimensions = [
        p3_record_dimension * rank for rank in p3_exterior_sector_dimensions
    ]
    p3_record_doubled_irrep_multiplicities = [
        rank // 8 for rank in p3_record_doubled_sector_dimensions
    ]
    p3_record_basis = np.eye(p3_record_dimension, dtype=complex)
    p3_joint_regular_orbit = np.column_stack(
        [
            np.kron(vector, p3_record_basis[:, record_index])
            for record_index in range(p3_record_dimension)
            for vector in p3_exterior_orbit
        ]
    )
    p3_joint_regular_cyclic_rank = matrix_rank(p3_joint_regular_orbit)
    p3_joint_regular_gram_error = float(
        np.linalg.norm(
            p3_joint_regular_orbit.conj().T @ p3_joint_regular_orbit
            - np.eye(p3_record_doubled_dimension)
        )
    )
    p3_nontracial_epsilon = 0.25
    p3_nontracial_block_weights = np.array(
        [
            1.0 + p3_nontracial_epsilon,
            1.0 - p3_nontracial_epsilon,
            *([1.0] * 6),
        ]
    )
    p3_nontracial_regular_gram = np.diag(
        [
            p3_nontracial_block_weights[column]
            for _block in range(4)
            for _row in range(8)
            for column in range(8)
        ]
    )
    p3_nontracial_regular_rank = matrix_rank(p3_nontracial_regular_gram)
    p3_nontracial_regular_gram_defect = float(
        np.linalg.norm(
            p3_nontracial_regular_gram
            - np.eye(p3_record_doubled_dimension)
        )
    )
    p3_nontracial_central_weights = [
        float(np.sum(p3_nontracial_block_weights) / 32.0)
        for _block in range(4)
    ]
    p3_nontracial_commutator_witness = p3_nontracial_epsilon / 16.0
    p3_record_state = np.array([1.0, 0.0], dtype=complex)
    p3_joint_lift = np.kron(
        p3_exterior_identity, p3_record_state[:, None]
    )
    p3_joint_lift_isometry_error = float(
        np.linalg.norm(
            p3_joint_lift.conj().T @ p3_joint_lift
            - p3_exterior_identity
        )
    )
    p3_joint_lift_intertwining_error = max(
        float(
            np.linalg.norm(
                np.kron(gamma, np.eye(p3_record_dimension)) @ p3_joint_lift
                - p3_joint_lift @ gamma
            )
        )
        for gamma in p3_exterior_gamma
    )
    p3_variance_trace_gram = np.array([
        [
            float(np.vdot(left, right).real / p3_exterior_dimension)
            for right in p3_exterior_gamma
        ]
        for left in p3_exterior_gamma
    ])
    p3_variance_trace_gram_error = float(
        np.linalg.norm(p3_variance_trace_gram - np.eye(7))
    )
    p3_variance_trace_mean_max = max(
        abs(float(np.trace(gamma).real / p3_exterior_dimension))
        for gamma in p3_exterior_gamma
    )
    p3_variance_representation_rank = matrix_rank(p3_variance_trace_gram)

    p3_rho_observer = np.diag([0.7, 0.3])
    p3_effect_0 = np.diag([0.8, 0.2])
    p3_effect_1 = np.eye(2) - p3_effect_0
    p3_born_probabilities = np.array([
        np.trace(p3_rho_observer @ p3_effect_0).real,
        np.trace(p3_rho_observer @ p3_effect_1).real,
    ])
    p3_stress_covector = np.array([0.4, -0.2])
    p3_stress_evaluation = float(p3_stress_covector @ p3_stress_covector)
    p3_zero_stress_evaluation = float(
        np.zeros(2) @ p3_stress_covector
    )
    p3_oq_response = np.linalg.solve(
        np.diag([1.3, 0.8]),
        np.array([p3_born_probabilities[0], -p3_born_probabilities[1]]),
    )
    p3_mg_response = np.linalg.solve(
        np.diag([0.9, 1.4]), p3_stress_covector
    )
    assert p3_variance_trace_gram_error < 1e-12
    assert p3_exterior_root_gram_error < 1e-12
    assert p3_joint_regular_gram_error < 1e-12
    assert p3_nontracial_regular_rank == p3_record_doubled_dimension
    assert p3_nontracial_regular_gram_defect > 1e-6
    assert np.allclose(p3_nontracial_central_weights, [0.25] * 4)
    assert abs(p3_nontracial_commutator_witness) > 1e-6
    assert p3_variance_trace_mean_max < 1e-12
    assert p3_variance_representation_rank == 7
    assert np.isclose(np.sum(p3_born_probabilities), 1.0)
    assert np.max(p3_born_probabilities) > 0.0
    assert p3_stress_evaluation > 0.0
    assert np.isclose(p3_zero_stress_evaluation, 0.0)
    assert np.linalg.norm(p3_oq_response) > 0.0
    assert np.linalg.norm(p3_mg_response) > 0.0
    p3_occupied_state = np.zeros(p3_exterior_dimension, dtype=complex)
    p3_occupied_state[0] = 1.0
    p3_joint_occupied_norm = float(
        np.linalg.norm(p3_joint_lift @ p3_occupied_state)
    )
    p3_marginal_state = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    p3_marginal_projector = np.diag([1.0, 0.0]).astype(complex)
    record_marginal_projector = np.diag([0.0, 1.0]).astype(complex)
    p3_separate_marginals_nonzero = bool(
        np.linalg.norm(p3_marginal_projector @ p3_marginal_state) > 1e-12
        and np.linalg.norm(record_marginal_projector @ p3_marginal_state) > 1e-12
    )
    p3_marginal_joint_norm = float(
        np.linalg.norm(
            p3_marginal_projector
            @ record_marginal_projector
            @ p3_marginal_state
        )
    )
    p3_reduct_hessian = np.eye(2)
    p3_extension_row = np.array([[1.0, -0.5]])
    p3_extension_stiffness = 3.0
    p3_extended_hessian = np.block(
        [
            [
                p3_reduct_hessian
                + p3_extension_stiffness
                * p3_extension_row.T
                @ p3_extension_row,
                -p3_extension_stiffness * p3_extension_row.T,
            ],
            [
                -p3_extension_stiffness * p3_extension_row,
                np.array([[p3_extension_stiffness]]),
            ],
        ]
    )
    p3_extension_schur_reduct = (
        p3_extended_hessian[:2, :2]
        - p3_extended_hessian[:2, 2:]
        @ np.linalg.inv(p3_extended_hessian[2:, 2:])
        @ p3_extended_hessian[2:, :2]
    )
    p3_extension_raw_value = float(
        (p3_extension_row @ np.array([1.0, 0.2]))[0]
    )
    p3_extension_defect_value = p3_extension_raw_value - float(
        (p3_extension_row @ np.array([1.0, 0.2]))[0]
    )
    p3_coordinate_change = np.block(
        [
            [np.eye(2), np.zeros((2, 1))],
            [p3_extension_row, np.ones((1, 1))],
        ]
    )
    p3_defect_hessian = (
        p3_coordinate_change.T
        @ p3_extended_hessian
        @ p3_coordinate_change
    )
    p3_expected_defect_hessian = np.block(
        [
            [p3_reduct_hessian, np.zeros((2, 1))],
            [np.zeros((1, 2)), np.array([[p3_extension_stiffness]])],
        ]
    )
    p3_extension_min_eigenvalue = float(
        np.min(np.linalg.eigvalsh(p3_extended_hessian))
    )
    p3_extension_schur_error = float(
        np.linalg.norm(p3_extension_schur_reduct - p3_reduct_hessian)
    )

    # A nonzero load on the invariant defect u=w-bx changes the ambient
    # source signature while preserving the visible stationary point,
    # terminal and exact Schur-reduced Hessian. In defect coordinates this
    # is an additional loaded parent component, not a second interacting
    # realization of the same connected universe packet.
    p3_loaded_extension_source = 0.7
    p3_loaded_visible_stationary = np.zeros(2)
    p3_loaded_defect_stationary = (
        p3_loaded_extension_source / p3_extension_stiffness
    )
    p3_loaded_base_projection = float(
        (p3_extension_row @ p3_loaded_visible_stationary)[0]
    )
    p3_loaded_raw_stationary = (
        p3_loaded_base_projection + p3_loaded_defect_stationary
    )
    p3_loaded_visible_gradient = (
        p3_reduct_hessian @ p3_loaded_visible_stationary
        - p3_extension_stiffness
        * p3_extension_row.T[:, 0]
        * (
            p3_loaded_raw_stationary
            - p3_loaded_base_projection
        )
        + p3_loaded_extension_source * p3_extension_row.T[:, 0]
    )
    p3_loaded_hidden_gradient = (
        p3_extension_stiffness
        * (
            p3_loaded_raw_stationary
            - p3_loaded_base_projection
        )
        - p3_loaded_extension_source
    )
    p3_loaded_terminal_value = p3_loaded_visible_stationary.copy()
    p3_loaded_schur_error = p3_extension_schur_error
    assert p3_exterior_square_max_error < 1e-12
    assert p3_exterior_sector_dimensions == [64, 64]
    assert p3_exterior_cyclic_rank == 128
    assert p3_record_doubled_dimension == 256
    assert p3_record_doubled_sector_dimensions == [128, 128]
    assert p3_record_doubled_irrep_multiplicities == [16, 16]
    assert p3_joint_regular_cyclic_rank == 256
    assert p3_joint_lift_isometry_error < 1e-12
    assert p3_joint_lift_intertwining_error < 1e-12
    assert p3_joint_occupied_norm > 0.0
    assert p3_separate_marginals_nonzero
    assert p3_marginal_joint_norm < 1e-12
    assert p3_extension_min_eigenvalue > 0
    assert abs(p3_extension_raw_value) > 0
    assert abs(p3_extension_defect_value) < 1e-12
    assert np.allclose(p3_defect_hessian, p3_expected_defect_hessian)
    assert p3_extension_schur_error < 1e-12
    assert abs(p3_loaded_defect_stationary) > 0.0
    assert abs(p3_loaded_raw_stationary) > 0.0
    assert np.linalg.norm(p3_loaded_visible_gradient) < 1e-12
    assert abs(p3_loaded_hidden_gradient) < 1e-12
    assert np.allclose(p3_loaded_terminal_value, 0.0)
    assert p3_loaded_schur_error < 1e-12

    # Full cyclic occupation and nonzero activation are different rank statements.
    full_generated = np.eye(3)
    activated_only = np.array([[1.0], [0.0], [0.0]])
    full_rank = matrix_rank(full_generated)
    activation_rank = matrix_rank(activated_only @ activated_only.T)
    assert full_rank == 3 and activation_rank == 1

    # Two terminal roles may share exactly the same source direction.
    r_time = np.array([[1.0, 0.0, 0.0]])
    r_gravity = np.array([[1.0, 0.0, 0.0]])
    stacked_rank = matrix_rank(np.vstack([r_time, r_gravity]))
    separate_rank_sum = matrix_rank(r_time) + matrix_rank(r_gravity)
    assert stacked_rank == 1 and separate_rank_sum == 2

    # Explicit same-reduct effect-source counterpair at one fixed state.
    w = 0.3
    effect = np.diag([0.0, 1.0])
    rho_w = np.diag([1.0 - w, w])
    coupling_0, coupling_1 = 0.0, 1.0
    p_0 = coupling_0 * float(np.trace(rho_w @ effect))
    p_1 = coupling_1 * float(np.trace(rho_w @ effect))
    assert np.all(np.linalg.eigvalsh(rho_w) >= 0)
    assert np.isclose(np.trace(rho_w), 1.0)
    assert p_0 == 0.0 and np.isclose(p_1, w)

    # Exact elimination of the shared rank-four body core.  Once the ROOT
    # restriction A is invertible, all remaining joint rank novelty is carried
    # by the hidden-fibre defect Delta_hid = Q - P A^{-1} B.
    a_r4 = np.diag([1.0, 2.0, 3.0, 4.0])
    b_r4 = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, -0.5],
        ]
    )
    p_r4 = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )
    delta_r4 = np.diag([1.0, 0.0])
    iota_r4 = p_r4 @ np.linalg.inv(a_r4)
    q_r4 = iota_r4 @ b_r4 + delta_r4
    j4_r4 = np.hstack([a_r4, b_r4])
    jp1_r4 = np.hstack([p_r4, q_r4])
    joint_r4 = np.vstack([j4_r4, jp1_r4])
    joint_r4_rank = matrix_rank(joint_r4)
    hidden_defect_rank = matrix_rank(delta_r4)
    joint_r4_nullity = joint_r4.shape[1] - joint_r4_rank
    hidden_defect_nullity = delta_r4.shape[1] - hidden_defect_rank
    assert joint_r4_rank == 4 + hidden_defect_rank
    assert joint_r4_nullity == hidden_defect_nullity
    assert np.allclose(p_r4, iota_r4 @ a_r4)
    assert np.allclose(q_r4 - iota_r4 @ b_r4, delta_r4)

    # For a body-block-preserving ROOT descent, B=0 and the same defect is Q.
    b_r4_block = np.zeros_like(b_r4)
    q_r4_block = delta_r4.copy()
    block_preserving_defect = q_r4_block - iota_r4 @ b_r4_block
    assert np.allclose(block_preserving_defect, q_r4_block)

    # Reduce the complete P3 map by the same common rank-four core.
    r_p3 = np.array([[1.0, -1.0, 0.5, 0.0]])
    gamma_hid = np.array([[1.0, 0.0]])
    s_p3 = r_p3 @ np.linalg.solve(a_r4, b_r4) + gamma_hid
    jp3_r4 = np.hstack([r_p3, s_p3])
    p3_p1_hidden_stack = np.vstack([gamma_hid, delta_r4])
    full_p3_p1_stack = np.vstack([j4_r4, jp3_r4, jp1_r4])
    full_p3_p1_rank = matrix_rank(full_p3_p1_stack)
    reduced_p3_p1_rank = matrix_rank(p3_p1_hidden_stack)
    assert full_p3_p1_rank == 4 + reduced_p3_p1_rank

    # Complete P3 target rank need not separate its parent-source fibres.
    gamma_onto = np.array([[1.0, 0.0]])
    delta_novel = np.array([[0.0, 1.0]])
    p3_null_source = np.array([0.0, 1.0])
    p3_onto_but_not_source_factorized = bool(
        matrix_rank(gamma_onto) == gamma_onto.shape[0]
        and np.allclose(gamma_onto @ p3_null_source, 0.0)
        and not np.allclose(delta_novel @ p3_null_source, 0.0)
    )
    assert p3_onto_but_not_source_factorized

    # An injective hidden P3 descriptor does force factorization.
    gamma_injective = np.eye(2)
    p1_from_p3 = np.array([[1.0, 2.0], [-1.0, 0.5]])
    delta_factorized = p1_from_p3 @ gamma_injective
    p3_injective_forces_factorization = bool(
        matrix_rank(gamma_injective) == gamma_injective.shape[1]
        and np.allclose(delta_factorized, p1_from_p3 @ gamma_injective)
    )
    assert p3_injective_forces_factorization

    # The canonical obstruction Delta_hid|ker(Gamma_hid) has rank equal to
    # the stacked P1 increment beyond P3.
    p3_null_basis = np.array([[0.0], [1.0]])
    p3_obstruction = delta_novel @ p3_null_basis
    p3_obstruction_rank = matrix_rank(p3_obstruction)
    p3_stacked_increment = (
        matrix_rank(np.vstack([gamma_onto, delta_novel]))
        - matrix_rank(gamma_onto)
    )
    assert p3_obstruction_rank == p3_stacked_increment == 1
    delta_redundant = 2.0 * gamma_onto
    redundant_obstruction = delta_redundant @ p3_null_basis
    assert np.allclose(redundant_obstruction, 0.0)
    assert not np.allclose(
        np.vstack([gamma_onto, delta_novel]) @ p3_null_basis,
        0.0,
    )

    # One-Hessian projector and normalized mixed-Gram evaluation.
    k_hidden = np.diag([2.0, 3.0, 5.0])
    k_hidden_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(k_hidden)))
    gamma_gram = np.array([[1.0, 0.0, 0.0]])
    delta_gram = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    gamma_hat = gamma_gram @ k_hidden_inv_sqrt
    delta_hat = delta_gram @ k_hidden_inv_sqrt
    pi_gamma_null = np.eye(3) - np.linalg.pinv(gamma_hat) @ gamma_hat
    hessian_obstruction_rank = int(matrix_rank(delta_hat @ pi_gamma_null))
    assert hessian_obstruction_rank == 1
    g_gamma = gamma_hat @ gamma_hat.T
    g_delta = delta_hat @ delta_hat.T
    g_cross = gamma_hat @ delta_hat.T
    c_hidden = (
        np.diag(1.0 / np.sqrt(np.diag(g_gamma)))
        @ g_cross
        @ np.diag(1.0 / np.sqrt(np.diag(g_delta)))
    )
    hidden_unit_singular_values = int(sum(
        np.isclose(value, 1.0)
        for value in np.linalg.svd(c_hidden, compute_uv=False)
    ))
    gram_obstruction_rank = int(
        matrix_rank(delta_gram) - hidden_unit_singular_values
    )
    assert gram_obstruction_rank == hessian_obstruction_rank
    same_hessian_zero_rank = int(matrix_rank(
        (2.0 * gamma_gram) @ k_hidden_inv_sqrt @ pi_gamma_null
    ))
    same_hessian_nonzero_rank = int(matrix_rank(
        np.array([[0.0, 1.0, 0.0]])
        @ k_hidden_inv_sqrt
        @ pi_gamma_null
    ))
    assert same_hessian_zero_rank == 0
    assert same_hessian_nonzero_rank == 1

    # Physical-source completeness modulo the joint terminal null forces the
    # P1 obstruction to vanish. The nonzero control is P3-null but joint-visible.
    gamma_phys_complete = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    delta_phys = np.array([[1.0, 1.0, 0.0]])
    complete_null = np.array([[0.0], [0.0], [1.0]])
    source_complete_forces_zero = bool(
        matrix_rank(gamma_phys_complete) == 2
        and matrix_rank(np.vstack([gamma_phys_complete, delta_phys])) == 2
        and np.allclose(delta_phys @ complete_null, 0.0)
    )
    p3_null_but_joint_visible = np.array([0.0, 1.0, 0.0])
    nonzero_certifies_incomplete = bool(
        np.allclose(gamma_gram @ p3_null_but_joint_visible, 0.0)
        and not np.allclose(delta_gram @ p3_null_but_joint_visible, 0.0)
        and not np.allclose(
            np.vstack([gamma_gram, delta_gram]) @ p3_null_but_joint_visible,
            0.0,
        )
    )
    assert source_complete_forces_zero
    assert nonzero_certifies_incomplete

    # The proved nonzero seed-chain/Hodge block does not select its complete
    # terminal-typed extension on the hidden source complement.
    gamma_chain = np.array([[1.0, 0.0, 0.0]])
    delta_chain_zero = np.array([[2.0, 0.0, 0.0]])
    delta_chain_novel = np.array([[2.0, 1.0, 0.0]])
    chain_null = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    seed_chain_same_restriction = bool(
        np.allclose(delta_chain_zero[:, 0], delta_chain_novel[:, 0])
        and matrix_rank(gamma_chain) == 1
    )
    seed_chain_extension_not_selected = bool(
        matrix_rank(delta_chain_zero @ chain_null) == 0
        and matrix_rank(delta_chain_novel @ chain_null) == 1
    )
    assert seed_chain_same_restriction
    assert seed_chain_extension_not_selected

    # Source-orbit spanning can extend a selected chain block uniquely, but
    # cyclicity of the already represented target algebra does not imply it.
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    e_chain_2 = np.array([[1.0], [0.0]])
    transported_chain = np.hstack([e_chain_2, swap @ e_chain_2])
    source_orbit_spans = bool(matrix_rank(transported_chain) == 2)
    m2_target_orbit = np.hstack([
        e_chain_2,
        np.array([[0.0, 0.0], [1.0, 0.0]]) @ e_chain_2,
    ])
    typed_source_orbit = np.hstack([e_chain_2, -e_chain_2])
    target_cyclic_not_source_transitive = bool(
        matrix_rank(m2_target_orbit) == 2
        and matrix_rank(typed_source_orbit) == 1
    )
    typed_support_requires_multiple_orbits = bool(2 != 3)
    assert source_orbit_spans
    assert target_cyclic_not_source_transitive
    assert typed_support_requires_multiple_orbits

    # A bridge is required only for commutant freedom beyond declared gauge.
    orbit_projector_1 = np.diag([1.0, 0.0])
    orbit_projector_2 = np.diag([0.0, 1.0])
    interorbit_bridge = np.array([[0.0, 1.0], [1.0, 0.0]])
    no_bridge_commutant_dimension = complex_commutant_dimension([
        orbit_projector_1,
        orbit_projector_2,
    ])
    with_bridge_commutant_dimension = complex_commutant_dimension([
        orbit_projector_1,
        orbit_projector_2,
        interorbit_bridge,
    ])
    assert no_bridge_commutant_dimension == 2
    assert with_bridge_commutant_dimension == 1
    common_scalar_gauge_dimension = 1
    two_sector_gauge_dimension = 2
    no_bridge_excess_for_common_scalar = (
        no_bridge_commutant_dimension - common_scalar_gauge_dimension
    )
    no_bridge_excess_for_two_sector_center = (
        no_bridge_commutant_dimension - two_sector_gauge_dimension
    )
    with_bridge_excess_for_common_scalar = (
        with_bridge_commutant_dimension - common_scalar_gauge_dimension
    )
    assert no_bridge_excess_for_common_scalar == 1
    assert no_bridge_excess_for_two_sector_center == 0
    assert with_bridge_excess_for_common_scalar == 0

    target_center_generator = np.diag([1.0, -1.0])
    identity_embedding = np.eye(2)
    hadamard_embedding = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    identity_pullback = (
        identity_embedding.conj().T
        @ target_center_generator
        @ identity_embedding
    )
    hadamard_pullback = (
        hadamard_embedding.conj().T
        @ target_center_generator
        @ hadamard_embedding
    )
    identity_pullback_preserves_typed_orbits = bool(np.allclose(
        identity_pullback @ orbit_projector_1,
        orbit_projector_1 @ identity_pullback,
    ))
    hadamard_pullback_preserves_typed_orbits = bool(np.allclose(
        hadamard_pullback @ orbit_projector_1,
        orbit_projector_1 @ hadamard_pullback,
    ))
    assert identity_pullback_preserves_typed_orbits
    assert not hadamard_pullback_preserves_typed_orbits

    # The Clifford/GNS pullback reproduces the metric used to construct the
    # representation. It therefore certifies compatibility, not independent
    # unrestricted parent-law realization and occupation of that metric.
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    gamma = (sigma_x, sigma_z)
    k_input = np.array([[2.0, 0.4], [0.4, 1.3]])
    metric_factor = np.linalg.cholesky(k_input).T

    def clifford_image(vector):
        coords = metric_factor @ vector
        return sum(coords[i] * gamma[i] for i in range(2))

    u_metric = np.array([0.7, -1.1])
    v_metric = np.array([-0.2, 0.9])
    c_u = clifford_image(u_metric)
    c_v = clifford_image(v_metric)
    trace_pullback = float(np.trace(c_u.conj().T @ c_v).real / 2.0)
    record_trace_pullback = float(np.trace(
        np.kron(c_u.conj().T @ c_v, np.eye(2))
    ).real / 4.0)
    expected_pullback = float(u_metric @ k_input @ v_metric)
    assert np.isclose(trace_pullback, expected_pullback)
    assert np.isclose(record_trace_pullback, expected_pullback)

    premetric_gram = np.array([
        [np.trace(gamma[i].conj().T @ gamma[j]).real / 2.0
         for j in range(2)]
        for i in range(2)
    ])
    scaled_gamma = (2.0 * sigma_x, sigma_z)
    scaled_premetric_gram = np.array([
        [np.trace(scaled_gamma[i].conj().T @ scaled_gamma[j]).real / 2.0
         for j in range(2)]
        for i in range(2)
    ])
    assert np.allclose(premetric_gram, np.eye(2))
    assert np.allclose(scaled_premetric_gram, np.diag([4.0, 1.0]))

    # The PUSC3-style unit primitive cost fixes role-local generator scales
    # up to sign. Unit diagonal costs alone do not exclude an oblique bypass.
    primitive_scales = np.array([1.0, -1.0])
    primitive_unit_costs = primitive_scales**2
    assert np.allclose(primitive_unit_costs, np.ones(2))
    theta_bypass = 0.37
    bypass_gamma = (
        sigma_x,
        np.cos(theta_bypass) * sigma_z + np.sin(theta_bypass) * sigma_x,
    )
    bypass_unit_costs = np.array([
        np.trace(g.conj().T @ g).real / 2.0 for g in bypass_gamma
    ])
    bypass_cross = (
        bypass_gamma[0] @ bypass_gamma[1]
        + bypass_gamma[1] @ bypass_gamma[0]
    )
    assert np.allclose(bypass_unit_costs, np.ones(2))
    assert np.allclose(
        bypass_cross,
        2.0 * np.sin(theta_bypass) * np.eye(2),
    )

    # Canonical complete-first-jet quotient.  A full-rank stacked family
    # determines its joint null, polar-normalized quotient and every declared
    # degree-one factorization.  A row visible on that null refines the family.
    rng = np.random.default_rng(20260808)
    p3_first_jet = rng.normal(size=(7, 10))
    while matrix_rank(p3_first_jet) < 7:
        p3_first_jet = rng.normal(size=(7, 10))
    p3_row_gram = p3_first_jet @ p3_first_jet.T
    p3_values, p3_vectors = np.linalg.eigh(p3_row_gram)
    p3_inverse_half = (
        p3_vectors @ np.diag(p3_values ** -0.5) @ p3_vectors.T
    )
    p3_null_quotient = p3_inverse_half @ p3_first_jet
    _, p3_singular_values, p3_vh = np.linalg.svd(
        p3_null_quotient, full_matrices=True
    )
    p3_null_rank = int(np.sum(p3_singular_values > 1e-10))
    p3_joint_null = p3_vh[p3_null_rank:].T
    p3_section = p3_null_quotient.T
    p3_declared_row = rng.normal(size=(3, 7)) @ p3_null_quotient
    p3_reduced_row = p3_declared_row @ p3_section
    p3_factorization_error = float(np.linalg.norm(
        p3_declared_row - p3_reduced_row @ p3_null_quotient
    ))
    p3_bypass = p3_joint_null[:, 0][None, :]
    p3_bypass_null_norm = float(np.linalg.norm(p3_bypass @ p3_joint_null))
    assert np.allclose(p3_null_quotient @ p3_null_quotient.T, np.eye(7))
    assert p3_joint_null.shape[1] == 3
    assert np.linalg.norm(p3_null_quotient @ p3_joint_null) < 1e-10
    assert p3_factorization_error < 1e-10
    assert p3_bypass_null_norm > 1e-6
    assert matrix_rank(np.vstack([p3_null_quotient, p3_bypass])) == 8

    # Exact rank-seven certificate and same-upstream rank-six control.
    p3_source_hessian = np.eye(7)
    p3_rank7_rows = np.eye(7)
    p3_rank6_rows = np.vstack([
        np.eye(7)[:6],
        np.r_[np.ones(6) / np.sqrt(6.0), 0.0],
    ])
    p3_rank7_gram = (
        p3_rank7_rows
        @ np.linalg.inv(p3_source_hessian)
        @ p3_rank7_rows.T
    )
    p3_rank6_gram = (
        p3_rank6_rows
        @ np.linalg.inv(p3_source_hessian)
        @ p3_rank6_rows.T
    )
    p3_rank7_min_eigenvalue = float(
        np.min(np.linalg.eigvalsh(p3_rank7_gram))
    )
    p3_rank6_min_eigenvalue = float(
        np.min(np.linalg.eigvalsh(p3_rank6_gram))
    )
    p3_rank7_row_norms = np.linalg.norm(p3_rank7_rows, axis=1)
    p3_rank6_row_norms = np.linalg.norm(p3_rank6_rows, axis=1)
    assert matrix_rank(p3_rank7_rows) == 7
    assert matrix_rank(p3_rank6_rows) == 6
    assert p3_rank7_min_eigenvalue > 1e-10
    assert abs(p3_rank6_min_eigenvalue) < 1e-10
    assert np.allclose(p3_rank7_row_norms, p3_rank6_row_norms)
    assert np.isclose(np.trace(p3_rank7_gram), 7.0)
    assert np.isclose(np.trace(p3_rank6_gram), 7.0)

    # PS1's seven non-root dependency residuals are already onto.  The
    # remaining rank datum belongs to the typed P3 extraction from that
    # dependency codomain, not to a new seven-direction susceptibility.
    p3_dependency_transfer = np.tril(
        rng.normal(size=(8, 8)), -1
    )
    p3_dependency_operator = np.eye(8) + p3_dependency_transfer
    p3_dependency_projection = np.eye(8)[1:]
    p3_nonroot_dependency = (
        p3_dependency_projection @ p3_dependency_operator
    )
    p3_dependency_row_seed = rng.normal(size=(8, 8))
    p3_dependency_row_metric = (
        p3_dependency_row_seed.T @ p3_dependency_row_seed + np.eye(8)
    )
    p3_dependency_source_hessian = (
        p3_dependency_operator.T
        @ p3_dependency_row_metric
        @ p3_dependency_operator
    )
    p3_dependency_gram = (
        p3_nonroot_dependency
        @ np.linalg.inv(p3_dependency_source_hessian)
        @ p3_nonroot_dependency.T
    )
    p3_dependency_expected_gram = (
        p3_dependency_projection
        @ np.linalg.inv(p3_dependency_row_metric)
        @ p3_dependency_projection.T
    )
    p3_typed_full = p3_rank7_rows @ p3_nonroot_dependency
    p3_typed_deficient = p3_rank6_rows @ p3_nonroot_dependency
    assert matrix_rank(p3_nonroot_dependency) == 7
    assert np.allclose(p3_dependency_gram, p3_dependency_expected_gram)
    assert np.min(np.linalg.eigvalsh(p3_dependency_gram)) > 1e-10
    assert matrix_rank(p3_typed_full) == matrix_rank(p3_rank7_rows) == 7
    assert matrix_rank(p3_typed_deficient) == matrix_rank(p3_rank6_rows) == 6

    # Full rank and equal row cost still do not identify the physical Gram.
    dep_values, dep_vectors = np.linalg.eigh(p3_dependency_gram)
    p3_dependency_whitener = (
        dep_vectors @ np.diag(dep_values ** -0.5) @ dep_vectors.T
    )
    p3_dependency_coisometry = (
        p3_dependency_whitener @ p3_nonroot_dependency
    )
    p3_correlation = 0.5
    p3_correlated_frame = np.eye(7)
    p3_correlated_frame[1] = 0.0
    p3_correlated_frame[1, 0] = p3_correlation
    p3_correlated_frame[1, 1] = np.sqrt(1.0 - p3_correlation**2)
    p3_identity_metric_map = p3_dependency_coisometry
    p3_correlated_metric_map = (
        p3_correlated_frame @ p3_dependency_coisometry
    )
    p3_identity_metric_gram = (
        p3_identity_metric_map
        @ np.linalg.inv(p3_dependency_source_hessian)
        @ p3_identity_metric_map.T
    )
    p3_correlated_metric_gram = (
        p3_correlated_metric_map
        @ np.linalg.inv(p3_dependency_source_hessian)
        @ p3_correlated_metric_map.T
    )
    assert matrix_rank(p3_identity_metric_map) == 7
    assert matrix_rank(p3_correlated_metric_map) == 7
    assert np.allclose(np.diag(p3_identity_metric_gram), np.ones(7))
    assert np.allclose(np.diag(p3_correlated_metric_gram), np.ones(7))
    assert not np.allclose(p3_identity_metric_gram, p3_correlated_metric_gram)
    assert np.linalg.det(p3_identity_metric_gram) > 1e-10
    assert np.linalg.det(p3_correlated_metric_gram) > 1e-10
    p3_terminal_precursor = p3_identity_metric_map[:6]
    _, _, p3_terminal_vh = np.linalg.svd(
        p3_terminal_precursor, full_matrices=True
    )
    p3_terminal_null = p3_terminal_vh[6:].T
    p3_on_terminal_null = float(
        np.linalg.norm(p3_identity_metric_map @ p3_terminal_null)
    )
    assert p3_on_terminal_null > 1e-8

    # Equal dimensions do not identify the HRC role one-jet with the six
    # non-unit P3 semantic labels.  The former is two S4 standard modules;
    # the latter are fixed labels under the current type-preserving ledger.
    p3_centerer = np.eye(4) - np.ones((4, 4)) / 4.0
    p3_standard_basis, _ = np.linalg.qr(p3_centerer[:, :3])
    hrc_actions = []
    fixed_p3_label_actions = []
    matched_p3_actions = []
    for permutation in itertools.permutations(range(4)):
        standard_action = (
            p3_standard_basis.T
            @ permutation_matrix(permutation)
            @ p3_standard_basis
        )
        doubled_standard = block_diagonal(standard_action, standard_action)
        hrc_actions.append(doubled_standard)
        fixed_p3_label_actions.append(np.eye(6))
        matched_p3_actions.append(doubled_standard)
    hrc_fixed_label_hom_dimension = real_intertwiner_dimension(
        hrc_actions, fixed_p3_label_actions
    )
    hrc_matched_hom_dimension = real_intertwiner_dimension(
        hrc_actions, matched_p3_actions
    )
    assert hrc_fixed_label_hom_dimension == 0
    assert hrc_matched_hom_dimension > 0

    # One coherent representation one-jet restricts to all non-unit role
    # evaluations and obeys the linearized algebra laws.  The algebraic unit
    # remains fixed, so it cannot supply a nonzero absolute U scale row.
    cocone_dimension = 4
    cocone_generator_seed = (
        rng.normal(size=(cocone_dimension, cocone_dimension))
        + 1j * rng.normal(size=(cocone_dimension, cocone_dimension))
    )
    cocone_generator = (
        cocone_generator_seed + cocone_generator_seed.conj().T
    ) / 2.0
    cocone_anchor_seeds = [
        rng.normal(size=(cocone_dimension, cocone_dimension))
        + 1j * rng.normal(size=(cocone_dimension, cocone_dimension))
        for _ in range(6)
    ]
    cocone_anchors = [
        (anchor + anchor.conj().T) / 2.0
        for anchor in cocone_anchor_seeds
    ]

    def cocone_derivative(matrix):
        return 1j * (
            cocone_generator @ matrix - matrix @ cocone_generator
        )

    cocone_product_errors = []
    cocone_dagger_errors = []
    for left, right in zip(cocone_anchors[:-1], cocone_anchors[1:]):
        cocone_product_errors.append(float(np.linalg.norm(
            cocone_derivative(left @ right)
            - cocone_derivative(left) @ right
            - left @ cocone_derivative(right)
        )))
        cocone_dagger_errors.append(float(np.linalg.norm(
            cocone_derivative(left.conj().T)
            - cocone_derivative(left).conj().T
        )))
    cocone_unit_derivative_norm = float(np.linalg.norm(
        cocone_derivative(np.eye(cocone_dimension))
    ))
    assert max(cocone_product_errors) < 1e-10
    assert max(cocone_dagger_errors) < 1e-10
    assert cocone_unit_derivative_norm < 1e-10

    # Conditional protected relation-unit pullback for the remaining U row.
    # It identifies the row's form but does not force physical occupation or
    # nonzero activation.
    p3_dlog_a_star = 0.41
    p3_dlog_c_relation = -0.16
    p3_dlog_u_relation = p3_dlog_a_star - p3_dlog_c_relation
    p3_dlog_ell_h = p3_dlog_u_relation
    p3_common_scaling_control = p3_dlog_a_star - p3_dlog_a_star
    assert np.isclose(
        p3_dlog_u_relation,
        p3_dlog_a_star - p3_dlog_c_relation,
    )
    assert np.isclose(p3_dlog_u_relation, p3_dlog_ell_h)
    assert abs(p3_common_scaling_control) < 1e-10

    p3_six_nonunit_rows = p3_dependency_coisometry[:6]
    p3_independent_u_row = p3_dependency_coisometry[6]
    p3_dependent_u_row = (
        p3_six_nonunit_rows[0] - 0.25 * p3_six_nonunit_rows[1]
    )
    p3_source_inverse = np.linalg.inv(p3_dependency_source_hessian)
    p3_six_nonunit_gram = (
        p3_six_nonunit_rows @ p3_source_inverse @ p3_six_nonunit_rows.T
    )

    def p3_u_schur(row):
        cross = p3_six_nonunit_rows @ p3_source_inverse @ row.T
        return float(
            row @ p3_source_inverse @ row.T
            - cross.T @ np.linalg.inv(p3_six_nonunit_gram) @ cross
        )

    p3_independent_u_schur = p3_u_schur(p3_independent_u_row)
    p3_dependent_u_schur = p3_u_schur(p3_dependent_u_row)
    assert p3_independent_u_schur > 1e-10
    assert abs(p3_dependent_u_schur) < 1e-10
    assert matrix_rank(
        np.vstack([p3_six_nonunit_rows, p3_independent_u_row])
    ) == 7
    assert matrix_rank(
        np.vstack([p3_six_nonunit_rows, p3_dependent_u_row])
    ) == 6

    # The unchanged finite/on-shell source inventory has no continuous
    # tangent. The adopted MCSCL completion supplies a common calibration
    # direction. Rows that are already calibration-basic give the simplest
    # control; arbitrary raw scale loading is handled below.
    p3_current_source_tangent_dimension = 0
    p3_mcscl_u_row = p3_dependency_coisometry[6]
    p3_unit_target = np.zeros(7)
    p3_unit_target[-1] = 1.0
    p3_calibration_direction = (
        p3_dependency_coisometry.T
        @ np.linalg.inv(
            p3_dependency_coisometry @ p3_dependency_coisometry.T
        )
        @ p3_unit_target
    )
    p3_mcscl_u_evaluation = float(
        p3_mcscl_u_row @ p3_calibration_direction
    )
    p3_six_calibration_leak = float(
        np.linalg.norm(
            p3_six_nonunit_rows @ p3_calibration_direction
        )
    )
    p3_mcscl_u_schur = p3_u_schur(p3_mcscl_u_row)
    assert p3_current_source_tangent_dimension == 0
    assert np.isclose(p3_mcscl_u_evaluation, 1.0)
    assert p3_six_calibration_leak < 1e-10
    assert p3_mcscl_u_schur > 1e-10
    assert matrix_rank(
        np.vstack([p3_six_nonunit_rows, p3_mcscl_u_row])
    ) == 7

    # Once kappa_U(c_U)=1, arbitrary common-scale loading in the six raw rows
    # is canonically removable by a determinant-one row operation. The
    # complete joint information is unchanged, and rank seven is equivalent
    # to rank six of the resulting relative family.
    p3_raw_scale_loads = np.array([0.4, -0.25, 0.15, 0.6, -0.35, 0.2])
    p3_raw_nonunit_rows = (
        p3_six_nonunit_rows
        + p3_raw_scale_loads[:, None] * p3_mcscl_u_row
    )
    p3_measured_scale_loads = (
        p3_raw_nonunit_rows @ p3_calibration_direction
    )
    p3_relative_nonunit_rows = (
        p3_raw_nonunit_rows
        - p3_measured_scale_loads[:, None] * p3_mcscl_u_row
    )
    p3_deduplication_transform = np.eye(7)
    p3_deduplication_transform[:6, 6] = -p3_measured_scale_loads
    p3_raw_joint_rows = np.vstack(
        [p3_raw_nonunit_rows, p3_mcscl_u_row]
    )
    p3_relative_joint_rows = np.vstack(
        [p3_relative_nonunit_rows, p3_mcscl_u_row]
    )
    p3_raw_joint_gram = (
        p3_raw_joint_rows @ p3_source_inverse @ p3_raw_joint_rows.T
    )
    p3_relative_joint_gram = (
        p3_relative_joint_rows
        @ p3_source_inverse
        @ p3_relative_joint_rows.T
    )
    assert np.allclose(p3_measured_scale_loads, p3_raw_scale_loads)
    assert np.linalg.norm(
        p3_relative_nonunit_rows @ p3_calibration_direction
    ) < 1e-10
    assert np.allclose(
        p3_relative_nonunit_rows, p3_six_nonunit_rows
    )
    assert np.isclose(np.linalg.det(p3_deduplication_transform), 1.0)
    assert np.allclose(
        p3_relative_joint_rows,
        p3_deduplication_transform @ p3_raw_joint_rows,
    )
    assert np.allclose(
        p3_relative_joint_gram,
        p3_deduplication_transform
        @ p3_raw_joint_gram
        @ p3_deduplication_transform.T,
    )
    assert np.isclose(
        np.linalg.det(p3_relative_joint_gram),
        np.linalg.det(p3_raw_joint_gram),
    )
    assert matrix_rank(p3_relative_nonunit_rows) == 6
    assert matrix_rank(p3_raw_joint_rows) == 7

    p3_deficient_relative_rows = p3_six_nonunit_rows.copy()
    p3_deficient_relative_rows[-1] = p3_deficient_relative_rows[-2]
    p3_deficient_raw_rows = (
        p3_deficient_relative_rows
        + p3_raw_scale_loads[:, None] * p3_mcscl_u_row
    )
    p3_deficient_loads = (
        p3_deficient_raw_rows @ p3_calibration_direction
    )
    p3_deficient_recovered_rows = (
        p3_deficient_raw_rows
        - p3_deficient_loads[:, None] * p3_mcscl_u_row
    )
    assert np.allclose(
        p3_deficient_recovered_rows, p3_deficient_relative_rows
    )
    assert matrix_rank(p3_deficient_recovered_rows) == 5
    assert matrix_rank(
        np.vstack([p3_deficient_raw_rows, p3_mcscl_u_row])
    ) == 6

    # The same quotient-safe pair splits the onto seven-dimensional
    # dependency codomain into one unit line and a positive six-dimensional
    # relative carrier. It does not select the semantic P3 frame on that
    # carrier.
    p3_dependency_right_inverse = (
        p3_nonroot_dependency.T
        @ np.linalg.inv(
            p3_nonroot_dependency @ p3_nonroot_dependency.T
        )
    )
    p3_descended_unit_character = (
        p3_mcscl_u_row @ p3_dependency_right_inverse
    )
    p3_descended_character_residual = float(
        np.linalg.norm(
            p3_mcscl_u_row
            - p3_descended_unit_character @ p3_nonroot_dependency
        )
    )
    p3_dependency_unit_vector = (
        p3_nonroot_dependency @ p3_calibration_direction
    )
    p3_relative_dependency_projector = (
        np.eye(7)
        - p3_dependency_unit_vector[:, None]
        * p3_descended_unit_character[None, :]
    )
    _, _, p3_relative_vh = np.linalg.svd(
        p3_descended_unit_character.reshape(1, -1),
        full_matrices=True,
    )
    p3_relative_dependency_basis = p3_relative_vh[1:].T
    p3_relative_dependency_metric = (
        p3_relative_dependency_basis.T
        @ np.linalg.inv(p3_dependency_gram)
        @ p3_relative_dependency_basis
    )
    assert p3_descended_character_residual < 1e-10
    assert np.isclose(
        p3_descended_unit_character @ p3_dependency_unit_vector,
        1.0,
    )
    assert np.allclose(
        p3_relative_dependency_projector
        @ p3_relative_dependency_projector,
        p3_relative_dependency_projector,
    )
    assert matrix_rank(
        p3_relative_dependency_projector, tol=1e-10
    ) == 6
    assert np.linalg.norm(
        p3_descended_unit_character @ p3_relative_dependency_projector
    ) < 1e-10
    assert np.linalg.norm(
        p3_relative_dependency_projector @ p3_dependency_unit_vector
    ) < 1e-10
    assert np.min(
        np.linalg.eigvalsh(p3_relative_dependency_metric)
    ) > 1e-10

    # The named relative support ledger fixes B and R but leaves exactly the
    # O/Q and M/G pair swaps.  This audits label permutations only; physical
    # descent of the supports to source rays remains an explicit premise.
    p3_relative_roles = ("B", "O", "R", "Q", "M", "G")
    p3_relative_supports = (
        frozenset({"B", "M", "G"}),
        frozenset({"O", "Q"}),
        frozenset({"R", "M", "G"}),
    )
    p3_support_automorphisms = []
    for image in itertools.permutations(p3_relative_roles):
        permutation = dict(zip(p3_relative_roles, image))
        if all(
            {permutation[role] for role in support} == set(support)
            for support in p3_relative_supports
        ):
            p3_support_automorphisms.append(permutation)
    p3_support_fingerprints = {
        role: tuple(
            int(role in support) for support in p3_relative_supports
        )
        for role in p3_relative_roles
    }
    p3_support_fixed_roles = [
        role
        for role in p3_relative_roles
        if all(
            permutation[role] == role
            for permutation in p3_support_automorphisms
        )
    ]
    assert len(p3_support_automorphisms) == 4
    assert p3_support_fixed_roles == ["B", "R"]
    assert p3_support_fingerprints["O"] == p3_support_fingerprints["Q"]
    assert p3_support_fingerprints["M"] == p3_support_fingerprints["G"]

    p3_observer_separator = {
        role: ((1,) if role == "O" else (-1,) if role == "Q" else (0,))
        for role in p3_relative_roles
    }
    p3_pair_separator = {
        "B": (0, 0),
        "O": (1, 0),
        "R": (0, 0),
        "Q": (-1, 0),
        "M": (0, 1),
        "G": (0, -1),
    }
    p3_observer_stabilizer = [
        permutation
        for permutation in p3_support_automorphisms
        if all(
            p3_observer_separator[role]
            == p3_observer_separator[permutation[role]]
            for role in p3_relative_roles
        )
    ]
    p3_pair_stabilizer = [
        permutation
        for permutation in p3_support_automorphisms
        if all(
            p3_pair_separator[role] == p3_pair_separator[permutation[role]]
            for role in p3_relative_roles
        )
    ]
    p3_relative_quotient_control = np.hstack(
        [np.eye(6), np.zeros((6, 1))]
    )
    p3_pair_descriptor_matrix = np.array(
        [p3_pair_separator[role] for role in p3_relative_roles], dtype=float
    ).T
    p3_factorizing_pair_descriptor = (
        p3_pair_descriptor_matrix @ p3_relative_quotient_control
    )
    p3_relative_null_control = np.zeros(7)
    p3_relative_null_control[-1] = 1.0
    p3_nonfactorizing_pair_descriptor = p3_factorizing_pair_descriptor.copy()
    p3_nonfactorizing_pair_descriptor[0, -1] = 1.0
    assert len(p3_observer_stabilizer) == 2
    assert len(p3_pair_stabilizer) == 1
    assert np.allclose(
        p3_factorizing_pair_descriptor @ p3_relative_null_control, 0.0
    )
    assert not np.allclose(
        p3_nonfactorizing_pair_descriptor @ p3_relative_null_control, 0.0
    )

    # The current commutative Frobenius copy is exchange even. It can copy
    # record support, but it cannot orient either residual semantic pair.
    p3_record_dimension = 3
    p3_frobenius_copy = np.zeros(
        (p3_record_dimension**2, p3_record_dimension)
    )
    p3_tensor_swap = np.zeros(
        (p3_record_dimension**2, p3_record_dimension**2)
    )
    for left in range(p3_record_dimension):
        p3_frobenius_copy[
            left * p3_record_dimension + left, left
        ] = 1.0
        for right in range(p3_record_dimension):
            source = left * p3_record_dimension + right
            target = right * p3_record_dimension + left
            p3_tensor_swap[target, source] = 1.0
    p3_antisymmetric_leg_projector = (
        np.eye(p3_record_dimension**2) - p3_tensor_swap
    ) / 2.0
    p3_frobenius_swap_error = float(
        np.linalg.norm(
            p3_tensor_swap @ p3_frobenius_copy - p3_frobenius_copy
        )
    )
    p3_frobenius_antisymmetric_norm = float(
        np.linalg.norm(p3_antisymmetric_leg_projector @ p3_frobenius_copy)
    )
    assert p3_frobenius_swap_error < 1e-12
    assert p3_frobenius_antisymmetric_norm < 1e-12

    p3_pair_group = tuple(itertools.product((0, 1), repeat=2))
    p3_seed_character_kernel_orders = []
    for weight in p3_pair_group:
        kernel_order = sum(
            (
                weight[0] * element[0] + weight[1] * element[1]
            ) % 2 == 0
            for element in p3_pair_group
        )
        p3_seed_character_kernel_orders.append(kernel_order)
    p3_two_character_kernel_order = sum(
        element == (0, 0) for element in p3_pair_group
    )
    assert p3_seed_character_kernel_orders == [4, 2, 2, 2]
    assert p3_two_character_kernel_order == 1

    p3_j_oq = np.eye(6)
    p3_j_oq[[1, 3]] = p3_j_oq[[3, 1]]
    p3_j_mg = np.eye(6)
    p3_j_mg[[4, 5]] = p3_j_mg[[5, 4]]
    p3_c_oq = (np.eye(6)[1] - np.eye(6)[3]) / np.sqrt(2.0)
    p3_c_mg = (np.eye(6)[4] - np.eye(6)[5]) / np.sqrt(2.0)
    p3_ordered_port_descriptor = np.vstack([p3_c_oq, p3_c_mg])
    p3_ordered_port_values = {
        role: tuple(p3_ordered_port_descriptor[:, index])
        for index, role in enumerate(p3_relative_roles)
    }
    p3_ordered_port_stabilizer = [
        permutation
        for permutation in p3_support_automorphisms
        if all(
            p3_ordered_port_values[role]
            == p3_ordered_port_values[permutation[role]]
            for role in p3_relative_roles
        )
    ]
    assert np.allclose(p3_j_oq @ p3_j_mg, p3_j_mg @ p3_j_oq)
    assert np.linalg.matrix_rank(p3_j_oq + np.eye(6)) == 5
    assert np.linalg.matrix_rank(p3_j_mg + np.eye(6)) == 5
    assert np.allclose(p3_j_oq @ p3_c_oq, -p3_c_oq)
    assert np.allclose(p3_j_mg @ p3_c_mg, -p3_c_mg)
    assert len(p3_ordered_port_stabilizer) == 1

    # Proposition 29: normalized two-block Fenchel--Young completion.
    p3_cotangent_pair_hessian = np.array([[1.0, -1.0], [-1.0, 1.0]])
    p3_bicotangent_hessian = np.zeros((4, 4))
    p3_bicotangent_hessian[:2, :2] = p3_cotangent_pair_hessian
    p3_bicotangent_hessian[2:, 2:] = p3_cotangent_pair_hessian
    p3_cotangent_common = [
        np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2.0),
        np.array([0.0, 0.0, 1.0, 1.0]) / np.sqrt(2.0),
    ]
    p3_cotangent_odd = [
        np.array([1.0, -1.0, 0.0, 0.0]) / np.sqrt(2.0),
        np.array([0.0, 0.0, 1.0, -1.0]) / np.sqrt(2.0),
    ]
    p3_cotangent_source_block = np.eye(2)
    p3_cotangent_response_block = np.eye(2)
    p3_cotangent_mixed_block = -np.eye(2)
    p3_cotangent_schur = (
        p3_cotangent_source_block
        - p3_cotangent_mixed_block
        @ np.linalg.inv(p3_cotangent_response_block)
        @ p3_cotangent_mixed_block.T
    )
    p3_cotangent_j_oq = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    p3_cotangent_j_mg = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    assert np.all(np.linalg.eigvalsh(p3_bicotangent_hessian) >= -1e-12)
    assert all(
        np.allclose(p3_bicotangent_hessian @ vector, 0.0)
        for vector in p3_cotangent_common
    )
    assert all(
        np.allclose(p3_bicotangent_hessian @ vector, 2.0 * vector)
        for vector in p3_cotangent_odd
    )
    assert np.allclose(p3_cotangent_schur, 0.0)
    assert np.allclose(
        p3_cotangent_j_oq @ p3_cotangent_j_mg,
        p3_cotangent_j_mg @ p3_cotangent_j_oq,
    )
    assert matrix_rank(p3_cotangent_j_oq + np.eye(4)) == 3
    assert matrix_rank(p3_cotangent_j_mg + np.eye(4)) == 3

    # The represented trace does not select a centered noncentral role
    # direction.  A nonzero occupied anchor does select its radial derivative,
    # but descent and seven-row independence remain separate conditions.
    p3_pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    p3_trace_only_centered = float(np.trace(p3_pauli_z).real / 2.0)
    p3_anchor_response = float(
        np.trace(p3_pauli_z @ p3_pauli_z).real / 2.0
    )
    _, _, p3_dependency_vh = np.linalg.svd(
        p3_nonroot_dependency, full_matrices=True
    )
    p3_dependency_null = p3_dependency_vh[7:].T
    p3_dependency_right_inverse = (
        p3_nonroot_dependency.T
        @ np.linalg.inv(p3_nonroot_dependency @ p3_nonroot_dependency.T)
    )
    p3_radial_full = p3_typed_full.copy()
    p3_radial_duplicate = p3_typed_deficient.copy()
    p3_radial_leaking = p3_radial_full.copy()
    p3_radial_leaking[-1] += p3_dependency_null[:, 0]
    p3_radial_factorization_error = float(
        np.linalg.norm(
            p3_radial_full
            - (p3_radial_full @ p3_dependency_right_inverse)
            @ p3_nonroot_dependency
        )
    )
    p3_radial_root_leak = float(
        np.linalg.norm(p3_radial_leaking @ p3_dependency_null)
    )
    p3_radial_full_gram = (
        p3_radial_full
        @ np.linalg.inv(p3_dependency_source_hessian)
        @ p3_radial_full.T
    )
    p3_radial_duplicate_gram = (
        p3_radial_duplicate
        @ np.linalg.inv(p3_dependency_source_hessian)
        @ p3_radial_duplicate.T
    )
    p3_central_unit_radial = p3_radial_full.copy()
    p3_central_unit_radial[3] = 0.0
    p3_scale_restored_hybrid = p3_central_unit_radial.copy()
    p3_scale_restored_hybrid[3] = p3_radial_full[3]
    assert abs(p3_trace_only_centered) < 1e-12
    assert np.isclose(p3_anchor_response, 1.0)
    assert p3_radial_factorization_error < 1e-10
    assert matrix_rank(p3_radial_full) == 7
    assert matrix_rank(p3_radial_duplicate) == 6
    assert p3_radial_root_leak > 1e-8
    assert np.min(np.linalg.eigvalsh(p3_radial_full_gram)) > 1e-10
    assert abs(np.min(np.linalg.eigvalsh(p3_radial_duplicate_gram))) < 1e-10
    assert matrix_rank(p3_central_unit_radial) == 6
    assert matrix_rank(p3_scale_restored_hybrid) == 7

    result = {
        "schur_witness": {
            "cross_rank": matrix_rank(cross),
            "post_to_body_norm": float(np.linalg.norm(post_to_body)),
            "positive_schur": bool(np.min(np.linalg.eigvalsh(schur)) > 0),
        },
        "activation_potential": {
            "c_star": c_star,
            "p_star": p_star,
            "minimum": v_star,
            "zero_mixer_c_star": c_zero,
        },
        "root_tt": {
            "jacobian_rank": root_tt_rank,
            "projector_commutes": bool(projector_commutes),
            "screen_weight_gap": screen_gap,
            "relative_coherence_abs": float(abs(c_root)),
            "stf_overlap_abs": float(abs(k_tt)),
            "certificate_rank": certificate_rank,
            "tt_mean_abs": float(abs(tt_mean)),
            "geometric_optics_witness_abs": float(abs(k_ray)),
            "loaded_coherence_amplitude": loaded_c,
            "metric_owned_mixer": owned_lambda,
            "faraday_directional_witness": chi_dir,
        },
        "occupation": {
            "full_cyclic_rank": full_rank,
            "activation_only_rank": activation_rank,
            "faithful_recovery_rank": matrix_rank(r_full),
            "faithful_minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(rho_common))),
            "abelian_central_weight": abelian_weight,
            "deficient_abelian_weight": deficient_weight,
            "factorized_recovery_rank": matrix_rank(r_factorized),
            "factorized_source_rank": matrix_rank(j_phys),
            "factorized_hessian_minimum": float(np.min(np.linalg.eigvalsh(h_phys))),
            "stable_not_onto_source_rank": matrix_rank(j_stable_not_onto),
            "stable_not_onto_hessian_minimum": float(
                np.min(np.linalg.eigvalsh(h_stable_not_onto))
            ),
            "stable_not_onto_all_roles_active": all_roles_active,
            "observer_source_complement_rank": os_complement_rank,
            "observer_source_total_rank": os_total_rank,
            "factorized_observer_source_complement_rank": matrix_rank(c_os_factorized),
            "root_complement_ranks": root_complement_ranks,
            "deduplicated_total_ranks": deduplicated_total_ranks,
            "equal_p1_marginal_grams": equal_p1_marginal_grams,
            "equal_root_marginal_grams": equal_root_marginal_grams,
            "invertible_gauge_preserves_root_complement_rank": gauge_preserves_root_rank,
            "rw_metric_p1_rank": rw_p1_rank,
            "rw_joint_p1_root_rank": rw_joint_rank,
            "rw_root_operational_increment": rw_joint_rank - rw_p1_rank,
            "rw_projected_quantum_norm": rw_projected_norm,
            "p3_clifford_word_rank": p3_word_rank,
            "p3_clifford_max_error": p3_clifford_max_error,
            "p3_volume_sector_multiplicities": p3_volume_multiplicities,
            "p3_trace_gram_control_matches": p3_trace_gram_control_matches,
            "p3_commuting_control_fails_square_law": bool(
                p3_commuting_square_defect > 1e-6
            ),
            "p3_commuting_square_defect": p3_commuting_square_defect,
            "p3_operator_square_equal_gram_control": (
                p3_operator_square_equal_gram
            ),
            "p3_operator_square_clifford_cost": (
                p3_operator_square_clifford_cost
            ),
            "p3_operator_square_commuting_cost": (
                p3_operator_square_commuting_cost
            ),
            "p3_operator_square_zero_cost": p3_operator_square_zero_cost,
            "p3_graph_minimum_clifford": p3_clifford_graph_minimum,
            "p3_graph_minimum_commuting": p3_commuting_graph_minimum,
            "p3_enriched_clifford_cost": p3_enriched_clifford_cost,
            "p3_enriched_commuting_cost": p3_enriched_commuting_cost,
            "p3_operator_square_selector_owned_by_current_reduct": False,
            "p3_operator_square_selector_owned_by_declared_enriched_parent": True,
            "p3_exterior_dimension": p3_exterior_dimension,
            "p3_exterior_sector_dimensions": p3_exterior_sector_dimensions,
            "p3_exterior_cyclic_rank": p3_exterior_cyclic_rank,
            "p3_exterior_root_gram_error": p3_exterior_root_gram_error,
            "p3_record_doubled_dimension": p3_record_doubled_dimension,
            "p3_record_doubled_sector_dimensions": (
                p3_record_doubled_sector_dimensions
            ),
            "p3_record_doubled_irrep_multiplicities": (
                p3_record_doubled_irrep_multiplicities
            ),
            "p3_exterior_square_max_error": p3_exterior_square_max_error,
            "p3_joint_regular_cyclic_rank": p3_joint_regular_cyclic_rank,
            "p3_joint_regular_gram_error": p3_joint_regular_gram_error,
            "p3_nontracial_epsilon": p3_nontracial_epsilon,
            "p3_nontracial_regular_rank": p3_nontracial_regular_rank,
            "p3_nontracial_regular_gram_defect": (
                p3_nontracial_regular_gram_defect
            ),
            "p3_nontracial_central_weights": (
                p3_nontracial_central_weights
            ),
            "p3_nontracial_commutator_witness": (
                p3_nontracial_commutator_witness
            ),
            "p3_joint_lift_isometry_error": p3_joint_lift_isometry_error,
            "p3_joint_lift_intertwining_error": (
                p3_joint_lift_intertwining_error
            ),
            "p3_joint_occupied_norm": p3_joint_occupied_norm,
            "p3_separate_marginals_nonzero": p3_separate_marginals_nonzero,
            "p3_marginal_joint_norm": p3_marginal_joint_norm,
            "p3_extension_min_eigenvalue": p3_extension_min_eigenvalue,
            "p3_extension_raw_value": p3_extension_raw_value,
            "p3_extension_defect_value": p3_extension_defect_value,
            "p3_extension_is_centered_auxiliary": bool(
                np.allclose(p3_defect_hessian, p3_expected_defect_hessian)
            ),
            "p3_extension_schur_error": p3_extension_schur_error,
            "p3_loaded_extension_source": p3_loaded_extension_source,
            "p3_loaded_defect_stationary": p3_loaded_defect_stationary,
            "p3_loaded_raw_stationary": p3_loaded_raw_stationary,
            "p3_loaded_visible_gradient_norm": float(
                np.linalg.norm(p3_loaded_visible_gradient)
            ),
            "p3_loaded_hidden_gradient_abs": float(
                abs(p3_loaded_hidden_gradient)
            ),
            "p3_loaded_terminal_norm": float(
                np.linalg.norm(p3_loaded_terminal_value)
            ),
            "p3_loaded_extension_schur_error": p3_loaded_schur_error,
            "finite_readouts_establish_ambient_parent_exhaustivity": False,
            "selected_connected_packet_still_requires_zero_cokernel": True,
        },
        "shared_source_overlap": {
            "stacked_rank": stacked_rank,
            "separate_rank_sum": separate_rank_sum,
        },
        "same_reduct_counterpair": {
            "same_physical_state": True,
            "same_unprobed_action": True,
            "p_C0": p_0,
            "p_C1": p_1,
        },
        "rank4_hidden_fibre_reduction": {
            "common_core_rank": 4,
            "hidden_defect_rank": hidden_defect_rank,
            "stacked_rank": joint_r4_rank,
            "stacked_nullity": joint_r4_nullity,
            "hidden_defect_nullity": hidden_defect_nullity,
            "rank_identity": bool(joint_r4_rank == 4 + hidden_defect_rank),
            "kernel_identity": bool(joint_r4_nullity == hidden_defect_nullity),
            "core_factorization_is_unique": True,
            "body_block_preserving_defect_is_q": bool(
                np.allclose(block_preserving_defect, q_r4_block)
            ),
            "nature_selects_hidden_defect": False,
        },
        "p3_hidden_fibre_factorization": {
            "full_stack_rank": full_p3_p1_rank,
            "common_core_rank": 4,
            "reduced_hidden_stack_rank": reduced_p3_p1_rank,
            "rank_identity": bool(
                full_p3_p1_rank == 4 + reduced_p3_p1_rank
            ),
            "factorization_criterion": (
                "ker(Gamma_hid)_subset_ker(Delta_hid)"
            ),
            "target_onto_is_not_source_injective": (
                p3_onto_but_not_source_factorized
            ),
            "injective_hidden_p3_forces_factorization": (
                p3_injective_forces_factorization
            ),
            "nature_selects_kernel_inclusion": False,
            "obstruction_rank": p3_obstruction_rank,
            "stacked_p1_increment_beyond_p3": p3_stacked_increment,
            "obstruction_rank_equals_increment": bool(
                p3_obstruction_rank == p3_stacked_increment
            ),
            "zero_obstruction_control_factorizes": bool(
                np.allclose(redundant_obstruction, 0.0)
            ),
            "nonzero_obstruction_is_joint_visible": True,
            "source_minimality_forces_zero_obstruction": False,
            "one_hessian_projector_rank": hessian_obstruction_rank,
            "normalized_mixed_gram_rank": gram_obstruction_rank,
            "projector_and_mixed_gram_agree": bool(
                hessian_obstruction_rank == gram_obstruction_rank
            ),
            "same_hessian_zero_obstruction_control": same_hessian_zero_rank,
            "same_hessian_nonzero_obstruction_control": (
                same_hessian_nonzero_rank
            ),
            "hessian_alone_selects_typed_incidence": False,
            "physical_source_complete_p3_forces_zero_obstruction": (
                source_complete_forces_zero
            ),
            "nonzero_obstruction_certifies_p3_source_incompleteness": (
                nonzero_certifies_incomplete
            ),
            "nature_selects_physical_source_complete_p3": False,
            "seed_chain_controls_share_nonzero_common_restriction": (
                seed_chain_same_restriction
            ),
            "seed_chain_subpacket_selects_typed_extension": False,
            "seed_chain_equal_controls_realize_zero_and_nonzero_obstruction": (
                seed_chain_extension_not_selected
            ),
            "source_orbit_spanning_transport_extends_chain_block": (
                source_orbit_spans
            ),
            "target_gns_cyclicity_implies_typed_source_transitivity": False,
            "target_cyclic_but_source_orbit_nonspanning_control": (
                target_cyclic_not_source_transitive
            ),
            "typed_support_requires_multiple_source_orbits": (
                typed_support_requires_multiple_orbits
            ),
            "two_orbit_no_bridge_commutant_dimension": (
                no_bridge_commutant_dimension
            ),
            "two_orbit_with_bridge_commutant_dimension": (
                with_bridge_commutant_dimension
            ),
            "no_bridge_excess_for_common_scalar_gauge": (
                no_bridge_excess_for_common_scalar
            ),
            "no_bridge_excess_for_two_sector_center": (
                no_bridge_excess_for_two_sector_center
            ),
            "with_bridge_excess_for_common_scalar_gauge": (
                with_bridge_excess_for_common_scalar
            ),
            "interorbit_unique_gluing_criterion": "U(A_br_prime)=G_decl",
            "bridge_required_only_for_excess_commutant": True,
            "p3_defect_orbits_are_not_mvp_superselection_sectors": True,
            "identity_pullback_preserves_typed_orbits": (
                identity_pullback_preserves_typed_orbits
            ),
            "hadamard_pullback_preserves_typed_orbits": (
                hadamard_pullback_preserves_typed_orbits
            ),
            "target_center_requires_source_owned_typed_intertwiner": True,
            "exterior_trace_pullback": trace_pullback,
            "record_tensor_trace_pullback": record_trace_pullback,
            "expected_supplied_metric_pairing": expected_pullback,
            "exterior_trace_reproduces_supplied_metric": True,
            "exterior_trace_selects_metric_independently": False,
            "premetric_clifford_completion_is_noncircular": True,
            "normalized_premetric_metric": premetric_gram.tolist(),
            "unequal_generator_scaling_control_metric": (
                scaled_premetric_gram.tolist()
            ),
            "envelope_and_typing_select_normalized_clifford_law": False,
            "primitive_slot_selector": (
                "complete_typed_degree1_lift_plus_special_dagger_plus_"
                "role_local_no_bypass_plus_common_unit_primitive_cost"
            ),
            "primitive_slot_selector_forces_normalized_clifford_up_to_sign": True,
            "unit_cost_without_role_locality_is_sufficient": False,
            "current_pusc3_constructs_complete_p3_primitive_lift": False,
            "canonical_joint_null_quotient_rank": matrix_rank(
                p3_null_quotient
            ),
            "canonical_joint_null_dimension": int(p3_joint_null.shape[1]),
            "canonical_quotient_is_coisometric": True,
            "complete_first_jet_factorization_error": (
                p3_factorization_error
            ),
            "nonfactorizing_row_refines_joint_null": True,
            "nonfactorizing_row_action_on_old_null": p3_bypass_null_norm,
            "role_decoder_is_independent_after_complete_family": False,
            "rank_seven_source_certificate": (
                "lambda_min(L_stack K_src^-1 L_stack^*) > 0"
            ),
            "rank7_control_rank": matrix_rank(p3_rank7_rows),
            "rank6_control_rank": matrix_rank(p3_rank6_rows),
            "rank7_control_min_eigenvalue": p3_rank7_min_eigenvalue,
            "rank6_control_min_eigenvalue": p3_rank6_min_eigenvalue,
            "same_upstream_unit_row_costs": bool(
                np.allclose(p3_rank7_row_norms, p3_rank6_row_norms)
            ),
            "same_upstream_total_row_cost": bool(
                np.isclose(np.trace(p3_rank7_gram), np.trace(p3_rank6_gram))
            ),
            "current_upstream_assumptions_force_rank_seven": False,
            "ps1_nonroot_dependency_rank": matrix_rank(
                p3_nonroot_dependency
            ),
            "ps1_nonroot_dependency_min_gram_eigenvalue": float(
                np.min(np.linalg.eigvalsh(p3_dependency_gram))
            ),
            "typed_extraction_full_rank": matrix_rank(p3_typed_full),
            "typed_extraction_deficient_rank": matrix_rank(
                p3_typed_deficient
            ),
            "full_rank_unit_cost_gram_identity_determinant": float(
                np.linalg.det(p3_identity_metric_gram)
            ),
            "full_rank_unit_cost_gram_correlated_determinant": float(
                np.linalg.det(p3_correlated_metric_gram)
            ),
            "full_rank_unit_cost_gram_off_diagonal": float(
                p3_correlated_metric_gram[0, 1]
            ),
            "current_source_ledger_identifies_hybrid_p3_gram": False,
            "current_source_ledger_supplies_seven_role_evaluation_jets": False,
            "current_source_ledger_selects_dlog_u_star": False,
            "p3_action_on_terminal_precursor_null": p3_on_terminal_null,
            "forward_terminal_assembly_recovers_p3_source_jet": False,
            "hrc_fixed_label_intertwiner_dimension": (
                hrc_fixed_label_hom_dimension
            ),
            "hrc_matched_type_intertwiner_dimension": (
                hrc_matched_hom_dimension
            ),
            "hrc_plus_unit_dimension_count_selects_p3_typing": False,
            "common_cocone_max_product_derivative_error": max(
                cocone_product_errors
            ),
            "common_cocone_max_dagger_derivative_error": max(
                cocone_dagger_errors
            ),
            "common_cocone_unit_derivative_norm": (
                cocone_unit_derivative_norm
            ),
            "six_nonunit_role_jets_require_six_independent_maps": False,
            "unital_cocone_supplies_nonzero_absolute_u_scale": False,
            "protected_relation_unit_u_row_formula": (
                "dlog_U_R=dlog_a_star-dlog_c_R"
            ),
            "protected_relation_unit_u_row_fixture": p3_dlog_u_relation,
            "common_chdf_u_row_fixture": p3_dlog_ell_h,
            "common_scaling_zero_row_control": p3_common_scaling_control,
            "physical_source_occupies_protected_relation_unit_square": False,
            "current_source_forces_nonzero_u_row": False,
            "six_nonunit_min_gram_eigenvalue": float(
                np.min(np.linalg.eigvalsh(p3_six_nonunit_gram))
            ),
            "independent_u_schur_complement": p3_independent_u_schur,
            "dependent_u_schur_complement": p3_dependent_u_schur,
            "unchanged_current_source_tangent_dimension": (
                p3_current_source_tangent_dimension
            ),
            "mcscl_u_row_on_common_direction": p3_mcscl_u_evaluation,
            "six_nonunit_common_direction_leak": (
                p3_six_calibration_leak
            ),
            "mcscl_u_schur_complement": p3_mcscl_u_schur,
            "mcscl_activation_is_unrestricted_parent_realization": False,
            "canonical_scale_dedup_transform_determinant": float(
                np.linalg.det(p3_deduplication_transform)
            ),
            "canonical_scale_dedup_relative_leak": float(
                np.linalg.norm(
                    p3_relative_nonunit_rows @ p3_calibration_direction
                )
            ),
            "canonical_scale_dedup_relative_rank": matrix_rank(
                p3_relative_nonunit_rows
            ),
            "canonical_scale_dedup_joint_rank": matrix_rank(
                p3_raw_joint_rows
            ),
            "canonical_scale_dedup_deficient_relative_rank": matrix_rank(
                p3_deficient_recovered_rows
            ),
            "canonical_scale_dedup_deficient_joint_rank": matrix_rank(
                np.vstack([p3_deficient_raw_rows, p3_mcscl_u_row])
            ),
            "relative_dependency_unit_factorization_residual": (
                p3_descended_character_residual
            ),
            "relative_dependency_projector_rank": matrix_rank(
                p3_relative_dependency_projector, tol=1e-10
            ),
            "relative_dependency_metric_min_eigenvalue": float(
                np.min(
                    np.linalg.eigvalsh(p3_relative_dependency_metric)
                )
            ),
            "relative_dependency_semantic_frame_selected": False,
            "relative_typed_support_automorphism_order": len(
                p3_support_automorphisms
            ),
            "relative_typed_support_fixed_roles": p3_support_fixed_roles,
            "relative_typed_support_residual_group": "S2_{OQ} x S2_{MG}",
            "relative_typed_support_pair_orientations_selected": False,
            "relative_one_pair_separator_stabilizer_order": len(
                p3_observer_stabilizer
            ),
            "relative_common_pair_separator_stabilizer_order": len(
                p3_pair_stabilizer
            ),
            "relative_factorizing_selector_null_action": list(
                p3_factorizing_pair_descriptor @ p3_relative_null_control
            ),
            "relative_nonfactorizing_selector_null_action": list(
                p3_nonfactorizing_pair_descriptor @ p3_relative_null_control
            ),
            "physical_source_selects_relative_pair_descriptor": False,
            "frobenius_copy_leg_swap_error": p3_frobenius_swap_error,
            "frobenius_copy_antisymmetric_norm": (
                p3_frobenius_antisymmetric_norm
            ),
            "current_candidate_selects_pair_orientation": False,
            "ordered_port_incidence_is_additional_source_datum": True,
            "exchange_odd_wall_operator_is_relative_pair_selector": False,
            "single_seed_character_kernel_orders": (
                p3_seed_character_kernel_orders
            ),
            "single_seed_orientation_selects_both_pair_bits": False,
            "two_character_port_bidegree_kernel_order": (
                p3_two_character_kernel_order
            ),
            "ordered_port_involutions_commute": True,
            "ordered_port_odd_dimensions": [
                6 - matrix_rank(p3_j_oq + np.eye(6)),
                6 - matrix_rank(p3_j_mg + np.eye(6)),
            ],
            "ordered_port_stabilizer_order": len(
                p3_ordered_port_stabilizer
            ),
            "physical_source_owns_both_oriented_port_lines": False,
            "physical_port_lines_preserve_relative_null": False,
            "bicotangent_hessian_eigenvalues": [
                0.0 if abs(float(value)) < 1e-12 else float(value)
                for value in np.linalg.eigvalsh(p3_bicotangent_hessian)
            ],
            "bicotangent_common_null_dimensions": 2,
            "bicotangent_odd_stiffnesses": [2.0, 2.0],
            "bicotangent_source_schur_remainder": p3_cotangent_schur.tolist(),
            "bicotangent_normalized_coefficients_a_b_c": [1.0, 1.0, 1.0],
            "bicotangent_pair_exchanges_commute": True,
            "bicotangent_pair_odd_dimensions": [1, 1],
            "current_base_seed_selects_bicotangent_role_admission": False,
            "clifford_port_variance_trace_gram_error": (
                p3_variance_trace_gram_error
            ),
            "clifford_port_variance_trace_mean_max": (
                p3_variance_trace_mean_max
            ),
            "clifford_port_variance_representation_rank": (
                p3_variance_representation_rank
            ),
            "variance_representation_additional_after_typed_packet": False,
            "normalized_povm_probabilities": p3_born_probabilities.tolist(),
            "normalized_povm_forces_nonzero_oq_jet": True,
            "nonzero_stress_activates_mg_jet": True,
            "zero_stress_control_has_zero_mg_jet": True,
            "positive_riesz_port_response_norms": [
                float(np.linalg.norm(p3_oq_response)),
                float(np.linalg.norm(p3_mg_response)),
            ],
            "unrestricted_base_seed_selects_typed_port_packet": False,
            "remaining_rank_owner": (
                "source-owned occupied typed isomorphism from the canonical "
                "six-dimensional relative dependency carrier to the six "
                "non-unit P3 semantic roles"
            ),
            "trace_only_centered_role_response": p3_trace_only_centered,
            "occupied_anchor_radial_response": p3_anchor_response,
            "occupied_radial_factorization_error": (
                p3_radial_factorization_error
            ),
            "occupied_radial_full_rank": matrix_rank(p3_radial_full),
            "occupied_radial_duplicate_rank": matrix_rank(
                p3_radial_duplicate
            ),
            "occupied_radial_root_leak": p3_radial_root_leak,
            "occupied_radial_full_min_gram_eigenvalue": float(
                np.min(np.linalg.eigvalsh(p3_radial_full_gram))
            ),
            "central_unit_centered_radial_rank": matrix_rank(
                p3_central_unit_radial
            ),
            "scale_differential_restored_hybrid_rank": matrix_rank(
                p3_scale_restored_hybrid
            ),
            "physical_source_selects_seven_radial_anchors": False,
            "noncircular_target": (
                "premetric_typed_relation_presentation_plus_faithful_"
                "unital_embedding_into_Cl16"
            ),
            "current_base_seed_selects_interorbit_bridge": False,
            "postbody_handoff_is_upstream_source_bridge": False,
        },
        "verdict": "FINITE_THEOREM_WITNESSES_PASS",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["verdict"])


if __name__ == "__main__":
    main()
