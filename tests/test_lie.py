"""
Tests for geope/lie.py.

Tested items:
  Classes:
    - Basis  (properties, linear_span, overlap, verify, apply_interaction_graph,
              apply_interaction_map, generate_parameter_list, generate_bounds,
              _generate_plot_labels, _generate_interaction_labels,
              _generate_interaction_qubits, _generate_interaction_graph,
              _generate_interaction_map, _remove_basis_elements, __len__)
    - Hamiltonian  (init, matrix, unitary, geodesic_hamiltonian, fidelity,
                    parameters_from_hamiltonian)
    - Unitary  (init, fidelity, parameters, geodesic_hamiltonian, __matmul__,
                _check_is_unitary, unitary_fidelity, parameters_from_unitary)
"""

import pytest
import numpy as np
import scipy.linalg as spla

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geope.lie import Basis, Hamiltonian, Unitary
from geope.utils import (
    construct_full_pauli_basis,
    construct_Heisenberg_pauli_basis,
    construct_two_body_pauli_basis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_qubit_basis():
    """Single-qubit Pauli basis (X, Y, Z) — 3 generators, 2×2."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return Basis(np.stack([X, Y, Z]), labels=["X", "Y", "Z"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basis_1q():
    return _single_qubit_basis()


@pytest.fixture
def full_basis_2q():
    return construct_full_pauli_basis(2)


@pytest.fixture
def heisenberg_2q():
    return construct_Heisenberg_pauli_basis(2)


@pytest.fixture
def identity_2x2():
    return np.eye(2, dtype=complex)


@pytest.fixture
def identity_4x4():
    return np.eye(4, dtype=complex)


@pytest.fixture
def hadamard():
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


@pytest.fixture
def cnot():
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )


# ===================================================================
# Tests — Basis
# ===================================================================


class TestBasisInit:
    def test_1q_shape(self, basis_1q):
        assert basis_1q.shape == (3, 2, 2)

    def test_2q_full_shape(self, full_basis_2q):
        assert full_basis_2q.shape == (15, 4, 4)

    def test_lie_algebra_dim(self, basis_1q):
        assert basis_1q.lie_algebra_dim == 3

    def test_dim(self, basis_1q):
        assert basis_1q.dim == 2

    def test_n_qubits(self, basis_1q):
        assert basis_1q.n == 1

    def test_n_qubits_2q(self, full_basis_2q):
        assert full_basis_2q.n == 2

    def test_local_dim_default(self, basis_1q):
        assert basis_1q.local_dim == 2

    def test_labels_stored(self, basis_1q):
        assert basis_1q.labels == ["X", "Y", "Z"]

    def test_len(self, basis_1q):
        assert len(basis_1q) == 3

    def test_len_2q(self, full_basis_2q):
        assert len(full_basis_2q) == 15

    def test_basis_ndim_assertion(self):
        with pytest.raises(AssertionError):
            Basis(np.eye(2, dtype=complex))  # 2-D, not 3-D

    def test_basis_property(self, basis_1q):
        assert np.array_equal(basis_1q.basis, basis_1q._basis)


class TestBasisPlotLabels:
    def test_1q_plot_labels(self, basis_1q):
        labels = basis_1q.plot_labels
        assert len(labels) == 3
        # Each label should be wrapped with $...$
        for lbl in labels:
            assert lbl.startswith("$")
            assert lbl.endswith("$")

    def test_2q_plot_labels(self, full_basis_2q):
        assert len(full_basis_2q.plot_labels) == 15


class TestBasisInteraction:
    def test_interaction_labels_1q(self, basis_1q):
        assert basis_1q.interaction_labels == ["x", "y", "z"]

    def test_interaction_qubits_1q(self, basis_1q):
        for q in basis_1q.interaction_qubits:
            assert isinstance(q, tuple)
            assert len(q) == 1

    def test_interaction_graph_1q(self, basis_1q):
        # single-qubit interactions have length 1, so graph should be empty
        assert basis_1q.interaction_graph == []

    def test_interaction_graph_2q(self, full_basis_2q):
        graph = full_basis_2q.interaction_graph
        assert len(graph) > 0
        for edge in graph:
            assert len(edge) == 2

    def test_interaction_map_1q(self, basis_1q):
        imap = basis_1q.interaction_map
        assert isinstance(imap, dict)
        # 1q basis has one key (1,) mapping to ['x','y','z']
        assert (1,) in imap

    def test_interaction_map_2q(self, full_basis_2q):
        imap = full_basis_2q.interaction_map
        assert isinstance(imap, dict)


class TestBasisApplyInteractionGraph:
    def test_removes_two_body_terms(self, full_basis_2q):
        original_dim = full_basis_2q.lie_algebra_dim
        # Only keep interactions between qubits 1-2
        full_basis_2q.apply_interaction_graph([(1, 2)])
        # Should have removed some 2-body terms that don't match graph
        # All remaining 2-body terms should have qubits (1,2)
        for iq in full_basis_2q.interaction_qubits:
            if len(iq) > 1:
                assert iq == (1, 2)

    def test_preserves_single_body(self, full_basis_2q):
        full_basis_2q.apply_interaction_graph([(1, 2)])
        single_body = [iq for iq in full_basis_2q.interaction_qubits if len(iq) == 1]
        assert len(single_body) > 0


class TestBasisApplyInteractionMap:
    def test_filters_by_map(self, full_basis_2q):
        imap = {(1,): ["x", "z"], (2,): ["x", "z"], (1, 2): ["xx", "zz"]}
        full_basis_2q.apply_interaction_map(imap)
        for iq, il in zip(
            full_basis_2q.interaction_qubits, full_basis_2q.interaction_labels
        ):
            assert iq in imap
            assert il in imap[iq]


class TestBasisLinearSpan:
    def test_zero_params(self, basis_1q):
        result = basis_1q.linear_span(np.zeros(3))
        assert np.allclose(result, 0)

    def test_single_param(self, basis_1q):
        params = np.array([1.0, 0.0, 0.0])
        result = basis_1q.linear_span(params)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        assert np.allclose(result, X)

    def test_shape(self, basis_1q):
        result = basis_1q.linear_span(np.ones(3))
        assert result.shape == (2, 2)


class TestBasisOverlap:
    def test_self_overlap_all_true(self, basis_1q):
        result = basis_1q.overlap(basis_1q)
        assert np.all(result)

    def test_subset_overlap(self, full_basis_2q, heisenberg_2q):
        result = heisenberg_2q.overlap(full_basis_2q)
        # full_basis_2q has 15 elements; heisenberg is a subset
        assert result.shape == (full_basis_2q.lie_algebra_dim,)
        # Heisenberg elements should appear in full basis
        assert result.sum() == heisenberg_2q.lie_algebra_dim


class TestBasisVerify:
    def test_full_pauli_basis_is_orthogonal(self, full_basis_2q):
        # Pauli matrices (divided by dim) are orthogonal under trace inner product
        assert full_basis_2q.verify()

    def test_1q_pauli_orthogonal(self, basis_1q):
        assert basis_1q.verify()


class TestBasisRemoveElements:
    def test_removes_element(self, basis_1q):
        original_dim = basis_1q.lie_algebra_dim
        basis_1q._remove_basis_elements([0])
        assert basis_1q.lie_algebra_dim == original_dim - 1
        assert len(basis_1q) == original_dim - 1

    def test_labels_updated(self, basis_1q):
        basis_1q._remove_basis_elements([0])
        assert "X" not in basis_1q.labels


class TestBasisGenerateParameterList:
    def test_basic_parameter_map(self, basis_1q):
        pmap = {1: {"x": 0.5, "y": 0.3, "z": 0.1}}
        result = basis_1q.generate_parameter_list(pmap)
        assert result == [0.5, 0.3, 0.1]

    def test_missing_interaction_gives_zero(self, basis_1q):
        pmap = {1: {"x": 0.5}}
        result = basis_1q.generate_parameter_list(pmap)
        assert result == [0.5, 0, 0]

    def test_missing_qubit_gives_zeros(self, basis_1q):
        pmap = {99: {"x": 1.0}}
        result = basis_1q.generate_parameter_list(pmap)
        assert result == [0, 0, 0]

    def test_2q_parameter_map(self, full_basis_2q):
        pmap = {1: {"x": 1.0}, 2: {"z": 0.5}, (1, 2): {"xx": 0.2}}
        result = full_basis_2q.generate_parameter_list(pmap)
        assert len(result) == full_basis_2q.lie_algebra_dim


class TestBasisGenerateBounds:
    def test_basic_bounds(self, basis_1q):
        bounds_map = {"x": (-1, 1), "y": (-2, 2), "z": (-3, 3)}
        lower, upper = basis_1q.generate_bounds(bounds_map, piecewise_steps=1)
        assert lower == [[-1, -2, -3]]
        assert upper == [[1, 2, 3]]

    def test_missing_label_gives_inf(self, basis_1q):
        bounds_map = {"x": (-1, 1)}
        lower, upper = basis_1q.generate_bounds(bounds_map, piecewise_steps=1)
        assert lower[0][0] == -1
        assert upper[0][0] == 1
        assert lower[0][1] == -jnp.inf
        assert upper[0][1] == jnp.inf

    def test_multiple_piecewise_steps(self, basis_1q):
        bounds_map = {"x": (-1, 1), "y": (-1, 1), "z": (-1, 1)}
        lower, upper = basis_1q.generate_bounds(bounds_map, piecewise_steps=3)
        assert len(lower) == 3
        assert len(upper) == 3
        for gate_lower, gate_upper in zip(lower, upper):
            assert len(gate_lower) == 3
            assert len(gate_upper) == 3


# ===================================================================
# Tests — Hamiltonian
# ===================================================================


class TestHamiltonian:
    def test_init_creates_matrix(self, basis_1q):
        params = np.array([1.0, 0.0, 0.0])
        h = Hamiltonian(basis_1q, params)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        assert np.allclose(h.matrix, X)

    def test_matrix_shape(self, basis_1q):
        h = Hamiltonian(basis_1q, np.ones(3))
        assert h.matrix.shape == (2, 2)

    def test_unitary_created(self, basis_1q):
        h = Hamiltonian(basis_1q, np.array([0.5, 0.0, 0.0]))
        assert isinstance(h.unitary, Unitary)
        assert h.unitary.matrix.shape == (2, 2)

    def test_unitary_is_unitary(self, basis_1q):
        h = Hamiltonian(basis_1q, np.array([0.3, 0.4, 0.5]))
        U = h.unitary.matrix
        assert np.allclose(U @ U.conj().T, np.eye(2), atol=1e-10)

    def test_zero_params_gives_identity_unitary(self, basis_1q):
        h = Hamiltonian(basis_1q, np.zeros(3))
        assert np.allclose(h.unitary.matrix, np.eye(2), atol=1e-12)

    def test_parameters_stored(self, basis_1q):
        params = np.array([1.0, 2.0, 3.0])
        h = Hamiltonian(basis_1q, params)
        assert np.array_equal(h.parameters, params)

    def test_basis_stored(self, basis_1q):
        h = Hamiltonian(basis_1q, np.zeros(3))
        assert h.basis is basis_1q


class TestHamiltonianGeodesic:
    def test_self_geodesic_is_zero(self, basis_1q):
        params = np.array([0.5, 0.3, 0.1])
        h = Hamiltonian(basis_1q, params)
        geo = h.geodesic_hamiltonian(h.unitary.matrix)
        assert isinstance(geo, Hamiltonian)
        assert np.allclose(geo.matrix, 0, atol=1e-10)

    def test_geodesic_returns_hamiltonian(self, basis_1q, hadamard):
        h = Hamiltonian(basis_1q, np.zeros(3))
        geo = h.geodesic_hamiltonian(hadamard)
        assert isinstance(geo, Hamiltonian)

    def test_geodesic_unitary_achieves_target(self, basis_1q, hadamard):
        h = Hamiltonian(basis_1q, np.zeros(3))
        geo = h.geodesic_hamiltonian(hadamard)
        # h.unitary @ geo.unitary should approximate target
        composed = h.unitary.matrix @ geo.unitary.matrix
        fid = float(Unitary.unitary_fidelity(composed, hadamard))
        assert fid > 0.99


class TestHamiltonianFidelity:
    def test_self_fidelity_one(self, basis_1q):
        h = Hamiltonian(basis_1q, np.array([0.5, 0.3, 0.1]))
        fid = h.fidelity(h.unitary.matrix)
        assert jnp.isclose(fid, 1.0, atol=1e-10)

    def test_identity_vs_other(self, basis_1q, hadamard):
        h = Hamiltonian(basis_1q, np.zeros(3))
        fid = h.fidelity(hadamard)
        assert 0 <= fid <= 1


class TestHamiltonianParametersFromHamiltonian:
    def test_roundtrip_1q(self, basis_1q):
        params_in = np.array([0.5, -0.3, 0.7])
        h = Hamiltonian(basis_1q, params_in)
        params_out = Hamiltonian.parameters_from_hamiltonian(h.matrix, basis_1q)
        assert np.allclose(params_out, params_in, atol=1e-10)

    def test_zero_hamiltonian(self, basis_1q):
        H = np.zeros((2, 2), dtype=complex)
        params = Hamiltonian.parameters_from_hamiltonian(H, basis_1q)
        assert np.allclose(params, 0)

    def test_output_length(self, full_basis_2q):
        H = np.zeros((4, 4), dtype=complex)
        params = Hamiltonian.parameters_from_hamiltonian(H, full_basis_2q)
        assert len(params) == full_basis_2q.lie_algebra_dim


# ===================================================================
# Tests — Unitary
# ===================================================================


class TestUnitary:
    def test_identity(self):
        u = Unitary(np.eye(2, dtype=complex))
        assert u.n == 1
        assert np.allclose(u.matrix, np.eye(2))

    def test_identity_4x4(self):
        u = Unitary(np.eye(4, dtype=complex))
        assert u.n == 2

    def test_hadamard(self, hadamard):
        u = Unitary(hadamard)
        assert u.n == 1

    def test_non_unitary_raises(self):
        with pytest.raises(ValueError, match="unitary"):
            Unitary(np.array([[1, 1], [0, 1]], dtype=complex))

    def test_non_square_raises(self):
        with pytest.raises((ValueError, IndexError)):
            Unitary(np.array([[1, 0, 0], [0, 1, 0]], dtype=complex))


class TestUnitaryFidelity:
    def test_self_fidelity(self, hadamard):
        u = Unitary(hadamard)
        fid = u.fidelity(hadamard)
        assert jnp.isclose(fid, 1.0, atol=1e-10)

    def test_identity_vs_hadamard(self, hadamard):
        u = Unitary(np.eye(2, dtype=complex))
        fid = u.fidelity(hadamard)
        assert 0 <= fid <= 1

    def test_static_fidelity(self, hadamard):
        fid = Unitary.unitary_fidelity(hadamard, hadamard)
        assert jnp.isclose(fid, 1.0, atol=1e-10)

    def test_static_fidelity_symmetry(self, identity_2x2, hadamard):
        f1 = Unitary.unitary_fidelity(identity_2x2, hadamard)
        f2 = Unitary.unitary_fidelity(hadamard, identity_2x2)
        assert jnp.isclose(f1, f2, atol=1e-12)


class TestUnitaryParameters:
    def test_identity_gives_zero_params(self, basis_1q):
        u = Unitary(np.eye(2, dtype=complex))
        params = u.parameters(basis_1q)
        assert np.allclose(params, 0, atol=1e-10)

    def test_roundtrip(self, basis_1q):
        params_in = np.array([0.3, -0.2, 0.5])
        h = Hamiltonian(basis_1q, params_in)
        params_out = h.unitary.parameters(basis_1q)
        assert np.allclose(params_out, params_in, atol=1e-8)

    def test_static_method(self, basis_1q):
        params = Unitary.parameters_from_unitary(np.eye(2, dtype=complex), basis_1q)
        assert np.allclose(params, 0, atol=1e-10)


class TestUnitaryGeodesic:
    def test_self_geodesic_zero(self, basis_1q):
        u = Unitary(np.eye(2, dtype=complex))
        geo = u.geodesic_hamiltonian(basis_1q, np.eye(2, dtype=complex))
        assert isinstance(geo, Hamiltonian)
        assert np.allclose(geo.matrix, 0, atol=1e-10)

    def test_returns_hamiltonian(self, basis_1q, hadamard):
        u = Unitary(np.eye(2, dtype=complex))
        geo = u.geodesic_hamiltonian(basis_1q, hadamard)
        assert isinstance(geo, Hamiltonian)


class TestUnitaryMatmul:
    def test_identity_compose(self):
        u1 = Unitary(np.eye(2, dtype=complex))
        u2 = Unitary(np.eye(2, dtype=complex))
        result = u1 @ u2
        assert isinstance(result, Unitary)
        assert np.allclose(result.matrix, np.eye(2))

    def test_compose_is_unitary(self, hadamard):
        u1 = Unitary(hadamard)
        u2 = Unitary(hadamard)
        result = u1 @ u2
        # H @ H = I
        assert np.allclose(result.matrix, np.eye(2), atol=1e-10)
