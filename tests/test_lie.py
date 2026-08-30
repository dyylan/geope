"""
Tests for geope/geometry/lie/basis.py.

Tested items:
  Classes:
    - Basis  (properties, linear_span, overlap, verify, apply_interaction_graph,
              apply_interaction_map, generate_parameter_list, generate_bounds,
              _generate_plot_labels, _generate_interaction_labels,
              _generate_interaction_qubits, _generate_interaction_graph,
              _generate_interaction_map, _remove_basis_elements, __len__)
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geope.geometry.lie import Basis
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
