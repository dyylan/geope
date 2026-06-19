"""
Tests for geope/engine.py.

Tested items:
  Functions:
    - fidelity
    - get_fidelity_fn
    - compute_matrices_params_list_fn
    - get_compute_matrices_params_list_fn
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geope.engine import (
    fidelity,
    get_fidelity_fn,
    compute_matrices_params_list_fn,
    get_compute_matrices_params_list_fn,
)
from geope.lie import Basis
from geope.utils import (
    construct_full_pauli_basis,
    construct_Heisenberg_pauli_basis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pauli_basis_1q():
    """Single-qubit Pauli basis (X, Y, Z)."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return np.stack([X, Y, Z])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def identity_2x2():
    return jnp.eye(2, dtype=complex)


@pytest.fixture
def identity_4x4():
    return jnp.eye(4, dtype=complex)


@pytest.fixture
def hadamard():
    return jnp.array([[1, 1], [1, -1]], dtype=complex) / jnp.sqrt(2)


@pytest.fixture
def cnot():
    return jnp.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )


@pytest.fixture
def full_basis_2q():
    return construct_full_pauli_basis(2)


@pytest.fixture
def projected_basis_2q():
    return construct_Heisenberg_pauli_basis(2)


# ---------------------------------------------------------------------------
# Tests — fidelity
# ---------------------------------------------------------------------------


class TestFidelity:
    def test_identity_with_itself(self, identity_2x2):
        assert jnp.isclose(fidelity(identity_2x2, identity_2x2), 1.0, atol=1e-12)

    def test_identity_with_itself_4x4(self, identity_4x4):
        assert jnp.isclose(fidelity(identity_4x4, identity_4x4), 1.0, atol=1e-12)

    def test_orthogonal_unitaries(self):
        """X and I are not orthogonal, but fidelity < 1."""
        I = jnp.eye(2, dtype=complex)
        X = jnp.array([[0, 1], [1, 0]], dtype=complex)
        fid = fidelity(X, I)
        assert fid < 1.0

    def test_same_unitary_fidelity_one(self, hadamard):
        assert jnp.isclose(fidelity(hadamard, hadamard), 1.0, atol=1e-12)

    def test_range_0_to_1(self, identity_4x4, cnot):
        fid = fidelity(identity_4x4, cnot)
        assert 0 <= fid <= 1.0

    def test_fidelity_symmetry(self, identity_2x2, hadamard):
        f1 = fidelity(identity_2x2, hadamard)
        f2 = fidelity(hadamard, identity_2x2)
        assert jnp.isclose(f1, f2, atol=1e-12)

    def test_cnot_with_itself(self, cnot):
        assert jnp.isclose(fidelity(cnot, cnot), 1.0, atol=1e-12)

    def test_global_phase_invariance(self, hadamard):
        """Fidelity of U and e^{iθ}U should be 1."""
        phase = jnp.exp(1j * 0.3)
        assert jnp.isclose(fidelity(hadamard, phase * hadamard), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Tests — get_fidelity_fn
# ---------------------------------------------------------------------------


class TestGetFidelityFn:
    def test_returns_callable(self, cnot):
        fn = get_fidelity_fn(cnot)
        assert callable(fn)

    def test_matches_direct_call(self, identity_4x4, cnot):
        fn = get_fidelity_fn(cnot)
        assert jnp.isclose(fn(identity_4x4), fidelity(identity_4x4, cnot), atol=1e-12)

    def test_self_target_gives_one(self, cnot):
        fn = get_fidelity_fn(cnot)
        assert jnp.isclose(fn(cnot), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Tests — compute_matrices_params_list_fn / get_compute_matrices_params_list_fn
# ---------------------------------------------------------------------------


class TestComputeMatricesParamsListFn:
    def test_zero_params_gives_identity(self):
        basis = _pauli_basis_1q()
        params = jnp.zeros((1, 3), dtype=complex)
        U = compute_matrices_params_list_fn(params, basis)
        assert jnp.allclose(U, jnp.eye(2), atol=1e-12)

    def test_output_is_unitary_1q(self):
        basis = _pauli_basis_1q()
        params = jnp.array([[0.3, -0.5, 0.7]], dtype=complex)
        U = compute_matrices_params_list_fn(params, basis)
        assert jnp.allclose(U @ U.conj().T, jnp.eye(2), atol=1e-10)

    def test_output_shape_1q(self):
        basis = _pauli_basis_1q()
        params = jnp.array([[0.1, 0.2, 0.3]], dtype=complex)
        U = compute_matrices_params_list_fn(params, basis)
        assert U.shape == (2, 2)

    def test_multi_gate(self):
        """Two gates composed: U2 @ U1."""
        basis = _pauli_basis_1q()
        params = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=complex)
        U = compute_matrices_params_list_fn(params, basis)
        assert U.shape == (2, 2)
        assert jnp.allclose(U @ U.conj().T, jnp.eye(2), atol=1e-10)

    def test_2q_basis(self, full_basis_2q):
        n = full_basis_2q.lie_algebra_dim
        params = jnp.zeros((1, n), dtype=complex)
        U = compute_matrices_params_list_fn(params, full_basis_2q.basis)
        assert U.shape == (4, 4)
        assert jnp.allclose(U, jnp.eye(4), atol=1e-12)


class TestGetComputeMatricesParamsListFn:
    def test_returns_callable(self):
        basis = _pauli_basis_1q()
        fn = get_compute_matrices_params_list_fn(basis)
        assert callable(fn)

    def test_matches_direct_call(self):
        basis = _pauli_basis_1q()
        fn = get_compute_matrices_params_list_fn(basis)
        params = jnp.array([[0.3, -0.1, 0.5]], dtype=complex)
        U_fn = fn(params)
        U_direct = compute_matrices_params_list_fn(params, basis)
        assert jnp.allclose(U_fn, U_direct, atol=1e-12)
