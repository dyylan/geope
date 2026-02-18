"""
Tests for geope/engine.py.

Tested items:
  Functions:
    - fidelity
    - get_fidelity_fn
    - compute_matrices_params_list_fn
    - get_compute_matrices_params_list_fn
  Classes:
    - Engine
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geope.engine import (
    Engine,
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
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]],
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
        params = jnp.array([[0.1, 0.2, 0.3],
                             [0.4, 0.5, 0.6]], dtype=complex)
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


# ---------------------------------------------------------------------------
# Tests — Engine
# ---------------------------------------------------------------------------

class TestEngine:
    def test_init_stores_bases(self, cnot, full_basis_2q, projected_basis_2q):
        eng = Engine(cnot, full_basis_2q, projected_basis_2q)
        assert eng.full_basis is full_basis_2q
        assert eng.projected_basis is projected_basis_2q
        assert eng.piecewise_steps == 1

    def test_projected_indices_shape(self, cnot, full_basis_2q, projected_basis_2q):
        eng = Engine(cnot, full_basis_2q, projected_basis_2q)
        assert eng.projected_indices.shape == (full_basis_2q.lie_algebra_dim,)
        assert eng.projected_indices.dtype == bool
        assert eng.projected_indices.sum() == projected_basis_2q.lie_algebra_dim

    def test_no_drift_gives_false_indices(self, cnot, full_basis_2q, projected_basis_2q):
        eng = Engine(cnot, full_basis_2q, projected_basis_2q)
        assert not np.any(eng.drift_indices)
        assert eng.drift_basis is None

    def test_proj_drift_indices_no_drift(self, cnot, full_basis_2q, projected_basis_2q):
        eng = Engine(cnot, full_basis_2q, projected_basis_2q)
        assert np.array_equal(eng.proj_drift_indices, eng.projected_indices)

    def test_proj_drift_basis_no_drift(self, cnot, full_basis_2q, projected_basis_2q):
        eng = Engine(cnot, full_basis_2q, projected_basis_2q)
        assert eng.proj_drift_basis.lie_algebra_dim == projected_basis_2q.lie_algebra_dim

    def test_with_drift_basis(self, cnot, full_basis_2q, projected_basis_2q):
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)
        drift = Basis(np.stack([np.kron(Z, I), np.kron(I, Z)]), labels=["ZI", "IZ"])
        eng = Engine(cnot, full_basis_2q, projected_basis_2q, drift_basis=drift)
        assert np.any(eng.drift_indices)
        assert eng.drift_basis is drift

    def test_with_drift_proj_drift_larger(self, cnot, full_basis_2q, projected_basis_2q):
        """proj_drift_basis should include both projected and drift generators."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)
        drift = Basis(np.stack([np.kron(Z, I), np.kron(I, Z)]), labels=["ZI", "IZ"])
        eng = Engine(cnot, full_basis_2q, projected_basis_2q, drift_basis=drift)
        assert eng.proj_drift_basis.lie_algebra_dim >= projected_basis_2q.lie_algebra_dim

    def test_gates_stored(self, cnot, full_basis_2q, projected_basis_2q):
        eng = Engine(cnot, full_basis_2q, projected_basis_2q, piecewise_steps=4)
        assert eng.piecewise_steps == 4

    def test_compute_U_fn_is_callable(self, cnot, full_basis_2q, projected_basis_2q):
        eng = Engine(cnot, full_basis_2q, projected_basis_2q)
        assert callable(eng.compute_U_fn)

    def test_fid_U_fn_is_callable(self, cnot, full_basis_2q, projected_basis_2q):
        eng = Engine(cnot, full_basis_2q, projected_basis_2q)
        assert callable(eng.fid_U_fn)

    def test_compute_U_fn_zero_params(self, cnot, full_basis_2q, projected_basis_2q):
        eng = Engine(cnot, full_basis_2q, projected_basis_2q)
        n = eng.proj_drift_basis.lie_algebra_dim
        params = jnp.zeros((1, n), dtype=complex)
        U = eng.compute_U_fn(params)
        assert jnp.allclose(U, jnp.eye(4), atol=1e-12)

    def test_fid_U_fn_target_self(self, cnot, full_basis_2q, projected_basis_2q):
        eng = Engine(cnot, full_basis_2q, projected_basis_2q)
        assert jnp.isclose(eng.fid_U_fn(cnot), 1.0, atol=1e-12)

    def test_fid_U_fn_identity_less_than_one(self, cnot, full_basis_2q, projected_basis_2q):
        eng = Engine(cnot, full_basis_2q, projected_basis_2q)
        fid = eng.fid_U_fn(jnp.eye(4, dtype=complex))
        assert fid < 1.0

    def test_proj_indices_projdrift_basis(self, cnot, full_basis_2q, projected_basis_2q):
        eng = Engine(cnot, full_basis_2q, projected_basis_2q)
        assert eng.proj_indices_projdrift_basis.dtype == bool

    def test_drift_indices_projdrift_basis(self, cnot, full_basis_2q, projected_basis_2q):
        eng = Engine(cnot, full_basis_2q, projected_basis_2q)
        assert eng.drift_indices_projdrift_basis.dtype == bool
        assert not np.any(eng.drift_indices_projdrift_basis)
