"""
Tests for the standalone function factories in geope/engine.py (issue #13).

These exercise the Jacobian, Hessian, gammas and omegas builders directly —
with no Engine or optimiser object — demonstrating that the individual
components are now independently testable and benchmarkable, and verifying them
against finite differences / ``jax`` references rather than against each other.
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geope.engine import (
    get_compute_matrices_params_list_fn,
    get_infidelity_fn,
    get_geodesic_hamiltonian_fn,
    get_jacobian_fn,
    get_gammas_fn,
    get_omegas_fn,
    get_gammas_and_omegas_fn,
    get_hessian_fn,
)
from geope.lie.pauli_projector import get_project_omegas_fn
from geope.parameters import Parameters
from geope.utils import (
    construct_full_pauli_basis,
    construct_Heisenberg_pauli_basis,
)


def _pauli_basis_1q():
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return np.stack([X, Y, Z])


CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


# ---------------------------------------------------------------------------
# Jacobian factory
# ---------------------------------------------------------------------------


class TestJacobianFactory:
    def test_matches_finite_difference_1q(self):
        basis = _pauli_basis_1q()
        compute_U = get_compute_matrices_params_list_fn(basis)
        jac = get_jacobian_fn(compute_U)
        x = jnp.array([[0.3, -0.2, 0.5]], dtype=complex)
        J = np.array(jac(x))  # (d, d, G, K)
        eps = 1e-6
        for k in range(3):
            dx = np.zeros((1, 3), dtype=complex)
            dx[0, k] = eps
            fd = (np.array(compute_U(x + dx)) - np.array(compute_U(x - dx))) / (2 * eps)
            np.testing.assert_allclose(J[:, :, 0, k], fd, atol=1e-5)

    def test_multi_gate_shape(self):
        basis = _pauli_basis_1q()
        compute_U = get_compute_matrices_params_list_fn(basis)
        jac = get_jacobian_fn(compute_U)
        x = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=complex)
        J = np.array(jac(x))
        assert J.shape == (2, 2, 2, 3)  # (d, d, G, K)


# ---------------------------------------------------------------------------
# Hessian factory
# ---------------------------------------------------------------------------


class TestHessianFactory:
    def test_matches_jax_hessian_quadratic(self):
        # f(y) = 0.5 yᵀ A y  ->  Hessian = A (symmetrised)
        A = jnp.array([[2.0, 0.5], [0.5, 3.0]])
        f = lambda y: 0.5 * jnp.vdot(y.reshape(-1), (A @ y.reshape(-1))).real
        hess = get_hessian_fn(f)
        y = jnp.array([0.7, -0.3])
        H = np.array(hess(y)).reshape(2, 2)
        np.testing.assert_allclose(H, np.array(A), atol=1e-8)

    def test_matches_jax_hessian_infidelity(self):
        basis = _pauli_basis_1q()
        compute_U = get_compute_matrices_params_list_fn(basis)
        target = jnp.array([[0, 1], [1, 0]], dtype=complex)  # X gate
        infid_U = get_infidelity_fn(target)
        infid = lambda x: infid_U(compute_U(x))
        hess = get_hessian_fn(infid)
        y = jnp.array([[0.2, -0.1, 0.4]])
        H = np.array(hess(y)).reshape(y.size, y.size)
        H_ref = np.array(jax.hessian(infid)(y)).reshape(y.size, y.size)
        np.testing.assert_allclose(H, H_ref, atol=1e-8)


# ---------------------------------------------------------------------------
# Gammas / Omegas factories — split halves match the combined function
# ---------------------------------------------------------------------------


class TestGammasOmegas:
    @pytest.fixture
    def pieces(self):
        # Source the (un-jitted) building blocks the way the optimiser does:
        # straight off a Parameters object — no engine involved.
        p = Parameters(
            basis=construct_full_pauli_basis(2),
            projected_basis=construct_Heisenberg_pauli_basis(2),
            target=CNOT,
            piecewise_steps=2,
            seed=0,
        )
        proj_indices = p.proj_indices_projdrift_basis
        has_pd = p.proj_drift_basis.lie_algebra_dim > 0
        K = p.proj_drift_basis.lie_algebra_dim
        free = jax.random.normal(jax.random.key(1), (2, K)).astype(jnp.complex128)
        return p, proj_indices, has_pd, free

    def test_split_matches_combined(self, pieces):
        p, proj_indices, has_pd, free = pieces
        key = jax.random.key(5)
        gammas = get_gammas_fn(p.compute_U_fn, p.geo_fn, p.project_omegas_fn)
        omegas = get_omegas_fn(p.jac_fn, p.project_omegas_fn, proj_indices, has_pd)
        combined = get_gammas_and_omegas_fn(
            p.compute_U_fn,
            p.jac_fn,
            p.geo_fn,
            p.project_omegas_fn,
            proj_indices,
            has_pd,
        )
        g_c, o_c = combined(free, key)
        np.testing.assert_allclose(
            np.array(gammas(free, key)), np.array(g_c), atol=1e-10
        )
        np.testing.assert_allclose(np.array(omegas(free)), np.array(o_c), atol=1e-10)

    def test_omega_restriction_shape(self, pieces):
        p, proj_indices, has_pd, free = pieces
        omegas = get_omegas_fn(p.jac_fn, p.project_omegas_fn, proj_indices, has_pd)
        out = np.array(omegas(free))
        # (piecewise_steps, n_projected, full_basis_dim)
        assert out.shape[0] == 2
        assert out.shape[1] == int(np.sum(proj_indices))


# ---------------------------------------------------------------------------
# Parameters-derived metadata (the algebraic index masks)
# ---------------------------------------------------------------------------


class TestParametersMetadata:
    def test_no_drift_masks(self):
        fb = construct_full_pauli_basis(2)
        pb = construct_Heisenberg_pauli_basis(2)
        p = Parameters(basis=fb, projected_basis=pb, target=CNOT)
        assert p.projected_indices.dtype == bool
        assert p.projected_indices.shape == (fb.lie_algebra_dim,)
        assert p.projected_indices.sum() == pb.lie_algebra_dim
        assert not np.any(p.drift_indices)
        assert np.array_equal(p.proj_drift_indices, p.projected_indices)
        assert p.proj_drift_basis.lie_algebra_dim == pb.lie_algebra_dim
        assert not np.any(p.drift_indices_projdrift_basis)

    def test_with_drift_masks(self):
        from geope.lie import Basis

        fb = construct_full_pauli_basis(2)
        pb = construct_Heisenberg_pauli_basis(2)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)
        drift = Basis(np.stack([np.kron(Z, I), np.kron(I, Z)]), labels=["ZI", "IZ"])
        p = Parameters(basis=fb, projected_basis=pb, drift_basis=drift, target=CNOT)
        assert np.any(p.drift_indices)
        assert p.proj_drift_basis.lie_algebra_dim >= pb.lie_algebra_dim
        # The within-combined-basis masks have the combined length, and the
        # drift mask marks exactly the drift generators.
        n_pd = p.proj_drift_basis.lie_algebra_dim
        assert p.proj_indices_projdrift_basis.shape == (n_pd,)
        assert p.drift_indices_projdrift_basis.shape == (n_pd,)
        assert p.drift_indices_projdrift_basis.sum() == int(p.drift_indices.sum())


# ---------------------------------------------------------------------------
# Lazy build / caching: functions are not built until accessed, then memoised
# ---------------------------------------------------------------------------


class TestLazyCaching:
    def test_functions_cached_on_params(self):
        p = Parameters(
            basis=construct_full_pauli_basis(2),
            projected_basis=construct_Heisenberg_pauli_basis(2),
            target=CNOT,
        )
        assert p.compute_U_fn is p.compute_U_fn  # cached (same object)
        assert p.gammas_and_omegas is p.gammas_and_omegas

    def test_geodesic_self_is_zero(self):
        p = Parameters(
            basis=construct_full_pauli_basis(2),
            projected_basis=construct_Heisenberg_pauli_basis(2),
            target=CNOT,
        )
        g = p.geo_fn(jnp.array(CNOT), key=jax.random.key(0))
        assert np.allclose(np.array(g), 0, atol=1e-10)
