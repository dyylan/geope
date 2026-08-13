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
from geope.geope import linear_comb_projected_coeffs_multigate
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
        omegas = get_omegas_fn(
            p.compute_U_fn, p.jac_fn, p.project_omegas_fn, proj_indices, has_pd
        )
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
        omegas = get_omegas_fn(
            p.compute_U_fn, p.jac_fn, p.project_omegas_fn, proj_indices, has_pd
        )
        out = np.array(omegas(free))
        # (piecewise_steps, n_projected, full_basis_dim)
        assert out.shape[0] == 2
        assert out.shape[1] == int(np.sum(proj_indices))


# ---------------------------------------------------------------------------
# Gammas / Omegas — the left-trivialisation invariant
#
# Both outputs must be projected *after* multiplying by U^dagger. The Pauli basis
# is Hermitian, so ``project_omegas`` keeps only the traceless-Hermitian part of
# its argument; the raw quantities are U * (traceless Hermitian), so projecting
# them directly is lossy in a U-dependent way and the downstream least squares
# stops being the Frobenius-orthogonal projection it is defined to be. See
# ``examples/left_trivialisation.py`` for the worked argument.
#
# Two things make these tests bite, and both are load-bearing:
#   * the Jacobian must be rank-deficient (8 dof against dim su(4) = 15). When it
#     is surjective the ambient solve happens to return the same direction, which
#     is why the benchmark suite never caught this.
#   * the iterate must be away from U = 1, where the ambient projection is very
#     nearly lossless and the two agree regardless.
# ---------------------------------------------------------------------------


class TestLeftTrivialisation:
    @pytest.fixture
    def deficient(self):
        p = Parameters(
            basis=construct_full_pauli_basis(2),
            control={1: ["x", "z"], 2: ["x", "z"]},
            drift={(1, 2): ["zz"]},
            drift_values={(1, 2): {"zz": 1.0}},
            target=CNOT,
            piecewise_steps=2,
            seed=0,
        )
        pidx = np.asarray(p.proj_indices_projdrift_basis)
        K = p.proj_drift_basis.lie_algebra_dim
        # Deliberately away from the identity: at U = 1 the bug is invisible.
        free = jnp.asarray(
            np.random.default_rng(7).normal(size=(p.piecewise_steps, K)) * 0.8
        ).astype(jnp.complex128)
        assert p.piecewise_steps * int(pidx.sum()) < p.basis.lie_algebra_dim
        return p, pidx, free

    @staticmethod
    def _left_trivialised(p, pidx, free, key):
        """The matrices the projection *should* be handed: A and i U^dag dU."""
        U = np.asarray(p.compute_U_fn(free))
        A = U.conj().T @ np.asarray(p.geo_fn(U, key=key))
        dU = np.transpose(np.asarray(p.jac_fn(free)), [2, 3, 0, 1])
        J = 1j * np.einsum("ab,gkbc->gkac", U.conj().T, dU)[:, pidx]
        return U, A, J

    def test_projections_are_of_the_left_trivialised_matrices(self, deficient):
        """gammas and omegas must be the projections of A and i U^dag dU.

        Not a Hermiticity check on the rebuilt matrices: ``project_omegas``
        returns *real* coefficients against a Hermitian basis, so any rebuild is
        Hermitian whatever it was handed. The invariant is that the coefficients
        are those of the left-trivialised matrices.
        """
        p, pidx, free = deficient
        key = jax.random.key(5)
        _, A, J = self._left_trivialised(p, pidx, free, key)
        gammas, omegas = p.gammas_and_omegas(free, key)
        d = p.proj_drift_basis.dim

        np.testing.assert_allclose(
            np.asarray(gammas),
            np.asarray(p.project_omegas_fn(A[None])).squeeze(0) / d,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(omegas),
            np.asarray(p.project_omegas_fn(J.reshape(-1, d, d))).reshape(
                J.shape[0], J.shape[1], -1
            ),
            atol=1e-12,
        )

    def test_search_direction_is_the_frobenius_projection(self, deficient):
        """Psi = P Omega: the solve must be the Frobenius-orthogonal projection.

        The geodesic step is *defined* as the projection of the geodesic tangent
        onto the reachable subspace, and the second-order line searches
        (``QuadraticArmijo``) leans on when it substitutes ``||Omega||^2`` for
        the intrinsic curvature term.
        """
        p, pidx, free = deficient
        key = jax.random.key(5)
        d = p.proj_drift_basis.dim
        _, A, J = self._left_trivialised(p, pidx, free, key)

        # Independent reference: least squares of A onto span(J) in matrix space,
        # under the real inner product Re tr(X^dagger Y).
        B = J.reshape(-1, d * d).T
        Br = np.concatenate([B.real, B.imag], axis=0)
        ar = np.concatenate([A.real.ravel(), A.imag.ravel()])
        r = Br @ np.linalg.lstsq(Br, ar, rcond=None)[0]
        PA = (r[: d * d] + 1j * r[d * d :]).reshape(d, d)
        # The fixture must be genuinely deficient, or this asserts nothing.
        assert np.linalg.norm(A - PA) / np.linalg.norm(A) > 1e-2

        gammas, omegas = p.gammas_and_omegas(free, key)
        sol = np.asarray(linear_comb_projected_coeffs_multigate(omegas, gammas, None))
        psi = np.einsum("gk,gkab->ab", sol, J)

        # Compare directions: gammas carry a 1/d and Geope renormalises anyway.
        cos = float(np.real(np.trace(PA.conj().T @ psi))) / (
            np.linalg.norm(PA) * np.linalg.norm(psi)
        )
        assert abs(cos - 1.0) < 1e-10, f"solve is not the projection: cos = {cos:.10f}"


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
