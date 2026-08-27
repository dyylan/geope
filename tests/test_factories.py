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
from geope.jax import su_hessian_quadratic_form
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
# Riemannian-Hessian quadratic form <Omega, K_A Omega>_F
# ---------------------------------------------------------------------------


def _rand_su(d, seed, norm=1.0):
    """A random traceless skew-Hermitian matrix scaled to a given Frobenius norm."""
    k1, k2 = jax.random.split(jax.random.key(seed))
    M = jax.random.normal(k1, (d, d)) + 1j * jax.random.normal(k2, (d, d))
    H = 0.5 * (M + M.conj().T)
    H = H - jnp.trace(H) / d * jnp.eye(d)
    A = 1j * H
    return A * (norm / jnp.linalg.norm(A))


def _dense_hessian_form(A, Omega):
    """Reference ``<Omega, K_A Omega>_F`` via an explicit operator matrix.

    Builds ``L(X) = -i(A X - X A) = ad_{-iA}(X)`` column by column by applying it
    to the elementary matrices, so no kron/vec convention can be got wrong. ``L``
    is Hermitian, so ``eigh`` diagonalises it and its real eigenvalues are the
    phase differences; ``K_A`` has eigenvalue ``h(delta)`` on the same vectors.
    This is an independent path from the eigenbasis trick under test.
    """
    d = A.shape[0]
    A = np.asarray(A)
    L = np.zeros((d * d, d * d), dtype=complex)
    for m in range(d * d):
        E = np.zeros((d, d), dtype=complex)
        E.flat[m] = 1.0
        L[:, m] = (-1j * (A @ E - E @ A)).reshape(-1)
    np.testing.assert_allclose(L, L.conj().T, atol=1e-10)

    lam, V = np.linalg.eigh(L)
    small = np.abs(lam) < 1e-12
    half = 0.5 * np.where(small, 1.0, lam)
    h = np.where(small, 1.0, half / np.tan(half))
    K = V @ np.diag(h) @ V.conj().T
    KOm = (K @ np.asarray(Omega).reshape(-1)).reshape(d, d)
    return float(np.real(np.trace(np.asarray(Omega).conj().T @ KOm)))


def _mu(rho):
    """The strong-convexity modulus mu(rho) = (rho/2) cot(rho/2)."""
    return 1.0 if rho < 1e-12 else float((rho / 2) / np.tan(rho / 2))


class TestSuHessianQuadraticForm:
    @pytest.mark.parametrize("d", [2, 4, 8])
    @pytest.mark.parametrize("norm", [0.3, 1.0, 2.0])
    def test_matches_dense_operator_reference(self, d, norm):
        # The eigenbasis evaluation must agree with an explicitly constructed
        # ad_A operator, for a generic (non-parallel) Omega.
        A = _rand_su(d, 0, norm=norm)
        Omega = _rand_su(d, 100)
        value, _ = su_hessian_quadratic_form(A, Omega)
        assert np.isclose(float(value), _dense_hessian_form(A, Omega), rtol=1e-9)

    @pytest.mark.parametrize("d", [2, 4, 8])
    def test_radial_direction_is_exact(self, d):
        # K_A A = A (radial eigenvalue 1), so the form collapses to ||A||_F^2.
        # This is the identity that makes the ||Omega||^2 surrogate valid under
        # exact tangent matching.
        A = _rand_su(d, 7, norm=1.7)
        value, _ = su_hessian_quadratic_form(A, A)
        A_norm2 = float(jnp.real(jnp.trace(A.conj().T @ A)))
        assert np.isclose(float(value), A_norm2, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("scale", [-2.5, 0.5, 3.0])
    def test_scale_multiple_of_A_matches_surrogate(self, scale):
        # Any scale multiple of A is still radial, so the form equals
        # ||Omega||^2 -- this is why the tangent-matching diagnostic must be
        # scale-invariant.
        A = _rand_su(4, 11, norm=1.1)
        Omega = scale * A
        value, _ = su_hessian_quadratic_form(A, Omega)
        omega_norm2 = float(jnp.real(jnp.trace(Omega.conj().T @ Omega)))
        assert np.isclose(float(value), omega_norm2, rtol=1e-12)

    def test_two_sided_bound_inside_convex_region(self):
        # mu(rho) I <= K_A <= I whenever rho < pi, so the surrogate ||Omega||^2
        # is an upper bound and mu(rho)||Omega||^2 a lower one.
        for seed in range(5):
            A = _rand_su(4, seed, norm=0.8)
            Omega = _rand_su(4, seed + 50)
            value, rho = su_hessian_quadratic_form(A, Omega)
            assert float(rho) < np.pi  # the regime the bound applies to
            omega_norm2 = float(jnp.real(jnp.trace(Omega.conj().T @ Omega)))
            assert float(value) <= omega_norm2 + 1e-9
            assert float(value) >= _mu(float(rho)) * omega_norm2 - 1e-9

    def test_zero_A_is_finite_and_equals_surrogate(self):
        # A = 0 puts every delta on the h(0) branch: the form must be exactly
        # ||Omega||^2 with no nan leaking from tan(0).
        Omega = _rand_su(4, 3)
        value, rho = su_hessian_quadratic_form(
            jnp.zeros((4, 4), dtype=jnp.complex128), Omega
        )
        omega_norm2 = float(jnp.real(jnp.trace(Omega.conj().T @ Omega)))
        assert bool(jnp.isfinite(value))
        assert np.isclose(float(value), omega_norm2, rtol=1e-12)
        assert np.isclose(float(rho), 0.0, atol=1e-12)

    def test_degenerate_spectrum_is_finite(self):
        # Repeated eigenphases give repeated zero deltas off the diagonal too.
        A = 1j * jnp.diag(jnp.array([0.4, 0.4, -0.4, -0.4], dtype=jnp.float64))
        A = A.astype(jnp.complex128)
        Omega = _rand_su(4, 21)
        value, rho = su_hessian_quadratic_form(A, Omega)
        assert bool(jnp.isfinite(value))
        assert np.isclose(float(value), _dense_hessian_form(A, Omega), rtol=1e-9)
        assert np.isclose(float(rho), 0.8, atol=1e-12)

    def test_rho_matches_eigenphase_spread(self):
        A = _rand_su(4, 13, norm=2.0)
        _, rho = su_hessian_quadratic_form(A, _rand_su(4, 14))
        theta = np.asarray(jnp.linalg.eigvalsh(-1j * A))
        assert np.isclose(float(rho), float(theta.max() - theta.min()), atol=1e-12)

    def test_jittable_and_differentiable(self):
        # Traced under jit, and the gradient in the Omega scale is finite (the
        # form is quadratic, so d/ds at s=1 must be 2x the value).
        A = _rand_su(4, 15, norm=0.5)
        Omega = _rand_su(4, 16)
        value, _ = jax.jit(su_hessian_quadratic_form)(A, Omega)
        grad = jax.grad(lambda s: su_hessian_quadratic_form(A, s * Omega)[0])(1.0)
        assert bool(jnp.isfinite(value))
        assert np.isclose(float(grad), 2.0 * float(value), rtol=1e-9)


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
# :func:`geope.engine.get_gammas_and_omegas_fn` for the worked argument.
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
        # XZ/ZX are outside the Heisenberg basis, keeping control and drift
        # disjoint as Parameters now requires.
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        drift = Basis(np.stack([np.kron(X, Z), np.kron(Z, X)]), labels=["XZ", "ZX"])
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
# Control/drift overlap guard (issue #26)
#
# A basis element shared by the control and drift bases used to have its
# control coefficient silently overwritten by the drift value, because every
# write path assigns the drift columns after the control columns. Parameters
# now rejects the configuration outright.
# ---------------------------------------------------------------------------


def _overlapping_drift_basis():
    """Drift basis on ZI/IZ — both inside the Heisenberg projected basis."""
    from geope.lie import Basis

    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    return Basis(np.stack([np.kron(Z, I), np.kron(I, Z)]), labels=["ZI", "IZ"])


def _disjoint_drift_basis():
    """Drift basis on XZ/ZX — outside the Heisenberg projected basis."""
    from geope.lie import Basis

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return Basis(np.stack([np.kron(X, Z), np.kron(Z, X)]), labels=["XZ", "ZX"])


class TestControlDriftOverlapGuard:
    def test_overlapping_bases_raise(self):
        fb = construct_full_pauli_basis(2)
        pb = construct_Heisenberg_pauli_basis(2)
        with pytest.raises(ValueError) as excinfo:
            Parameters(
                basis=fb,
                projected_basis=pb,
                drift_basis=_overlapping_drift_basis(),
                target=CNOT,
            )
        msg = str(excinfo.value)
        # The message names the offending elements and points at the fix.
        assert "ZI" in msg and "IZ" in msg
        assert "param_transform" in msg

    def test_overlapping_bases_raise_under_param_transform(self):
        # The dead-gradient case from issue #26: the guard is unconditional,
        # so it fires in experimental space too.
        fb = construct_full_pauli_basis(2)
        pb = construct_Heisenberg_pauli_basis(2)
        with pytest.raises(ValueError, match="must be disjoint"):
            Parameters(
                basis=fb,
                projected_basis=pb,
                drift_basis=_overlapping_drift_basis(),
                target=CNOT,
                param_transform=lambda x: x,
                n_experimental_params=pb.lie_algebra_dim,
            )

    def test_disjoint_bases_are_accepted(self):
        fb = construct_full_pauli_basis(2)
        pb = construct_Heisenberg_pauli_basis(2)
        p = Parameters(
            basis=fb,
            projected_basis=pb,
            drift_basis=_disjoint_drift_basis(),
            target=CNOT,
        )
        assert not np.any(p.projected_indices & p.drift_indices)

    def test_no_drift_basis_is_accepted(self):
        fb = construct_full_pauli_basis(2)
        pb = construct_Heisenberg_pauli_basis(2)
        p = Parameters(basis=fb, projected_basis=pb, target=CNOT)
        assert not np.any(p.drift_indices)


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
