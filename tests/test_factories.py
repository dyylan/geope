"""
Tests for the standalone function factories in geope/engine.py (issue #13).

These exercise the Jacobian and Hessian builders directly — with no optimiser
object — demonstrating that the individual components are independently testable
and benchmarkable, and verifying them against finite differences / ``jax``
references rather than against each other.

The per-step geometry the GEOPE step is built from (gammas, omegas and the
left-trivialisation invariant they must satisfy) is assembled by
``geope.geometry``, so those tests read it off a
``Manifold.context``; the manifold and tangent-space primitives themselves are
covered in tests/test_geometry.py.
"""

import pytest
import numpy as np
import scipy.linalg as spla

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geope.geometry.chart import (
    get_compute_matrices_params_list_fn,
    get_jacobian_fn,
)
from geope.geometry import StateSphere
from geope.geometry.lie import groups
from geope.geometry.lie.basis import get_project_omegas_fn
from geope.geometry.lie.groups import infidelity
from geope.jax.hessian import get_hessian_fn
from geope.geope import linear_comb_projected_coeffs_multigate
from geope.jax import stiefel_hessian_quadratic_form, su_hessian_quadratic_form
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
        compute_point = get_compute_matrices_params_list_fn(basis)
        jac = get_jacobian_fn(compute_point)
        x = jnp.array([[0.3, -0.2, 0.5]], dtype=complex)
        J = np.array(jac(x))  # (d, d, G, K)
        eps = 1e-6
        for k in range(3):
            dx = np.zeros((1, 3), dtype=complex)
            dx[0, k] = eps
            fd = (np.array(compute_point(x + dx)) - np.array(compute_point(x - dx))) / (
                2 * eps
            )
            np.testing.assert_allclose(J[:, :, 0, k], fd, atol=1e-5)

    def test_multi_gate_shape(self):
        basis = _pauli_basis_1q()
        compute_point = get_compute_matrices_params_list_fn(basis)
        jac = get_jacobian_fn(compute_point)
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
        compute_point = get_compute_matrices_params_list_fn(basis)
        target = jnp.array([[0, 1], [1, 0]], dtype=complex)  # X gate
        infid = lambda x: infidelity(compute_point(x), target)
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
# Canonical-Stiefel Riemannian-Hessian quadratic form
# ---------------------------------------------------------------------------


def _rand_stiefel(n, m, seed, norm=0.6):
    """A random frame in St_m(C^n) with a tangent at it, scaled to ``norm``."""
    rng = np.random.default_rng(seed)
    q = np.linalg.qr(rng.normal(size=(n, m)) + 1j * rng.normal(size=(n, m)))[0]
    weight = np.eye(n) - 0.5 * q @ q.conj().T

    def tangent(scale):
        z = rng.normal(size=(n, m)) + 1j * rng.normal(size=(n, m))
        overlap = q.conj().T @ z
        t = z - q @ (0.5 * (overlap + overlap.conj().T))
        return scale * t / np.sqrt(np.real(np.trace(t.conj().T @ weight @ t)))

    return q, tangent(norm), tangent(1.0)


def _dense_stiefel_hessian_form(q, delta, xi):
    r"""Reference form via the *un-reduced* block exponential on all of T_Q.

    Builds L_S and M_S column by column on the full ``2Nm - m^2``-dimensional
    horizontal space, using a complete orthonormal complement ``Q_perp`` with no
    regard for where ``B``'s support lies, then takes ``E_12^{-1} E_11`` on that
    whole space. It therefore validates the **two-sector split** itself — not
    merely the arithmetic inside each sector — since the implementation under
    test never forms an operator of this size and never materialises ``Q_2``.
    """
    n, m = q.shape
    q_perp = np.linalg.qr(np.concatenate([q, delta], axis=1), mode="complete")[0][:, m:]
    a = q.conj().T @ delta
    b = q_perp.conj().T @ delta
    c = q.conj().T @ xi
    d = q_perp.conj().T @ xi
    k = n - m

    # A real basis of {(C, D) : C skew-Hermitian m x m, D in C^{k x m}}.
    basis = []
    for j in range(m):
        for i in range(m):
            e = np.zeros((m, m), complex)
            if i == j:
                e[i, i] = 1j
            elif i < j:
                e[i, j], e[j, i] = 1.0, -1.0
            else:
                e[i, j] = e[j, i] = 1j
            basis.append((e, np.zeros((k, m), complex)))
    for i in range(k):
        for j in range(m):
            for unit in (1.0, 1j):
                blk = np.zeros((k, m), complex)
                blk[i, j] = unit
                basis.append((np.zeros((m, m), complex), blk))
    basis = [(c_, d_) for c_, d_ in basis if np.any(c_) or np.any(d_)]
    dim = len(basis)

    def inner(x, y):
        return 0.5 * np.real(np.trace(x[0].conj().T @ y[0])) + np.real(
            np.trace(x[1].conj().T @ y[1])
        )

    # Gram-Schmidt, so the coordinate dot product *is* the canonical metric.
    frame = []
    for vec in basis:
        for prev in frame:
            proj = inner(prev, vec)
            vec = (vec[0] - proj * prev[0], vec[1] - proj * prev[1])
        norm = np.sqrt(inner(vec, vec))
        if norm > 1e-9:
            frame.append((vec[0] / norm, vec[1] / norm))
    dim = len(frame)
    assert dim == 2 * n * m - m * m

    def l_op(cc, dd):
        return (a @ cc - cc @ a - b.conj().T @ dd + dd.conj().T @ b, b @ cc - dd @ a)

    def m_op(cc, dd):
        f = dd @ b.conj().T - b @ dd.conj().T
        return (np.zeros((m, m), complex), -f @ b)

    def coords(vec):
        return np.array([inner(e, vec) for e in frame])

    l_mat = np.column_stack([coords(l_op(*e)) for e in frame])
    m_mat = np.column_stack([coords(m_op(*e)) for e in frame])
    block = np.block(
        [[np.zeros((dim, dim)), np.eye(dim)], [m_mat, -l_mat]],
    )
    exp = spla.expm(block)
    e11, e12 = exp[:dim, :dim], exp[:dim, dim:]
    z = coords((c, d))
    return float(z @ np.linalg.solve(e12, e11 @ z))


class TestStiefelHessianQuadraticForm:
    @pytest.mark.parametrize(
        "n, m",
        [(6, 2), (4, 2), (5, 3), (6, 3), (5, 1), (4, 4)],
        ids="6x2 4x2 5x3 6x3 5x1 4x4".split(),
    )
    @pytest.mark.parametrize("norm", [0.3, 0.9])
    def test_matches_the_unreduced_operator_reference(self, n, m, norm):
        # The reduced two-sector evaluation must agree with the full-space block
        # exponential the note prescribes, across every regime of p = min(m, N-m).
        q, delta, xi = _rand_stiefel(n, m, seed=n * 100 + m, norm=norm)
        value, _ = stiefel_hessian_quadratic_form(
            jnp.asarray(q), jnp.asarray(delta), jnp.asarray(xi)
        )
        assert np.isclose(
            float(value), _dense_stiefel_hessian_form(q, delta, xi), rtol=1e-8
        )

    @pytest.mark.parametrize("n, m", [(6, 2), (5, 3), (4, 4)])
    def test_radial_direction_is_exact(self, n, m):
        # K_S S = S: the Jacobi field with z(0) = S, z(1) = 0 is just (1-t)S, so
        # the form collapses to ||Delta||_Q^2 with no operator error at all.
        q, delta, _ = _rand_stiefel(n, m, seed=7 * n + m, norm=0.8)
        q_j, d_j = jnp.asarray(q), jnp.asarray(delta)
        value, _ = stiefel_hessian_quadratic_form(q_j, d_j, d_j)
        weight = np.eye(n) - 0.5 * q @ q.conj().T
        norm2 = float(np.real(np.trace(delta.conj().T @ weight @ delta)))
        assert np.isclose(float(value), norm2, rtol=1e-11)

    def test_zero_delta_is_finite_and_equals_the_surrogate(self):
        # Delta = 0 gives L = M = 0, so K = I exactly and no tan/solve branch may
        # leak a nan into either sector.
        q, _, xi = _rand_stiefel(6, 2, seed=3)
        zero = jnp.zeros((6, 2), dtype=jnp.complex128)
        value, spread = stiefel_hessian_quadratic_form(
            jnp.asarray(q), zero, jnp.asarray(xi)
        )
        weight = np.eye(6) - 0.5 * q @ q.conj().T
        norm2 = float(np.real(np.trace(xi.conj().T @ weight @ xi)))
        assert bool(jnp.isfinite(value))
        assert np.isclose(float(value), norm2, rtol=1e-11)
        assert np.isclose(float(spread), 0.0, atol=1e-12)

    def test_rho_matches_the_lift_eigenphase_spread(self):
        q, delta, xi = _rand_stiefel(6, 2, seed=21, norm=1.1)
        _, rho = stiefel_hessian_quadratic_form(
            jnp.asarray(q), jnp.asarray(delta), jnp.asarray(xi)
        )
        # The full N x N lift, including the ambient zeros the reduction drops.
        q_perp = np.linalg.qr(np.concatenate([q, delta], axis=1), mode="complete")[0][
            :, 2:
        ]
        a, b = q.conj().T @ delta, q_perp.conj().T @ delta
        lift = np.block([[a, -b.conj().T], [b, np.zeros((4, 4), complex)]])
        phases = np.linalg.eigvalsh(-1j * lift)
        assert np.isclose(float(rho), float(phases.max() - phases.min()), atol=1e-10)

    def test_survives_a_second_independent_trace(self):
        """The skew basis depends only on ``m``, so it is cached — but the cache
        must hold *numpy*. Memoising a `jax.Array` built inside the first trace
        leaks it into every later one, and the second `jit` at the same frame
        size dies with `UnexpectedTracerError`. One shape, two traces is the
        only arrangement that catches it.
        """
        q, delta, xi = _rand_stiefel(6, 2, seed=44, norm=0.5)
        args = (jnp.asarray(q), jnp.asarray(delta), jnp.asarray(xi))
        first = jax.jit(lambda *a: stiefel_hessian_quadratic_form(*a)[0])(*args)
        # A distinct jit: same shapes, same m, a separately traced function.
        second = jax.jit(lambda *a: 1.0 * stiefel_hessian_quadratic_form(*a)[0])(*args)
        assert np.isclose(float(first), float(second), rtol=1e-12)

    def test_jittable_and_differentiable(self):
        q, delta, xi = _rand_stiefel(6, 2, seed=15, norm=0.5)
        q_j, d_j, x_j = jnp.asarray(q), jnp.asarray(delta), jnp.asarray(xi)
        value, _ = jax.jit(stiefel_hessian_quadratic_form)(q_j, d_j, x_j)
        grad = jax.grad(lambda s: stiefel_hessian_quadratic_form(q_j, d_j, s * x_j)[0])(
            1.0
        )
        assert bool(jnp.isfinite(value))
        # The form is quadratic in Omega, so d/ds at s = 1 is twice the value.
        assert np.isclose(float(grad), 2.0 * float(value), rtol=1e-9)


# ---------------------------------------------------------------------------
# Gammas / Omegas — shapes and laziness, read off a GeometricContext
# ---------------------------------------------------------------------------


class TestGammasOmegas:
    @pytest.fixture
    def pieces(self):
        # Source the geometry the way the optimiser does: one context off the
        # Parameters' bound manifold.
        p = Parameters(
            basis=construct_full_pauli_basis(2),
            projected_basis=construct_Heisenberg_pauli_basis(2),
            target=CNOT,
            piecewise_steps=2,
            seed=0,
        )
        K = p.proj_drift_basis.lie_algebra_dim
        free = jax.random.normal(jax.random.key(1), (2, K)).astype(jnp.complex128)
        return p, free

    def test_omega_restriction_shape(self, pieces):
        p, free = pieces
        out = np.array(p.manifold.context(free).omegas)
        # (piecewise_steps, n_projected, full_basis_dim)
        assert out.shape[0] == 2
        assert out.shape[1] == int(np.sum(p.proj_indices_projdrift_basis))
        assert out.shape[2] == p.basis.lie_algebra_dim

    def test_gammas_shape(self, pieces):
        p, free = pieces
        assert np.array(p.manifold.context(free).gammas).shape == (
            p.basis.lie_algebra_dim,
        )

    def test_both_come_from_one_propagator_and_one_jacobian(self, pieces):
        # The combined evaluation the old ``gammas_and_omegas`` existed to
        # provide is now just tier-0 memoisation: gammas and omegas share the
        # same ``unitary``, and omegas is the only reader of the Jacobian.
        p, free = pieces
        ctx = p.manifold.context(free)
        _ = ctx.gammas
        point_after_gammas = ctx.point
        _ = ctx.omegas
        assert ctx.point is point_after_gammas


# ---------------------------------------------------------------------------
# Gammas / Omegas — the left-trivialisation invariant
#
# Both outputs must be projected *after* multiplying by U^dagger. The Pauli basis
# is Hermitian, so ``project_omegas`` keeps only the traceless-Hermitian part of
# its argument; the raw quantities are U * (traceless Hermitian), so projecting
# them directly is lossy in a U-dependent way and the downstream least squares
# stops being the Frobenius-orthogonal projection it is defined to be. See
# the geodesic step's coefficient contract for the worked argument.
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
    def _left_trivialised(p, pidx, free):
        """The matrices the projection *should* be handed: A and i U^dag dU.

        Rebuilt here from the manifold's own primitives, independently of the
        context that is under test.
        """
        m = p.manifold
        U = np.asarray(m.compute_point(free))
        # The context's convention: the tangent at U, pointing away from the
        # target, so that the slope of the distance objective is positive.
        A = -np.asarray(m.log(U, m.target))
        dU = np.transpose(np.asarray(m.tangent.jacobian(free)), [2, 3, 0, 1])
        J = 1j * np.einsum("ab,gkbc->gkac", U.conj().T, dU)[:, pidx]
        return U, A, J

    def test_projections_are_of_the_left_trivialised_matrices(self, deficient):
        """gammas and omegas must be the projections of iA and i U^dag dU.

        Not a Hermiticity check on the rebuilt matrices: the projector returns
        *real* coefficients against a Hermitian basis, so any rebuild is
        Hermitian whatever it was handed. The invariant is that the coefficients
        are those of the left-trivialised matrices.
        """
        p, pidx, free = deficient
        _, A, J = self._left_trivialised(p, pidx, free)
        ctx = p.manifold.context(free)
        d = p.proj_drift_basis.dim
        # Built here rather than read off the bundle, so the test pins the
        # coefficient map and not where the projector happens to live.
        project = get_project_omegas_fn(p.basis)

        # gammas and omegas go through the same coefficient map, with no extra
        # normalisation on either side.
        np.testing.assert_allclose(
            np.asarray(ctx.gammas),
            np.asarray(project(1j * A[None])).squeeze(0),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(ctx.omegas),
            np.asarray(project(J.reshape(-1, d, d))).reshape(
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
        d = p.proj_drift_basis.dim
        _, A, J = self._left_trivialised(p, pidx, free)
        # Both sides Hermitian, as the projection sees them: J is already
        # i U^dag dU, so the geodesic tangent enters as iA (see
        # ``TangentSpace.coefficients``).
        Ah = 1j * A

        # Independent reference: least squares of iA onto span(J) in matrix
        # space, under the real inner product Re tr(X^dagger Y).
        B = J.reshape(-1, d * d).T
        Br = np.concatenate([B.real, B.imag], axis=0)
        ar = np.concatenate([Ah.real.ravel(), Ah.imag.ravel()])
        r = Br @ np.linalg.lstsq(Br, ar, rcond=None)[0]
        PA = (r[: d * d] + 1j * r[d * d :]).reshape(d, d)
        # The fixture must be genuinely deficient, or this asserts nothing.
        assert np.linalg.norm(Ah - PA) / np.linalg.norm(Ah) > 1e-2

        ctx = p.manifold.context(free)
        sol = np.asarray(
            linear_comb_projected_coeffs_multigate(ctx.omegas, ctx.gammas, None)
        )
        psi = np.einsum("gk,gkab->ab", sol, J)

        # Compare directions only: Geope renormalises the solution anyway.
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
        from geope.geometry.lie import Basis

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
    from geope.geometry.lie import Basis

    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    return Basis(np.stack([np.kron(Z, I), np.kron(I, Z)]), labels=["ZI", "IZ"])


def _disjoint_drift_basis():
    """Drift basis on XZ/ZX — outside the Heisenberg projected basis."""
    from geope.geometry.lie import Basis

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return Basis(np.stack([np.kron(X, Z), np.kron(Z, X)]), labels=["XZ", "ZX"])


class TestTargetlessParameters:
    """`target=None` is a supported signature: the chart binds, the target does not."""

    def test_a_targetless_parameters_still_constructs(self):
        p = Parameters(
            basis=construct_full_pauli_basis(2),
            projected_basis=construct_Heisenberg_pauli_basis(2),
            piecewise_steps=2,
        )
        assert not p.manifold.is_bound
        # the chart and its differentials are still there and still work
        assert p.manifold.compute_point(p.free()).shape == (4, 4)
        assert p.manifold.tangent.jacobian(p.free()) is not None
        # ... but nothing that needs a target is
        with pytest.raises(ValueError, match="needs a bound manifold"):
            p.manifold.fidelity_at(p.free())


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
# The binding is built once, in __init__, and shared by everything downstream
# ---------------------------------------------------------------------------


class TestBindingIsBuiltOnce:
    def test_functions_cached_on_params(self):
        p = Parameters(
            basis=construct_full_pauli_basis(2),
            projected_basis=construct_Heisenberg_pauli_basis(2),
            target=CNOT,
        )
        assert p.manifold is p.manifold  # cached (same object)
        assert p.manifold.compute_point is p.manifold.compute_point
        assert p.manifold.hessian is p.manifold.hessian

    def test_geodesic_self_is_zero(self):
        p = Parameters(
            basis=construct_full_pauli_basis(2),
            projected_basis=construct_Heisenberg_pauli_basis(2),
            target=CNOT,
        )
        m = p.manifold
        assert np.allclose(np.array(m.log(m.target, jnp.array(CNOT))), 0, atol=1e-10)


class TestNoProjectorForFrameFreeManifolds:
    """A manifold that coordinatises without an ambient frame builds no projector.

    Before the frame became `TangentBundle` data, `Parameters` built a Pauli
    projector for *every* manifold and handed it over, including to the two
    Stiefel manifolds whose `coefficients` is a real/imaginary split and never
    reads it. Above 5 qubits that was not free: `get_project_omegas_fn_otf`
    eagerly materialises all $4^n-1$ Pauli index rows, ~40 MB at $n = 10$.
    """

    @staticmethod
    def _spy(monkeypatch):
        calls = []
        for name in ("get_project_omegas_fn", "get_project_omegas_fn_otf"):
            real = getattr(groups, name)

            def spy(basis, *args, _real=real, _name=name, **kwargs):
                calls.append(_name)
                return _real(basis, *args, **kwargs)

            monkeypatch.setattr(groups, name, spy)
        return calls

    def _sphere(self):
        psi = np.zeros(4, dtype=complex)
        psi[0] = 1.0
        return Parameters(
            basis=construct_full_pauli_basis(2),
            projected_basis=construct_Heisenberg_pauli_basis(2),
            target=np.asarray(CNOT) @ psi,
            manifold=StateSphere(dim=4, base_point=psi),
        )

    def test_the_sphere_builds_none(self, monkeypatch):
        calls = self._spy(monkeypatch)
        p = self._sphere()
        # A full step's worth of coefficient work: gammas and omegas both.
        ctx = p.manifold.context(p.free())
        _ = ctx.gammas, ctx.omegas
        assert calls == []
        assert p.manifold.tangent.frame is p.basis  # carried as data, unused

    def test_the_group_still_builds_one(self, monkeypatch):
        """The control: the same spy fires on a manifold that does need a frame."""
        calls = self._spy(monkeypatch)
        p = Parameters(
            basis=construct_full_pauli_basis(2),
            projected_basis=construct_Heisenberg_pauli_basis(2),
            target=CNOT,
        )
        assert calls == []  # not at construction either
        ctx = p.manifold.context(p.free())
        _ = ctx.gammas
        assert calls == ["get_project_omegas_fn"]
