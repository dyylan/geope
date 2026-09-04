"""
Tests for geope/geometry/chart.py — the pulse model.

The product-of-exponentials chart every manifold is coordinatised by. The
fidelity formulas that used to share ``engine.py`` with it now live with the
group that owns them, and are tested in tests/test_manifolds.py.

Tested items:
  Functions:
    - compute_matrices_params_list_fn
    - get_compute_matrices_params_list_fn
    - get_chart_jacobian_fn / get_chart_vjp_fn / get_chart_hessian_fn
"""

from types import SimpleNamespace

import pytest
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geope.geometry.chart import (
    compute_matrices_params_list_fn,
    get_chart_fn,
    get_chart_hessian_fn,
    get_chart_jacobian_fn,
    get_chart_vjp_fn,
    get_compute_matrices_params_list_fn,
    get_jacobian_fn,
)
from geope.geometry.basis import Basis
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


# ---------------------------------------------------------------------------
# Tests — the landed jet: DPhi, DPhi^T and D^2Phi from the propagators
# ---------------------------------------------------------------------------
#
# These are the live paths, and each is checked against the autodiff equivalent
# it replaced. The parametrisation over ``base_point`` is the point: the manual
# propagator is shaped ``(G, d, d, K)`` while the ``TangentBundle`` contract is
# ``(*ambient_shape, G, K)``, and the landing has to survive a base point that is
# absent (a group), a vector (a state) or a matrix (a frame).


BASE_POINT_CASES = ["None", "(d,)", "(d, m)"]


def _base_point(case, d, key):
    if case == "None":
        return None
    shape = (d,) if case == "(d,)" else (d, 2)
    parts = jax.random.normal(key, (2, *shape))
    return parts[0] + 1j * parts[1]


@pytest.fixture(params=BASE_POINT_CASES)
def jet(request):
    """A 2-qubit, 3-step chart landed on each kind of base point."""
    basis = np.asarray(construct_full_pauli_basis(2).basis)
    d, K = basis.shape[1], basis.shape[0]
    key_p, key_b = jax.random.split(jax.random.key(11))
    params = (jax.random.normal(key_p, (3, K)) * 0.3).astype(jnp.complex128)
    base = _base_point(request.param, d, key_b)
    chart = get_chart_fn(basis, base)
    return SimpleNamespace(
        case=request.param,
        basis=basis,
        base=base,
        params=params,
        chart=chart,
        ambient=chart(params).shape,
    )


class TestChartJacobian:
    def test_matches_autodiff(self, jet):
        manual = get_chart_jacobian_fn(jet.basis, jet.base)(jet.params)
        auto = get_jacobian_fn(jet.chart)(jet.params)
        assert jnp.allclose(manual, auto, atol=1e-9)

    def test_layout_is_ambient_then_parameters(self, jet):
        """The contract `GeometricContext` slices with ``moveaxis((-2,-1),(0,1))``."""
        manual = get_chart_jacobian_fn(jet.basis, jet.base)(jet.params)
        assert manual.shape == (*jet.ambient, *jet.params.shape)


class TestChartVjp:
    def test_matches_contracting_the_jacobian(self, jet):
        cot_parts = jax.random.normal(jax.random.key(12), (2, *jet.ambient))
        cot = cot_parts[0] + 1j * cot_parts[1]
        jac = get_jacobian_fn(jet.chart)(jet.params)
        expected = jnp.einsum("...,...gk->gk", jnp.conj(cot), jac)
        _, pullback = get_chart_vjp_fn(jet.basis, jet.base)(jet.params)
        got = pullback(cot)
        assert got.shape == jet.params.shape
        assert jnp.allclose(got, expected, atol=1e-9)

    def test_returns_the_landed_point_with_the_pullback(self, jet):
        """The value is shared with the pullback, so a gradient is one pass."""
        point, _ = get_chart_vjp_fn(jet.basis, jet.base)(jet.params)
        assert point.shape == jet.ambient
        assert jnp.allclose(point, jet.chart(jet.params), atol=1e-12)

    def test_is_conjugate_linear_in_the_covector(self, jet):
        """It pairs through ``Tr(C^dagger .)``, so ``i C`` scales it by ``-i``."""
        _, pullback = get_chart_vjp_fn(jet.basis, jet.base)(jet.params)
        cot = jnp.ones(jet.ambient, dtype=jnp.complex128)
        assert jnp.allclose(pullback(1j * cot), -1j * pullback(cot))


class TestChartHessian:
    def test_matches_autodiff(self, jet):
        manual = get_chart_hessian_fn(jet.basis, jet.base)(jet.params)
        auto = jax.jacfwd(jax.jacrev(jet.chart, holomorphic=True), holomorphic=True)(
            jet.params
        )
        # (G, G, *ambient, K, K) -> (*ambient, G, K, G, K)
        assert jnp.allclose(jnp.einsum("ij...kl->...ikjl", manual), auto, atol=1e-8)

    def test_layout_is_gates_ambient_coefficients(self, jet):
        manual = get_chart_hessian_fn(jet.basis, jet.base)(jet.params)
        n_gates, n_coeffs = jet.params.shape
        assert manual.shape == (n_gates, n_gates, *jet.ambient, n_coeffs, n_coeffs)
