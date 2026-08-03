"""
Tests for geope/geope.py and geope/jacobian_propagator.py.

The Gecko tests live in ``test_gecko.py`` and the History tests in
``test_history.py``.

Tested items:
  Functions (geope.engine):
    - geodesic_hamiltonian
    - get_geodesic_hamiltonian_fn
    - linear_comb_projected_coeffs_multigate
    - hvp_forward_over_reverse
  Classes:
    - Geope
  Functions:
    - build_pulse_expander
  jacobian_propagator:
    - Ui / get_Ui_fn
    - jacobian_propagator
    - get_jacobian_propagator
    - jvp_propagator / get_jvp_propagator 
  jax primitives:
    - dexpm / dexpm_eig (per-step derivative)
    - d2expm / d2expm_eig (per-step second derivative)
    - expm_jvp / expm_hvp (+ _eig) 
    - hessian_propagator / get_hessian_propagator (propagator Hessian)
    - hvp_propagator / get_hvp_propagator
    - get_hessian_propagator_fn (cost propagator Hessian)
"""

import dataclasses
from dataclasses import FrozenInstanceError

import pytest
import numpy as np
import scipy.linalg as spla

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geope.geope import (
    Geope,
    build_pulse_expander,
    linear_comb_projected_coeffs_multigate,
    DEFAULT_PRECISION,
    DEFAULT_MAX_STEP_SIZE,
    DEFAULT_GRAM_SCHMIDT_STEP_SIZE,
)
from geope.line_searches import (
    Adam,
    GoldenSection,
    LineSearch,
    QuadraticArmijo,
)
from geope.engine import (
    geodesic_hamiltonian,
    get_geodesic_hamiltonian_fn,
    hvp_forward_over_reverse,
    get_compute_matrices_params_list_fn,
    get_jacobian_fn,
    get_hessian_fn,
    get_hessian_propagator_fn,
    get_infidelity_fn,
    get_infidelity_full_fn,
)
from geope.gecko import Gecko
from geope.parameters import Parameters
from geope.utils.history import History
from geope.lie import Basis, Hamiltonian, Unitary
from geope.engine import fidelity
from geope.utils import (
    construct_full_pauli_basis,
    construct_Heisenberg_pauli_basis,
    construct_restricted_pauli_basis,
)


def _params_2q(
    cnot,
    full_basis_2q,
    projected_basis_2q,
    *,
    drift_basis=None,
    drift_values=None,
    init_values=None,
    constraints=None,
    piecewise_steps=1,
    seed=42,
    init_spread=0.1,
    pulse_constraints=None,
    projective=True,
    param_transform=None,
    n_experimental_params=None,
):
    """Build a Parameters bundle from the raw test fixtures.

    Helper for tests that need to construct a ``Geope`` from the
    Heisenberg / full Pauli basis fixtures rather than from a
    control dict.
    """
    return Parameters(
        basis=full_basis_2q,
        projected_basis=projected_basis_2q,
        drift_basis=drift_basis,
        drift_values=drift_values,
        init_values=init_values,
        target=cnot,
        piecewise_steps=piecewise_steps,
        constraints=constraints,
        pulse_constraints=pulse_constraints,
        init_spread=init_spread,
        seed=seed,
        projective=projective,
        param_transform=param_transform,
        n_experimental_params=n_experimental_params,
    )


from geope.jax.jacobian import (
    Ui,
    get_Ui_fn,
    jacobian_propagator,
    get_jacobian_propagator,
    jvp_propagator,
    get_jvp_propagator,
)
from geope.jax.dexpm import get_dexpm, dexpm, dexpm_eig, dexpm_eig_batched
from geope.jax.dexpm import d2expm, d2expm_eig, d2expm_eig_batched
from geope.jax.dexpm import expm_jvp, expm_jvp_eig, expm_hvp, expm_hvp_eig
from geope.jax.hessian import (
    hessian_propagator,
    get_hessian_propagator,
    hvp_propagator,
    get_hvp_propagator,
)
from geope.utils import qft_unitary

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
    """Full 2-qubit Pauli basis (15 elements)."""
    return construct_full_pauli_basis(2)


@pytest.fixture
def projected_basis_2q():
    """Heisenberg 2-qubit basis (9 elements ⊂ 15) — a proper subset of the full basis."""
    return construct_Heisenberg_pauli_basis(2)


@pytest.fixture
def params_2q(cnot, full_basis_2q, projected_basis_2q):
    return _params_2q(cnot, full_basis_2q, projected_basis_2q)


@pytest.fixture
def geope_2q(params_2q):
    return Geope(params_2q)


# ---------------------------------------------------------------------------
# Helpers — small bases for jacobian_propagator tests
# ---------------------------------------------------------------------------


def _pauli_basis_1q():
    """Single-qubit Pauli basis (X, Y, Z) — 3 generators, 2×2."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return jnp.stack([X, Y, Z])


# ---------------------------------------------------------------------------
# Tests — jacobian_propagator.Ui / get_Ui_fn
# ---------------------------------------------------------------------------


class TestUi:
    def test_zero_params_gives_identity(self):
        basis = _pauli_basis_1q()
        U = Ui(jnp.zeros(3), basis)
        assert jnp.allclose(U, jnp.eye(2), atol=1e-12)

    def test_output_is_unitary(self):
        basis = _pauli_basis_1q()
        params = jnp.array([0.3, -0.5, 0.7])
        U = Ui(params, basis)
        assert jnp.allclose(U @ U.conj().T, jnp.eye(2), atol=1e-10)

    def test_shape(self):
        basis = _pauli_basis_1q()
        U = Ui(jnp.ones(3), basis)
        assert U.shape == (2, 2)

    def test_get_Ui_fn_matches_direct(self):
        basis = _pauli_basis_1q()
        fn = get_Ui_fn(basis)
        params = jnp.array([0.1, 0.2, 0.3])
        assert jnp.allclose(fn(params), Ui(params, basis))

    def test_get_Ui_fn_is_callable(self):
        basis = _pauli_basis_1q()
        assert callable(get_Ui_fn(basis))


# ---------------------------------------------------------------------------
# Tests — jacobian_propagator
# ---------------------------------------------------------------------------


class TestDexpmEig:
    """The spectral derivative must match the block-exponential `dexpm`."""

    def test_matches_block_method_1q(self):
        basis = _pauli_basis_1q()
        x = jnp.array([0.4, -0.2, 0.6], dtype=complex)
        assert jnp.allclose(dexpm_eig(x, basis), dexpm(x, basis), atol=1e-9)

    def test_matches_block_method_2q(self):
        basis = jnp.asarray(construct_full_pauli_basis(2).basis)  # (15, 4, 4)
        x = jax.random.normal(jax.random.key(7), (basis.shape[0],)).astype(complex)
        assert jnp.allclose(dexpm_eig(x, basis), dexpm(x, basis), atol=1e-9)

    def test_complex_coeffs_need_hermitian_false(self):
        """For genuinely complex coefficients the default (eigh) is invalid; the
        hermitian=False fallback (general eig) must match the block method."""
        basis = jnp.asarray(construct_full_pauli_basis(2).basis)
        K = basis.shape[0]
        x = jax.random.normal(jax.random.key(20), (K,)) + 1j * jax.random.normal(
            jax.random.key(21), (K,)
        )
        ref = dexpm(x, basis)  # block method handles non-Hermitian A
        assert jnp.allclose(dexpm_eig(x, basis, hermitian=False), ref, atol=1e-8)
        assert not jnp.allclose(dexpm_eig(x, basis), ref, atol=1e-3)

    def test_zero_params_gives_generators(self):
        """At x=0 the derivative of expm(iA) w.r.t. x_k is i*B_k."""
        basis = _pauli_basis_1q()
        x = jnp.zeros(3, dtype=complex)
        out = dexpm_eig(x, basis)  # (2, 2, 3)
        expected = jnp.moveaxis(1j * basis, 0, -1)
        assert jnp.allclose(out, expected, atol=1e-9)

    def test_batched_matches_full(self):
        """Chunking the directions must not change the result."""
        basis = jnp.asarray(construct_full_pauli_basis(2).basis)  # K=15
        x = jax.random.normal(jax.random.key(9), (basis.shape[0],)).astype(complex)
        full = dexpm_eig(x, basis)
        for batch_size in (1, 4, basis.shape[0]):
            assert jnp.allclose(
                dexpm_eig_batched(x, basis, batch_size), full, atol=1e-9
            )


class TestJacobianPropagator:
    def test_output_shape_single_gate(self):
        basis = _pauli_basis_1q()
        Ui_fn = get_Ui_fn(basis)
        jac_fn = get_dexpm(basis)
        params = jnp.array([[0.1, 0.2, 0.3]])
        result = jacobian_propagator(params, Ui_fn, jac_fn)
        # shape: (n_gates, dim, dim, n_params)
        assert result.shape == (1, 2, 2, 3)

    def test_output_shape_multi_gate(self):
        basis = _pauli_basis_1q()
        Ui_fn = get_Ui_fn(basis)
        jac_fn = get_dexpm(basis)
        params = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        result = jacobian_propagator(params, Ui_fn, jac_fn)
        assert result.shape == (2, 2, 2, 3)

    def test_zero_params_derivatives_nonzero(self):
        """At identity, derivatives of expm are the generators themselves."""
        basis = _pauli_basis_1q()
        Ui_fn = get_Ui_fn(basis)
        jac_fn = get_dexpm(basis)
        params = jnp.array([[0.0, 0.0, 0.0]])
        result = jacobian_propagator(params, Ui_fn, jac_fn)
        # Should not be all zeros — derivative of expm(i*0) w.r.t. params gives i*basis
        assert not jnp.allclose(result, 0, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests — get_jacobian_propagator
# ---------------------------------------------------------------------------


class TestGetJacobianPropagator:
    def test_returns_callable(self):
        basis = _pauli_basis_1q()
        fn = get_jacobian_propagator(basis)
        assert callable(fn)

    @pytest.mark.parametrize("method", ["eig", "block"])
    def test_call_produces_correct_shape(self, method):
        basis = _pauli_basis_1q()
        fn = get_jacobian_propagator(basis, method=method)
        params = jnp.array([[0.1, 0.2, 0.3]])
        result = fn(params)
        assert result.shape == (1, 2, 2, 3)

    def test_matches_jacobian_propagator_direct(self):
        basis = _pauli_basis_1q()
        fn = get_jacobian_propagator(basis)
        Ui_fn = get_Ui_fn(basis)
        jac_fn = get_dexpm(basis)
        params = jnp.array([[0.5, -0.3, 0.1]])
        assert jnp.allclose(
            fn(params), jacobian_propagator(params, Ui_fn, jac_fn), atol=1e-10
        )

    @pytest.mark.parametrize("method", ["eig", "block"])
    def test_agrees_with_jax_jacobian(self, method):
        """Compare jacobian propagator against jax.jacobian for a single gate."""
        basis = _pauli_basis_1q()
        fn_propagator = get_jacobian_propagator(basis, method=method)
        Ui_fn = get_Ui_fn(basis)

        params = jnp.array([[0.4, -0.2, 0.6]], dtype=complex)
        jac_propagator = fn_propagator(params)  # (1, 2, 2, 3)

        # jax.jacobian over full compute
        def compute_U(p):
            A = jnp.tensordot(p[0], basis, axes=[[-1], [0]])
            return jax.scipy.linalg.expm(1j * A)

        jac_auto = jax.jacobian(compute_U, holomorphic=True)(params)  # (2,2,1,3)
        # manual shape is (1,2,2,3), auto shape is (2,2,1,3) — rearrange
        jac_auto_rearranged = jnp.transpose(jac_auto, (2, 0, 1, 3))  # (1,2,2,3)
        assert jnp.allclose(jac_propagator, jac_auto_rearranged, atol=1e-8)

    @pytest.mark.parametrize("method", ["eig", "block"])
    def test_agrees_with_autodiff_multigate_multiqubit(self, method):
        """The prefix/suffix Jacobian must match full-sequence autodiff for
        the general G>1, n>1 case, not just a single 1-qubit gate."""
        basis = jnp.asarray(construct_full_pauli_basis(2).basis)  # (15, 4, 4)
        K = basis.shape[0]
        params = jax.random.normal(jax.random.key(3), (3, K)).astype(jnp.complex128)

        jac_propagator = get_jacobian_propagator(basis, method=method)(
            params
        )  # (3, 4, 4, 15)

        compute_U = get_compute_matrices_params_list_fn(basis)
        jac_auto = get_jacobian_fn(compute_U)(params)  # (4, 4, 3, 15)
        jac_auto = jnp.transpose(jac_auto, (2, 0, 1, 3))  # (3, 4, 4, 15)

        assert jac_propagator.shape == (3, 4, 4, K)
        assert jnp.allclose(jac_propagator, jac_auto, atol=1e-8)

    def test_block_method_matches_eig(self):
        """The ``method`` switch: block and eig must agree (real params)."""
        basis = jnp.asarray(construct_full_pauli_basis(2).basis)  # (15, 4, 4)
        params = jax.random.normal(jax.random.key(30), (3, basis.shape[0])) * 0.3
        eig = get_jacobian_propagator(basis, method="eig")(params)
        block = get_jacobian_propagator(basis, method="block")(params)
        assert jnp.allclose(eig, block, atol=1e-8)

    def test_unknown_method_raises(self):
        basis = _pauli_basis_1q()
        with pytest.raises(ValueError, match="Unknown method"):
            get_jacobian_propagator(basis, method="nope")


# ---------------------------------------------------------------------------
# Tests — d2expm (per-step second derivative)
# ---------------------------------------------------------------------------


class TestD2expm:
    """Second-derivative primitives vs autodiff and each other."""

    def _autodiff(self, basis, x):
        Ui_fn = get_Ui_fn(basis)
        return jax.jacfwd(jax.jacrev(Ui_fn, holomorphic=True), holomorphic=True)(x)

    def test_block_matches_autodiff(self):
        basis = _pauli_basis_1q()
        x = jnp.array([0.4, -0.2, 0.6], dtype=complex)
        assert jnp.allclose(d2expm(x, basis), self._autodiff(basis, x), atol=1e-8)

    def test_eig_matches_autodiff_2q(self):
        basis = jnp.asarray(construct_full_pauli_basis(2).basis)
        x = jax.random.normal(jax.random.key(11), (basis.shape[0],)).astype(complex)
        assert jnp.allclose(d2expm_eig(x, basis), self._autodiff(basis, x), atol=1e-8)

    def test_block_matches_eig(self):
        basis = jnp.asarray(construct_full_pauli_basis(2).basis)
        x = jax.random.normal(jax.random.key(12), (basis.shape[0],)).astype(complex)
        assert jnp.allclose(d2expm(x, basis), d2expm_eig(x, basis), atol=1e-8)

    def test_symmetric_in_kl(self):
        basis = jnp.asarray(construct_full_pauli_basis(2).basis)
        x = jax.random.normal(jax.random.key(13), (basis.shape[0],)).astype(complex)
        out = d2expm_eig(x, basis)  # (d, d, K, K)
        assert jnp.allclose(out, jnp.swapaxes(out, -1, -2), atol=1e-12)

    def test_batched_matches_full(self):
        basis = jnp.asarray(construct_full_pauli_basis(2).basis)
        x = jax.random.normal(jax.random.key(14), (basis.shape[0],)).astype(complex)
        full = d2expm_eig(x, basis)
        for bs in (1, 4, basis.shape[0]):
            assert jnp.allclose(d2expm_eig_batched(x, basis, bs), full, atol=1e-9)

    def test_complex_coeffs_need_hermitian_false(self):
        """Complex coefficients require the hermitian=False (general eig) path."""
        basis = jnp.asarray(construct_full_pauli_basis(2).basis)
        K = basis.shape[0]
        x = jax.random.normal(jax.random.key(22), (K,)) + 1j * jax.random.normal(
            jax.random.key(23), (K,)
        )
        ref = d2expm(x, basis)  # block method handles non-Hermitian A
        assert jnp.allclose(d2expm_eig(x, basis, hermitian=False), ref, atol=1e-8)


# ---------------------------------------------------------------------------
# Tests — hessian_propagator (propagator Hessian)
# ---------------------------------------------------------------------------


class TestHessianPropagator:
    def _autodiff(self, basis, params):
        compute_U = get_compute_matrices_params_list_fn(basis)
        h = jax.jacfwd(jax.jacrev(compute_U, holomorphic=True), holomorphic=True)
        return jnp.transpose(h(params), (2, 4, 0, 1, 3, 5))  # -> (i, j, a, c, k, l)

    @pytest.mark.parametrize("method", ["eig", "block"])
    def test_shape_and_value_single_gate(self, method):
        basis = _pauli_basis_1q()
        params = jnp.array([[0.4, -0.2, 0.6]], dtype=complex)
        H = get_hessian_propagator(basis, method=method)(params)
        assert H.shape == (1, 1, 2, 2, 3, 3)
        assert jnp.allclose(H, self._autodiff(basis, params), atol=1e-8)

    @pytest.mark.parametrize("method", ["eig", "block"])
    def test_agrees_with_autodiff_multigate_multiqubit(self, method):
        basis = jnp.asarray(construct_full_pauli_basis(2).basis)  # (15, 4, 4)
        K = basis.shape[0]
        params = jax.random.normal(jax.random.key(15), (3, K)).astype(complex)
        H = get_hessian_propagator(basis, method=method)(params)
        assert H.shape == (3, 3, 4, 4, K, K)
        assert jnp.allclose(H, self._autodiff(basis, params), atol=1e-8)

    @pytest.mark.parametrize("method", ["eig", "block"])
    def test_symmetric_under_pair_exchange(self, method):
        basis = jnp.asarray(construct_full_pauli_basis(1).basis)
        params = jax.random.normal(jax.random.key(16), (2, 3)).astype(complex)
        H = get_hessian_propagator(basis, method=method)(params)  # (G, G, d, d, K, K)
        # H[i,j,:,:,k,l] == H[j,i,:,:,l,k]
        swapped = jnp.swapaxes(jnp.swapaxes(H, 0, 1), -1, -2)
        assert jnp.allclose(H, swapped, atol=1e-10)

    def test_block_method_matches_eig(self):
        """The ``method`` switch: block and eig must agree (real params)."""
        basis = jnp.asarray(construct_full_pauli_basis(2).basis)  # (15, 4, 4)
        params = jax.random.normal(jax.random.key(31), (3, basis.shape[0])) * 0.3
        eig = get_hessian_propagator(basis, method="eig")(params)
        block = get_hessian_propagator(basis, method="block")(params)
        assert jnp.allclose(eig, block, atol=1e-8)

    def test_unknown_method_raises(self):
        basis = _pauli_basis_1q()
        with pytest.raises(ValueError, match="Unknown method"):
            get_hessian_propagator(basis, method="nope")


# ---------------------------------------------------------------------------
# Tests — per-gate directional primitives (expm_jvp / expm_hvp)
# ---------------------------------------------------------------------------


class TestExpmDirectionalPrimitives:
    """Directional (single-``p``) per-gate value/derivatives vs the full stacks.

    ``expm_jvp*`` returns ``(U, E)`` and ``expm_hvp*`` returns ``(U, E, G)`` for
    a single direction ``p``; these must equal the ``p``-contractions of the
    full per-parameter ``dexpm`` / ``d2expm`` tensors.
    """

    def _xp(self, K, seed):
        x = jax.random.normal(jax.random.key(seed), (K,)).astype(complex)
        p = jax.random.normal(jax.random.key(seed + 1), (K,)).astype(complex)
        return x, p

    def test_jvp_matches_dexpm_contraction(self):
        basis = jnp.asarray(construct_full_pauli_basis(2).basis)  # (15, 4, 4)
        x, p = self._xp(basis.shape[0], 40)
        U_ref = Ui(x, basis)
        E_ref = jnp.einsum("dek,k->de", dexpm(x, basis), p)
        for U, E in (expm_jvp(x, p, basis), expm_jvp_eig(x, p, basis)):
            assert jnp.allclose(U, U_ref, atol=1e-9)
            assert jnp.allclose(E, E_ref, atol=1e-8)

    def test_hvp_matches_dexpm_d2expm_contraction(self):
        basis = jnp.asarray(construct_full_pauli_basis(2).basis)
        x, p = self._xp(basis.shape[0], 42)
        U_ref = Ui(x, basis)
        E_ref = jnp.einsum("dek,k->de", dexpm(x, basis), p)
        G_ref = jnp.einsum("dekl,k,l->de", d2expm(x, basis), p, p)
        for U, E, G in (expm_hvp(x, p, basis), expm_hvp_eig(x, p, basis)):
            assert jnp.allclose(U, U_ref, atol=1e-9)
            assert jnp.allclose(E, E_ref, atol=1e-8)
            assert jnp.allclose(G, G_ref, atol=1e-8)

    def test_block_matches_eig(self):
        basis = jnp.asarray(construct_full_pauli_basis(2).basis)
        x, p = self._xp(basis.shape[0], 44)
        Ub, Eb, Gb = expm_hvp(x, p, basis)
        Ue, Ee, Ge = expm_hvp_eig(x, p, basis)
        assert jnp.allclose(Ub, Ue, atol=1e-8)
        assert jnp.allclose(Eb, Ee, atol=1e-8)
        assert jnp.allclose(Gb, Ge, atol=1e-8)

    def test_complex_coeffs_need_hermitian_false(self):
        """Genuinely complex ``x`` requires the general-eig path; the default
        (eigh) is invalid, the block method is the reference."""
        basis = jnp.asarray(construct_full_pauli_basis(2).basis)
        K = basis.shape[0]
        x = jax.random.normal(jax.random.key(46), (K,)) + 1j * jax.random.normal(
            jax.random.key(47), (K,)
        )
        p = jax.random.normal(jax.random.key(48), (K,)).astype(complex)
        _, _, G_ref = expm_hvp(x, p, basis)  # block handles non-Hermitian A
        _, _, G_eig = expm_hvp_eig(x, p, basis, hermitian=False)
        assert jnp.allclose(G_eig, G_ref, atol=1e-8)


# ---------------------------------------------------------------------------
# Tests — jvp_propagator / hvp_propagator (directional product derivatives)
# ---------------------------------------------------------------------------


def _autodiff_dir_derivs(basis, params, p):
    """Reference ``(phi, Dphi[p], D2phi[p,p])`` via jvp-of-jvp through compute_U."""
    compute_U = get_compute_matrices_params_list_fn(basis)
    X = compute_U(params)
    V = jax.jvp(compute_U, (params,), (p,))[1]
    W = jax.jvp(lambda z: jax.jvp(compute_U, (z,), (p,))[1], (params,), (p,))[1]
    return X, V, W


class TestJvpPropagator:
    """First-order directional propagator vs forward-mode autodiff."""

    @pytest.mark.parametrize("method", ["eig", "block"])
    @pytest.mark.parametrize("n,G", [(1, 2), (2, 3)])
    def test_matches_autodiff(self, method, n, G):
        basis = jnp.asarray(construct_full_pauli_basis(n).basis)
        K = basis.shape[0]
        params = jax.random.normal(jax.random.key(50), (G, K)).astype(complex)
        p = jax.random.normal(jax.random.key(51), (G, K)).astype(complex)

        X, V = get_jvp_propagator(basis, method=method)(params, p)
        X_ref, V_ref, _ = _autodiff_dir_derivs(basis, params, p)

        d = 2**n
        assert X.shape == (d, d) and V.shape == (d, d)
        assert jnp.allclose(X, X_ref, atol=1e-9)
        assert jnp.allclose(V, V_ref, atol=1e-8)

    def test_finite_difference(self):
        """Note §11: V ≈ (phi(θ+hp) − phi(θ−hp)) / 2h."""
        basis = jnp.asarray(construct_full_pauli_basis(2).basis)
        compute_U = get_compute_matrices_params_list_fn(basis)
        params = jax.random.normal(jax.random.key(52), (3, basis.shape[0])) * 0.3
        p = jax.random.normal(jax.random.key(53), (3, basis.shape[0])) * 0.3
        _, V = get_jvp_propagator(basis)(params.astype(complex), p.astype(complex))
        h = 1e-6
        V_fd = (compute_U(params + h * p) - compute_U(params - h * p)) / (2 * h)
        assert jnp.allclose(V, V_fd, atol=1e-6)

    def test_unknown_method_raises(self):
        basis = _pauli_basis_1q()
        with pytest.raises(ValueError, match="Unknown method"):
            get_jvp_propagator(basis, method="nope")


class TestHvpPropagator:
    """Second-order directional propagator vs forward-over-forward autodiff."""

    @pytest.mark.parametrize("method", ["eig", "block"])
    @pytest.mark.parametrize("n,G", [(1, 2), (2, 3)])
    def test_matches_autodiff(self, method, n, G):
        basis = jnp.asarray(construct_full_pauli_basis(n).basis)
        K = basis.shape[0]
        params = jax.random.normal(jax.random.key(54), (G, K)).astype(complex)
        p = jax.random.normal(jax.random.key(55), (G, K)).astype(complex)

        X, V, W = get_hvp_propagator(basis, method=method)(params, p)
        X_ref, V_ref, W_ref = _autodiff_dir_derivs(basis, params, p)

        d = 2**n
        assert X.shape == (d, d) and V.shape == (d, d) and W.shape == (d, d)
        assert jnp.allclose(X, X_ref, atol=1e-9)
        assert jnp.allclose(V, V_ref, atol=1e-8)
        assert jnp.allclose(W, W_ref, atol=1e-8)

    def test_first_order_matches_jvp_propagator(self):
        """X and V from the HVP must equal the JVP propagator's outputs."""
        basis = jnp.asarray(construct_full_pauli_basis(2).basis)
        params = jax.random.normal(jax.random.key(56), (3, basis.shape[0])).astype(
            complex
        )
        p = jax.random.normal(jax.random.key(57), (3, basis.shape[0])).astype(complex)
        X, V, _ = get_hvp_propagator(basis)(params, p)
        Xj, Vj = get_jvp_propagator(basis)(params, p)
        assert jnp.allclose(X, Xj, atol=1e-10)
        assert jnp.allclose(V, Vj, atol=1e-10)

    def test_finite_difference(self):
        """Note §11: W ≈ (phi(θ+hp) − 2phi(θ) + phi(θ−hp)) / h^2."""
        basis = jnp.asarray(construct_full_pauli_basis(2).basis)
        compute_U = get_compute_matrices_params_list_fn(basis)
        params = jax.random.normal(jax.random.key(58), (3, basis.shape[0])) * 0.3
        p = jax.random.normal(jax.random.key(59), (3, basis.shape[0])) * 0.3
        _, _, W = get_hvp_propagator(basis)(params.astype(complex), p.astype(complex))
        h = 1e-4
        W_fd = (
            compute_U(params + h * p)
            - 2 * compute_U(params)
            + compute_U(params - h * p)
        ) / h**2
        assert jnp.allclose(W, W_fd, atol=1e-4)

    def test_unknown_method_raises(self):
        basis = _pauli_basis_1q()
        with pytest.raises(ValueError, match="Unknown method"):
            get_hvp_propagator(basis, method="nope")


# ---------------------------------------------------------------------------
# Tests — cost propagator Hessian (Goodwin–Kuprov NR-GRAPE)
# ---------------------------------------------------------------------------


class TestCostHessianPropagator:
    """Infidelity Hessian propatator must match the autodiff get_hessian_fn."""

    @pytest.mark.parametrize("projective", [True, False])
    @pytest.mark.parametrize("method", ["eig", "block"])
    @pytest.mark.parametrize("n,G", [(1, 2), (2, 3)])
    def test_matches_autodiff(self, projective, method, n, G):
        basis = jnp.asarray(construct_full_pauli_basis(n).basis)
        K = basis.shape[0]
        target = jnp.asarray(qft_unitary(n))
        compute_U = get_compute_matrices_params_list_fn(basis)
        # GRAPE parameters are real-valued.
        y = jax.random.normal(jax.random.key(17), (G, K)) * 0.3

        infid_U = (
            get_infidelity_fn(target) if projective else get_infidelity_full_fn(target)
        )
        infid = lambda x: infid_U(compute_U(x))
        H_auto = get_hessian_fn(infid)(y).reshape(G * K, G * K)
        H_man = get_hessian_propagator_fn(
            basis, target, projective=projective, method=method
        )(y)
        assert H_man.shape == (G * K, G * K)
        assert jnp.allclose(H_man, H_auto, atol=1e-7)

    def test_hessian_is_symmetric(self):
        basis = jnp.asarray(construct_full_pauli_basis(2).basis)
        target = jnp.asarray(qft_unitary(2))
        y = jax.random.normal(jax.random.key(18), (3, basis.shape[0])) * 0.3
        H = get_hessian_propagator_fn(basis, target, projective=True)(y)
        assert jnp.allclose(H, H.T, atol=1e-9)


# ---------------------------------------------------------------------------
# Tests — geodesic_hamiltonian
# ---------------------------------------------------------------------------


class TestGeodesicHamiltonian:
    def test_identity_to_identity_2x2(self, identity_2x2):
        """Same unitary and target ⇒ geodesic hamiltonian ≈ 0."""
        result = geodesic_hamiltonian(identity_2x2, identity_2x2)
        assert result.shape == (2, 2)
        assert jnp.allclose(result, 0, atol=1e-10)

    def test_identity_to_identity_4x4(self, identity_4x4):
        result = geodesic_hamiltonian(identity_4x4, identity_4x4)
        assert result.shape == (4, 4)
        assert jnp.allclose(result, 0, atol=1e-10)

    def test_output_shape_2x2(self, identity_2x2, hadamard):
        result = geodesic_hamiltonian(identity_2x2, hadamard)
        assert result.shape == (2, 2)

    def test_output_shape_4x4(self, identity_4x4, cnot):
        result = geodesic_hamiltonian(identity_4x4, cnot)
        assert result.shape == (4, 4)

    def test_nonzero_for_different_unitaries(self, identity_2x2, hadamard):
        result = geodesic_hamiltonian(identity_2x2, hadamard)
        assert not jnp.allclose(result, 0, atol=1e-5)

    def test_traceless_after_global_phase_removal(self, identity_2x2, hadamard):
        """The function removes the global phase, so U†·result should be traceless."""
        result = geodesic_hamiltonian(identity_2x2, hadamard)
        # For unitary = I, U† @ result = result
        trace_val = jnp.trace(result)
        assert jnp.abs(trace_val) < 1e-8

    def test_target_equal_unitary_4x4(self, cnot):
        result = geodesic_hamiltonian(cnot, cnot)
        assert jnp.allclose(result, 0, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests — get_geodesic_hamiltonian_fn
# ---------------------------------------------------------------------------


class TestGetGeodesicHamiltonianFn:
    def test_returns_callable(self, hadamard):
        fn = get_geodesic_hamiltonian_fn(hadamard)
        assert callable(fn)

    def test_partial_matches_direct_call(self, identity_2x2, hadamard):
        fn = get_geodesic_hamiltonian_fn(hadamard)
        result_partial = fn(identity_2x2)
        result_direct = geodesic_hamiltonian(identity_2x2, hadamard)
        assert jnp.allclose(result_partial, result_direct)

    def test_partial_preserves_target(self, identity_4x4, cnot):
        fn = get_geodesic_hamiltonian_fn(cnot)
        result = fn(identity_4x4)
        assert result.shape == (4, 4)


# ---------------------------------------------------------------------------
# Tests — linear_comb_projected_coeffs_multigate
# ---------------------------------------------------------------------------


class TestLinearCombProjectedCoeffsMultigate:
    def test_identity_system_no_expander(self):
        """With identity-like combo vectors, lstsq should recover the target."""
        comb_vecs = jnp.array(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            ]
        )
        target = jnp.array([0.5, 0.3, 0.1, 0.0])
        result = linear_comb_projected_coeffs_multigate(comb_vecs, target, None)
        assert result.shape == (1, 3)
        assert jnp.allclose(result[0], jnp.array([0.5, 0.3, 0.1]), atol=1e-10)

    def test_with_expander(self):
        comb_vecs = jnp.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ]
        )
        target = jnp.array([0.5, 0.3, 0.0])
        expander = jnp.eye(2, dtype=float)
        result = linear_comb_projected_coeffs_multigate(comb_vecs, target, expander)
        assert result.shape == (1, 2)

    def test_output_shape_multigate(self):
        n_gates, n_params, n_elements = 3, 4, 5
        comb_vecs = jnp.ones((n_gates, n_params, n_elements))
        target = jnp.ones(n_elements)
        result = linear_comb_projected_coeffs_multigate(comb_vecs, target, None)
        assert result.shape == (n_gates, n_params)

    def test_zero_target(self):
        comb_vecs = jnp.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        )
        target = jnp.zeros(2)
        result = linear_comb_projected_coeffs_multigate(comb_vecs, target, None)
        assert jnp.allclose(result, 0, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests — hvp_forward_over_reverse
# ---------------------------------------------------------------------------


class TestHvpForwardOverReverse:
    def test_quadratic_function(self):
        """f(x) = 0.5 x^T A x  ⇒  H = A  ⇒  Hv = A·v."""
        A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        f = lambda x: 0.5 * x @ A @ x
        params = jnp.array([1.0, 2.0])
        v = jnp.array([1.0, 0.0])
        result = hvp_forward_over_reverse(f, params, v)
        expected = A @ v
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_output_shape(self):
        f = lambda x: jnp.sum(x**2)
        params = jnp.array([1.0, 2.0, 3.0])
        v = jnp.ones(3)
        result = hvp_forward_over_reverse(f, params, v)
        assert result.shape == params.shape

    def test_identity_hessian(self):
        """f(x) = 0.5 ||x||^2  ⇒  H = I  ⇒  Hv = v."""
        f = lambda x: 0.5 * jnp.sum(x**2)
        params = jnp.array([1.0, 2.0])
        v = jnp.array([3.0, 4.0])
        result = hvp_forward_over_reverse(f, params, v)
        assert jnp.allclose(result, v, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests — build_pulse_expander (control-format pulse_constraints)
# ---------------------------------------------------------------------------


class TestBuildPulseExpander:
    """`pulse_constraints` uses the control-format dict, same as `control`."""

    @pytest.fixture(scope="class")
    @staticmethod
    def pulse_setup_3q():
        proj = construct_restricted_pauli_basis(3, ["x", "z", "zz"])
        return proj, 4  # (projected_basis, piecewise_steps)

    def test_control_dict_selects_expected_zz_indices(self, pulse_setup_3q):
        proj, L = pulse_setup_3q
        labels = list(proj.labels)
        n_proj = proj.lie_algebra_dim
        proj_params = np.random.default_rng(0).standard_normal((L, n_proj))

        constraints = {(1, 2): ["zz"], (2, 3): ["zz"], (1, 3): ["zz"]}
        E, templates = build_pulse_expander(
            L, proj, constraints, False, n_proj, proj_params
        )

        # The dict selects exactly the three two-body ZZ terms.
        expected = {labels.index(lbl) for lbl in ("ZZI", "IZZ", "ZIZ")}
        assert set(templates.keys()) == expected

        # Each template is the unit-normalised time profile of its column.
        for k, tmpl in templates.items():
            ref = proj_params[:, k] / np.linalg.norm(proj_params[:, k])
            np.testing.assert_allclose(tmpl, ref, atol=1e-12)
            assert np.isclose(np.linalg.norm(tmpl), 1.0)

        # One free column per constrained term + L columns per free term.
        n_free = L * (n_proj - len(expected)) + len(expected)
        assert E.shape == (L * n_proj, n_free)

    def test_single_qubit_key(self, pulse_setup_3q):
        proj, L = pulse_setup_3q
        labels = list(proj.labels)
        n_proj = proj.lie_algebra_dim
        proj_params = np.random.default_rng(1).standard_normal((L, n_proj))

        _, templates = build_pulse_expander(
            L, proj, {1: ["x"]}, False, n_proj, proj_params
        )
        assert set(templates.keys()) == {labels.index("XII")}

    def test_absent_interaction_raises(self, pulse_setup_3q):
        proj, L = pulse_setup_3q
        n_proj = proj.lie_algebra_dim
        proj_params = np.zeros((L, n_proj))

        # 'yy' is not in the restricted basis -> strict check raises.
        with pytest.raises(ValueError, match="not present in the basis"):
            build_pulse_expander(L, proj, {(1, 2): ["yy"]}, False, n_proj, proj_params)

    def test_wrong_qubit_index_raises(self, pulse_setup_3q):
        proj, L = pulse_setup_3q
        n_proj = proj.lie_algebra_dim
        proj_params = np.zeros((L, n_proj))

        # Qubit 4 does not exist on a 3-qubit system.
        with pytest.raises(ValueError, match="not present in the basis"):
            build_pulse_expander(L, proj, {(1, 4): ["zz"]}, False, n_proj, proj_params)

    def test_list_form_now_rejected(self, pulse_setup_3q):
        proj, L = pulse_setup_3q
        n_proj = proj.lie_algebra_dim
        proj_params = np.zeros((L, n_proj))
        # The legacy list-of-Pauli-labels form is no longer accepted in
        # projected space.
        with pytest.raises(TypeError):
            build_pulse_expander(L, proj, ["ZZI"], False, n_proj, proj_params)


class TestParametersPulseConstraintsValidation:
    """`Parameters` validates a dict `pulse_constraints` at construction."""

    @staticmethod
    def _control_3q():
        return {
            1: ["x", "z"],
            2: ["x", "z"],
            3: ["x", "z"],
            (1, 2): ["zz"],
            (2, 3): ["zz"],
            (1, 3): ["zz"],
        }

    def test_valid_dict_constructs(self):
        p = Parameters(
            basis=construct_full_pauli_basis(3),
            control=self._control_3q(),
            target=np.eye(8, dtype=complex),
            piecewise_steps=4,
            pulse_constraints={(1, 2): ["zz"], (2, 3): ["zz"], (1, 3): ["zz"]},
        )
        assert p.pulse_constraints == {(1, 2): ["zz"], (2, 3): ["zz"], (1, 3): ["zz"]}

    def test_absent_interaction_raises_at_construction(self):
        with pytest.raises(ValueError, match="not present in the basis"):
            Parameters(
                basis=construct_full_pauli_basis(3),
                control=self._control_3q(),
                target=np.eye(8, dtype=complex),
                piecewise_steps=4,
                pulse_constraints={(1, 2): ["xx"]},  # only zz is controllable
            )


# ---------------------------------------------------------------------------
# Tests — Geope
# ---------------------------------------------------------------------------


class TestGeope:
    # --- initialisation ---------------------------------------------------

    def test_init_default(self, params_2q):
        g = Geope(params_2q, history=History())
        n = params_2q.basis.lie_algebra_dim
        assert g.params.parameters.shape == (1, n)
        assert g.params.fidelity is not None
        assert len(g.history) == 1

    def test_init_fidelity_in_range(self, params_2q):
        g = Geope(params_2q)
        assert 0 <= g.params.fidelity <= 1

    def test_init_infidelity_complement(self, params_2q):
        """infidelity = 1 − fidelity."""
        g = Geope(params_2q)
        assert jnp.isclose(g.params.fidelity + g.params.infidelity, 1.0, atol=1e-10)

    def test_init_with_custom_params(self, cnot, full_basis_2q, projected_basis_2q):
        n = full_basis_2q.lie_algebra_dim
        init = np.zeros(n)
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q, init_values=init)
        g = Geope(p)
        assert g.params.parameters.shape == (1, n)

    def test_init_with_gate_shaped_params(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        n = full_basis_2q.lie_algebra_dim
        init = np.zeros((2, n))
        p = _params_2q(
            cnot, full_basis_2q, projected_basis_2q, piecewise_steps=2, init_values=init
        )
        g = Geope(p)
        assert g.params.parameters.shape == (2, n)

    def test_init_bad_params_shape_raises(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        p = _params_2q(
            cnot, full_basis_2q, projected_basis_2q, init_values=np.zeros((5, 5, 5))
        )
        with pytest.raises(ValueError):
            Geope(p)

    def test_verbose_flag(self, params_2q):
        g = Geope(params_2q, verbose=True)
        assert g.verbose is True

    # --- line search (object API) ----------------------------------------

    def test_optimize_default_is_golden_section(self, params_2q):
        # line_search defaults to GoldenSection(); max_steps=0 configures
        # without running an iteration.
        g = Geope(params_2q)
        g.optimize(max_steps=0)
        assert isinstance(g.line_search, GoldenSection)

    def test_line_search_unset_before_optimize(self, params_2q):
        # The line search and its state are unset until optimize() configures them.
        g = Geope(params_2q)
        assert g.line_search is None
        assert g.line_search_state is None

    def test_optimize_with_adam_runs(self, params_2q):
        # Primary acceptance criterion: the Adam line-search object runs end to end.
        g = Geope(params_2q)
        g.optimize(max_steps=5, line_search=Adam(1e-2))
        assert isinstance(g.line_search, Adam)

    def test_adam_valid_fidelities(self, cnot, full_basis_2q, projected_basis_2q):
        # both gradient modes must run inside the real loop and stay valid
        for ls in (Adam(1e-2), Adam(1e-2, finite_difference=False)):
            p = _params_2q(cnot, full_basis_2q, projected_basis_2q)
            g = Geope(p, history=History())
            g.optimize(max_steps=5, line_search=ls)
            for f in g.history.fidelities:
                assert 0 <= f <= 1

    def test_adam_improves_fidelity(self, cnot, full_basis_2q, projected_basis_2q):
        for ls in (Adam(1e-2), Adam(1e-2, finite_difference=False)):
            p = _params_2q(cnot, full_basis_2q, projected_basis_2q)
            g = Geope(p, history=History())
            f0 = float(g.params.fidelity)
            g.optimize(max_steps=60, line_search=ls)
            assert g.history.best_fidelity > f0

    def test_line_search_state_threads_and_updates(self, params_2q):
        # gram_schmidt_step_size=0 (falsy) skips the fallback, so g.step_size is
        # exactly the line-search dt. With warm_start the threaded state carries
        # the last step's dt — proving the pytree threads and updates within a
        # run (not reset every step).
        g = Geope(params_2q)
        g.optimize(
            max_steps=5,
            line_search=Adam(1e-2, warm_start=True),
            gram_schmidt_step_size=0,
        )
        assert jnp.allclose(g.line_search_state["t_prev"], g.step_size)

    def test_optimize_resets_state_between_calls(self, params_2q):
        # Issue #1: the per-run init() reset is decoupled from compile reuse.
        g = Geope(params_2q)
        g.optimize(max_steps=3, line_search=Adam(1e-2, warm_start=True))
        # Poison the state, then a 0-step run: only the per-run init() reset can
        # have cleared the sentinel — without it this reads 999.0.
        g.line_search_state = {"t_prev": jnp.asarray(999.0)}
        g.optimize(max_steps=0, line_search=Adam(1e-2, warm_start=True))
        assert g.line_search_state["t_prev"] == 0.0

    def test_goldensection_state_has_n_eval(self, params_2q):
        # The zeroth-order search threads only the base {"n_eval"} pytree (the
        # per-step evaluation count), not None and not an empty dict.
        g = Geope(params_2q)
        g.optimize(max_steps=3)
        assert set(g.line_search_state) == {"n_eval"}
        # Golden section spends at least the two initial f1/f2 probes.
        assert int(g.line_search_state["n_eval"]) >= 2

    def test_repeated_optimize_reuses_compiled_fn(self, params_2q):
        # Two optimize() calls with an equal default GoldenSection() reuse the
        # compiled update_step (compile memo via the dataclass __eq__), so reset
        # and recompile-avoidance coexist.
        g = Geope(params_2q)
        g.optimize(max_steps=0)
        first = g.update_step
        g.optimize(max_steps=0)
        assert g.update_step is first

    def test_line_search_eq_and_hash(self):
        # Frozen-dataclass value semantics drive the compile memo and keep
        # hyperparameter sweeps correct (issue #2).
        assert Adam(1e-2) == Adam(1e-2)
        assert hash(Adam(1e-2)) == hash(Adam(1e-2))
        assert Adam(1e-2) != Adam(2e-2)
        assert dataclasses.replace(Adam(1e-2), lr=2e-2) == Adam(2e-2)
        # usable as a set member / dict key
        assert len({Adam(1e-2), Adam(1e-2), GoldenSection()}) == 2
        # immutable
        ls = Adam(1e-2)
        with pytest.raises(FrozenInstanceError):
            ls.lr = 0.5

    def test_optimize_pulse_constrained_threads_state(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        # The pulse-constrained rebuild get_update_step(expander_override=...) is
        # not covered by the compile memo (issue #5); confirm it threads the
        # state too.
        p = _params_2q(
            cnot,
            full_basis_2q,
            projected_basis_2q,
            piecewise_steps=3,
            pulse_constraints={(1, 2): ["zz"]},
        )
        g = Geope(p)
        g.optimize(
            max_steps=4,
            line_search=Adam(1e-2, warm_start=True),
            gram_schmidt_step_size=0,
        )
        assert jnp.allclose(g.line_search_state["t_prev"], g.step_size)

    def test_line_search_history_records_attrs(self, params_2q):
        # History integration: a logging_fn reads line_search attributes, with no
        # change to History. line_search is None at step 0 (before optimize
        # configures it), so the fn guards against that.
        g = Geope(
            params_2q,
            history=History(
                logging_fn=lambda gg: {
                    "name": gg.line_search.name if gg.line_search else None,
                    "lr": getattr(gg.line_search, "lr", None),
                }
            ),
        )
        g.optimize(max_steps=3, line_search=Adam(1e-2))
        assert g.history["name"][-1] == "adam"
        assert g.history["lr"][-1] == 1e-2

    # --- quadratic-seeded Armijo (geometry-aware line search) ------------

    def test_quadratic_armijo_value_semantics(self):
        # Frozen-dataclass value equality (drives the compile memo).
        assert QuadraticArmijo() == QuadraticArmijo()
        assert QuadraticArmijo(c1=1e-3) != QuadraticArmijo()

    def test_quadratic_armijo_runs_and_threads_n_eval(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        # The geometry-aware line search runs end to end, improves fidelity, and
        # threads its per-step evaluation count as an integer state.
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q, piecewise_steps=6)
        g = Geope(p, history=History())
        f0 = float(g.params.fidelity)
        g.optimize(max_steps=60, line_search=QuadraticArmijo())
        assert isinstance(g.line_search, QuadraticArmijo)
        assert g.history.best_fidelity > f0
        for f in g.history.fidelities:
            assert 0 <= f <= 1
        # n_eval populated: at least the seed evaluation each step.
        assert int(g.line_search_state["n_eval"]) >= 1

    def test_quadratic_armijo_converges(self, cnot, full_basis_2q, projected_basis_2q):
        # With enough piecewise steps the geometry-seeded step drives synthesis to
        # high fidelity, matching the golden-section baseline it shares a
        # direction with.
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q, piecewise_steps=6)
        g = Geope(p, history=History())
        g.optimize(max_steps=200, precision=0.9999999, line_search=QuadraticArmijo())
        assert g.history.best_fidelity > 0.999

    # --- run-control knobs (optimize() arguments) ------------------------

    def test_run_knobs_stored_from_optimize(self, params_2q):
        g = Geope(params_2q)
        g.optimize(
            max_steps=0, precision=0.999, max_step_size=0.5, gram_schmidt_step_size=1.5
        )
        assert g.precision == 0.999
        assert g.max_step_size == 0.5
        assert g.gram_schmidt_step_size == 1.5

    def test_run_knobs_default_before_optimize(self, params_2q):
        g = Geope(params_2q)
        assert g.precision == DEFAULT_PRECISION
        assert g.max_step_size == DEFAULT_MAX_STEP_SIZE
        assert g.gram_schmidt_step_size == DEFAULT_GRAM_SCHMIDT_STEP_SIZE

    def test_max_step_size_is_memo_keyed(self, params_2q):
        # max_step_size is baked into the jitted closure, so a changed value
        # rebuilds update_step while repeating the same value reuses it.
        g = Geope(params_2q)
        g.optimize(max_steps=0, max_step_size=0.9)
        first = g.update_step
        g.optimize(max_steps=0, max_step_size=0.9)
        assert g.update_step is first
        g.optimize(max_steps=0, max_step_size=0.5)
        assert g.update_step is not first

    def test_precision_from_optimize_controls_stopping(self, params_2q):
        # precision=0.0 → fidelity < 0.0 is never true → the loop runs 0 steps.
        g = Geope(params_2q, history=History())
        g.optimize(max_steps=1000, precision=0.0)
        assert len(g.history) == 1  # only step 0 recorded

    def test_non_parameters_arg_rejected(self):
        """Passing anything other than a Parameters must raise TypeError."""
        with pytest.raises(TypeError):
            Geope("not a Parameters object")

    # --- reinit -----------------------------------------------------------

    def test_reinit_resets(self, params_2q):
        g = Geope(params_2q, history=History())
        g.init(seed=99)
        assert len(g.history) == 1
        assert g.params.fidelity is not None

    def test_reinit_different_seed(self, params_2q):
        g = Geope(params_2q)
        params_42 = np.array(g.params.parameters)
        g.init(seed=99)
        params_99 = np.array(g.params.parameters)
        # Very unlikely to be identical with different seeds
        assert not np.allclose(params_42, params_99)

    # --- optimize ---------------------------------------------------------

    def test_optimize_runs(self, params_2q):
        g = Geope(params_2q)
        result = g.optimize(max_steps=3)
        # optimize() returns the bound Parameters object
        assert result is params_2q

    def test_optimize_increases_steps(self, params_2q):
        g = Geope(params_2q, history=History())
        g.optimize(max_steps=3)
        assert len(g.history) > 1
        assert g.history.steps[-1] > 0

    def test_optimize_fidelity_tracking(self, params_2q):
        g = Geope(params_2q, history=History())
        g.optimize(max_steps=5)
        n = len(g.history)
        assert len(g.history.fidelities) == n
        assert len(g.history.infidelities) == n
        assert len(g.history.step_sizes) == n
        assert len(g.history.steps) == n

    def test_optimize_all_fidelities_valid(self, params_2q):
        g = Geope(params_2q, history=History())
        g.optimize(max_steps=5)
        for f in g.history.fidelities:
            assert 0 <= f <= 1

    def test_optimize_infidelity_consistency(self, params_2q):
        g = Geope(params_2q, history=History())
        g.optimize(max_steps=5)
        for f, inf in zip(g.history.fidelities, g.history.infidelities):
            assert jnp.isclose(f + inf, 1.0, atol=1e-10)

    def test_optimize_logs_into_history(self, params_2q):
        """History lives on geope.history, not mirrored onto Parameters."""
        # precision=0.0 → converges immediately without running the geodesic step.
        g = Geope(params_2q, history=History())
        g.optimize(max_steps=1, precision=0.0)
        assert g.history.best_fidelity == max(g.history.fidelities)
        # the current/final answer lives on Parameters
        assert params_2q.fidelity is not None

    def test_optimize_returns_params_when_converged(self, params_2q):
        """With precision=0, optimize converges immediately and returns the Parameters."""
        g = Geope(params_2q, history=History())
        result = g.optimize(max_steps=1, precision=0.0)
        assert result is params_2q
        assert g.history.best_fidelity is not None

    def test_optimize_repeated_accumulates_history(self, params_2q):
        """Repeated optimize() calls keep accumulating into the same History."""
        g = Geope(params_2q, history=History())
        g.optimize(max_steps=2)
        n1 = len(g.history)
        g.optimize(max_steps=2)
        n2 = len(g.history)
        assert n2 >= n1

    def test_optimize_verbose(self, cnot, full_basis_2q, projected_basis_2q, capsys):
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q)
        g = Geope(p, verbose=True)
        g.optimize(max_steps=2)
        captured = capsys.readouterr()
        # verbose mode prints progress lines
        assert len(captured.out) > 0

    # --- add_parameters ---------------------------------------------------

    def test_add_parameters_full_shape(self, params_2q):
        g = Geope(params_2q, history=History())
        n = g.params.basis.lie_algebra_dim
        new_params = np.zeros((g.params.piecewise_steps, n))
        fid = g.add_parameters(new_params)
        assert 0 <= fid <= 1
        assert len(g.history) == 2

    def test_add_parameters_proj_drift_shape(self, params_2q):
        g = Geope(params_2q)
        n = g.params.proj_drift_basis.lie_algebra_dim
        new_params = np.zeros((g.params.piecewise_steps, n))
        fid = g.add_parameters(new_params)
        assert 0 <= fid <= 1

    def test_add_parameters_projected_shape(self, params_2q):
        g = Geope(params_2q)
        n = g.params.projected_basis.lie_algebra_dim
        new_params = np.zeros((g.params.piecewise_steps, n))
        fid = g.add_parameters(new_params)
        assert 0 <= fid <= 1

    def test_add_parameters_with_fidelity(self, params_2q):
        g = Geope(params_2q)
        n = g.params.basis.lie_algebra_dim
        new_params = np.zeros((g.params.piecewise_steps, n))
        g.add_parameters(new_params, fidelity=0.75, step_size=0.1)
        assert g.params.fidelity == 0.75
        assert g.step_size == 0.1

    def test_add_parameters_step_tracking(self, params_2q):
        g = Geope(params_2q, history=History())
        n = g.params.basis.lie_algebra_dim
        for _ in range(3):
            g.add_parameters(np.zeros((g.params.piecewise_steps, n)))
        assert len(g.history) == 4  # initial + 3
        assert g.history.steps[-1] == 3

    # --- constraints ------------------------------------------------------

    def test_init_with_constraints(self, cnot, full_basis_2q, projected_basis_2q):
        n_proj = projected_basis_2q.lie_algebra_dim
        constraint = np.zeros(n_proj)
        constraint[0] = 1
        constraint[1] = 1
        p = _params_2q(
            cnot, full_basis_2q, projected_basis_2q, constraints=[constraint]
        )
        g = Geope(p)
        assert g.constraint_expander is not None
        assert g.constraint_expander.shape[0] == n_proj
        # One constraint merges two params ⇒ one fewer column
        assert g.constraint_expander.shape[1] == n_proj - 1

    def test_no_constraint_expander_is_none(self, params_2q):
        g = Geope(params_2q)
        assert g.constraint_expander is None

    # --- with drift -------------------------------------------------------

    def test_init_with_drift(self, cnot, full_basis_2q, projected_basis_2q):
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I2 = np.eye(2, dtype=complex)
        drift_basis = Basis(
            np.stack([np.kron(Z, I2), np.kron(I2, Z)]), labels=["ZI", "IZ"]
        )
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q, drift_basis=drift_basis)
        g = Geope(p)
        assert 0 <= g.params.fidelity <= 1

    def test_init_with_drift_custom_params(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I2 = np.eye(2, dtype=complex)
        drift_basis = Basis(
            np.stack([np.kron(Z, I2), np.kron(I2, Z)]), labels=["ZI", "IZ"]
        )
        p = _params_2q(
            cnot,
            full_basis_2q,
            projected_basis_2q,
            drift_basis=drift_basis,
            drift_values=[0.5, 0.5],
        )
        g = Geope(p)
        assert np.allclose(g.drift_parameters, [0.5, 0.5])

    # --- gram_schmidt (via optimize when geodesic gives negative update) --

    def test_gram_schmidt_seeded_reproducible(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        # The Gram-Schmidt fallback draws from a seeded per-instance RNG, so two
        # runs with the same seed produce identical fidelity trajectories, while
        # a different seed yields a different one (confirming the fallback fires).
        def run(seed):
            p = _params_2q(cnot, full_basis_2q, projected_basis_2q, seed=seed)
            g = Geope(p, history=History())
            g.optimize(max_steps=80, precision=0.9999)
            return [float(f) for f in g.history.fidelities]

        assert run(42) == run(42)
        assert run(42) != run(7)

    def _drifted_params(self):
        """Controls on x/y, drift on zz — disjoint, with a non-unit drift value."""
        return Parameters(
            basis=construct_full_pauli_basis(2),
            control={1: ["x", "y"], 2: ["x", "y"]},
            drift={(1, 2): ["zz"]},
            drift_values={(1, 2): {"zz": 0.9}},
            target=np.array(
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=complex,
            ),
            piecewise_steps=3,
            seed=0,
        )

    def _projected_direction(self, p, scale=0.3):
        rng = np.random.default_rng(0)
        coeffs = np.zeros((p.piecewise_steps, p.proj_drift_basis.lie_algebra_dim))
        n_proj = int(p.proj_indices_projdrift_basis.sum())
        coeffs[:, p.proj_indices_projdrift_basis] = (
            rng.normal(size=(p.piecewise_steps, n_proj)) * scale
        )
        return coeffs

    def test_gram_schmidt_fidelity_matches_returned_parameters(self):
        # Regression for issue #27: the fallback evaluated the trial pulse with
        # the drift added a second time, so the reported fidelity described a
        # different pulse from the one it returned.
        p = self._drifted_params()
        g = Geope(p)
        new_params, reported, _ = g.gram_schmidt(self._projected_direction(p))
        true_fid = float(
            p.fid_U_fn(p.compute_U_fn(jnp.array(new_params, dtype=jnp.complex128)))
        )
        assert np.isclose(float(reported), true_fid, rtol=0, atol=1e-12)

    def test_gram_schmidt_leaves_drift_untouched(self):
        # The direction is a displacement on the projected columns only; the
        # drift columns must come back exactly as configured.
        p = self._drifted_params()
        g = Geope(p)
        new_params, _, _ = g.gram_schmidt(self._projected_direction(p))
        assert np.allclose(
            new_params[:, p.drift_indices_projdrift_basis],
            np.array(p.drift_parameters),
            atol=1e-15,
        )

    # --- null-space passes now live on Gecko, not Geope ------------------

    def test_smooth_is_callable(self, params_2q):
        gk = Gecko(Geope(params_2q).params)
        assert callable(gk.smooth)

    def test_bound_is_callable(self, params_2q):
        gk = Gecko(Geope(params_2q).params)
        assert callable(gk.bound)

    def test_geope_has_no_null_space_methods(self, geope_2q):
        for name in (
            "smooth",
            "smooth_frequency",
            "filter_frequency",
            "speed",
            "length",
            "robust",
            "bound",
        ):
            assert not hasattr(geope_2q, name)

    # --- get_update_linesearch (internal helper exposed on instance) ------

    def test_update_linesearch_returns_callable(self, geope_2q):
        # Built lazily by optimize(); max_steps=0 configures without iterating.
        geope_2q.optimize(max_steps=0)
        assert callable(geope_2q.update_linesearch)

    def test_gammas_and_omegas_returns_callable(self, geope_2q):
        assert callable(geope_2q.params.gammas_and_omegas)

    def test_update_step_returns_callable(self, geope_2q):
        # Built lazily by optimize(); max_steps=0 configures without iterating.
        geope_2q.optimize(max_steps=0)
        assert callable(geope_2q.update_step)
