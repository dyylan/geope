"""
Tests for geope/geope.py and geope/jacobian_propagator.py.

The Gecko tests live in ``test_gecko.py`` and the History tests in
``test_history.py``.

Tested items:
  Functions (geope.geometry):
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
    PROGRESS_RTOL,
)
from geope.line_searches import (
    Adam,
    ApproximateQuadraticArmijo,
    Armijo,
    GoldenSection,
    LineSearch,
    LineSearchResult,
    QuadraticArmijo,
)
from geope.geometry.chart import (
    get_compute_matrices_params_list_fn,
    get_jacobian_fn,
)
from geope.geometry.lie.groups import (
    get_hessian_propagator_fn,
    infidelity,
    infidelity_full,
)
from geope.jax.hessian import get_hessian_fn, hvp_forward_over_reverse
from geope.gecko import Gecko
from geope.parameters import Parameters
from geope.utils.history import History
from geope.geometry.lie import Basis
from geope.geometry.lie.groups import fidelity
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


@dataclasses.dataclass(frozen=True)
class _GeometryProbe(ApproximateQuadraticArmijo):
    """``ApproximateQuadraticArmijo`` that also reports its geometry.

    The geometry is computed inside the jitted update and normally never leaves,
    so this widens the threaded line-search state to carry the fields a test
    wants to assert on. Every context quantity is memoised per step, so reading
    them here costs nothing extra.
    """

    name = "geometry_probe"

    def init(self):
        return {
            **super().init(),
            "q": jnp.asarray(0.0, jnp.float64),
            "q_exact": jnp.asarray(0.0, jnp.float64),
            "rho": jnp.asarray(0.0, jnp.float64),
            "xi_rel": jnp.asarray(0.0, jnp.float64),
        }

    def __call__(self, ctx, a, b, state):
        result = super().__call__(ctx, a, b, state)
        extra = {
            "q": ctx.q,
            "q_exact": ctx.q_exact,
            "rho": ctx.rho,
            "xi_rel": ctx.xi_rel,
        }
        return result._replace(state={**result.state, **extra})


@dataclasses.dataclass(frozen=True)
class _ReportingSearch(LineSearch):
    """Stubs the line-search contract to exercise the loop's progress test.

    Takes a zero step and *reports* ``factor * value0`` as its objective there,
    so what is under test is the loop's accept/reject arithmetic rather than any
    real 1-D minimiser. ``objective`` is a field here (it is a plain class
    attribute on the real searches) so a single stub can play either role.
    """

    name = "reporting"
    factor: float = 1.0
    objective: str = "infidelity"

    def __call__(self, ctx, a, b, state):
        value0 = ctx.F0 if self.objective == "distance" else ctx.infidelity
        return LineSearchResult(
            jnp.asarray(0.0, jnp.float64),
            self.factor * value0,
            {"n_eval": jnp.asarray(1, jnp.int32)},
        )


def _run_with_geometry_probe(params, max_steps):
    """Optimise with the probe, returning one dict of geometry per step."""
    rows = []

    def record(step, history, geope):
        st = geope.line_search_state or {}
        if "q" in st:
            rows.append({k: float(st[k]) for k in ("q", "q_exact", "rho", "xi_rel")})
        return True

    Geope(params, history=History()).optimize(
        max_steps=max_steps, line_search=_GeometryProbe(), callbacks=(record,)
    )
    return rows


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
        def compute_point(p):
            A = jnp.tensordot(p[0], basis, axes=[[-1], [0]])
            return jax.scipy.linalg.expm(1j * A)

        jac_auto = jax.jacobian(compute_point, holomorphic=True)(params)  # (2,2,1,3)
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

        compute_point = get_compute_matrices_params_list_fn(basis)
        jac_auto = get_jacobian_fn(compute_point)(params)  # (4, 4, 3, 15)
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
        compute_point = get_compute_matrices_params_list_fn(basis)
        h = jax.jacfwd(jax.jacrev(compute_point, holomorphic=True), holomorphic=True)
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
    """Reference ``(phi, Dphi[p], D2phi[p,p])`` via jvp-of-jvp through compute_point."""
    compute_point = get_compute_matrices_params_list_fn(basis)
    X = compute_point(params)
    V = jax.jvp(compute_point, (params,), (p,))[1]
    W = jax.jvp(lambda z: jax.jvp(compute_point, (z,), (p,))[1], (params,), (p,))[1]
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
        compute_point = get_compute_matrices_params_list_fn(basis)
        params = jax.random.normal(jax.random.key(52), (3, basis.shape[0])) * 0.3
        p = jax.random.normal(jax.random.key(53), (3, basis.shape[0])) * 0.3
        _, V = get_jvp_propagator(basis)(params.astype(complex), p.astype(complex))
        h = 1e-6
        V_fd = (compute_point(params + h * p) - compute_point(params - h * p)) / (2 * h)
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
        compute_point = get_compute_matrices_params_list_fn(basis)
        params = jax.random.normal(jax.random.key(58), (3, basis.shape[0])) * 0.3
        p = jax.random.normal(jax.random.key(59), (3, basis.shape[0])) * 0.3
        _, _, W = get_hvp_propagator(basis)(params.astype(complex), p.astype(complex))
        h = 1e-4
        W_fd = (
            compute_point(params + h * p)
            - 2 * compute_point(params)
            + compute_point(params - h * p)
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
        compute_point = get_compute_matrices_params_list_fn(basis)
        # GRAPE parameters are real-valued.
        y = jax.random.normal(jax.random.key(17), (G, K)) * 0.3

        infid_U = infidelity if projective else infidelity_full
        infid = lambda x: infid_U(compute_point(x), target)
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


# The geodesic tangent itself is a manifold primitive now
# (``Manifold.log``); its tests live in tests/test_geometry.py.

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
# Tests — linear_comb_projected_coeffs_multigate diagnostics
# ---------------------------------------------------------------------------

_LS_DIAGNOSTIC_KEYS = {"residual", "residual_rel", "rank", "cond"}


class TestLinearCombProjectedCoeffsDiagnostics:
    def test_default_returns_bare_array(self):
        """Without the flag the return value stays a plain array.

        Regression guard for existing callers (the exploration notebooks call
        this positionally and index the result directly).
        """
        comb_vecs = jnp.ones((2, 3, 4))
        target = jnp.ones(4)
        result = linear_comb_projected_coeffs_multigate(comb_vecs, target, None)
        assert isinstance(result, jnp.ndarray)
        assert result.shape == (2, 3)

    def test_returns_solution_and_diagnostics(self):
        comb_vecs = jnp.ones((2, 3, 4))
        target = jnp.ones(4)
        sol, diag = linear_comb_projected_coeffs_multigate(
            comb_vecs, target, None, return_diagnostics=True
        )
        assert sol.shape == (2, 3)
        assert set(diag) == _LS_DIAGNOSTIC_KEYS
        # The solution must be identical to the no-diagnostics call.
        assert jnp.allclose(
            sol, linear_comb_projected_coeffs_multigate(comb_vecs, target, None)
        )

    def test_exactly_solvable_system(self):
        """A consistent system has ~zero residual and full rank."""
        comb_vecs = jnp.array(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            ]
        )
        target = jnp.array([0.5, 0.3, 0.1, 0.0])
        sol, diag = linear_comb_projected_coeffs_multigate(
            comb_vecs, target, None, return_diagnostics=True
        )
        assert jnp.allclose(sol[0], jnp.array([0.5, 0.3, 0.1]), atol=1e-10)
        assert diag["residual"] < 1e-10
        assert diag["residual_rel"] < 1e-10
        assert int(diag["rank"]) == 3

    def test_residual_matches_direct_computation(self):
        """An inconsistent system's residual equals ||A x - b|| recomputed."""
        # Column space spans only e0/e1, so the e2 component is unreachable.
        comb_vecs = jnp.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ]
        )
        target = jnp.array([0.5, 0.3, 0.4])
        sol, diag = linear_comb_projected_coeffs_multigate(
            comb_vecs, target, None, return_diagnostics=True
        )
        A = jnp.concatenate(comb_vecs, axis=0).T
        expected = jnp.linalg.norm(A @ sol.reshape(-1) - target)
        assert jnp.allclose(diag["residual"], expected, atol=1e-10)
        # The unreachable component is exactly the residual here.
        assert jnp.allclose(diag["residual"], 0.4, atol=1e-10)
        assert jnp.allclose(
            diag["residual_rel"], expected / jnp.linalg.norm(target), atol=1e-10
        )

    def test_zero_target_relative_residual_is_zero_not_nan(self):
        """The 0/0 guard: at convergence the geodesic tangent vanishes.

        ``residual_rel`` divides by ``||target||``, which goes to zero as the
        optimisation converges, so it must be defined (0.0) rather than NaN.
        """
        comb_vecs = jnp.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        )
        target = jnp.zeros(2)
        sol, diag = linear_comb_projected_coeffs_multigate(
            comb_vecs, target, None, return_diagnostics=True
        )
        assert jnp.allclose(sol, 0, atol=1e-10)
        assert not jnp.isnan(diag["residual_rel"])
        assert diag["residual_rel"] == 0.0
        assert diag["residual"] < 1e-10

    def test_rank_deficient_input(self):
        """Duplicate columns drop the rank and keep ``cond`` finite."""
        # Two identical omega rows -> the stacked matrix is rank 1.
        comb_vecs = jnp.array(
            [
                [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            ]
        )
        target = jnp.array([1.0, 1.0, 0.0])
        _, diag = linear_comb_projected_coeffs_multigate(
            comb_vecs, target, None, return_diagnostics=True
        )
        assert int(diag["rank"]) < 2
        assert jnp.isfinite(diag["cond"])

    def test_ill_conditioned_reports_large_cond(self):
        comb_vecs = jnp.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1e-8, 0.0]],
            ]
        )
        target = jnp.array([1.0, 1.0, 0.0])
        _, diag = linear_comb_projected_coeffs_multigate(
            comb_vecs, target, None, return_diagnostics=True
        )
        assert diag["cond"] > 1e6

    def test_with_expander(self):
        """The residual is the constrained one (solve is on A @ E)."""
        comb_vecs = jnp.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ]
        )
        target = jnp.array([0.5, 0.3, 0.0])
        # Expander tying the two coefficients together: only the symmetric
        # combination is reachable, so the fit cannot be exact.
        expander = jnp.array([[1.0], [1.0]])
        sol, diag = linear_comb_projected_coeffs_multigate(
            comb_vecs, target, expander, return_diagnostics=True
        )
        A = jnp.concatenate(comb_vecs, axis=0).T
        expected = jnp.linalg.norm(A @ sol.reshape(-1) - target)
        assert jnp.allclose(diag["residual"], expected, atol=1e-10)
        assert diag["residual"] > 1e-3

    def test_diagnostics_under_jit(self):
        """The real call site is jitted, so the diagnostics must be traceable."""
        comb_vecs = jnp.ones((2, 3, 4))
        target = jnp.arange(4, dtype=jnp.float64)

        @jax.jit
        def solve(cv, t):
            return linear_comb_projected_coeffs_multigate(
                cv, t, None, return_diagnostics=True
            )

        sol, diag = solve(comb_vecs, target)
        assert sol.shape == (2, 3)
        assert set(diag) == _LS_DIAGNOSTIC_KEYS
        assert all(jnp.isfinite(diag[k]) for k in ("residual", "residual_rel", "cond"))

    def test_complex_input_gives_real_diagnostics(self):
        """The standard (non-param_transform) path runs in complex128."""
        comb_vecs = jnp.ones((2, 3, 4), dtype=jnp.complex128)
        target = jnp.array([1.0, 1.0j, 0.5, 0.0], dtype=jnp.complex128)
        _, diag = linear_comb_projected_coeffs_multigate(
            comb_vecs, target, None, return_diagnostics=True
        )
        for key in ("residual", "residual_rel", "cond"):
            assert not jnp.iscomplexobj(diag[key]), key
            assert jnp.isfinite(diag[key]), key


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

    # --- non-quadratic Armijo (first-order, no HVP) ----------------------

    def test_armijo_value_semantics(self):
        # Frozen-dataclass value equality (drives the compile memo).
        assert Armijo() == Armijo()
        assert Armijo(c1=1e-3) != Armijo()
        assert Armijo() != QuadraticArmijo()

    def test_armijo_runs_and_threads_n_eval(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        # The first-order Armijo runs end to end, improves fidelity, and threads
        # its per-step evaluation count as an integer state.
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q, piecewise_steps=6)
        g = Geope(p, history=History())
        f0 = float(g.params.fidelity)
        g.optimize(max_steps=60, line_search=Armijo())
        assert isinstance(g.line_search, Armijo)
        assert g.history.best_fidelity > f0
        for f in g.history.fidelities:
            assert 0 <= f <= 1
        # n_eval populated: at least the seed evaluation. F0 comes off the
        # context now, so the probe the old implementation spent is gone.
        assert int(g.line_search_state["n_eval"]) >= 1

    def test_armijo_converges(self, cnot, full_basis_2q, projected_basis_2q):
        # Seeding at the full bracket step and backtracking is enough to drive
        # synthesis to high fidelity — no curvature needed.
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q, piecewise_steps=6)
        g = Geope(p, history=History())
        g.optimize(max_steps=200, precision=0.9999999, line_search=Armijo())
        assert g.history.best_fidelity > 0.999

    def test_armijo_steps_within_bracket(self, cnot, full_basis_2q, projected_basis_2q):
        # Every line-search step must land in [-max_step_size / G, 0]. The
        # Gram-Schmidt fallback is disabled so that every recorded step size
        # really is one the line search chose (the fallback steps by
        # +/- gram_schmidt_step_size, either sign).
        max_step_size, steps = 0.9, 6
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q, piecewise_steps=steps)
        g = Geope(p, history=History())
        g.optimize(
            max_steps=30,
            line_search=Armijo(),
            max_step_size=max_step_size,
            gram_schmidt_step_size=0,
        )
        a = -max_step_size / steps
        for dt in np.asarray(g.history.step_sizes[1:], dtype=float):
            assert a - 1e-12 <= dt <= 0.0

    def test_armijo_works_under_param_transform(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        # The payoff of never calling ctx.geometry(): Armijo runs in the float64
        # param_transform mode, where QuadraticArmijo raises NotImplementedError.
        def transform(phi):
            out = jnp.zeros(full_basis_2q.lie_algebra_dim)
            out = out.at[0].set(jnp.cos(phi[0]))
            out = out.at[1].set(jnp.sin(phi[1]))
            out = out.at[2].set(phi[2] ** 2)
            return out

        def make():
            return Geope(
                _params_2q(
                    cnot,
                    full_basis_2q,
                    projected_basis_2q,
                    piecewise_steps=2,
                    param_transform=transform,
                    n_experimental_params=3,
                ),
                history=History(),
            )

        g = make()
        g.optimize(max_steps=5, line_search=Armijo())
        assert isinstance(g.line_search, Armijo)
        for f in g.history.fidelities:
            assert np.isfinite(f)

        with pytest.raises(NotImplementedError):
            make().optimize(max_steps=5, line_search=QuadraticArmijo())

    # --- residual-aware quadratic Armijo ---------------------------------

    def test_approx_quadratic_armijo_value_semantics(self):
        # Frozen-dataclass value equality (drives the compile memo), and it must
        # not compare equal to the surrogate-curvature variant.
        assert ApproximateQuadraticArmijo() == ApproximateQuadraticArmijo()
        assert ApproximateQuadraticArmijo(c1=1e-3) != ApproximateQuadraticArmijo()
        assert ApproximateQuadraticArmijo() != QuadraticArmijo()

    def test_approx_quadratic_armijo_runs_and_threads_n_eval(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q, piecewise_steps=6)
        g = Geope(p, history=History())
        f0 = float(g.params.fidelity)
        g.optimize(max_steps=60, line_search=ApproximateQuadraticArmijo())
        assert isinstance(g.line_search, ApproximateQuadraticArmijo)
        assert g.history.best_fidelity > f0
        for f in g.history.fidelities:
            assert 0 <= f <= 1
        assert int(g.line_search_state["n_eval"]) >= 1

    def test_approx_quadratic_armijo_converges(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q, piecewise_steps=6)
        g = Geope(p, history=History())
        g.optimize(
            max_steps=200,
            precision=0.9999999,
            line_search=ApproximateQuadraticArmijo(),
        )
        assert g.history.best_fidelity > 0.999

    def test_q_exact_never_exceeds_q(self, cnot, full_basis_2q, projected_basis_2q):
        # K_A <= I, so the exact intrinsic curvature can never exceed the
        # ||Omega||^2 surrogate. Checked on every step of a real run.
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q, piecewise_steps=6)
        rows = _run_with_geometry_probe(p, max_steps=8)
        assert rows  # the probe actually recorded something
        for r in rows:
            assert r["q_exact"] <= r["q"] + 1e-9
            assert np.isfinite(r["q_exact"]) and np.isfinite(r["rho"])
            assert 0.0 <= r["xi_rel"] <= 1.0

    def test_zero_residual_leaves_q_exact_equal_to_q(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        # With 6 x 9 = 54 unknowns against 15 equations the least-squares solve is
        # underdetermined, so it fits the geodesic tangent exactly: the direction
        # is radial, xi_rel vanishes and the two curvatures coincide. This is the
        # regime in which ApproximateQuadraticArmijo is a no-op.
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q, piecewise_steps=6)
        rows = _run_with_geometry_probe(p, max_steps=5)
        for r in rows:
            assert r["xi_rel"] < 1e-6
            assert np.isclose(r["q_exact"], r["q"], rtol=1e-9, atol=1e-9)

    def test_nonzero_residual_separates_the_two_curvatures(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        # One piecewise step gives 9 unknowns against 15 equations, so the
        # geodesic tangent is genuinely unreachable and the residual is non-zero.
        # This is the regime the exact curvature exists for, so assert the
        # correction actually does something here.
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q, piecewise_steps=1)
        rows = _run_with_geometry_probe(p, max_steps=4)
        assert any(r["xi_rel"] > 1e-3 for r in rows), "expected a non-zero residual"
        assert any(
            r["q"] - r["q_exact"] > 1e-9 for r in rows
        ), "exact curvature should differ from the surrogate when xi_rel > 0"

    def test_approx_quadratic_armijo_rejects_param_transform(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        # Geometry-aware, so it inherits QuadraticArmijo's restriction.
        def transform(phi):
            out = jnp.zeros(full_basis_2q.lie_algebra_dim)
            out = out.at[0].set(jnp.cos(phi[0]))
            out = out.at[1].set(jnp.sin(phi[1]))
            out = out.at[2].set(phi[2] ** 2)
            return out

        p = _params_2q(
            cnot,
            full_basis_2q,
            projected_basis_2q,
            piecewise_steps=2,
            param_transform=transform,
            n_experimental_params=3,
        )
        with pytest.raises(NotImplementedError):
            Geope(p).optimize(max_steps=5, line_search=ApproximateQuadraticArmijo())

    # --- accepting on the line search's own objective --------------------

    def test_a_stalled_step_triggers_the_gram_schmidt_fallback(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        # A search that does not move cannot have made progress on any
        # objective, so the fallback must replace the step. Its signature is the
        # step size: +/- gram_schmidt_step_size rather than a bracket step.
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q, piecewise_steps=3)
        g = Geope(p, history=History())
        g.optimize(max_steps=3, line_search=_ReportingSearch(factor=1.0))
        for dt in np.asarray(g.history.step_sizes[1:], dtype=float):
            assert np.isclose(abs(dt), DEFAULT_GRAM_SCHMIDT_STEP_SIZE)

    def test_a_stalled_step_without_the_fallback_does_not_move(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q, piecewise_steps=3)
        g = Geope(p, history=History())
        before = np.array(p.parameters)
        g.optimize(
            max_steps=3,
            line_search=_ReportingSearch(factor=1.0),
            gram_schmidt_step_size=0,
        )
        np.testing.assert_allclose(np.array(p.parameters), before, atol=1e-12)

    def test_progress_below_the_threshold_counts_as_a_stall(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        # A relative improvement under PROGRESS_RTOL is noise, not progress.
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q, piecewise_steps=3)
        g = Geope(p, history=History())
        g.optimize(
            max_steps=2, line_search=_ReportingSearch(factor=1.0 - PROGRESS_RTOL / 100)
        )
        for dt in np.asarray(g.history.step_sizes[1:], dtype=float):
            assert np.isclose(abs(dt), DEFAULT_GRAM_SCHMIDT_STEP_SIZE)

    def test_progress_above_the_threshold_keeps_the_step(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        # Reported improvement well above the threshold: the step is kept, so no
        # fallback runs and the recorded step size is the search's own zero.
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q, piecewise_steps=3)
        g = Geope(p, history=History())
        g.optimize(max_steps=2, line_search=_ReportingSearch(factor=0.5))
        for dt in np.asarray(g.history.step_sizes[1:], dtype=float):
            assert np.isclose(dt, 0.0)

    def test_a_distance_objective_step_is_judged_on_the_distance(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        # The regression the rule exists for: a search minimising the geodesic
        # distance is judged on the distance. Reporting a halved distance keeps
        # the step even though the fidelity did not move at all (dt = 0).
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q, piecewise_steps=3)
        g = Geope(p, history=History())
        g.optimize(
            max_steps=2, line_search=_ReportingSearch(factor=0.5, objective="distance")
        )
        assert g.line_search.objective == "distance"
        for dt in np.asarray(g.history.step_sizes[1:], dtype=float):
            assert np.isclose(dt, 0.0)

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

    # --- least-squares diagnostics ---------------------------------------

    def test_ls_diagnostics_sentinels_before_any_solve(self, params_2q):
        """Available (as sentinels) from construction, before any solve."""
        g = Geope(params_2q)
        assert set(g.ls_diagnostics) == _LS_DIAGNOSTIC_KEYS
        assert np.isnan(g.ls_diagnostics["residual"])
        assert np.isnan(g.ls_diagnostics["residual_rel"])
        assert np.isnan(g.ls_diagnostics["cond"])
        assert g.ls_diagnostics["rank"] == -1

    def test_ls_diagnostics_populated_after_optimize(self, params_2q):
        g = Geope(params_2q)
        g.optimize(max_steps=3)
        diag = g.ls_diagnostics
        assert np.isfinite(diag["residual"])
        assert np.isfinite(diag["residual_rel"])
        assert 0.0 <= diag["residual_rel"] <= 1.0 + 1e-9
        assert diag["rank"] > 0
        # Host-side plain scalars, not device arrays.
        assert isinstance(diag["residual"], float)
        assert isinstance(diag["rank"], int)

    def test_ls_diagnostics_reset_by_init(self, params_2q):
        """init() re-seeds the sentinels so a fresh run never sees stale values."""
        g = Geope(params_2q)
        g.optimize(max_steps=2)
        assert np.isfinite(g.ls_diagnostics["residual"])
        g.init(g.init_parameters, g.drift_parameters, None, 42)
        assert np.isnan(g.ls_diagnostics["residual"])
        assert g.ls_diagnostics["rank"] == -1

    def test_ls_diagnostics_logged_via_logging_fn(self, params_2q):
        """The documented opt-in recipe, and the step-0 raggedness trap."""
        g = Geope(
            params_2q,
            history=History(
                logging_fn=lambda gg: {
                    "fidelities": gg.params.fidelity,
                    "ls_residual_rel": gg.ls_diagnostics["residual_rel"],
                    "ls_rank": gg.ls_diagnostics["rank"],
                    "ls_cond": gg.ls_diagnostics["cond"],
                }
            ),
        )
        g.optimize(max_steps=4)
        # Every column must be equal-length (step 0 records the sentinel rather
        # than starting the column late), else to_dataframe() would raise.
        lengths = {k: len(v) for k, v in g.history.logs.items()}
        assert len(set(lengths.values())) == 1, lengths
        residuals = g.history["ls_residual_rel"]
        assert np.isnan(residuals[0])  # step-0 sentinel: no solve yet
        assert all(np.isfinite(r) for r in residuals[1:])
        assert all(r > 0 for r in g.history["ls_rank"][1:])

    def test_ls_diagnostics_decrease_as_fidelity_converges(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        """The relative residual shrinks toward 0 as the run converges.

        Also the convergence guard end-to-end: the geodesic tangent vanishes as
        fidelity -> 1, so residual_rel must stay finite (not 0/0 NaN) there.
        """
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q)
        g = Geope(
            p,
            history=History(
                logging_fn=lambda gg: {
                    "fidelities": float(gg.params.fidelity),
                    "ls_residual_rel": gg.ls_diagnostics["residual_rel"],
                }
            ),
        )
        g.optimize(max_steps=300)
        assert g.params.fidelity > 0.999
        residuals = g.history["ls_residual_rel"][1:]
        assert all(np.isfinite(r) for r in residuals)
        # Converged: the geodesic direction is now essentially reachable.
        assert residuals[-1] < 1e-2
        assert residuals[-1] < residuals[0]

    def test_ls_diagnostics_with_pulse_constraints(self):
        """The expander_override update_step also threads the diagnostics."""
        proj = construct_restricted_pauli_basis(3, ["x", "z", "zz"])
        p = Parameters(
            basis=construct_full_pauli_basis(3),
            projected_basis=proj,
            target=qft_unitary(3),
            piecewise_steps=4,
            pulse_constraints={(1, 2): ["zz"]},
            seed=7,
            init_spread=0.3,
        )
        g = Geope(p)
        g.optimize(max_steps=3)
        assert np.isfinite(g.ls_diagnostics["residual_rel"])
        assert g.ls_diagnostics["rank"] > 0

    def test_ls_diagnostics_with_param_transform(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        """The float64 (param_transform) path yields real, finite diagnostics."""
        n_exp = 3

        def transform(phi):
            out = jnp.zeros(full_basis_2q.lie_algebra_dim)
            out = out.at[0].set(jnp.cos(phi[0]))
            out = out.at[1].set(jnp.sin(phi[1]))
            out = out.at[2].set(phi[2] ** 2)
            return out

        p = _params_2q(
            cnot,
            full_basis_2q,
            projected_basis_2q,
            piecewise_steps=2,
            param_transform=transform,
            n_experimental_params=n_exp,
        )
        g = Geope(p)
        g.optimize(max_steps=3)
        assert np.isfinite(g.ls_diagnostics["residual"])
        assert np.isfinite(g.ls_diagnostics["residual_rel"])
        assert g.ls_diagnostics["rank"] > 0

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
        # XZ/ZX are outside the Heisenberg projected basis, so control and
        # drift stay disjoint (see Parameters' overlap guard).
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        drift_basis = Basis(
            np.stack([np.kron(X, Z), np.kron(Z, X)]), labels=["XZ", "ZX"]
        )
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q, drift_basis=drift_basis)
        g = Geope(p)
        assert 0 <= g.params.fidelity <= 1

    def test_init_with_drift_custom_params(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        drift_basis = Basis(
            np.stack([np.kron(X, Z), np.kron(Z, X)]), labels=["XZ", "ZX"]
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
            p.manifold.fidelity_at(jnp.asarray(new_params, dtype=jnp.complex128))
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

    # --- the jitted update step (built lazily by optimize) ----------------

    def test_update_step_is_built_by_optimize(self, geope_2q):
        # max_steps=0 configures the line search without iterating.
        assert geope_2q.update_step is None
        geope_2q.optimize(max_steps=0)
        assert callable(geope_2q.update_step)

    def test_manifold_is_bound_and_shared(self, geope_2q):
        m = geope_2q.params.manifold
        assert m.is_bound
        assert geope_2q.params.manifold is m  # cached on Parameters

    def test_update_step_returns_callable(self, geope_2q):
        # Built lazily by optimize(); max_steps=0 configures without iterating.
        geope_2q.optimize(max_steps=0)
        assert callable(geope_2q.update_step)
