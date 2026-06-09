"""
Tests for geope/geope.py and geope/jacobian_manual.py.

Tested items:
  Functions (geope.geope):
    - geodesic_hamiltonian
    - get_geodesic_hamiltonian_fn
    - linear_comb_projected_coeffs_multigate
    - hvp_forward_over_reverse
  Functions (geope.gecko):
    - find_null_space
    - piecewise_smoothing
    - piecewise_bounding_mp
    - piecewise_bounding_pg
  Classes:
    - GeopeEngine
    - Geope
    - Gecko
  jacobian_manual:
    - Ui / get_Ui_fn
    - scan_single_switch_matmul
    - get_apply_branch
    - scan_branch / get_scan_branch
    - manual_jacobian
    - get_jacobian_manual
"""

import pytest
import numpy as np
import scipy.linalg as spla

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geope.geope import (
    GeopeEngine,
    Geope,
    geodesic_hamiltonian,
    get_geodesic_hamiltonian_fn,
    linear_comb_projected_coeffs_multigate,
    hvp_forward_over_reverse,
)
from geope.gecko import (
    Gecko,
    find_null_space,
    piecewise_smoothing,
    piecewise_bounding_mp,
    piecewise_bounding_pg,
)
from geope.parameters import Parameters
from geope.history import History
from geope.lie import Basis, Hamiltonian, Unitary
from geope.engine import Engine, fidelity
from geope.utils import (
    construct_full_pauli_basis,
    construct_Heisenberg_pauli_basis,
    construct_restricted_pauli_basis,
)


def _params_2q(cnot, full_basis_2q, projected_basis_2q, *,
               drift_basis=None, drift_values=None,
               init_values=None, constraints=None,
               piecewise_steps=1, seed=42, init_spread=0.1,
               pulse_constraints=None, projective=True,
               param_transform=None, n_experimental_params=None):
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
from geope.jacobian_manual import (
    Ui,
    get_Ui_fn,
    scan_single_switch_matmul,
    get_apply_branch,
    scan_branch,
    get_scan_branch,
    manual_jacobian,
    get_jacobian_manual,
)
from geope.dexpm import get_dexpm


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
    """Full 2-qubit Pauli basis (15 elements)."""
    return construct_full_pauli_basis(2)


@pytest.fixture
def projected_basis_2q():
    """Heisenberg 2-qubit basis (9 elements ⊂ 15) — a proper subset of the full basis."""
    return construct_Heisenberg_pauli_basis(2)


@pytest.fixture
def engine_2q(cnot, full_basis_2q, projected_basis_2q):
    return GeopeEngine(
        target_unitary=cnot,
        full_basis=full_basis_2q,
        projected_basis=projected_basis_2q,
        piecewise_steps=1,
    )


@pytest.fixture
def params_2q(cnot, full_basis_2q, projected_basis_2q):
    return _params_2q(cnot, full_basis_2q, projected_basis_2q)


@pytest.fixture
def geope_2q(params_2q):
    return Geope(
        params_2q,
        precision=0.9999,
    )


# ---------------------------------------------------------------------------
# Helpers — small bases for jacobian_manual tests
# ---------------------------------------------------------------------------

def _pauli_basis_1q():
    """Single-qubit Pauli basis (X, Y, Z) — 3 generators, 2×2."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return jnp.stack([X, Y, Z])


# ---------------------------------------------------------------------------
# Tests — jacobian_manual.Ui / get_Ui_fn
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
# Tests — scan_single_switch_matmul
# ---------------------------------------------------------------------------

class TestScanSingleSwitchMatmul:
    def test_applies_gate_when_idx_false(self):
        U = jnp.eye(2, dtype=complex)
        jac = jnp.zeros((2, 2), dtype=complex)
        gate = jnp.array([[0, 1], [1, 0]], dtype=complex)  # X gate
        (U_out, _), _ = scan_single_switch_matmul((U, jac), (False, gate))
        # Should have applied gate: X @ I = X
        assert jnp.allclose(U_out, gate, atol=1e-12)

    def test_applies_jacobian_when_idx_true(self):
        U = jnp.eye(2, dtype=complex)
        jac = jnp.array([[0, -1j], [1j, 0]], dtype=complex)  # Y
        gate = jnp.array([[0, 1], [1, 0]], dtype=complex)    # X (ignored)
        (U_out, _), _ = scan_single_switch_matmul((U, jac), (True, gate))
        assert jnp.allclose(U_out, jac, atol=1e-12)


# ---------------------------------------------------------------------------
# Tests — get_apply_branch
# ---------------------------------------------------------------------------

class TestGetApplyBranch:
    def test_single_gate_identity(self):
        gate = jnp.eye(2, dtype=complex).reshape(1, 2, 2)
        fn = get_apply_branch(gate)
        jac = jnp.array([[1, 0], [0, -1]], dtype=complex)  # Z
        idx = jnp.array([True])
        U_out, _ = fn(idx, jac)
        # With a single identity gate switched to jac: result = jac @ I = jac
        assert U_out.shape == (2, 2)
        assert jnp.allclose(U_out, jac, atol=1e-12)

    def test_two_gates_switch(self):
        X = jnp.array([[0, 1], [1, 0]], dtype=complex)
        I = jnp.eye(2, dtype=complex)
        gates = jnp.stack([X, I])
        fn = get_apply_branch(gates)
        jac = jnp.array([[1, 0], [0, -1]], dtype=complex)
        # Switch 1st gate to jac, 2nd stays I: I @ (jac @ eye) = jac
        idx = jnp.array([True, False])
        U_out, _ = fn(idx, jac)
        assert U_out.shape == (2, 2)
        assert jnp.allclose(U_out, jac, atol=1e-12)

    def test_returns_callable(self):
        gates = jnp.eye(2, dtype=complex).reshape(1, 2, 2)
        assert callable(get_apply_branch(gates))


# ---------------------------------------------------------------------------
# Tests — scan_branch / get_scan_branch
# ---------------------------------------------------------------------------

class TestScanBranch:
    def test_output_shape(self):
        I = jnp.eye(2, dtype=complex)
        gates = I.reshape(1, 2, 2)
        branch_fn = get_apply_branch(gates)
        # jac with last axis = number of parameters
        n_params = 3
        jac = jnp.stack([jnp.eye(2, dtype=complex)] * n_params, axis=-1)
        idx = jnp.array([True])
        result = scan_branch(jac, idx, branch_fn)
        assert result.shape == (2, 2, n_params)

    def test_get_scan_branch_returns_callable(self):
        gates = jnp.eye(2, dtype=complex).reshape(1, 2, 2)
        branch_fn = get_apply_branch(gates)
        fn = get_scan_branch(branch_fn)
        assert callable(fn)


# ---------------------------------------------------------------------------
# Tests — manual_jacobian
# ---------------------------------------------------------------------------

class TestManualJacobian:
    def test_output_shape_single_gate(self):
        basis = _pauli_basis_1q()
        Ui_fn = get_Ui_fn(basis)
        jac_fn = get_dexpm(basis)
        params = jnp.array([[0.1, 0.2, 0.3]])
        result = manual_jacobian(params, Ui_fn, jac_fn)
        # shape: (n_gates, dim, dim, n_params)
        assert result.shape == (1, 2, 2, 3)

    def test_output_shape_multi_gate(self):
        basis = _pauli_basis_1q()
        Ui_fn = get_Ui_fn(basis)
        jac_fn = get_dexpm(basis)
        params = jnp.array([[0.1, 0.2, 0.3],
                            [0.4, 0.5, 0.6]])
        result = manual_jacobian(params, Ui_fn, jac_fn)
        assert result.shape == (2, 2, 2, 3)

    def test_zero_params_derivatives_nonzero(self):
        """At identity, derivatives of expm are the generators themselves."""
        basis = _pauli_basis_1q()
        Ui_fn = get_Ui_fn(basis)
        jac_fn = get_dexpm(basis)
        params = jnp.array([[0.0, 0.0, 0.0]])
        result = manual_jacobian(params, Ui_fn, jac_fn)
        # Should not be all zeros — derivative of expm(i*0) w.r.t. params gives i*basis
        assert not jnp.allclose(result, 0, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests — get_jacobian_manual
# ---------------------------------------------------------------------------

class TestGetJacobianManual:
    def test_returns_callable(self):
        basis = _pauli_basis_1q()
        fn = get_jacobian_manual(basis)
        assert callable(fn)

    def test_call_produces_correct_shape(self):
        basis = _pauli_basis_1q()
        fn = get_jacobian_manual(basis)
        params = jnp.array([[0.1, 0.2, 0.3]])
        result = fn(params)
        assert result.shape == (1, 2, 2, 3)

    def test_matches_manual_jacobian_direct(self):
        basis = _pauli_basis_1q()
        fn = get_jacobian_manual(basis)
        Ui_fn = get_Ui_fn(basis)
        jac_fn = get_dexpm(basis)
        params = jnp.array([[0.5, -0.3, 0.1]])
        assert jnp.allclose(fn(params), manual_jacobian(params, Ui_fn, jac_fn), atol=1e-10)

    def test_agrees_with_jax_jacobian(self):
        """Compare manual jacobian against jax.jacobian for a single gate."""
        basis = _pauli_basis_1q()
        fn_manual = get_jacobian_manual(basis)
        Ui_fn = get_Ui_fn(basis)

        params = jnp.array([[0.4, -0.2, 0.6]], dtype=complex)
        jac_manual = fn_manual(params)  # (1, 2, 2, 3)

        # jax.jacobian over full compute
        def compute_U(p):
            A = jnp.tensordot(p[0], basis, axes=[[-1], [0]])
            return jax.scipy.linalg.expm(1j * A)

        jac_auto = jax.jacobian(compute_U, holomorphic=True)(params)  # (2,2,1,3)
        # manual shape is (1,2,2,3), auto shape is (2,2,1,3) — rearrange
        jac_auto_rearranged = jnp.transpose(jac_auto, (2, 0, 1, 3))  # (1,2,2,3)
        assert jnp.allclose(jac_manual, jac_auto_rearranged, atol=1e-8)


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
        comb_vecs = jnp.array([
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0]],
        ])
        target = jnp.array([0.5, 0.3, 0.1, 0.0])
        result = linear_comb_projected_coeffs_multigate(comb_vecs, target, None)
        assert result.shape == (1, 3)
        assert jnp.allclose(result[0], jnp.array([0.5, 0.3, 0.1]), atol=1e-10)

    def test_with_expander(self):
        comb_vecs = jnp.array([
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0]],
        ])
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
        comb_vecs = jnp.array([
            [[1.0, 0.0],
             [0.0, 1.0]],
        ])
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
        f = lambda x: jnp.sum(x ** 2)
        params = jnp.array([1.0, 2.0, 3.0])
        v = jnp.ones(3)
        result = hvp_forward_over_reverse(f, params, v)
        assert result.shape == params.shape

    def test_identity_hessian(self):
        """f(x) = 0.5 ||x||^2  ⇒  H = I  ⇒  Hv = v."""
        f = lambda x: 0.5 * jnp.sum(x ** 2)
        params = jnp.array([1.0, 2.0])
        v = jnp.array([3.0, 4.0])
        result = hvp_forward_over_reverse(f, params, v)
        assert jnp.allclose(result, v, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests — find_null_space
# ---------------------------------------------------------------------------

class TestFindNullSpace:
    def test_rank_deficient(self):
        """Rank-2 matrix in 3-col space ⇒ 1-D null space."""
        omegas = jnp.array([
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0]],
        ])
        vh, num = find_null_space(omegas, None)
        assert int(num) == 2
        assert vh.shape[0] == 3

    def test_full_rank(self):
        omegas = jnp.array([
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]],
        ])
        vh, num = find_null_space(omegas, None)
        assert int(num) == 3

    def test_with_expander(self):
        omegas = jnp.array([
            [[1.0, 0.0],
             [0.0, 1.0]],
        ])
        expander = jnp.eye(2)
        vh, num = find_null_space(omegas, expander)
        assert int(num) == 2

    def test_all_zero_matrix(self):
        """All-zero matrix has rank 0."""
        omegas = jnp.zeros((1, 3, 4))
        vh, num = find_null_space(omegas, None)
        assert int(num) == 0

    def test_returns_vh_and_num(self):
        omegas = jnp.array([
            [[1.0, 2.0],
             [3.0, 4.0]],
        ])
        vh, num = find_null_space(omegas, None)
        assert vh.ndim == 2
        assert num.ndim == 0  # scalar


# ---------------------------------------------------------------------------
# Tests — piecewise_smoothing
# ---------------------------------------------------------------------------

class TestPiecewiseSmoothing:
    def test_output_shape(self):
        phi = jnp.ones((2, 3), dtype=jnp.float64)
        null_space = jnp.eye(6, 2, dtype=jnp.float64)
        result, diff = piecewise_smoothing(phi, null_space, None, smoothing_rate=0.01)
        assert result.shape == phi.shape
        assert diff.shape == ()

    def test_diff_nonnegative(self):
        phi = jnp.array([[0.5, 0.3, 0.1],
                          [0.4, 0.2, 0.6]], dtype=jnp.float64)
        null_space = jnp.eye(6, 3, dtype=jnp.float64)
        _, diff = piecewise_smoothing(phi, null_space, None, smoothing_rate=0.01)
        assert diff >= 0

    def test_uniform_params_small_diff(self):
        """Identical piecewise parameters ⇒ small diff on cross terms."""
        single = jnp.array([0.5, 0.3, 0.1], dtype=jnp.float64)
        phi = jnp.stack([single, single])
        null_space = jnp.eye(6, 2, dtype=jnp.float64)
        _, diff = piecewise_smoothing(phi, null_space, None, smoothing_rate=0.01)
        assert diff >= 0

    def test_with_expander(self):
        phi = jnp.ones((2, 3), dtype=jnp.float64)
        null_space = jnp.eye(6, 2, dtype=jnp.float64)
        expander = jnp.eye(6, dtype=jnp.float64)
        result, diff = piecewise_smoothing(phi, null_space, expander, smoothing_rate=0.01)
        assert result.shape == phi.shape


# ---------------------------------------------------------------------------
# Tests — piecewise_bounding_mp
# ---------------------------------------------------------------------------

class TestPiecewiseBoundingMp:
    def _make_inputs(self, n_gates=2, n_params=3, phi_val=0.5):
        phi = jnp.full((n_gates, n_params), phi_val, dtype=jnp.float64)
        null_space = jnp.eye(n_gates * n_params, 2, dtype=jnp.float64)
        lower = jnp.zeros((n_gates, n_params), dtype=jnp.float64)
        upper = jnp.ones((n_gates, n_params), dtype=jnp.float64)
        return phi, null_space, lower, upper

    def test_output_shape(self):
        phi, ns, lo, hi = self._make_inputs()
        result, diff = piecewise_bounding_mp(phi, ns, None,
                                             bounding_rate=0.01,
                                             lower_bounds=lo,
                                             upper_bounds=hi)
        assert result.shape == phi.shape
        assert diff.shape == ()

    def test_diff_nonnegative(self):
        phi, ns, lo, hi = self._make_inputs()
        _, diff = piecewise_bounding_mp(phi, ns, None,
                                        bounding_rate=0.01,
                                        lower_bounds=lo,
                                        upper_bounds=hi)
        assert diff >= 0

    def test_with_expander(self):
        phi, ns, lo, hi = self._make_inputs()
        expander = jnp.eye(phi.size, dtype=jnp.float64)
        result, _ = piecewise_bounding_mp(phi, ns, expander,
                                          bounding_rate=0.01,
                                          lower_bounds=lo,
                                          upper_bounds=hi)
        assert result.shape == phi.shape


# ---------------------------------------------------------------------------
# Tests — piecewise_bounding_pg
# ---------------------------------------------------------------------------

class TestPiecewiseBoundingPg:
    def _make_inputs(self, n_gates=2, n_params=3, phi_val=0.5):
        phi = jnp.full((n_gates, n_params), phi_val, dtype=jnp.float64)
        null_space = jnp.eye(n_gates * n_params, 2, dtype=jnp.float64)
        lower = jnp.zeros((n_gates, n_params), dtype=jnp.float64)
        upper = jnp.ones((n_gates, n_params), dtype=jnp.float64)
        return phi, null_space, lower, upper

    def test_output_shape(self):
        phi, ns, lo, hi = self._make_inputs(phi_val=2.0)
        result, val = piecewise_bounding_pg(phi, ns, None,
                                            bounding_rate=0.01,
                                            lower_bounds=lo,
                                            upper_bounds=hi)
        assert result.shape == phi.shape
        assert val.shape == ()

    def test_within_bounds_zero_cost(self):
        phi, ns, lo, hi = self._make_inputs(phi_val=0.5)
        _, val = piecewise_bounding_pg(phi, ns, None,
                                       bounding_rate=0.01,
                                       lower_bounds=lo,
                                       upper_bounds=hi)
        assert jnp.isclose(val, 0.0, atol=1e-10)

    def test_outside_bounds_positive_cost(self):
        phi, ns, lo, hi = self._make_inputs(phi_val=2.0)
        _, val = piecewise_bounding_pg(phi, ns, None,
                                       bounding_rate=0.01,
                                       lower_bounds=lo,
                                       upper_bounds=hi)
        assert val > 0

    def test_with_expander(self):
        phi, ns, lo, hi = self._make_inputs(phi_val=2.0)
        expander = jnp.eye(phi.size, dtype=jnp.float64)
        result, _ = piecewise_bounding_pg(phi, ns, expander,
                                          bounding_rate=0.01,
                                          lower_bounds=lo,
                                          upper_bounds=hi)
        assert result.shape == phi.shape


# ---------------------------------------------------------------------------
# Tests — GeopeEngine
# ---------------------------------------------------------------------------

class TestGeopeEngine:
    def test_init_has_expected_attributes(self, engine_2q):
        for attr in ("compute_U_fn", "fid_U_fn", "jac_fn", "geo_fn",
                      "project_omegas_fn", "infid_fn", "grad_fn"):
            assert hasattr(engine_2q, attr), f"Missing attribute: {attr}"

    def test_gates_stored(self, engine_2q):
        assert engine_2q.piecewise_steps == 1

    def test_projected_indices_shape(self, engine_2q, full_basis_2q, projected_basis_2q):
        assert engine_2q.projected_indices.shape == (full_basis_2q.lie_algebra_dim,)
        assert engine_2q.projected_indices.dtype == bool
        assert engine_2q.projected_indices.sum() == projected_basis_2q.lie_algebra_dim

    def test_no_drift_indices(self, engine_2q):
        assert not np.any(engine_2q.drift_indices)

    def test_proj_drift_basis(self, engine_2q, projected_basis_2q):
        """Without drift, proj_drift_basis should match projected_basis dimensions."""
        assert engine_2q.proj_drift_basis.lie_algebra_dim == projected_basis_2q.lie_algebra_dim

    def test_compute_U_fn_zero_params(self, engine_2q):
        """Zero parameters ⇒ identity unitary."""
        n = engine_2q.proj_drift_basis.lie_algebra_dim
        params = jnp.zeros((1, n), dtype=complex)
        U = engine_2q.compute_U_fn(params)
        assert U.shape == (4, 4)
        assert jnp.allclose(U, jnp.eye(4), atol=1e-10)

    def test_compute_U_fn_returns_unitary(self, engine_2q):
        """Random parameters ⇒ result should be unitary."""
        n = engine_2q.proj_drift_basis.lie_algebra_dim
        rng = np.random.default_rng(0)
        params = jnp.array(rng.standard_normal((1, n)), dtype=complex)
        U = engine_2q.compute_U_fn(params)
        # U U† ≈ I
        assert jnp.allclose(U @ U.conj().T, jnp.eye(4), atol=1e-10)

    def test_fid_U_fn_self(self, engine_2q, cnot):
        """Fidelity of target with itself = 1."""
        assert jnp.isclose(engine_2q.fid_U_fn(cnot), 1.0, atol=1e-10)

    def test_fid_U_fn_identity_less_than_one(self, engine_2q):
        fid = engine_2q.fid_U_fn(jnp.eye(4, dtype=complex))
        assert fid < 1.0

    def test_fid_U_fn_range(self, engine_2q):
        rng = np.random.default_rng(1)
        n = engine_2q.proj_drift_basis.lie_algebra_dim
        params = jnp.array(rng.standard_normal((1, n)), dtype=complex)
        U = engine_2q.compute_U_fn(params)
        fid = engine_2q.fid_U_fn(U)
        assert 0 <= fid <= 1.0

    def test_geo_fn_self(self, engine_2q, cnot):
        result = engine_2q.geo_fn(cnot)
        assert result.shape == (4, 4)
        assert jnp.allclose(result, 0, atol=1e-10)

    def test_project_omegas_fn_shape(self, engine_2q):
        dim = engine_2q.full_basis.dim
        x = jnp.eye(dim, dtype=complex).reshape(1, dim, dim)
        result = engine_2q.project_omegas_fn(x)
        assert result.shape[0] == 1

    def test_jac_fn_callable(self, engine_2q):
        assert callable(engine_2q.jac_fn)

    def test_with_drift_basis(self, cnot, full_basis_2q, projected_basis_2q):
        """Engine with an explicit drift basis."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)
        drift_matrices = np.stack([np.kron(Z, I), np.kron(I, Z)])
        drift_basis = Basis(drift_matrices, labels=["ZI", "IZ"])

        eng = GeopeEngine(
            target_unitary=cnot,
            full_basis=full_basis_2q,
            projected_basis=projected_basis_2q,
            drift_basis=drift_basis,
            piecewise_steps=1,
        )
        assert np.any(eng.drift_indices)
        assert eng.drift_basis is not None

    def test_multiple_gates(self, cnot, full_basis_2q, projected_basis_2q):
        eng = GeopeEngine(
            target_unitary=cnot,
            full_basis=full_basis_2q,
            projected_basis=projected_basis_2q,
            piecewise_steps=3,
        )
        assert eng.piecewise_steps == 3


# ---------------------------------------------------------------------------
# Tests — build_pulse_expander (control-format pulse_constraints)
# ---------------------------------------------------------------------------

class TestBuildPulseExpander:
    """`pulse_constraints` uses the control-format dict, same as `control`."""

    @pytest.fixture(scope="class")
    def engine_3q(self):
        full = construct_full_pauli_basis(3)
        proj = construct_restricted_pauli_basis(3, ['x', 'z', 'zz'])
        eng = GeopeEngine(
            target_unitary=jnp.eye(8, dtype=complex),
            full_basis=full,
            projected_basis=proj,
            piecewise_steps=4,
        )
        return eng, proj

    def test_control_dict_selects_expected_zz_indices(self, engine_3q):
        eng, proj = engine_3q
        labels = list(proj.labels)
        n_proj = proj.lie_algebra_dim
        L = eng.piecewise_steps
        proj_params = np.random.default_rng(0).standard_normal((L, n_proj))

        constraints = {(1, 2): ['zz'], (2, 3): ['zz'], (1, 3): ['zz']}
        E, templates = eng.build_pulse_expander(constraints, False, n_proj, proj_params)

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

    def test_single_qubit_key(self, engine_3q):
        eng, proj = engine_3q
        labels = list(proj.labels)
        n_proj = proj.lie_algebra_dim
        L = eng.piecewise_steps
        proj_params = np.random.default_rng(1).standard_normal((L, n_proj))

        _, templates = eng.build_pulse_expander({1: ['x']}, False, n_proj, proj_params)
        assert set(templates.keys()) == {labels.index("XII")}

    def test_absent_interaction_raises(self, engine_3q):
        eng, proj = engine_3q
        n_proj = proj.lie_algebra_dim
        proj_params = np.zeros((eng.piecewise_steps, n_proj))

        # 'yy' is not in the restricted basis -> strict check raises.
        with pytest.raises(ValueError, match="not present in the basis"):
            eng.build_pulse_expander({(1, 2): ['yy']}, False, n_proj, proj_params)

    def test_wrong_qubit_index_raises(self, engine_3q):
        eng, proj = engine_3q
        n_proj = proj.lie_algebra_dim
        proj_params = np.zeros((eng.piecewise_steps, n_proj))

        # Qubit 4 does not exist on a 3-qubit system.
        with pytest.raises(ValueError, match="not present in the basis"):
            eng.build_pulse_expander({(1, 4): ['zz']}, False, n_proj, proj_params)

    def test_list_form_now_rejected(self, engine_3q):
        eng, proj = engine_3q
        n_proj = proj.lie_algebra_dim
        proj_params = np.zeros((eng.piecewise_steps, n_proj))
        # The legacy list-of-Pauli-labels form is no longer accepted in
        # projected space.
        with pytest.raises(TypeError):
            eng.build_pulse_expander(["ZZI"], False, n_proj, proj_params)


class TestParametersPulseConstraintsValidation:
    """`Parameters` validates a dict `pulse_constraints` at construction."""

    @staticmethod
    def _control_3q():
        return {1: ['x', 'z'], 2: ['x', 'z'], 3: ['x', 'z'],
                (1, 2): ['zz'], (2, 3): ['zz'], (1, 3): ['zz']}

    def test_valid_dict_constructs(self):
        p = Parameters(
            basis=construct_full_pauli_basis(3),
            control=self._control_3q(),
            target=np.eye(8, dtype=complex),
            piecewise_steps=4,
            pulse_constraints={(1, 2): ['zz'], (2, 3): ['zz'], (1, 3): ['zz']},
        )
        assert p.pulse_constraints == {(1, 2): ['zz'], (2, 3): ['zz'], (1, 3): ['zz']}

    def test_absent_interaction_raises_at_construction(self):
        with pytest.raises(ValueError, match="not present in the basis"):
            Parameters(
                basis=construct_full_pauli_basis(3),
                control=self._control_3q(),
                target=np.eye(8, dtype=complex),
                piecewise_steps=4,
                pulse_constraints={(1, 2): ['xx']},  # only zz is controllable
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

    def test_init_with_gate_shaped_params(self, cnot, full_basis_2q, projected_basis_2q):
        n = full_basis_2q.lie_algebra_dim
        init = np.zeros((2, n))
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q,
                       piecewise_steps=2, init_values=init)
        g = Geope(p)
        assert g.params.parameters.shape == (2, n)

    def test_init_bad_params_shape_raises(self, cnot, full_basis_2q, projected_basis_2q):
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q,
                       init_values=np.zeros((5, 5, 5)))
        with pytest.raises(ValueError):
            Geope(p)

    def test_precision_stored(self, params_2q):
        g = Geope(params_2q, precision=0.999)
        assert g.precision == 0.999

    def test_verbose_flag(self, params_2q):
        g = Geope(params_2q, verbose=True)
        assert g.verbose is True

    def test_line_search_method(self, params_2q):
        # line-search config is an optimize() argument; max_steps=0 configures
        # without running an iteration.
        g = Geope(params_2q)
        g.optimize(max_steps=0, line_search_method="golden_section")
        assert g.line_search_method == "golden_section"

    def test_line_search_method_adam_stored(self, params_2q):
        for m in ("adam", "adam_fd", "adam_grad"):
            g = Geope(params_2q)
            g.optimize(max_steps=0, line_search_method=m)
            assert g.line_search_method == m
            assert g.adam_lr == 0.05
            assert g.adam_steps == 3

    def test_line_search_method_adam_custom_hparams(self, params_2q):
        g = Geope(params_2q)
        g.optimize(max_steps=0, line_search_method="adam_fd",
                   adam_lr=0.1, adam_steps=12)
        assert g.adam_lr == 0.1
        assert g.adam_steps == 12

    def test_line_search_unset_before_optimize(self, params_2q):
        # The line-search attributes are unset until optimize() configures them.
        g = Geope(params_2q)
        assert g.line_search_method is None
        assert g.adam_lr is None
        assert g.adam_steps is None

    def test_adam_optimize_valid_fidelities(self, cnot, full_basis_2q, projected_basis_2q):
        # both gradient modes must run inside the real loop and stay valid
        for m in ("adam_fd", "adam_grad"):
            p = _params_2q(cnot, full_basis_2q, projected_basis_2q)
            g = Geope(p, history=History())
            g.optimize(max_steps=5, line_search_method=m)
            for f in g.history.fidelities:
                assert 0 <= f <= 1

    def test_adam_optimize_improves_fidelity(self, cnot, full_basis_2q, projected_basis_2q):
        for m in ("adam_fd", "adam_grad"):
            p = _params_2q(cnot, full_basis_2q, projected_basis_2q)
            g = Geope(p, history=History())
            f0 = float(g.params.fidelity)
            g.optimize(max_steps=60, line_search_method=m)
            assert g.history.best_fidelity > f0

    def test_adam_alias_matches_adam_fd(self, params_2q):
        # "adam" routes to the same finite-difference line search as "adam_fd".
        # Compare the line-search output directly on identical inputs: the full
        # optimize loop has a stochastic Gram-Schmidt fallback (global
        # np.random), so comparing two optimize runs would be non-deterministic.
        g_alias = Geope(params_2q)
        g_alias.optimize(max_steps=0, line_search_method="adam")
        g_fd = Geope(params_2q)
        g_fd.optimize(max_steps=0, line_search_method="adam_fd")
        steps = g_fd.engine.piecewise_steps
        params_arr = params_2q.parameters
        free_params = params_arr[:, g_fd.engine.proj_drift_indices].astype(np.complex128)
        # a real, correctly-shaped search direction (deterministic geodesic step)
        coeffs, *_ = g_fd.update_step(free_params, params_arr, steps)
        _, fid_alias, dt_alias = g_alias.update_linesearch(params_arr, coeffs, steps)
        _, fid_fd, dt_fd = g_fd.update_linesearch(params_arr, coeffs, steps)
        assert jnp.isclose(dt_alias, dt_fd)
        assert jnp.isclose(fid_alias, fid_fd)

    def test_line_search_method_unknown_raises(self, params_2q):
        # unknown methods are rejected when the line search first runs
        g = Geope(params_2q)
        with pytest.raises(ValueError):
            g.optimize(max_steps=1, line_search_method="not_a_method")

    def test_engine_arg_rejected(self, engine_2q):
        """Passing a raw GeopeEngine must raise TypeError now."""
        with pytest.raises(TypeError):
            Geope(engine_2q)

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
        g = Geope(params_2q, history=History(), precision=0.0)
        g.optimize(max_steps=1)
        assert g.history.best_fidelity == max(g.history.fidelities)
        # the current/final answer lives on Parameters
        assert params_2q.fidelity is not None

    def test_optimize_returns_params_when_converged(self, params_2q):
        """With precision=0, optimize converges immediately and returns the Parameters."""
        g = Geope(params_2q, history=History(), precision=0.0)
        result = g.optimize(max_steps=1)
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
        n = g.engine.full_basis.lie_algebra_dim
        new_params = np.zeros((g.engine.piecewise_steps, n))
        fid = g.add_parameters(new_params)
        assert 0 <= fid <= 1
        assert len(g.history) == 2

    def test_add_parameters_proj_drift_shape(self, params_2q):
        g = Geope(params_2q)
        n = g.engine.proj_drift_basis.lie_algebra_dim
        new_params = np.zeros((g.engine.piecewise_steps, n))
        fid = g.add_parameters(new_params)
        assert 0 <= fid <= 1

    def test_add_parameters_projected_shape(self, params_2q):
        g = Geope(params_2q)
        n = g.engine.projected_basis.lie_algebra_dim
        new_params = np.zeros((g.engine.piecewise_steps, n))
        fid = g.add_parameters(new_params)
        assert 0 <= fid <= 1

    def test_add_parameters_with_fidelity(self, params_2q):
        g = Geope(params_2q)
        n = g.engine.full_basis.lie_algebra_dim
        new_params = np.zeros((g.engine.piecewise_steps, n))
        g.add_parameters(new_params, fidelity=0.75, step_size=0.1)
        assert g.params.fidelity == 0.75
        assert g.step_size == 0.1

    def test_add_parameters_step_tracking(self, params_2q):
        g = Geope(params_2q, history=History())
        n = g.engine.full_basis.lie_algebra_dim
        for _ in range(3):
            g.add_parameters(np.zeros((g.engine.piecewise_steps, n)))
        assert len(g.history) == 4  # initial + 3
        assert g.history.steps[-1] == 3

    # --- constraints ------------------------------------------------------

    def test_init_with_constraints(self, cnot, full_basis_2q, projected_basis_2q):
        n_proj = projected_basis_2q.lie_algebra_dim
        constraint = np.zeros(n_proj)
        constraint[0] = 1
        constraint[1] = 1
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q,
                       constraints=[constraint])
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
        drift_basis = Basis(np.stack([np.kron(Z, I2), np.kron(I2, Z)]),
                            labels=["ZI", "IZ"])
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q,
                       drift_basis=drift_basis)
        g = Geope(p)
        assert 0 <= g.params.fidelity <= 1

    def test_init_with_drift_custom_params(self, cnot, full_basis_2q, projected_basis_2q):
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I2 = np.eye(2, dtype=complex)
        drift_basis = Basis(np.stack([np.kron(Z, I2), np.kron(I2, Z)]),
                            labels=["ZI", "IZ"])
        p = _params_2q(cnot, full_basis_2q, projected_basis_2q,
                       drift_basis=drift_basis, drift_values=[0.5, 0.5])
        g = Geope(p)
        assert np.allclose(g.drift_parameters, [0.5, 0.5])

    # --- gram_schmidt (via optimize when geodesic gives negative update) --

    def test_gram_schmidt_step_size_attribute(self, params_2q):
        g = Geope(params_2q, gram_schmidt_step_size=1.5)
        assert g.gram_schmidt_step_size == 1.5

    def test_gram_schmidt_seeded_reproducible(self, cnot, full_basis_2q, projected_basis_2q):
        # The Gram-Schmidt fallback draws from a seeded per-instance RNG, so two
        # runs with the same seed produce identical fidelity trajectories, while
        # a different seed yields a different one (confirming the fallback fires).
        def run(seed):
            p = _params_2q(cnot, full_basis_2q, projected_basis_2q, seed=seed)
            g = Geope(p, history=History(), precision=0.9999)
            g.optimize(max_steps=80)
            return [float(f) for f in g.history.fidelities]

        assert run(42) == run(42)
        assert run(42) != run(7)

    # --- null-space passes now live on Gecko, not Geope ------------------

    def test_smooth_is_callable(self, params_2q):
        gk = Gecko(geope=Geope(params_2q))
        assert callable(gk.smooth)

    def test_bound_is_callable(self, params_2q):
        gk = Gecko(geope=Geope(params_2q))
        assert callable(gk.bound)

    def test_geope_has_no_null_space_methods(self, geope_2q):
        for name in ("smooth", "smooth_frequency", "filter_frequency",
                     "speed", "length", "robust", "bound"):
            assert not hasattr(geope_2q, name)

    # --- get_update_linesearch (internal helper exposed on instance) ------

    def test_update_linesearch_returns_callable(self, geope_2q):
        # Built lazily by optimize(); max_steps=0 configures without iterating.
        geope_2q.optimize(max_steps=0)
        assert callable(geope_2q.update_linesearch)

    def test_gammas_and_omegas_returns_callable(self, geope_2q):
        assert callable(geope_2q.engine.gammas_and_omegas)

    def test_update_step_returns_callable(self, geope_2q):
        # Built lazily by optimize(); max_steps=0 configures without iterating.
        geope_2q.optimize(max_steps=0)
        assert callable(geope_2q.update_step)


# ---------------------------------------------------------------------------
# Tests — Gecko (null-space / auxiliary-cost optimiser)
# ---------------------------------------------------------------------------

class TestGecko:
    # --- construction modes ----------------------------------------------

    def test_from_geope_reuses_engine_and_params(self, geope_2q):
        gk = Gecko(geope=geope_2q)
        assert gk.engine is geope_2q.engine
        assert gk.params is geope_2q.params

    def test_from_params_builds_own_engine(self, params_2q):
        gk = Gecko(params=params_2q)
        assert isinstance(gk.engine, GeopeEngine)
        assert gk.params is params_2q

    def test_both_compatible_reuses_engine(self, geope_2q):
        gk = Gecko(params=geope_2q.params, geope=geope_2q)
        assert gk.engine is geope_2q.engine

    def test_both_incompatible_target_raises(self, geope_2q, identity_4x4,
                                             full_basis_2q, projected_basis_2q):
        other = _params_2q(identity_4x4, full_basis_2q, projected_basis_2q)
        with pytest.raises(ValueError):
            Gecko(params=other, geope=geope_2q)

    def test_both_incompatible_projective_raises(self, geope_2q, cnot,
                                                 full_basis_2q, projected_basis_2q):
        other = _params_2q(cnot, full_basis_2q, projected_basis_2q, projective=False)
        with pytest.raises(ValueError):
            Gecko(params=other, geope=geope_2q)

    def test_neither_raises(self):
        with pytest.raises(ValueError):
            Gecko()

    # --- fidelity preservation + step-count consistency ------------------

    def test_smooth_preserves_fidelity_and_subdivides(self, params_2q):
        g = Geope(params_2q, precision=0.9999)
        g.optimize(max_steps=400)
        f0 = float(g.params.fidelity)
        original_steps = g.engine.piecewise_steps

        gk = Gecko(geope=g)
        gk.smooth(piecewise_steps_multiplier=3, max_smoothing_steps=30)

        assert abs(float(gk.params.fidelity) - f0) < 5e-3
        new_steps = 3 * original_steps
        assert g.params.piecewise_steps == new_steps
        assert g.params.parameters.shape[0] == new_steps
        assert g.engine.piecewise_steps == new_steps

    def test_params_mode_from_subdivided_params(self, params_2q):
        g = Geope(params_2q, precision=0.9999)
        g.optimize(max_steps=400)
        Gecko(geope=g).smooth(piecewise_steps_multiplier=2, max_smoothing_steps=10)
        # A fresh engine sized from the subdivided params must construct and run.
        gk2 = Gecko(params=g.params)
        assert gk2.engine.piecewise_steps == g.params.piecewise_steps
        gk2.smooth(piecewise_steps_multiplier=1, max_smoothing_steps=5)

    # --- experimental parameters (param_transform) -----------------------

    def _exp_params(self, cnot, full_basis_2q, projected_basis_2q):
        n_exp = projected_basis_2q.lie_algebra_dim
        return _params_2q(
            cnot, full_basis_2q, projected_basis_2q,
            param_transform=lambda phi: phi,
            n_experimental_params=n_exp,
        )

    def test_experimental_geope_mode(self, cnot, full_basis_2q, projected_basis_2q):
        params = self._exp_params(cnot, full_basis_2q, projected_basis_2q)
        g = Geope(params, precision=0.9999)
        g.optimize(max_steps=400)
        f0 = float(g.params.fidelity)
        gk = Gecko(geope=g)
        assert gk._real_params is True
        gk.speed(parameter_indices=(0,), max_optimization_steps=10)
        assert abs(float(gk.params.fidelity) - f0) < 5e-3

    def test_experimental_params_mode_rewraps(self, cnot, full_basis_2q, projected_basis_2q):
        params = self._exp_params(cnot, full_basis_2q, projected_basis_2q)
        g = Geope(params, precision=0.9999)
        g.optimize(max_steps=400)
        gk = Gecko(params=g.params)
        assert gk._real_params is True
        # labels are not allowed under param_transform
        with pytest.raises(ValueError):
            gk.speed(parameter_labels=["XX"], max_optimization_steps=5)


# ---------------------------------------------------------------------------
# Tests — History (opt-in run log)
# ---------------------------------------------------------------------------

class TestHistory:
    def test_no_history_is_none(self, params_2q):
        g = Geope(params_2q)
        assert g.history is None
        # the final result is still available on Parameters
        g.optimize(max_steps=3)
        assert g.params.fidelity is not None

    def test_default_columns(self, params_2q):
        g = Geope(params_2q, history=History())
        g.optimize(max_steps=3)
        assert set(g.history.keys()) == {
            "parameters", "fidelities", "infidelities", "step_sizes", "steps"}

    def test_attribute_is_item(self, params_2q):
        g = Geope(params_2q, history=History())
        g.optimize(max_steps=3)
        assert g.history.fidelities is g.history["fidelities"]

    def test_unknown_column_raises(self, params_2q):
        g = Geope(params_2q, history=History())
        with pytest.raises(AttributeError):
            _ = g.history.not_a_column

    def test_to_dataframe_length(self, params_2q):
        g = Geope(params_2q, history=History())
        g.optimize(max_steps=4)
        assert len(g.history.to_dataframe()) == len(g.history)

    def test_reset_empties(self, params_2q):
        g = Geope(params_2q, history=History())
        g.optimize(max_steps=3)
        g.history.reset()
        assert len(g.history) == 0
        assert g.history.best_fidelity is None

    def test_backref_to_params(self, params_2q):
        g = Geope(params_2q, history=History())
        assert g.history.params is params_2q

    def test_best_fidelity_is_max(self, params_2q):
        g = Geope(params_2q, history=History())
        g.optimize(max_steps=5)
        assert g.history.best_fidelity == max(g.history.fidelities)

    def test_custom_logging_fn(self, params_2q):
        g = Geope(params_2q,
                  history=History(logging_fn=lambda gg: {"fid": float(gg.params.fidelity)}),
                  precision=0.0)
        g.optimize(max_steps=5)
        # only the custom column is logged
        assert list(g.history.keys()) == ["fid"]
        # the loop still converges (reads params.fidelity, not a column)
        assert g.params.fidelity is not None
        # best-helpers degrade gracefully when the default columns are absent
        assert g.history.best_fidelity is None
        assert g.history.to_dict() == {}

    def test_params_to_dict_reflects_current(self, params_2q):
        g = Geope(params_2q)
        g.optimize(max_steps=5)
        # to_dict over the current params is a non-empty control dict
        assert params_2q.to_dict() != {}

    def test_best_basis_coefficients_requires_backref(self, full_basis_2q):
        # A bare History with logged columns but no back-ref must raise.
        n = full_basis_2q.lie_algebra_dim
        h = History()
        h.logs = {"fidelities": [0.5], "parameters": [np.zeros((1, n))]}
        with pytest.raises(ValueError):
            _ = h.best_basis_coefficients
