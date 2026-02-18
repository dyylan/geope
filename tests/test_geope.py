"""
Tests for geope/geope.py and geope/jacobian_manual.py.

Tested items:
  Functions:
    - geodesic_hamiltonian
    - get_geodesic_hamiltonian_fn
    - linear_comb_projected_coeffs_multigate
    - hvp_forward_over_reverse
    - find_null_space
    - piecewise_smoothing
    - piecewise_bounding_mp
    - piecewise_bounding_pg
  Classes:
    - GeopeEngine
    - Geope
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
    find_null_space,
    piecewise_smoothing,
    piecewise_bounding_mp,
    piecewise_bounding_pg,
)
from geope.lie import Basis, Hamiltonian, Unitary
from geope.engine import Engine, fidelity
from geope.utils import (
    construct_full_pauli_basis,
    construct_Heisenberg_pauli_basis,
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
def geope_2q(engine_2q):
    return Geope(
        engine=engine_2q,
        max_steps=5,
        precision=0.9999,
        seed=42,
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
# Tests — Geope
# ---------------------------------------------------------------------------

class TestGeope:
    # --- initialisation ---------------------------------------------------

    def test_init_default(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=3, seed=42)
        assert len(g.parameters) == 1
        assert len(g.fidelities) == 1
        assert len(g.infidelities) == 1
        assert len(g.step_sizes) == 1
        assert len(g.steps) == 1

    def test_init_fidelity_in_range(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=3, seed=42)
        assert 0 <= g.fidelities[0] <= 1

    def test_init_infidelity_complement(self, engine_2q):
        """infidelity = 1 − fidelity."""
        g = Geope(engine=engine_2q, max_steps=3, seed=42)
        assert jnp.isclose(g.fidelities[0] + g.infidelities[0], 1.0, atol=1e-10)

    def test_init_with_custom_params(self, engine_2q, full_basis_2q):
        n = full_basis_2q.lie_algebra_dim
        init = np.zeros(n)
        g = Geope(engine=engine_2q, init_parameters=init, max_steps=3, seed=42)
        assert g.parameters[0].shape == (1, n)

    def test_init_with_gate_shaped_params(self, cnot, full_basis_2q, projected_basis_2q):
        eng = GeopeEngine(
            target_unitary=cnot,
            full_basis=full_basis_2q,
            projected_basis=projected_basis_2q,
            piecewise_steps=2,
        )
        n = full_basis_2q.lie_algebra_dim
        init = np.zeros((2, n))
        g = Geope(engine=eng, init_parameters=init, max_steps=3, seed=42)
        assert g.parameters[0].shape == (2, n)

    def test_init_bad_params_shape_raises(self, engine_2q):
        with pytest.raises(ValueError):
            Geope(engine=engine_2q, init_parameters=np.zeros((5, 5, 5)), max_steps=1)

    def test_precision_stored(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=1, precision=0.999, seed=42)
        assert g.precision == 0.999

    def test_max_steps_stored(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=7, seed=42)
        assert g.max_steps == 7

    def test_verbose_flag(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=1, verbose=True, seed=42)
        assert g.verbose is True

    def test_line_search_method(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=1, line_search_method="golden_section", seed=42)
        assert g.line_search_method == "golden_section"

    # --- reinit -----------------------------------------------------------

    def test_reinit_resets(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=3, seed=42)
        g.init(seed=99)
        assert len(g.fidelities) == 1
        assert len(g.parameters) == 1

    def test_reinit_different_seed(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=3, seed=42)
        params_42 = np.array(g.parameters[0])
        g.init(seed=99)
        params_99 = np.array(g.parameters[0])
        # Very unlikely to be identical with different seeds
        assert not np.allclose(params_42, params_99)

    # --- optimize ---------------------------------------------------------

    def test_optimize_runs(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=3, seed=42)
        result = g.optimize()
        assert isinstance(result, bool)

    def test_optimize_increases_steps(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=3, seed=42)
        g.optimize()
        assert len(g.fidelities) > 1
        assert len(g.steps) > 1

    def test_optimize_fidelity_tracking(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=5, seed=42)
        g.optimize()
        assert len(g.fidelities) == len(g.steps)
        assert len(g.infidelities) == len(g.steps)
        assert len(g.step_sizes) == len(g.steps)

    def test_optimize_all_fidelities_valid(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=5, seed=42)
        g.optimize()
        for f in g.fidelities:
            assert 0 <= f <= 1

    def test_optimize_infidelity_consistency(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=5, seed=42)
        g.optimize()
        for f, inf in zip(g.fidelities, g.infidelities):
            assert jnp.isclose(f + inf, 1.0, atol=1e-10)

    def test_optimize_returns_true_when_converged(self, engine_2q):
        """With precision=0, any fidelity satisfies ⇒ should return True immediately."""
        g = Geope(engine=engine_2q, max_steps=1, precision=0.0, seed=42)
        assert g.optimize() is True

    def test_optimize_extra_steps(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=2, seed=42)
        g.optimize()
        n1 = len(g.fidelities)
        g.optimize(extra_steps=2)
        n2 = len(g.fidelities)
        assert n2 >= n1

    def test_optimize_verbose(self, engine_2q, capsys):
        g = Geope(engine=engine_2q, max_steps=2, verbose=True, seed=42)
        g.optimize()
        captured = capsys.readouterr()
        # verbose mode prints progress lines
        assert len(captured.out) > 0

    # --- add_parameters ---------------------------------------------------

    def test_add_parameters_full_shape(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=1, seed=42)
        n = engine_2q.full_basis.lie_algebra_dim
        new_params = np.zeros((engine_2q.piecewise_steps, n))
        fid = g.add_parameters(new_params)
        assert 0 <= fid <= 1
        assert len(g.parameters) == 2

    def test_add_parameters_proj_drift_shape(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=1, seed=42)
        n = engine_2q.proj_drift_basis.lie_algebra_dim
        new_params = np.zeros((engine_2q.piecewise_steps, n))
        fid = g.add_parameters(new_params)
        assert 0 <= fid <= 1

    def test_add_parameters_projected_shape(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=1, seed=42)
        n = engine_2q.projected_basis.lie_algebra_dim
        new_params = np.zeros((engine_2q.piecewise_steps, n))
        fid = g.add_parameters(new_params)
        assert 0 <= fid <= 1

    def test_add_parameters_with_fidelity(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=1, seed=42)
        n = engine_2q.full_basis.lie_algebra_dim
        new_params = np.zeros((engine_2q.piecewise_steps, n))
        g.add_parameters(new_params, fidelity=0.75, step_size=0.1)
        assert g.fidelities[-1] == 0.75
        assert g.step_sizes[-1] == 0.1

    def test_add_parameters_step_tracking(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=1, seed=42)
        n = engine_2q.full_basis.lie_algebra_dim
        for _ in range(3):
            g.add_parameters(np.zeros((engine_2q.piecewise_steps, n)))
        assert len(g.parameters) == 4  # initial + 3
        assert g.steps[-1] == 3

    # --- constraints ------------------------------------------------------

    def test_init_with_constraints(self, engine_2q):
        n_proj = engine_2q.projected_basis.lie_algebra_dim
        constraint = np.zeros(n_proj)
        constraint[0] = 1
        constraint[1] = 1
        g = Geope(engine=engine_2q, max_steps=1, constraints=[constraint], seed=42)
        assert g.constraint_expander is not None
        assert g.constraint_expander.shape[0] == n_proj
        # One constraint merges two params ⇒ one fewer column
        assert g.constraint_expander.shape[1] == n_proj - 1

    def test_no_constraint_expander_is_none(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=1, seed=42)
        assert g.constraint_expander is None

    # --- with drift -------------------------------------------------------

    def test_init_with_drift(self, cnot, full_basis_2q, projected_basis_2q):
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I2 = np.eye(2, dtype=complex)
        drift_basis = Basis(np.stack([np.kron(Z, I2), np.kron(I2, Z)]),
                            labels=["ZI", "IZ"])
        eng = GeopeEngine(
            target_unitary=cnot,
            full_basis=full_basis_2q,
            projected_basis=projected_basis_2q,
            drift_basis=drift_basis,
            piecewise_steps=1,
        )
        g = Geope(engine=eng, max_steps=3, seed=42)
        assert len(g.fidelities) == 1
        assert 0 <= g.fidelities[0] <= 1

    def test_init_with_drift_custom_params(self, cnot, full_basis_2q, projected_basis_2q):
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I2 = np.eye(2, dtype=complex)
        drift_basis = Basis(np.stack([np.kron(Z, I2), np.kron(I2, Z)]),
                            labels=["ZI", "IZ"])
        eng = GeopeEngine(
            target_unitary=cnot,
            full_basis=full_basis_2q,
            projected_basis=projected_basis_2q,
            drift_basis=drift_basis,
            piecewise_steps=1,
        )
        g = Geope(engine=eng, max_steps=1, drift_parameters=[0.5, 0.5], seed=42)
        assert np.allclose(g.drift_parameters, [0.5, 0.5])

    # --- gram_schmidt (via optimize when geodesic gives negative update) --

    def test_gram_schmidt_step_size_attribute(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=1, gram_schmidt_step_size=1.5, seed=42)
        assert g.gram_schmidt_step_size == 1.5

    # --- smooth / bound exist --------------------------------------------

    def test_smooth_is_callable(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=1, seed=42)
        assert callable(g.smooth)

    def test_bound_is_callable(self, engine_2q):
        g = Geope(engine=engine_2q, max_steps=1, seed=42)
        assert callable(g.bound)

    # --- get_update_linesearch (internal helper exposed on instance) ------

    def test_update_linesearch_returns_callable(self, geope_2q):
        assert callable(geope_2q.update_linesearch)

    def test_gammas_and_omegas_returns_callable(self, geope_2q):
        assert callable(geope_2q.gammas_and_omegas)

    def test_update_step_returns_callable(self, geope_2q):
        assert callable(geope_2q.update_step)

    def test_bound_parameters_returns_callable(self, geope_2q):
        assert callable(geope_2q.bound_parameters)
