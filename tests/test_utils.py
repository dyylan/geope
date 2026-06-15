"""
Tests for geope/utils.py.

Tested items:
  Functions:
    - trace_dot_jit
    - traces
    - check_xy_comb
    - check_Heisenberg_comb
    - check_2_local_comb
    - restriction_function
    - restriction_order_function
    - construct_restricted_pauli_basis
    - construct_Heisenberg_pauli_basis
    - construct_two_body_pauli_basis
    - construct_full_pauli_basis
    - creation_annihilation_operators
    - construct_full_spin_boson_basis
    - construct_restricted_spin_boson_basis
    - prepare_random_parameters
    - multikron
    - multicontrol_unitary
    - qft_unitary
    - golden_section_search_np
    - golden_section_search
    - adam_line_search
    - merge_constraints
"""

import pytest
import numpy as np
import scipy.linalg as spla

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geope.utils import (
    trace_dot_jit,
    traces,
    check_xy_comb,
    check_Heisenberg_comb,
    check_2_local_comb,
    restriction_function,
    restriction_order_function,
    construct_restricted_pauli_basis,
    construct_Heisenberg_pauli_basis,
    construct_two_body_pauli_basis,
    construct_full_pauli_basis,
    creation_annihilation_operators,
    construct_full_spin_boson_basis,
    construct_restricted_spin_boson_basis,
    prepare_random_parameters,
    multikron,
    multicontrol_unitary,
    qft_unitary,
    golden_section_search_np,
    golden_section_search,
    adam_line_search,
    merge_constraints,
)
from geope.lie import Basis


# ===================================================================
# Tests — trace_dot_jit
# ===================================================================

class TestTraceDotJit:
    def test_identity(self):
        I = jnp.eye(2, dtype=complex)
        assert jnp.isclose(trace_dot_jit(I, I), 2.0)

    def test_orthogonal_paulis(self):
        X = jnp.array([[0, 1], [1, 0]], dtype=complex)
        Z = jnp.array([[1, 0], [0, -1]], dtype=complex)
        assert jnp.isclose(trace_dot_jit(X, Z), 0.0)

    def test_same_pauli(self):
        X = jnp.array([[0, 1], [1, 0]], dtype=complex)
        assert jnp.isclose(trace_dot_jit(X, X), 2.0)

    def test_zero_matrix(self):
        Z = jnp.zeros((3, 3), dtype=complex)
        assert jnp.isclose(trace_dot_jit(Z, Z), 0.0)


# ===================================================================
# Tests — traces
# ===================================================================

class TestTraces:
    def test_self_trace(self):
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        b = np.stack([X, Y])
        result = traces(b, b)
        assert result.shape == (2, 2)
        # Diagonal should be Tr(X@X)=2, Tr(Y@Y)=2
        assert jnp.isclose(result[0, 0], 2.0)
        assert jnp.isclose(result[1, 1], 2.0)
        # Off-diagonal should be 0
        assert jnp.isclose(result[0, 1], 0.0, atol=1e-10)

    def test_different_size_bases(self):
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        b1 = np.stack([X])
        b2 = np.stack([X, Y, Z])
        result = traces(b1, b2)
        assert result.shape == (1, 3)


# ===================================================================
# Tests — check_xy_comb
# ===================================================================

class TestCheckXYComb:
    def test_single_body(self):
        assert check_xy_comb((1, 0)) is True
        assert check_xy_comb((0, 2)) is True
        assert check_xy_comb((0, 3)) is True

    def test_same_pair_xx(self):
        assert check_xy_comb((1, 1)) is True

    def test_same_pair_yy(self):
        assert check_xy_comb((2, 2)) is True

    def test_zz_rejected(self):
        assert check_xy_comb((3, 3)) is False

    def test_mixed_pair_rejected(self):
        assert check_xy_comb((1, 2)) is False

    def test_three_body_rejected(self):
        assert check_xy_comb((1, 2, 3)) is False

    def test_identity(self):
        # All zeros except one → single body
        assert check_xy_comb((0, 0, 1)) is True


# ===================================================================
# Tests — check_Heisenberg_comb
# ===================================================================

class TestCheckHeisenbergComb:
    def test_single_body(self):
        assert check_Heisenberg_comb((1, 0)) is True
        assert check_Heisenberg_comb((0, 3)) is True

    def test_same_pair(self):
        assert check_Heisenberg_comb((1, 1)) is True
        assert check_Heisenberg_comb((2, 2)) is True
        assert check_Heisenberg_comb((3, 3)) is True

    def test_different_pair_rejected(self):
        assert check_Heisenberg_comb((1, 2)) is False
        assert check_Heisenberg_comb((1, 3)) is False

    def test_three_body_rejected(self):
        assert check_Heisenberg_comb((1, 2, 3)) is False


# ===================================================================
# Tests — check_2_local_comb
# ===================================================================

class TestCheck2LocalComb:
    def test_single_body(self):
        assert check_2_local_comb((1, 0)) is True

    def test_two_body(self):
        assert check_2_local_comb((1, 2)) is True
        assert check_2_local_comb((3, 1)) is True

    def test_three_body_rejected(self):
        assert check_2_local_comb((1, 2, 3)) is False


# ===================================================================
# Tests — restriction_function
# ===================================================================

class TestRestrictionFunction:
    def test_single_restriction(self):
        fn = restriction_function(["x"])
        assert fn((1, 0)) is True
        assert fn((2, 0)) is False

    def test_multiple_restrictions(self):
        fn = restriction_function(["x", "z"])
        assert fn((1, 0)) is True
        assert fn((3, 0)) is True
        assert fn((2, 0)) is False

    def test_pair_restriction(self):
        fn = restriction_function(["xx"])
        assert fn((1, 1)) is True
        assert fn((1, 2)) is False


# ===================================================================
# Tests — restriction_order_function
# ===================================================================

class TestRestrictionOrderFunction:
    def test_single_qubit(self):
        restriction = {1: ["x"], 2: ["z"]}
        fn = restriction_order_function(2, restriction)
        assert fn((1, 0)) is True   # X on qubit 1
        assert fn((0, 3)) is True   # Z on qubit 2
        assert fn((0, 1)) is False  # X on qubit 2

    def test_two_body(self):
        restriction = {(1, 2): ["xz"]}
        fn = restriction_order_function(2, restriction)
        assert fn((1, 3)) is True
        assert fn((3, 1)) is False


# ===================================================================
# Tests — construct_*_pauli_basis
# ===================================================================

class TestConstructFullPauliBasis:
    def test_1q(self):
        b = construct_full_pauli_basis(1)
        assert b.lie_algebra_dim == 3
        assert b.dim == 2

    def test_2q(self):
        b = construct_full_pauli_basis(2)
        assert b.lie_algebra_dim == 15
        assert b.dim == 4

    def test_3q(self):
        b = construct_full_pauli_basis(3)
        assert b.lie_algebra_dim == 63
        assert b.dim == 8

    def test_labels_count(self):
        b = construct_full_pauli_basis(2)
        assert len(b.labels) == 15

    def test_orthogonality(self):
        b = construct_full_pauli_basis(2)
        assert b.verify()


class TestConstructHeisenbergPauliBasis:
    def test_2q(self):
        b = construct_Heisenberg_pauli_basis(2)
        # 6 single-body (3 per qubit) + 3 Heisenberg pairs (XX, YY, ZZ)
        assert b.lie_algebra_dim == 9

    def test_subset_of_full(self):
        full = construct_full_pauli_basis(2)
        heis = construct_Heisenberg_pauli_basis(2)
        assert heis.lie_algebra_dim < full.lie_algebra_dim

    def test_orthogonality(self):
        b = construct_Heisenberg_pauli_basis(2)
        assert b.verify()


class TestConstructTwoBodyPauliBasis:
    def test_2q(self):
        b = construct_two_body_pauli_basis(2)
        # All 2-qubit Paulis are at most 2-body → same as full for 2 qubits
        assert b.lie_algebra_dim == 15

    def test_3q_fewer_than_full(self):
        full = construct_full_pauli_basis(3)
        twob = construct_two_body_pauli_basis(3)
        assert twob.lie_algebra_dim < full.lie_algebra_dim

    def test_orthogonality(self):
        b = construct_two_body_pauli_basis(2)
        assert b.verify()


class TestConstructRestrictedPauliBasis:
    def test_list_restriction(self):
        b = construct_restricted_pauli_basis(2, ["x", "z", "xx", "zz"])
        for label in b.labels:
            non_I = label.replace("I", "")
            assert set(non_I).issubset({"X", "Z"})

    def test_dict_restriction(self):
        restriction = {1: ["x"], 2: ["z"], (1, 2): ["xz"]}
        b = construct_restricted_pauli_basis(2, restriction)
        assert b.lie_algebra_dim == 3

    def test_orthogonality(self):
        b = construct_restricted_pauli_basis(2, ["x", "y", "z", "xx", "yy", "zz"])
        assert b.verify()


# ===================================================================
# Tests — creation_annihilation_operators
# ===================================================================

class TestCreationAnnihilationOperators:
    def test_shapes(self):
        a0, am, ap = creation_annihilation_operators(3)
        assert a0.shape == (4, 4)
        assert am.shape == (4, 4)
        assert ap.shape == (4, 4)

    def test_identity(self):
        a0, _, _ = creation_annihilation_operators(2)
        assert np.allclose(a0, np.eye(3))

    def test_adjoint_relation(self):
        """a_plus = a_minus†."""
        _, am, ap = creation_annihilation_operators(3)
        assert np.allclose(ap, am.T)

    def test_commutation_truncated(self):
        """[a_minus, a_plus] ≈ I for low-lying states."""
        _, am, ap = creation_annihilation_operators(5)
        comm = am @ ap - ap @ am
        # Exact commutation is I only for infinite truncation; check approximate
        assert np.allclose(comm[0, 0], 1.0, atol=1e-10)


# ===================================================================
# Tests — construct_*_spin_boson_basis
# ===================================================================

class TestConstructSpinBosonBasis:
    def test_full_spin_boson_shape(self):
        b = construct_full_spin_boson_basis(1, 1, boson_truncation=2)
        assert isinstance(b, Basis)
        assert b.lie_algebra_dim > 0
        # dim = 2^1 * (2+1) = 6
        assert b.dim == 6

    def test_full_spin_boson_labels(self):
        b = construct_full_spin_boson_basis(1, 1, boson_truncation=2)
        assert len(b.labels) == b.lie_algebra_dim

    def test_restricted_subset(self):
        full = construct_full_spin_boson_basis(1, 1, boson_truncation=2)
        restricted = construct_restricted_spin_boson_basis(1, 1, ["x", "z"], boson_truncation=2)
        assert restricted.lie_algebra_dim <= full.lie_algebra_dim

    def test_restricted_dict(self):
        restriction = {1: ["x", "z"]}
        b = construct_restricted_spin_boson_basis(1, 1, restriction, boson_truncation=2)
        assert b.lie_algebra_dim > 0


# ===================================================================
# Tests — prepare_random_parameters
# ===================================================================

class TestPrepareRandomParameters:
    def test_output_shape(self):
        proj = np.array([True, False, True, False, True])
        result = prepare_random_parameters(proj)
        assert result.shape == proj.shape

    def test_zero_where_not_projected(self):
        proj = np.array([True, False, True, False, True])
        result = prepare_random_parameters(proj, key=jax.random.key(42))
        assert result[1] == 0.0
        assert result[3] == 0.0

    def test_nonzero_where_projected(self):
        proj = np.array([True, False, True])
        result = prepare_random_parameters(proj, key=jax.random.key(42))
        assert result[0] != 0.0
        assert result[2] != 0.0

    def test_seed_reproducibility(self):
        proj = np.array([True, True, True])
        r1 = prepare_random_parameters(proj, key=jax.random.key(42))
        r2 = prepare_random_parameters(proj, key=jax.random.key(42))
        assert np.allclose(r1, r2)

    def test_different_seeds_differ(self):
        proj = np.array([True, True, True])
        r1 = prepare_random_parameters(proj, key=jax.random.key(42))
        r2 = prepare_random_parameters(proj, key=jax.random.key(99))
        assert not np.allclose(r1, r2)

    def test_spread(self):
        proj = np.array([True] * 100)
        result = prepare_random_parameters(proj, spread=0.5, key=jax.random.key(0))
        assert np.all(np.abs(result) <= 0.5 + 1e-10)

    def test_with_expander(self):
        proj = np.array([True, True, True, True])
        expander = np.eye(4, 3)
        result = prepare_random_parameters(proj, expander=expander, key=jax.random.key(42))
        assert result.shape == (4,)


# ===================================================================
# Tests — multikron
# ===================================================================

class TestMultikron:
    def test_two_matrices(self):
        A = np.array([[1, 0], [0, 2]])
        B = np.array([[3, 0], [0, 4]])
        result = multikron([A, B])
        expected = np.kron(A, B)
        assert np.allclose(result, expected)

    def test_three_matrices(self):
        I = np.eye(2)
        result = multikron([I, I, I])
        assert np.allclose(result, np.eye(8))

    def test_single_matrix(self):
        A = np.array([[1, 2], [3, 4]])
        assert np.array_equal(multikron([A]), A)

    def test_shape(self):
        A = np.eye(2)
        B = np.eye(3)
        result = multikron([A, B])
        assert result.shape == (6, 6)


# ===================================================================
# Tests — multicontrol_unitary
# ===================================================================

class TestMulticontrolUnitary:
    def test_single_control(self):
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        result = multicontrol_unitary(X, 1)
        assert result.shape == (4, 4)
        # Should be CNOT-like: top-left is I, bottom-right is X
        assert np.allclose(result[:2, :2], np.eye(2))
        assert np.allclose(result[2:, 2:], X)

    def test_two_controls(self):
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        result = multicontrol_unitary(X, 2)
        assert result.shape == (8, 8)
        # Upper 6×6 block should be identity-like
        assert np.allclose(result[:6, :6], np.eye(6))

    def test_is_unitary(self):
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        result = multicontrol_unitary(H, 1)
        assert np.allclose(result @ result.conj().T, np.eye(4), atol=1e-10)


# ===================================================================
# Tests — qft_unitary
# ===================================================================

class TestQftUnitary:
    def test_shape(self):
        U = qft_unitary(2)
        assert U.shape == (4, 4)

    def test_single_qubit_is_hadamard(self):
        U = qft_unitary(1)
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        assert np.allclose(U, H, atol=1e-10)

    def test_is_unitary(self):
        U = qft_unitary(3)
        assert np.allclose(U @ U.conj().T, np.eye(8), atol=1e-10)

    def test_3q_shape(self):
        U = qft_unitary(3)
        assert U.shape == (8, 8)


# ===================================================================
# Tests — golden_section_search_np
# ===================================================================

class TestGoldenSectionSearchNp:
    def test_returns_tuple(self):
        f = lambda x: (x - 2) ** 2
        result = golden_section_search_np(f, 0, 5, tol=1e-6)
        assert len(result) == 2

    def test_x_within_bounds(self):
        f = lambda x: (x - 2) ** 2
        x, fx = golden_section_search_np(f, 0, 5, tol=1e-6)
        assert 0 <= x <= 5

    def test_f_matches_x(self):
        f = lambda x: (x - 2) ** 2
        x, fx = golden_section_search_np(f, 0, 5, tol=1e-6)
        assert np.isclose(fx, f(x), atol=1e-10)

    def test_narrow_interval(self):
        f = lambda x: x ** 2
        x, fx = golden_section_search_np(f, -0.01, 0.01, tol=1e-8)
        assert -0.01 <= x <= 0.01


# ===================================================================
# Tests — golden_section_search (JAX version)
# ===================================================================

class TestGoldenSectionSearch:
    def test_returns_tuple(self):
        f = lambda x: (x - 2.0) ** 2
        result = golden_section_search(f, 0.0, 5.0, tol=1e-6)
        assert len(result) == 2

    def test_x_within_bounds(self):
        f = lambda x: (x - 2.0) ** 2
        x, fx = golden_section_search(f, 0.0, 5.0, tol=1e-6)
        assert 0.0 <= x <= 5.0

    def test_agrees_with_numpy_version(self):
        f_np = lambda x: (x - 1.5) ** 2
        f_jax = lambda x: (x - 1.5) ** 2
        x_np, _ = golden_section_search_np(f_np, 0, 3, tol=1e-6)
        x_jax, _ = golden_section_search(f_jax, 0.0, 3.0, tol=1e-6)
        assert jnp.abs(x_np - x_jax) < 1e-3

    def test_f_matches_x(self):
        f = lambda x: (x + 1.0) ** 2
        x, fx = golden_section_search(f, -3.0, 1.0, tol=1e-6)
        assert jnp.isclose(fx, f(x), atol=1e-8)


# ===================================================================
# Tests — adam_line_search
# ===================================================================

@pytest.mark.parametrize("fd", [True, False], ids=["adam_fd", "adam_grad"])
class TestAdamLineSearch:
    def test_returns_tuple(self, fd):
        f = lambda x: (x - 2.0) ** 2
        result = adam_line_search(f, 0.0, 5.0, finite_difference=fd)
        assert len(result) == 2

    def test_x_within_bounds(self, fd):
        f = lambda x: (x - 2.0) ** 2
        x, fx = adam_line_search(f, 0.0, 5.0, finite_difference=fd)
        assert 0.0 <= float(x) <= 5.0

    def test_minimises_quadratic(self, fd):
        # interior minimum at x = 2, reachable from t_init=0
        f = lambda x: (x - 2.0) ** 2
        x, fx = adam_line_search(f, 0.0, 5.0, lr=0.05, num_steps=500,
                                 finite_difference=fd)
        assert jnp.isclose(x, 2.0, atol=0.1)
        assert float(fx) < 1e-2
        assert jnp.isclose(fx, f(x), atol=1e-8)

    def test_clips_to_boundary_when_min_outside(self, fd):
        # unconstrained min at x=2, but the interval caps at 0 -> best is x=0
        f = lambda x: (x - 2.0) ** 2
        x, fx = adam_line_search(f, -0.9, 0.0, lr=0.1, num_steps=100,
                                 finite_difference=fd)
        assert -0.9 <= float(x) <= 0.0
        assert float(fx) <= f(0.0) + 1e-6

    def test_returns_best_not_worse_than_start(self, fd):
        # a large lr can overshoot; best-so-far must never exceed f(t_init)
        f = lambda x: (x - 2.0) ** 2
        x, fx = adam_line_search(f, 0.0, 5.0, lr=0.9, num_steps=50,
                                 finite_difference=fd)
        assert jnp.isclose(fx, f(x), atol=1e-8)
        assert float(fx) <= f(0.0) + 1e-9

    def test_jittable(self, fd):
        f = lambda x: (x - 2.0) ** 2
        x, fx = jax.jit(
            lambda: adam_line_search(f, 0.0, 5.0, finite_difference=fd)
        )()
        assert bool(jnp.isfinite(x)) and bool(jnp.isfinite(fx))


def test_adam_fd_and_grad_agree():
    # both gradient modes should converge to the same interior minimum
    f = lambda x: (x - 2.0) ** 2
    x_fd, _ = adam_line_search(f, 0.0, 5.0, lr=0.05, num_steps=500,
                               finite_difference=True)
    x_grad, _ = adam_line_search(f, 0.0, 5.0, lr=0.05, num_steps=500,
                                 finite_difference=False)
    assert jnp.isclose(x_fd, 2.0, atol=0.1)
    assert jnp.isclose(x_grad, 2.0, atol=0.1)
    assert jnp.abs(x_fd - x_grad) < 0.1


# ===================================================================
# Tests — merge_constraints
# ===================================================================

class TestMergeConstraints:
    def test_no_overlap(self):
        c1 = [1, 0, 0]
        c2 = [0, 1, 0]
        result = merge_constraints([c1, c2])
        assert len(result) == 2

    def test_full_overlap_merge(self):
        c1 = [1, 1, 0]
        c2 = [2, 2, 0]
        result = merge_constraints([c1, c2])
        assert len(result) == 1

    def test_partial_overlap_merge(self):
        c1 = [1, 1, 0]
        c2 = [0, 1, 1]
        result = merge_constraints([c1, c2])
        assert len(result) == 1
        merged = result[0]
        assert merged[0] != 0
        assert merged[1] != 0
        assert merged[2] != 0

    def test_inconsistent_raises(self):
        c1 = [1, 1, 0]
        c2 = [1, 2, 0]  # Overlaps but inconsistent ratio
        with pytest.raises(ValueError, match="Inconsistent"):
            merge_constraints([c1, c2])

    def test_single_constraint(self):
        result = merge_constraints([[1, 2, 3]])
        assert len(result) == 1
        assert result[0] == [1, 2, 3]

    def test_three_constraints_chain_merge(self):
        c1 = [1, 1, 0, 0]
        c2 = [0, 1, 1, 0]
        c3 = [0, 0, 1, 1]
        result = merge_constraints([c1, c2, c3])
        assert len(result) == 1

    def test_returns_list_of_lists(self):
        result = merge_constraints([[1, 0], [0, 1]])
        assert isinstance(result, list)
        for r in result:
            assert isinstance(r, list)
