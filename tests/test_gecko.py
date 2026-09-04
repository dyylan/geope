"""
Tests for geope/gecko.py.

Covers the Gecko null-space post-processor, which moves parameters within the
Jacobian null space to improve a secondary objective while preserving fidelity.

Tested items:
  Functions (geope.gecko):
    - find_null_space
    - piecewise_smoothing
    - piecewise_bounding_mp
    - piecewise_bounding_pg
  Classes:
    - Gecko
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geope.geope import Geope
from geope.gecko import (
    Gecko,
    find_null_space,
    piecewise_smoothing,
    piecewise_bounding_mp,
    piecewise_bounding_pg,
)
from geope.parameters import Parameters
from geope.utils.history import History
from geope.utils import (
    construct_full_pauli_basis,
    construct_Heisenberg_pauli_basis,
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
    param_transform=None,
    n_experimental_params=None,
    manifold=None,
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
        manifold=manifold,
        param_transform=param_transform,
        n_experimental_params=n_experimental_params,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
# Tests — find_null_space
# ---------------------------------------------------------------------------


class TestFindNullSpace:
    def test_rank_deficient(self):
        """Rank-2 matrix in 3-col space ⇒ 1-D null space."""
        omegas = jnp.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        )
        vh, num = find_null_space(omegas, None)
        assert int(num) == 2
        assert vh.shape[0] == 3

    def test_full_rank(self):
        omegas = jnp.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ]
        )
        vh, num = find_null_space(omegas, None)
        assert int(num) == 3

    def test_with_expander(self):
        omegas = jnp.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        )
        expander = jnp.eye(2)
        vh, num = find_null_space(omegas, expander)
        assert int(num) == 2

    def test_all_zero_matrix(self):
        """All-zero matrix has rank 0."""
        omegas = jnp.zeros((1, 3, 4))
        vh, num = find_null_space(omegas, None)
        assert int(num) == 0

    def test_returns_vh_and_num(self):
        omegas = jnp.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
            ]
        )
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
        phi = jnp.array([[0.5, 0.3, 0.1], [0.4, 0.2, 0.6]], dtype=jnp.float64)
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
        result, diff = piecewise_smoothing(
            phi, null_space, expander, smoothing_rate=0.01
        )
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
        result, diff = piecewise_bounding_mp(
            phi, ns, None, bounding_rate=0.01, lower_bounds=lo, upper_bounds=hi
        )
        assert result.shape == phi.shape
        assert diff.shape == ()

    def test_diff_nonnegative(self):
        phi, ns, lo, hi = self._make_inputs()
        _, diff = piecewise_bounding_mp(
            phi, ns, None, bounding_rate=0.01, lower_bounds=lo, upper_bounds=hi
        )
        assert diff >= 0

    def test_with_expander(self):
        phi, ns, lo, hi = self._make_inputs()
        expander = jnp.eye(phi.size, dtype=jnp.float64)
        result, _ = piecewise_bounding_mp(
            phi, ns, expander, bounding_rate=0.01, lower_bounds=lo, upper_bounds=hi
        )
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
        result, val = piecewise_bounding_pg(
            phi, ns, None, bounding_rate=0.01, lower_bounds=lo, upper_bounds=hi
        )
        assert result.shape == phi.shape
        assert val.shape == ()

    def test_within_bounds_zero_cost(self):
        phi, ns, lo, hi = self._make_inputs(phi_val=0.5)
        _, val = piecewise_bounding_pg(
            phi, ns, None, bounding_rate=0.01, lower_bounds=lo, upper_bounds=hi
        )
        assert jnp.isclose(val, 0.0, atol=1e-10)

    def test_outside_bounds_positive_cost(self):
        phi, ns, lo, hi = self._make_inputs(phi_val=2.0)
        _, val = piecewise_bounding_pg(
            phi, ns, None, bounding_rate=0.01, lower_bounds=lo, upper_bounds=hi
        )
        assert val > 0

    def test_with_expander(self):
        phi, ns, lo, hi = self._make_inputs(phi_val=2.0)
        expander = jnp.eye(phi.size, dtype=jnp.float64)
        result, _ = piecewise_bounding_pg(
            phi, ns, expander, bounding_rate=0.01, lower_bounds=lo, upper_bounds=hi
        )
        assert result.shape == phi.shape


# ---------------------------------------------------------------------------
# Tests — Gecko (null-space / auxiliary-cost optimiser)
# ---------------------------------------------------------------------------


class TestGecko:
    # --- construction modes ----------------------------------------------

    def test_from_params(self, params_2q):
        gk = Gecko(params_2q)
        assert gk.params is params_2q

    def test_shares_geope_params(self, geope_2q):
        gk = Gecko(geope_2q.params)
        assert gk.params is geope_2q.params

    def test_reuses_geope_cached_functions(self, geope_2q):
        # Sharing the Parameters reuses the cached (and thus already-compiled)
        # bound manifold instead of rebuilding it.
        gk = Gecko(geope_2q.params)
        assert gk.params.manifold is geope_2q.params.manifold
        assert (
            gk.params.manifold.compute_point is geope_2q.params.manifold.compute_point
        )

    def test_non_parameters_raises(self, geope_2q):
        with pytest.raises(TypeError):
            Gecko(geope_2q)  # a Geope, not its Parameters

    def test_missing_params_raises(self):
        with pytest.raises(TypeError):
            Gecko()

    # --- fidelity preservation + step-count consistency ------------------

    def test_smooth_preserves_fidelity_and_subdivides(self, params_2q):
        g = Geope(params_2q)
        g.optimize(max_steps=400, precision=0.9999)
        f0 = float(g.params.fidelity)
        original_steps = g.params.piecewise_steps

        gk = Gecko(g.params)
        gk.smooth(piecewise_steps_multiplier=3, max_smoothing_steps=30)

        assert abs(float(gk.params.fidelity) - f0) < 5e-3
        new_steps = 3 * original_steps
        # Gecko shares g.params, so subdivision advances the source Geope too.
        assert g.params.piecewise_steps == new_steps
        assert g.params.parameters.shape[0] == new_steps

    def test_params_mode_from_subdivided_params(self, params_2q):
        g = Geope(params_2q)
        g.optimize(max_steps=400, precision=0.9999)
        Gecko(g.params).smooth(piecewise_steps_multiplier=2, max_smoothing_steps=10)
        # A Gecko sized from the subdivided params must construct and run.
        gk2 = Gecko(g.params)
        assert gk2.params.piecewise_steps == g.params.piecewise_steps
        gk2.smooth(piecewise_steps_multiplier=1, max_smoothing_steps=5)

    # --- experimental parameters (param_transform) -----------------------

    def _exp_params(self, cnot, full_basis_2q, projected_basis_2q):
        n_exp = projected_basis_2q.lie_algebra_dim
        return _params_2q(
            cnot,
            full_basis_2q,
            projected_basis_2q,
            param_transform=lambda phi: phi,
            n_experimental_params=n_exp,
        )

    def test_experimental_geope_mode(self, cnot, full_basis_2q, projected_basis_2q):
        params = self._exp_params(cnot, full_basis_2q, projected_basis_2q)
        g = Geope(params)
        g.optimize(max_steps=400, precision=0.9999)
        f0 = float(g.params.fidelity)
        gk = Gecko(g.params)
        assert gk._real_params is True
        gk.speed(parameter_indices=(0,), max_optimization_steps=10)
        assert abs(float(gk.params.fidelity) - f0) < 5e-3

    def test_experimental_params_mode_rewraps(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        params = self._exp_params(cnot, full_basis_2q, projected_basis_2q)
        g = Geope(params)
        g.optimize(max_steps=400, precision=0.9999)
        gk = Gecko(params=g.params)
        assert gk._real_params is True
        # labels are not allowed under param_transform
        with pytest.raises(ValueError):
            gk.speed(parameter_labels=["XX"], max_optimization_steps=5)
