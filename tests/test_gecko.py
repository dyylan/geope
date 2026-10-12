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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from geope.gecko import (
    Gecko,
    find_null_space,
    piecewise_bounding_mp,
    piecewise_bounding_pg,
    piecewise_smoothing,
)
from geope.geope import Geope
from geope.grape import Grape
from geope.optimizers import Adam
from geope.parameters import Parameters
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
        _, num = find_null_space(omegas, None)
        assert int(num) == 3

    def test_with_expander(self):
        omegas = jnp.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        )
        expander = jnp.eye(2)
        _, num = find_null_space(omegas, expander)
        assert int(num) == 2

    def test_all_zero_matrix(self):
        """All-zero matrix has rank 0."""
        omegas = jnp.zeros((1, 3, 4))
        _, num = find_null_space(omegas, None)
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
        result, _ = piecewise_smoothing(phi, null_space, expander, smoothing_rate=0.01)
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

    def test_smooth_respects_verbose(self, params_2q, capsys):
        g = Geope(params_2q)
        g.optimize(max_steps=400, precision=0.9999)
        capsys.readouterr()

        # Quiet by default: `verbose` gates every progress line.
        Gecko(g.params).smooth(piecewise_steps_multiplier=2, max_smoothing_steps=5)
        assert capsys.readouterr().out == ""

        Gecko(g.params, verbose=True).smooth(
            piecewise_steps_multiplier=1, max_smoothing_steps=5
        )
        assert "Smoothing" in capsys.readouterr().out

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


# ---------------------------------------------------------------------------
# Tests — delta_t: subdivision divides the duration, not the parameters (#25)
# ---------------------------------------------------------------------------


class TestDeltaT:
    """Subdivision must be exact for any parametrisation.

    ``expm(dt A) == expm((dt/m) A)^m``, so replicating the segments and
    dividing ``delta_t`` preserves the pulse exactly — including under a
    nonlinear ``param_transform``, where dividing the parameter values is
    wrong (``tau(phi/m) != tau(phi)/m``).
    """

    def _nonlinear_params(self, cnot, full_basis_2q, projected_basis_2q, **kw):
        n_proj = projected_basis_2q.lie_algebra_dim

        def transform(phi):
            # Genuinely nonlinear: tau(phi/m) != tau(phi)/m.
            return 1.5 * jnp.sin(phi)

        return _params_2q(
            cnot,
            full_basis_2q,
            projected_basis_2q,
            piecewise_steps=4,
            param_transform=transform,
            n_experimental_params=n_proj,
            **kw,
        )

    def _fresh_fidelity(self, params):
        return float(
            params.manifold.fidelity_at(
                params.free(), jnp.asarray(params.delta_t, jnp.float64)
            )
        )

    def test_subdivision_is_exact_under_a_nonlinear_transform(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        params = self._nonlinear_params(cnot, full_basis_2q, projected_basis_2q)
        Geope(params).optimize(max_steps=200, precision=0.999)
        f0 = self._fresh_fidelity(params)
        total0 = params.total_time

        gk = Gecko(params)
        gk.smooth(piecewise_steps_multiplier=2, max_smoothing_steps=0)
        assert params.delta_t == pytest.approx(0.5)
        assert abs(self._fresh_fidelity(params) - f0) < 1e-10

        # Chained subdivisions stay exact and keep the total time invariant.
        gk.smooth(piecewise_steps_multiplier=3, max_smoothing_steps=0)
        assert params.delta_t == pytest.approx(1.0 / 6.0)
        assert params.total_time == pytest.approx(total0)
        assert abs(self._fresh_fidelity(params) - f0) < 1e-10

    def test_subdivision_with_drift_is_exact(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        from geope.utils import construct_restricted_pauli_basis

        # Disjoint from the Heisenberg projected basis, which owns IZ/ZI/ZZ.
        drift_basis = construct_restricted_pauli_basis(2, {(1, 2): ["xy", "yx"]})
        params = self._nonlinear_params(
            cnot,
            full_basis_2q,
            projected_basis_2q,
            drift_basis=drift_basis,
            drift_values=[0.7, -0.3],
        )
        Geope(params).optimize(max_steps=200, precision=0.999)
        f0 = self._fresh_fidelity(params)
        Gecko(params).smooth(piecewise_steps_multiplier=2, max_smoothing_steps=0)
        # The drift row is scaled by delta_t along with everything else, so a
        # run *with* drift subdivides exactly too.
        assert abs(self._fresh_fidelity(params) - f0) < 1e-10

    def test_smoothing_still_runs_on_the_subdivided_pulse(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        params = self._nonlinear_params(cnot, full_basis_2q, projected_basis_2q)
        Geope(params).optimize(max_steps=200, precision=0.999)
        f0 = self._fresh_fidelity(params)
        Gecko(params).smooth(piecewise_steps_multiplier=2, max_smoothing_steps=20)
        # Null-space smoothing is first order, so allow the usual slow drift.
        assert abs(float(params.fidelity) - f0) < 5e-3

    def test_zero_step_pass_reports_the_true_fidelity(self, params_2q):
        g = Geope(params_2q)
        g.optimize(max_steps=200, precision=0.999)
        f0 = float(params_2q.fidelity)
        Gecko(params_2q).smooth(piecewise_steps_multiplier=2, max_smoothing_steps=0)
        # Previously a subdivide-only pass overwrote the fidelity with 0.0.
        assert float(params_2q.fidelity) == pytest.approx(f0)
        assert params_2q.delta_t == pytest.approx(0.5)

    def test_bound_experimental_space_is_index_keyed(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        n_proj = projected_basis_2q.lie_algebra_dim
        params = _params_2q(
            cnot,
            full_basis_2q,
            projected_basis_2q,
            piecewise_steps=4,
            param_transform=lambda phi: phi,
            n_experimental_params=n_proj,
        )
        Geope(params).optimize(max_steps=200, precision=0.999)
        f0 = float(params.fidelity)
        gk = Gecko(params)
        with pytest.raises(ValueError, match="integer-indexed"):
            gk.bound({"XX": (-0.5, 0.5)}, max_bounding_steps=5)
        bound_value = 0.9 * float(jnp.max(jnp.abs(jnp.real(params.parameters[:, 0]))))

        def violation():
            column = jnp.abs(jnp.real(params.parameters[:, 0]))
            return float(jnp.max(jnp.clip(column - bound_value, 0.0)))

        before = violation()
        gk.bound({0: (-bound_value, bound_value)}, max_bounding_steps=50)
        # Null-space bounding is first order and fidelity-preserving, so it
        # reduces the violation rather than hard-clamping it.
        assert violation() < before
        assert abs(float(params.fidelity) - f0) < 5e-3


# ---------------------------------------------------------------------------
# Tests — delta_t: single-qubit regressions under a nonlinear transform
# ---------------------------------------------------------------------------


def _rabi(phi):
    """Nonlinear transform: tau(phi/m) != tau(phi)/m."""
    return jnp.array([phi[0] * jnp.cos(phi[1]), phi[0] * jnp.sin(phi[1]), 0.0])


def _rx(theta):
    return np.array(
        [
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)],
        ],
        dtype=complex,
    )


def _fidelity_now(p):
    """Fidelity recomputed from live state, independent of stored bookkeeping."""
    return float(p.manifold.fidelity_at(p.free(), jnp.asarray(p.delta_t, jnp.float64)))


def _solved_1q(transform=_rabi, *, steps=4):
    """A converged single-qubit solution in experimental space."""
    params = Parameters(
        basis=construct_full_pauli_basis(1),
        control={1: ["x", "y", "z"]},
        target=_rx(np.pi / 3),
        piecewise_steps=steps,
        param_transform=transform,
        n_experimental_params=2,
        init_spread=0.3,
        seed=0,
    )
    Geope(params).optimize(max_steps=400, precision=1 - 1e-9)
    return params


class TestDeltaTRegressions:
    """Subdivision and a changed delta_t, end to end through every consumer."""

    def test_subdivision_is_exact_in_projected_space_with_drift(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        """Without a transform the drift columns used to be divided by m too."""
        drift_basis = construct_full_pauli_basis(2)
        labels = list(drift_basis.labels)
        keep = [i for i, lab in enumerate(labels) if lab in ("XY", "YZ")]
        params = _params_2q(
            cnot,
            full_basis_2q,
            projected_basis_2q,
            drift_basis=type(drift_basis)(
                np.asarray(drift_basis.basis)[keep], labels=[labels[i] for i in keep]
            ),
            drift_values=np.array([0.2, -0.1]),
            piecewise_steps=2,
        )
        Geope(params)  # writes the drift into the drift columns of `parameters`
        assert np.any(params.parameters[:, params.drift_indices] != 0)
        before = _fidelity_now(params)
        Gecko(params).smooth(piecewise_steps_multiplier=3, max_smoothing_steps=0)
        assert abs(_fidelity_now(params) - before) < 1e-12

    def test_parameters_are_replicated_not_divided(self):
        """phi is untouched; the duration absorbs the 1/m."""
        p = _solved_1q()
        phi_before = np.array(p.parameters)
        Gecko(p).smooth(piecewise_steps_multiplier=2, max_smoothing_steps=0)
        np.testing.assert_array_equal(p.parameters[0], phi_before[0])
        np.testing.assert_array_equal(p.parameters[1], phi_before[0])
        assert p.delta_t == pytest.approx(0.5)

    def test_robust_evaluates_the_subdivided_chart(self):
        """`robust` must score the pulse at the duration after subdivision."""
        p = _solved_1q()
        before = _fidelity_now(p)
        Gecko(p).robust(
            parameter_indices=(0,),
            piecewise_steps_multiplier=2,
            max_optimization_steps=3,
            optimization_rate=1e-3,
        )
        assert float(p.fidelity) == pytest.approx(_fidelity_now(p), abs=1e-12)
        assert abs(_fidelity_now(p) - before) < 5e-3

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
    def test_assigning_a_non_positive_delta_t_raises(self, params_2q, bad):
        with pytest.raises(ValueError, match="delta_t must be positive"):
            params_2q.delta_t = bad
        assert params_2q.delta_t == 1.0

    def test_assigning_delta_t_refreshes_the_live_fidelity(self, geope_2q):
        """The live fidelity describes (parameters, delta_t), so it moves too."""
        p = geope_2q.params
        assert p.fidelity is not None
        p.delta_t = 0.5
        assert float(p.fidelity) == pytest.approx(_fidelity_now(p), abs=1e-13)

    def test_geope_uses_a_new_delta_t(self):
        """A Geope run after ``params.delta_t`` changed optimises the new
        duration: the fidelity it reports matches a fresh evaluation there."""
        p = _solved_1q()
        g = Geope(p)
        g.optimize(max_steps=5)
        p.delta_t = 0.5
        g.optimize(max_steps=1)
        assert float(p.fidelity) == pytest.approx(_fidelity_now(p), abs=1e-9)

    def test_grape_uses_a_new_delta_t(self):
        """Grape reports the fidelity *at* the updated parameters, so a stale
        duration shows up as a mismatch against a fresh evaluation."""
        p = _solved_1q()
        g = Grape(p)
        g.optimize(max_steps=1, optimizer=Adam(learning_rate=0.01))
        p.delta_t = 0.5
        g.optimize(max_steps=1, optimizer=Adam(learning_rate=0.01))
        assert float(p.fidelity) == pytest.approx(_fidelity_now(p), abs=1e-9)

    def test_unlisted_bound_indices_are_unbounded(self):
        p = _solved_1q()
        gk = Gecko(p)
        gk.bound({0: (-0.4, 0.4)}, max_bounding_steps=5)
        assert np.isneginf(np.array(gk.lower_bounds)[:, 1]).all()
        assert np.isposinf(np.array(gk.upper_bounds)[:, 1]).all()
