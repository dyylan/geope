"""
Tests for geope/gecko.py.

Covers the Gecko null-space post-processor, which moves parameters within the
Jacobian null space to improve a secondary objective while preserving fidelity.

Tested items:
  Functions (geope.gecko):
    - find_null_space
    - _null_space_gradient_step
  Classes:
    - Gecko (including ``optimize`` over weighted sums of objectives)

The objectives' costs themselves are covered in ``test_objectives.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from geope.gecko import (
    Gecko,
    _null_space_gradient_step,
    find_null_space,
)
from geope.geope import Geope
from geope.objectives import (
    BoundViolation,
    CustomObjective,
    PeakAmplitude,
    PulseLength,
    Smoothness,
    as_weighted_sum,
)
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
# Tests — the shared null-space gradient step
# ---------------------------------------------------------------------------


def _quartic(phi):
    return jnp.sum(phi**4)


class TestNullSpaceGradientStep:
    """The one step every objective, and every weighted sum, goes through."""

    def _inputs(self, n_gates=3, n_params=4, n_null=5):
        rng = np.random.default_rng(1)
        phi = jnp.asarray(rng.normal(size=(n_gates, n_params)))
        q, _ = np.linalg.qr(rng.normal(size=(phi.size, phi.size)))
        return phi, jnp.asarray(q[:, :n_null])

    def _step(self, objective):
        return _null_space_gradient_step(as_weighted_sum(objective).build(None))

    def test_returns_the_moved_pulse_and_the_costs_at_the_old_one(self):
        phi, ns = self._inputs()
        objective = Smoothness() + 2.0 * CustomObjective(_quartic, "quartic")
        new_phi, total, values = self._step(objective)(phi, ns, None, 0.01)
        assert new_phi.shape == phi.shape
        assert values.shape == (2,)
        smooth = Smoothness().build(None)(phi)
        np.testing.assert_allclose(values, [smooth, _quartic(phi)])
        assert total == pytest.approx(float(smooth + 2.0 * _quartic(phi)))

    def test_moves_a_fixed_length_inside_the_null_space(self):
        phi, ns = self._inputs()
        new_phi, _, _ = self._step(Smoothness())(phi, ns, None, 0.03)
        update = (new_phi - phi).reshape(-1)
        assert float(jnp.linalg.norm(update)) == pytest.approx(0.03)
        # Orthonormal null-space basis: the update is its own projection.
        np.testing.assert_allclose(ns @ (ns.T @ update), update, atol=1e-14)

    def test_direction_of_a_sum_is_the_projected_weighted_gradient(self):
        phi, ns = self._inputs()
        objective = 0.7 * Smoothness() + 0.2 * CustomObjective(_quartic, "quartic")
        new_phi, _, _ = self._step(objective)(phi, ns, None, 0.01)
        gradient = 0.7 * jax.grad(Smoothness().build(None))(phi) + 0.2 * jax.grad(
            _quartic
        )(phi)
        direction = -(ns @ (ns.T @ gradient.reshape(-1)))
        expected = 0.01 * direction / jnp.linalg.norm(direction)
        np.testing.assert_allclose((new_phi - phi).reshape(-1), expected, atol=1e-14)

    def test_with_expander(self):
        phi, ns = self._inputs()
        expander = jnp.eye(phi.size)
        new_phi, _, _ = self._step(Smoothness())(phi, ns, expander, 0.01)
        assert new_phi.shape == phi.shape


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
# Tests — Gecko.optimize over weighted sums of objectives
# ---------------------------------------------------------------------------

_CNOT = np.eye(4)[[0, 1, 3, 2]].astype(complex)
_XX = 4  # column of "XX" in the Heisenberg projected basis


def _four_segment_params(init_values=None):
    return _params_2q(
        _CNOT,
        construct_full_pauli_basis(2),
        construct_Heisenberg_pauli_basis(2),
        piecewise_steps=4,
        init_values=init_values,
    )


@pytest.fixture(scope="module")
def solved_pulse():
    """A converged 4-segment CNOT pulse, solved once for the optimize tests."""
    params = _four_segment_params()
    Geope(params).optimize(max_steps=200, precision=0.999)
    return np.array(params.parameters)


class TestOptimize:
    def _gecko(self, solved_pulse):
        return Gecko(_four_segment_params(init_values=solved_pulse))

    def _projected(self, gk):
        return jnp.real(
            jnp.asarray(gk.params.parameters[:, gk.params.projected_indices])
        )

    def test_a_weighted_sum_lowers_its_cost_and_keeps_the_fidelity(self, solved_pulse):
        gk = self._gecko(solved_pulse)
        f0 = float(gk.params.fidelity)
        objective = Smoothness() + 0.1 * PulseLength()
        # A zero-step pass only subdivides, so the costs compare like for like.
        gk.optimize(objective, piecewise_steps_multiplier=2, max_steps=0)
        cost = objective.build(gk.params)
        before = float(cost(self._projected(gk))[0])

        gk.optimize(objective, max_steps=30, diff_tol=0.0)

        assert gk.params.piecewise_steps == 8
        assert float(cost(self._projected(gk))[0]) < before
        assert abs(float(gk.params.fidelity) - f0) < 5e-3
        assert set(gk.costs) == {"smoothness", "pulse_length"}

    def test_a_zero_weight_term_is_reported_but_inert(self, solved_pulse):
        alone, monitored = self._gecko(solved_pulse), self._gecko(solved_pulse)
        alone.optimize(Smoothness(), max_steps=10, diff_tol=0.0)
        monitored.optimize(
            Smoothness() + 0 * PeakAmplitude(parameter_indices=(0,)),
            max_steps=10,
            diff_tol=0.0,
        )
        np.testing.assert_allclose(
            monitored.params.parameters, alone.params.parameters, rtol=0, atol=1e-12
        )
        assert monitored.costs["smoothness"] == alone.costs["smoothness"]
        assert monitored.costs["peak_amplitude"] > 0.0

    def test_only_the_weight_ratios_matter(self, solved_pulse):
        a, b = self._gecko(solved_pulse), self._gecko(solved_pulse)
        a.optimize(Smoothness() + 0.5 * PulseLength(), max_steps=10, diff_tol=0.0)
        b.optimize(4 * Smoothness() + 2 * PulseLength(), max_steps=10, diff_tol=0.0)
        np.testing.assert_allclose(
            b.params.parameters, a.params.parameters, rtol=0, atol=1e-10
        )

    def test_bounds_hold_across_a_subdivision(self, solved_pulse):
        gk = self._gecko(solved_pulse)
        f0 = float(gk.params.fidelity)
        limit = 0.9 * float(jnp.max(jnp.abs(self._projected(gk)[:, _XX])))
        violation = BoundViolation({"xx": (-limit, limit)})
        before = float(violation.build(gk.params)(self._projected(gk)))

        gk.optimize(violation, piecewise_steps_multiplier=2, max_steps=50, diff_tol=0.0)

        assert gk.params.piecewise_steps == 8
        assert float(violation.build(gk.params)(self._projected(gk))) < before
        assert abs(float(gk.params.fidelity) - f0) < 5e-3

    def test_a_custom_objective(self, solved_pulse):
        gk = self._gecko(solved_pulse)
        energy = CustomObjective(lambda phi: jnp.sum(phi**2), name="energy")
        before = float(jnp.sum(self._projected(gk) ** 2))
        gk.optimize(energy, max_steps=10, diff_tol=0.0)
        assert "energy" in gk.costs
        assert float(jnp.sum(self._projected(gk) ** 2)) < before

    def test_a_failed_build_leaves_the_parameters_untouched(self, solved_pulse):
        gk = self._gecko(solved_pulse)
        parameters = gk.params.parameters
        with pytest.raises(ValueError):
            gk.optimize(
                PeakAmplitude(parameter_labels=["QQ"]), piecewise_steps_multiplier=2
            )
        # The subdivision ran before the build and was rolled back.
        assert gk.params.piecewise_steps == 4
        assert gk.params.delta_t == 1.0
        assert gk.params.parameters is parameters

    def test_rejects_a_non_objective(self, solved_pulse):
        gk = self._gecko(solved_pulse)
        with pytest.raises(TypeError, match="Objective"):
            gk.optimize(lambda phi: 0.0)

    def test_experimental_space(self, cnot, full_basis_2q, projected_basis_2q):
        params = _params_2q(
            cnot,
            full_basis_2q,
            projected_basis_2q,
            piecewise_steps=4,
            param_transform=lambda phi: phi,
            n_experimental_params=projected_basis_2q.lie_algebra_dim,
        )
        Geope(params).optimize(max_steps=200, precision=0.999)
        f0 = float(params.fidelity)
        limit = 0.9 * float(jnp.max(jnp.abs(jnp.real(params.parameters[:, 0]))))
        gk = Gecko(params)
        gk.optimize(
            PeakAmplitude(parameter_indices=(1,))
            + BoundViolation({0: (-limit, limit)}),
            piecewise_steps_multiplier=2,
            max_steps=20,
        )
        assert set(gk.costs) == {"peak_amplitude", "bound_violation"}
        assert abs(float(params.fidelity) - f0) < 5e-3
