"""
Tests for geope/grape.py.

Covers the GRAPE optimiser after its public API was aligned with Geope:
  - Parameters-only constructor (legacy GrapeEngine input removed),
  - an Optimizer config object / max_steps passed to optimize(),
  - Geope-style result model (params.parameters is the current array,
    params.fidelity is a scalar, trajectory in an optional History),
  - the compile memo, driven by the config object's value __eq__,
  - reproducibility from an integer seed and a jax.random.key seed,
  - the param_transform path via the bound params.manifold.

The update rules themselves are pinned in test_optimizers.py; what is tested
here is the loop around them.
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geope.grape import Grape
from geope.line_searches import GoldenSection
from geope.optimizers import Adam, GradientDescent, NewtonRFO, NewtonTRM
from geope.parameters import Parameters
from geope.utils import (
    construct_full_pauli_basis,
    construct_Heisenberg_pauli_basis,
)
from geope.utils.history import History

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
    return construct_full_pauli_basis(2)


@pytest.fixture
def projected_basis_2q():
    return construct_Heisenberg_pauli_basis(2)


def _params(cnot, full_basis_2q, projected_basis_2q, *, seed=42, piecewise_steps=1):
    return Parameters(
        basis=full_basis_2q,
        projected_basis=projected_basis_2q,
        target=cnot,
        piecewise_steps=piecewise_steps,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Tests — constructor / usage parity with Geope
# ---------------------------------------------------------------------------


class TestGrapeConstructor:
    def test_requires_parameters(self):
        with pytest.raises(TypeError):
            Grape("not a Parameters object")

    def test_init_sets_geope_style_state(self, cnot, full_basis_2q, projected_basis_2q):
        p = _params(cnot, full_basis_2q, projected_basis_2q)
        g = Grape(p)
        # params.parameters is the current array, not a list
        assert isinstance(g.params.parameters, np.ndarray)
        assert g.params.parameters.shape == (1, full_basis_2q.lie_algebra_dim)
        # params.fidelity is a scalar
        assert np.ndim(g.params.fidelity) == 0
        # optimiser is built lazily
        assert g.update_step is None


class TestGrapeOptimize:
    def test_optimize_improves_fidelity(self, cnot, full_basis_2q, projected_basis_2q):
        p = _params(cnot, full_basis_2q, projected_basis_2q)
        g = Grape(p, history=History())
        f0 = float(g.params.fidelity)
        # delta=0.1 keeps the trust-region step well-regularised; a pathologically
        # small delta (e.g. 1e-3) makes NewtonTRM take near-pure-Newton steps.
        out = g.optimize(max_steps=100, optimizer=NewtonTRM(delta=0.1))
        assert out is p
        # The *final* iterate, not the best over the trajectory: the Armijo test
        # seeds off ctx.slope, a genuine descent slope, so this is a real
        # sufficient-decrease test and the run cannot end worse than it started.
        assert float(g.params.fidelity) > f0
        # result is still a current array + scalar fidelity
        assert g.params.parameters.shape == (1, full_basis_2q.lie_algebra_dim)
        assert np.ndim(g.params.fidelity) == 0

    def test_newton_is_monotone(self, cnot, full_basis_2q, projected_basis_2q):
        # The acceptance test for the Armijo slope. Pairing the direction with
        # itself gives a *positive* offset on the right-hand side of the
        # sufficient-decrease test, which permits an increase proportional to the
        # step; pairing it with the gradient does not.
        for optimizer in (NewtonTRM(delta=0.1), NewtonRFO(kappa=100.0)):
            p = _params(cnot, full_basis_2q, projected_basis_2q)
            g = Grape(p, history=History())
            g.optimize(max_steps=60, optimizer=optimizer)
            infidelities = np.array([float(f) for f in g.history.infidelities])
            # Skip the step-0 row, recorded before any step was taken.
            assert np.all(np.diff(infidelities[1:]) <= 1e-12), optimizer.name

    def test_fidelity_describes_the_stored_parameters(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        # params.fidelity and params.parameters must describe the same pulse. The
        # reported infidelity is measured at the step just taken, not at the point
        # it was taken from.
        for optimizer in (NewtonTRM(delta=0.1), Adam(0.1), GradientDescent(0.1)):
            p = _params(cnot, full_basis_2q, projected_basis_2q)
            g = Grape(p)
            g.optimize(max_steps=5, optimizer=optimizer)
            recomputed = float(p.manifold.fidelity_at(p.free()))
            assert np.isclose(float(p.fidelity), recomputed, atol=1e-12), optimizer.name

    def test_history_records_trajectory(self, cnot, full_basis_2q, projected_basis_2q):
        p = _params(cnot, full_basis_2q, projected_basis_2q)
        g = Grape(p, history=History())
        g.optimize(max_steps=40, optimizer=NewtonTRM(delta=1e-3))
        assert len(g.history) > 1  # step 0 + iterations
        assert g.history.best_fidelity >= float(g.params.fidelity) - 1e-9

    def test_step_sizes_are_logged(self, cnot, full_basis_2q, projected_basis_2q):
        # History's step_sizes column used to be all zeros for Grape; it now
        # carries the accepted step, which is negative on GEOPE's convention.
        p = _params(cnot, full_basis_2q, projected_basis_2q)
        g = Grape(p, history=History())
        g.optimize(max_steps=5, optimizer=NewtonTRM(delta=0.1))
        assert all(s < 0 for s in g.history.step_sizes[1:])

    @pytest.mark.parametrize(
        "optimizer",
        [GradientDescent(0.1), Adam(0.1), NewtonTRM(delta=0.1), NewtonRFO(kappa=100.0)],
        ids=lambda o: o.name,
    )
    def test_every_rule_runs_end_to_end(
        self, optimizer, cnot, full_basis_2q, projected_basis_2q
    ):
        p = _params(cnot, full_basis_2q, projected_basis_2q)
        g = Grape(p, history=History())
        f0 = float(g.params.fidelity)
        g.optimize(max_steps=40, optimizer=optimizer)
        assert g.method == optimizer.name
        assert g.history.best_fidelity >= f0
        for f in g.history.fidelities:
            assert 0 <= f <= 1

    def test_defaults_to_newton_trm(self, cnot, full_basis_2q, projected_basis_2q):
        g = Grape(_params(cnot, full_basis_2q, projected_basis_2q))
        g.optimize(max_steps=1)
        assert isinstance(g.optimizer, NewtonTRM)
        assert g.method == "newton_trm"

    def test_non_optimizer_raises(self, cnot, full_basis_2q, projected_basis_2q):
        g = Grape(_params(cnot, full_basis_2q, projected_basis_2q))
        with pytest.raises(TypeError, match="geope.optimizers.Optimizer"):
            g.optimize(max_steps=1, optimizer="nr-trm")
        # A LineSearch tunes a scalar step along a direction GEOPE solved for; it
        # is not an update rule, and the mistake is makeable now that both
        # families are frozen dataclasses passed to an `optimize` keyword.
        with pytest.raises(TypeError, match="geope.optimizers.Optimizer"):
            g.optimize(max_steps=1, optimizer=GoldenSection())


class TestGrapeCompileMemo:
    def test_equal_optimizers_reuse_the_compiled_step(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        g = Grape(_params(cnot, full_basis_2q, projected_basis_2q))
        g.optimize(max_steps=0, optimizer=NewtonTRM(delta=0.1))
        first = g.update_step
        g.optimize(max_steps=0, optimizer=NewtonTRM(delta=0.1))
        assert g.update_step is first

    def test_changed_hyperparameter_rebuilds(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        g = Grape(_params(cnot, full_basis_2q, projected_basis_2q))
        g.optimize(max_steps=0, optimizer=NewtonTRM(delta=0.1))
        first = g.update_step
        g.optimize(max_steps=0, optimizer=NewtonTRM(delta=0.2))
        assert g.update_step is not first

    def test_state_is_reinitialised_each_run(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        # Decoupled from compile reuse, as in Geope.optimize: the memo may reuse
        # the compiled step while Adam's moments restart.
        g = Grape(_params(cnot, full_basis_2q, projected_basis_2q))
        g.optimize(max_steps=3, optimizer=Adam(0.1))
        assert int(g.optimizer_state["count"]) == 3
        g.optimize(max_steps=2, optimizer=Adam(0.1))
        assert int(g.optimizer_state["count"]) == 2

    def test_optimizer_unset_before_optimize(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        g = Grape(_params(cnot, full_basis_2q, projected_basis_2q))
        assert g.optimizer is None
        assert g.optimizer_state is None
        assert g.method is None


class TestGrapeReproducibility:
    def test_same_seed_same_result(self, cnot, full_basis_2q, projected_basis_2q):
        g1 = Grape(_params(cnot, full_basis_2q, projected_basis_2q, seed=7))
        g1.optimize(max_steps=30, optimizer=NewtonTRM(delta=1e-3))
        g2 = Grape(_params(cnot, full_basis_2q, projected_basis_2q, seed=7))
        g2.optimize(max_steps=30, optimizer=NewtonTRM(delta=1e-3))
        assert np.allclose(g1.params.parameters, g2.params.parameters)
        assert np.isclose(float(g1.params.fidelity), float(g2.params.fidelity))

    def test_different_seed_differs(self, cnot, full_basis_2q, projected_basis_2q):
        g1 = Grape(_params(cnot, full_basis_2q, projected_basis_2q, seed=7))
        g2 = Grape(_params(cnot, full_basis_2q, projected_basis_2q, seed=8))
        assert not np.allclose(g1.init_parameters, g2.init_parameters)

    def test_jax_key_seed_matches_int_seed(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        g_int = Grape(_params(cnot, full_basis_2q, projected_basis_2q, seed=7))
        g_key = Grape(
            _params(cnot, full_basis_2q, projected_basis_2q, seed=jax.random.key(7))
        )
        assert np.allclose(g_int.init_parameters, g_key.init_parameters)


class TestGrapeParamTransform:
    """The param_transform path wraps the manifold's chart, so
    ``params.manifold`` operates in experimental space."""

    def _exp_params(self, cnot, full_basis_2q, projected_basis_2q, seed=42):
        n_exp = projected_basis_2q.lie_algebra_dim
        return Parameters(
            basis=full_basis_2q,
            projected_basis=projected_basis_2q,
            target=cnot,
            piecewise_steps=1,
            seed=seed,
            param_transform=lambda phi: phi,
            n_experimental_params=n_exp,
        )

    def test_param_transform_uses_experimental_space(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        p = self._exp_params(cnot, full_basis_2q, projected_basis_2q)
        g = Grape(p)
        n_exp = projected_basis_2q.lie_algebra_dim
        assert g._real_params is True
        # Experimental space: drift is folded into compute_point, and every column
        # is a free parameter.
        assert g.drift_parameters is None
        assert g._proj_drift_mask().sum() == n_exp
        assert g.params.parameters.shape == (1, n_exp)

    def test_optimize_improves_fidelity(self, cnot, full_basis_2q, projected_basis_2q):
        p = self._exp_params(cnot, full_basis_2q, projected_basis_2q)
        g = Grape(p, history=History())
        f0 = float(g.params.fidelity)
        # Experimental space exercises manifold.hessian's autodiff fallback (the
        # param_transform chart has no second differential), so the Newton path is
        # what to run here.
        g.optimize(max_steps=100, optimizer=NewtonTRM(delta=0.1))
        assert float(g.params.fidelity) > f0
