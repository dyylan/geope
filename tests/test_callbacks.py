"""
Tests for the ``callbacks`` argument on the optimise loops.

Covers the callback hook shared by ``Geope.optimize``, ``Grape.optimize`` and
the ``Gecko`` public methods (which forward into
``Gecko._null_space_optimisation``). A callback has the signature
``callback(step, history, optimizer) -> bool``; all callbacks run at the end of
every step, and the loop stops early if any returns a falsy value.

Also exercises the shared helper in ``geope.utils.callbacks``.
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geope.geope import Geope
from geope.grape import Grape
from geope.gecko import Gecko
from geope.optimizers import Adam, NewtonTRM
from geope.parameters import Parameters
from geope.utils.history import History
from geope.utils.callbacks import normalize_callbacks, run_callbacks
from geope.utils import (
    construct_full_pauli_basis,
    construct_Heisenberg_pauli_basis,
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
# Shared helper
# ---------------------------------------------------------------------------


class TestCallbackHelper:
    def test_normalize_none(self):
        assert normalize_callbacks(None) == ()

    def test_normalize_single_callable(self):
        fn = lambda step, history, opt: True
        assert normalize_callbacks(fn) == (fn,)

    def test_normalize_list_and_tuple(self):
        a = lambda step, history, opt: True
        b = lambda step, history, opt: True
        assert normalize_callbacks([a, b]) == (a, b)
        assert normalize_callbacks((a, b)) == (a, b)

    def test_normalize_rejects_non_callable(self):
        with pytest.raises(TypeError):
            normalize_callbacks(5)
        with pytest.raises(TypeError):
            normalize_callbacks([lambda s, h, o: True, 5])

    def test_run_callbacks_empty_continues(self):
        assert run_callbacks((), 1, None, None) is True

    def test_run_callbacks_all_run_and_stops_on_falsy(self):
        seen = []

        def a(step, history, opt):
            seen.append("a")
            return False  # requests stop

        def b(step, history, opt):
            seen.append("b")
            return True

        # Both callbacks must run even though the first requests a stop.
        assert run_callbacks((a, b), 1, None, None) is False
        assert seen == ["a", "b"]

    def test_run_callbacks_none_return_stops(self):
        assert run_callbacks((lambda s, h, o: None,), 1, None, None) is False


# ---------------------------------------------------------------------------
# Geope
# ---------------------------------------------------------------------------


class TestGeopeCallbacks:
    def test_invocation_and_args(self, cnot, full_basis_2q, projected_basis_2q):
        p = _params(cnot, full_basis_2q, projected_basis_2q)
        g = Geope(p, history=History())
        calls = []

        def cb(step, history, optimizer):
            calls.append((step, history, optimizer))
            return True

        g.optimize(max_steps=5, callbacks=cb, precision=1.0)
        assert len(calls) == 5  # precision=1.0 is unreachable -> runs the budget
        # step increments 1..N
        assert [c[0] for c in calls] == [1, 2, 3, 4, 5]
        # history is the optimiser's History, optimizer is the Geope itself
        assert all(c[1] is g.history for c in calls)
        assert all(c[2] is g for c in calls)

    def test_early_stop(self, cnot, full_basis_2q, projected_basis_2q):
        p = _params(cnot, full_basis_2q, projected_basis_2q)
        g = Geope(p)
        seen = []

        def cb(step, history, optimizer):
            seen.append(step)
            return step != 3  # falsy at step 3 -> stop

        g.optimize(max_steps=100, callbacks=cb, precision=1.0)
        assert seen == [1, 2, 3]

    def test_none_return_stops_after_first_step(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        p = _params(cnot, full_basis_2q, projected_basis_2q)
        g = Geope(p)
        seen = []

        def cb(step, history, optimizer):
            seen.append(step)
            # no explicit return -> None -> stop

        g.optimize(max_steps=100, callbacks=cb, precision=1.0)
        assert seen == [1]

    def test_multiple_callbacks(self, cnot, full_basis_2q, projected_basis_2q):
        p = _params(cnot, full_basis_2q, projected_basis_2q)
        g = Geope(p)
        a_steps, b_steps = [], []

        def a(step, history, optimizer):
            a_steps.append(step)
            return True

        def b(step, history, optimizer):
            b_steps.append(step)
            return step < 2  # falsy at step 2 -> stop

        g.optimize(max_steps=100, callbacks=[a, b], precision=1.0)
        # both callbacks run each step, including the stopping step
        assert a_steps == [1, 2]
        assert b_steps == [1, 2]

    def test_single_vs_list_equivalent(self, cnot, full_basis_2q, projected_basis_2q):
        def cb(step, history, optimizer):
            return step < 4

        p1 = _params(cnot, full_basis_2q, projected_basis_2q)
        g1 = Geope(p1)
        g1.optimize(max_steps=100, callbacks=cb, precision=1.0)

        p2 = _params(cnot, full_basis_2q, projected_basis_2q)
        g2 = Geope(p2)
        g2.optimize(max_steps=100, callbacks=[cb], precision=1.0)

        assert float(g1.params.fidelity) == pytest.approx(float(g2.params.fidelity))
        np.testing.assert_allclose(g1.params.parameters, g2.params.parameters)


# ---------------------------------------------------------------------------
# Grape
# ---------------------------------------------------------------------------


class TestGrapeCallbacks:
    def test_invocation_and_args(self, cnot, full_basis_2q, projected_basis_2q):
        p = _params(cnot, full_basis_2q, projected_basis_2q)
        g = Grape(p, precision=1.0, history=History())
        calls = []

        def cb(step, history, optimizer):
            calls.append((step, history, optimizer))
            return True

        g.optimize(max_steps=5, optimizer=NewtonTRM(delta=0.1), callbacks=cb)
        assert [c[0] for c in calls] == [1, 2, 3, 4, 5]
        assert all(c[1] is g.history for c in calls)
        assert all(c[2] is g for c in calls)

    def test_early_stop(self, cnot, full_basis_2q, projected_basis_2q):
        p = _params(cnot, full_basis_2q, projected_basis_2q)
        g = Grape(p, precision=1.0)
        seen = []

        def cb(step, history, optimizer):
            seen.append(step)
            return step != 3

        g.optimize(max_steps=100, optimizer=NewtonTRM(delta=0.1), callbacks=cb)
        assert seen == [1, 2, 3]

    def test_callbacks_not_swallowed_by_kwargs(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        # callbacks must be a named parameter, not swallowed by the optimizer slot
        p = _params(cnot, full_basis_2q, projected_basis_2q)
        g = Grape(p, precision=1.0)
        seen = []

        g.optimize(
            max_steps=4,
            optimizer=Adam(0.1),
            callbacks=lambda s, h, o: seen.append(s) or True,
        )
        assert seen == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Gecko
# ---------------------------------------------------------------------------


class TestGeckoCallbacks:
    def _solved_gecko(self, cnot, full_basis_2q, projected_basis_2q):
        p = _params(cnot, full_basis_2q, projected_basis_2q)
        g = Geope(p)
        g.optimize(max_steps=400, precision=0.9999)
        return g

    def test_smooth_invocation_and_args(self, cnot, full_basis_2q, projected_basis_2q):
        g = self._solved_gecko(cnot, full_basis_2q, projected_basis_2q)
        gk = Gecko(g.params, history=History())
        calls = []

        def cb(step, history, optimizer):
            calls.append((step, history, optimizer))
            return True

        # diff_tol tiny so the loop does not converge before max steps
        gk.smooth(
            piecewise_steps_multiplier=3,
            max_smoothing_steps=4,
            diff_tol=1e-12,
            callbacks=cb,
        )
        assert [c[0] for c in calls] == [1, 2, 3, 4]
        assert all(c[1] is gk.history for c in calls)
        assert all(c[2] is gk for c in calls)

    def test_smooth_early_stop(self, cnot, full_basis_2q, projected_basis_2q):
        g = self._solved_gecko(cnot, full_basis_2q, projected_basis_2q)
        gk = Gecko(g.params)
        seen = []

        def cb(step, history, optimizer):
            seen.append(step)
            return step != 3

        success, iters = gk.smooth(
            piecewise_steps_multiplier=3,
            max_smoothing_steps=30,
            diff_tol=1e-12,
            callbacks=cb,
        )
        assert seen == [1, 2, 3]
        assert iters == 3
