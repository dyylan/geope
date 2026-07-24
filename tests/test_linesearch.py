"""
Standalone tests for the 1-D line-search methods, exercised **outside** the
GEOPE loop — directly on scalar objectives.

Covers:
  Raw minimisers (``geope.utils``):
    - golden_section_search   (golden-ratio bracketing)
    - quadratic_line_search   (derivative-free parabolic / second-order)
  LineSearch objects (``geope.line_searches``):
    - GoldenSection
    - Quadratic

Both raw minimisers share the ``(t_best, f_best, n_eval)`` contract of
:func:`golden_section_search`; both LineSearch objects return
``(dt, infid, state)`` and expose the evaluation count as ``state["n_eval"]``.
"""

import pytest

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geope.utils import (
    golden_section_search,
    golden_section_search_np,
    quadratic_line_search,
    adam_line_search,
)
from geope.line_searches import GoldenSection, Quadratic


# ---------------------------------------------------------------------------
# Raw minimisers — shared behaviour, parametrised over both methods
# ---------------------------------------------------------------------------

# Uniform ``search(f, a, b) -> (t_best, f_best, n_eval)`` wrappers.
RAW = {
    "golden": lambda f, a, b: golden_section_search(f, a, b, tol=1e-7),
    "quadratic": lambda f, a, b: quadratic_line_search(f, a, b, num_iters=8),
}


@pytest.fixture(params=list(RAW), ids=list(RAW))
def search(request):
    return RAW[request.param]


class TestRawLineSearches:
    def test_returns_finite_triple(self, search):
        f = lambda x: (x - 2.0) ** 2
        result = search(f, 0.0, 5.0)
        assert len(result) == 3
        t, fx, n = result
        assert bool(jnp.isfinite(t)) and bool(jnp.isfinite(fx))
        assert int(n) > 0

    def test_interior_minimum(self, search):
        # convex quadratic with its minimum at x=2, strictly inside [0, 5]
        f = lambda x: (x - 2.0) ** 2
        t, fx, _ = search(f, 0.0, 5.0)
        assert jnp.isclose(t, 2.0, atol=1e-3)
        assert float(fx) < 1e-5

    def test_interior_minimum_negative_bracket(self, search):
        # geope-style bracket [a, 0] with a < 0; minimum at t = -0.4
        f = lambda x: (x + 0.4) ** 2
        t, fx, _ = search(f, -1.0, 0.0)
        assert jnp.isclose(t, -0.4, atol=1e-3)
        assert float(fx) < 1e-5

    def test_clips_to_boundary_when_min_outside(self, search):
        # unconstrained min at x=2, but the bracket caps at 0 -> best is x=0
        f = lambda x: (x - 2.0) ** 2
        t, fx, _ = search(f, -1.0, 0.0)
        assert -1.0 <= float(t) <= 0.0
        assert jnp.isclose(t, 0.0, atol=1e-2)
        assert float(fx) <= float(f(0.0)) + 1e-6

    def test_within_bounds(self, search):
        f = lambda x: (x - 1.5) ** 2
        t, _, _ = search(f, 0.0, 3.0)
        assert 0.0 <= float(t) <= 3.0

    def test_f_best_matches_t_best(self, search):
        f = lambda x: (x + 1.0) ** 2
        t, fx, _ = search(f, -3.0, 1.0)
        assert jnp.isclose(fx, f(t), atol=1e-8)

    def test_never_worse_than_bracket_midpoint(self, search):
        # non-parabolic unimodal objective: result must beat the midpoint probe
        f = lambda x: jnp.abs(x - 1.2) ** 1.5
        a, b = 0.0, 3.0
        t, fx, _ = search(f, a, b)
        assert float(fx) <= float(f(0.5 * (a + b))) + 1e-8
        assert 0.0 <= float(t) <= 3.0

    def test_jittable(self, search):
        f = lambda x: (x - 2.0) ** 2
        t, fx, n = jax.jit(lambda: search(f, 0.0, 5.0))()
        assert bool(jnp.isfinite(t)) and bool(jnp.isfinite(fx))


def test_methods_agree_on_interior_minimum():
    f = lambda x: (x - 1.5) ** 2
    tg, _, _ = golden_section_search(f, 0.0, 3.0, tol=1e-7)
    tq, _, _ = quadratic_line_search(f, 0.0, 3.0, num_iters=8)
    assert jnp.abs(tg - tq) < 1e-3


def test_quadratic_is_exact_on_a_parabola():
    # the parabola through the three seed points *is* f, so the vertex is exact
    f = lambda x: 3.0 * (x - 1.7) ** 2 + 0.5
    t, fx, _ = quadratic_line_search(f, -1.0, 4.0, num_iters=1)
    assert jnp.isclose(t, 1.7, atol=1e-9)
    assert jnp.isclose(fx, 0.5, atol=1e-9)


@pytest.mark.parametrize("num_iters", [1, 3, 8])
def test_quadratic_n_eval_is_three_plus_num_iters(num_iters):
    f = lambda x: (x - 2.0) ** 2
    _, _, n = quadratic_line_search(f, 0.0, 5.0, num_iters=num_iters)
    assert int(n) == 3 + num_iters


def test_golden_n_eval_positive():
    f = lambda x: (x - 2.0) ** 2
    _, _, n = golden_section_search(f, 0.0, 5.0, tol=1e-6)
    assert int(n) > 0


# ---------------------------------------------------------------------------
# LineSearch objects — the pluggable interface used by Geope
# ---------------------------------------------------------------------------

OBJ = {"GoldenSection": GoldenSection, "Quadratic": Quadratic}


@pytest.fixture(params=list(OBJ), ids=list(OBJ))
def line_search_cls(request):
    return OBJ[request.param]


class TestLineSearchObjects:
    def test_init_state_has_n_eval(self, line_search_cls):
        assert line_search_cls().init() == {"n_eval": 0}

    def test_call_contract(self, line_search_cls):
        f = lambda x: (x - 2.0) ** 2
        ls = line_search_cls()
        state = ls.init()
        dt, infid, new_state = ls(f, 0.0, 5.0, state)
        assert 0.0 <= float(dt) <= 5.0
        assert jnp.isclose(infid, f(dt), atol=1e-8)
        assert int(new_state["n_eval"]) > 0

    def test_finds_minimum(self, line_search_cls):
        f = lambda x: (x - 2.0) ** 2
        ls = line_search_cls()
        dt, infid, _ = ls(f, 0.0, 5.0, ls.init())
        assert jnp.isclose(dt, 2.0, atol=1e-3)
        assert float(infid) < 1e-5

    def test_frozen_config_is_hashable_and_value_equal(self, line_search_cls):
        # frozen dataclasses -> value __eq__/__hash__ (drives Geope's compile memo)
        assert line_search_cls() == line_search_cls()
        assert hash(line_search_cls()) == hash(line_search_cls())


def test_quadratic_num_iters_controls_eval_count():
    f = lambda x: (x - 2.0) ** 2
    for k in (2, 5):
        ls = Quadratic(num_iters=k)
        _, _, state = ls(f, 0.0, 5.0, ls.init())
        assert int(state["n_eval"]) == 3 + k


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
        f = lambda x: x**2
        x, fx = golden_section_search_np(f, -0.01, 0.01, tol=1e-8)
        assert -0.01 <= x <= 0.01


# ===================================================================
# Tests — golden_section_search (JAX version)
# ===================================================================


class TestGoldenSectionSearch:
    def test_agrees_with_numpy_version(self):
        # cross-check the JAX golden-section against the numpy reference; the
        # value/bounds/f-matching behaviour is covered by TestRawLineSearches
        f = lambda x: (x - 1.5) ** 2
        x_np, _ = golden_section_search_np(f, 0, 3, tol=1e-6)
        x_jax, _, _ = golden_section_search(f, 0.0, 3.0, tol=1e-6)
        assert jnp.abs(x_np - x_jax) < 1e-3


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
        x, fx = adam_line_search(
            f, 0.0, 5.0, lr=0.05, num_steps=500, finite_difference=fd
        )
        assert jnp.isclose(x, 2.0, atol=0.1)
        assert float(fx) < 1e-2
        assert jnp.isclose(fx, f(x), atol=1e-8)

    def test_clips_to_boundary_when_min_outside(self, fd):
        # unconstrained min at x=2, but the interval caps at 0 -> best is x=0
        f = lambda x: (x - 2.0) ** 2
        x, fx = adam_line_search(
            f, -0.9, 0.0, lr=0.1, num_steps=100, finite_difference=fd
        )
        assert -0.9 <= float(x) <= 0.0
        assert float(fx) <= f(0.0) + 1e-6

    def test_returns_best_not_worse_than_start(self, fd):
        # a large lr can overshoot; best-so-far must never exceed f(t_init)
        f = lambda x: (x - 2.0) ** 2
        x, fx = adam_line_search(
            f, 0.0, 5.0, lr=0.9, num_steps=50, finite_difference=fd
        )
        assert jnp.isclose(fx, f(x), atol=1e-8)
        assert float(fx) <= f(0.0) + 1e-9

    def test_jittable(self, fd):
        f = lambda x: (x - 2.0) ** 2
        x, fx = jax.jit(lambda: adam_line_search(f, 0.0, 5.0, finite_difference=fd))()
        assert bool(jnp.isfinite(x)) and bool(jnp.isfinite(fx))


def test_adam_fd_and_grad_agree():
    # both gradient modes should converge to the same interior minimum
    f = lambda x: (x - 2.0) ** 2
    x_fd, _ = adam_line_search(
        f, 0.0, 5.0, lr=0.05, num_steps=500, finite_difference=True
    )
    x_grad, _ = adam_line_search(
        f, 0.0, 5.0, lr=0.05, num_steps=500, finite_difference=False
    )
    assert jnp.isclose(x_fd, 2.0, atol=0.1)
    assert jnp.isclose(x_grad, 2.0, atol=0.1)
    assert jnp.abs(x_fd - x_grad) < 0.1
