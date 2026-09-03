"""
Tests for geope/line_searches.py.

Tested items:
  Functions:
    - _golden_section_search
    - _adam_line_search
    - _quadratic_armijo_line_search
    - _armijo_line_search
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geope.line_searches import (
    _golden_section_search,
    _adam_line_search,
    _quadratic_armijo_line_search,
    _armijo_line_search,
)


# ===================================================================
# Tests — _golden_section_search (JAX version)
# ===================================================================


class TestGoldenSectionSearch:
    def test_returns_triple(self):
        f = lambda x: (x - 2.0) ** 2
        result = _golden_section_search(f, 0.0, 5.0, tol=1e-6)
        assert len(result) == 3

    def test_x_within_bounds(self):
        f = lambda x: (x - 2.0) ** 2
        x, fx, n = _golden_section_search(f, 0.0, 5.0, tol=1e-6)
        assert 0.0 <= x <= 5.0
        # At least the two initial f1/f2 probes were spent.
        assert int(n) >= 2

    def test_f_matches_x(self):
        f = lambda x: (x + 1.0) ** 2
        x, fx, n = _golden_section_search(f, -3.0, 1.0, tol=1e-6)
        assert jnp.isclose(fx, f(x), atol=1e-8)


# ===================================================================
# Tests — _adam_line_search
# ===================================================================


@pytest.mark.parametrize("fd", [True, False], ids=["adam_fd", "adam_grad"])
class TestAdamLineSearch:
    def test_returns_triple(self, fd):
        f = lambda x: (x - 2.0) ** 2
        result = _adam_line_search(f, 0.0, 5.0, finite_difference=fd)
        assert len(result) == 3

    def test_x_within_bounds(self, fd):
        f = lambda x: (x - 2.0) ** 2
        x, fx, n = _adam_line_search(f, 0.0, 5.0, num_steps=30, finite_difference=fd)
        assert 0.0 <= float(x) <= 5.0
        # Fixed schedule: num_steps body evals + f(t0) + final f(t).
        assert int(n) == 30 + 2

    def test_minimises_quadratic(self, fd):
        # interior minimum at x = 2, reachable from t_init=0
        f = lambda x: (x - 2.0) ** 2
        x, fx, n = _adam_line_search(
            f, 0.0, 5.0, lr=0.05, num_steps=500, finite_difference=fd
        )
        assert jnp.isclose(x, 2.0, atol=0.1)
        assert float(fx) < 1e-2
        assert jnp.isclose(fx, f(x), atol=1e-8)

    def test_clips_to_boundary_when_min_outside(self, fd):
        # unconstrained min at x=2, but the interval caps at 0 -> best is x=0
        f = lambda x: (x - 2.0) ** 2
        x, fx, n = _adam_line_search(
            f, -0.9, 0.0, lr=0.1, num_steps=100, finite_difference=fd
        )
        assert -0.9 <= float(x) <= 0.0
        assert float(fx) <= f(0.0) + 1e-6

    def test_returns_best_not_worse_than_start(self, fd):
        # a large lr can overshoot; best-so-far must never exceed f(t_init)
        f = lambda x: (x - 2.0) ** 2
        x, fx, n = _adam_line_search(
            f, 0.0, 5.0, lr=0.9, num_steps=50, finite_difference=fd
        )
        assert jnp.isclose(fx, f(x), atol=1e-8)
        assert float(fx) <= f(0.0) + 1e-9

    def test_jittable(self, fd):
        f = lambda x: (x - 2.0) ** 2
        x, fx, n = jax.jit(
            lambda: _adam_line_search(f, 0.0, 5.0, finite_difference=fd)
        )()
        assert bool(jnp.isfinite(x)) and bool(jnp.isfinite(fx))


def test_adam_fd_and_grad_agree():
    # both gradient modes should converge to the same interior minimum
    x_fd, _, _ = _adam_line_search(
        lambda x: (x - 2.0) ** 2,
        0.0,
        5.0,
        lr=0.05,
        num_steps=500,
        finite_difference=True,
    )
    x_grad, _, _ = _adam_line_search(
        lambda x: (x - 2.0) ** 2,
        0.0,
        5.0,
        lr=0.05,
        num_steps=500,
        finite_difference=False,
    )
    assert jnp.isclose(x_fd, 2.0, atol=0.1)
    assert jnp.isclose(x_grad, 2.0, atol=0.1)
    assert jnp.abs(x_fd - x_grad) < 0.1


# ===================================================================
# Tests — _quadratic_armijo_line_search
# ===================================================================
#
# The routine consumes the exact slope s = psi'(0) and curvature q = psi''(0)
# supplied by the caller, on GEOPE's one-sided bracket [a, 0] with a < 0 and a
# descent direction (s > 0, so the model minimiser -s/q is negative).


class TestQuadraticArmijoLineSearch:
    def test_returns_triple(self):
        fF = lambda t: 1.0 + 1.0 * t + 0.5 * 4.0 * t**2
        result = _quadratic_armijo_line_search(fF, -1.0, 1.0, 4.0, 1.0)
        assert len(result) == 3

    def test_exact_seed_on_parabola(self):
        # For a parabola, the quadratic seed -s/q is the exact minimiser; when it
        # lies in the bracket it satisfies Armijo immediately (n_eval == 1).
        s, q, F0 = 1.0, 4.0, 1.0
        fF = lambda t: F0 + s * t + 0.5 * q * t**2
        t, F, n = _quadratic_armijo_line_search(fF, -1.0, s, q, F0)
        assert jnp.isclose(t, -s / q)  # = -0.25, in [-1, 0]
        assert int(n) == 1
        assert jnp.isclose(F, fF(t))

    def test_seed_clipped_to_bracket(self):
        # -s/q = -2.0 lies outside [-1, 0]; the seed is clipped to a = -1.0.
        s, q, F0 = 1.0, 0.5, 1.0
        fF = lambda t: F0 + s * t + 0.5 * q * t**2
        t, F, n = _quadratic_armijo_line_search(fF, -1.0, s, q, F0)
        assert jnp.isclose(t, -1.0)
        assert int(n) == 1

    def test_nonpositive_curvature_falls_back_to_full_step(self):
        # q <= 0: the model is concave, so it has no minimiser -> full step
        # t0 = a, accepted on a purely linear (descent) objective.
        s, F0 = 1.0, 1.0
        fF = lambda t: F0 + s * t  # linear, min on [-1, 0] at t = -1
        t, F, n = _quadratic_armijo_line_search(fF, -1.0, s, -1.0, F0)
        assert jnp.isclose(t, -1.0)
        assert int(n) == 1
        assert jnp.isclose(F, fF(-1.0))

    def test_backtracks_when_seed_overshoots(self):
        # True objective is far steeper than the model (q=4) used for the seed, so
        # the seed fails sufficient decrease and the search backtracks (n_eval > 1),
        # returning a step that does satisfy Armijo.
        s, q_model, F0 = 1.0, 4.0, 1.0
        c1 = 1e-4
        fF = lambda t: F0 + s * t + 0.5 * 100.0 * t**2  # steep true curvature
        t, F, n = _quadratic_armijo_line_search(fF, -1.0, s, q_model, F0, c1=c1)
        assert int(n) > 1
        assert float(F) <= F0 + c1 * float(t) * s + 1e-9  # Armijo holds at return

    def test_jittable(self):
        s, q, F0 = 1.0, 4.0, 1.0
        fF = lambda t: F0 + s * t + 0.5 * q * t**2
        t, F, n = jax.jit(lambda: _quadratic_armijo_line_search(fF, -1.0, s, q, F0))()
        assert bool(jnp.isfinite(t)) and bool(jnp.isfinite(F))


# ===================================================================
# Tests — _armijo_line_search
# ===================================================================
#
# The non-quadratic sibling: same one-sided bracket [a, 0] with a < 0 and s > 0,
# but the seed is the full bracket step t0 = a and no curvature is consumed. With
# s omitted it defaults to the radial slope s = 2 * F0, so the objectives below
# use F0 = 1.0 and a true slope of 2.0 to be consistent with that.


class TestArmijoLineSearch:
    def test_returns_triple(self):
        fF = lambda t: 1.0 + 2.0 * t
        result = _armijo_line_search(fF, -1.0)
        assert len(result) == 3

    def test_takes_full_step_when_accepted(self):
        # Linear descent objective: the full bracket step satisfies Armijo
        # outright, so no backtracking happens. n_eval == 2: the F0 probe plus
        # the seed evaluation.
        fF = lambda t: 1.0 + 2.0 * t
        t, F, n = _armijo_line_search(fF, -1.0)
        assert jnp.isclose(t, -1.0)
        assert int(n) == 2
        assert jnp.isclose(F, fF(-1.0))

    def test_backtracks_when_full_step_overshoots(self):
        # A steep upward curvature makes the full step *increase* the objective,
        # so the search must contract until sufficient decrease holds.
        c1 = 1e-4
        F0 = 1.0
        s = 2.0 * F0  # the radial slope the routine will infer
        fF = lambda t: F0 + s * t + 0.5 * 100.0 * t**2
        t, F, n = _armijo_line_search(fF, -1.0, c1=c1)
        assert int(n) > 2
        assert -1.0 <= float(t) <= 0.0
        assert float(F) <= F0 + c1 * float(t) * s + 1e-9  # Armijo holds at return

    def test_explicit_F0_and_s_skip_the_probe(self):
        # Supplying F0 saves its evaluation (n_eval == 1), and passing the same
        # values the routine would have inferred reproduces the same step.
        fF = lambda t: 1.0 + 2.0 * t
        t_auto, F_auto, n_auto = _armijo_line_search(fF, -1.0)
        t, F, n = _armijo_line_search(fF, -1.0, F0=1.0, s=2.0)
        assert int(n) == 1
        assert int(n_auto) == 2
        assert jnp.isclose(t, t_auto) and jnp.isclose(F, F_auto)

    def test_explicit_slope_changes_the_threshold(self):
        # The Armijo bound is F0 + c1 * t * s with t < 0, so a *larger* s is a
        # *stricter* demand and buys a shorter step. c1 is raised here because at
        # the default 1e-4 the slope barely moves the threshold at all.
        F0, c1 = 1.0, 0.4
        fF = lambda t: F0 + 2.0 * t + 0.5 * 6.0 * t**2  # fF(-1) = 2.0 > F0
        t_radial, _, _ = _armijo_line_search(fF, -1.0, c1=c1)  # radial s = 2 * F0
        t_steep, F_steep, _ = _armijo_line_search(fF, -1.0, F0=F0, s=4.0, c1=c1)
        assert abs(float(t_steep)) < abs(float(t_radial))
        assert float(F_steep) <= F0 + c1 * float(t_steep) * 4.0 + 1e-9

    def test_respects_t_min(self):
        # No descent anywhere on the bracket: Armijo can never pass, so the search
        # contracts to the floor and gives up there rather than looping forever.
        t_min = 1e-2
        fF = lambda t: 1.0 - 2.0 * t  # increases as t goes negative
        t, F, n = _armijo_line_search(fF, -1.0, t_min=t_min)
        assert t_min <= abs(float(t))
        assert -1.0 <= float(t) <= 0.0
        assert jnp.isclose(F, fF(t))

    def test_converged_objective_accepts_seed(self):
        # F0 <= 0 makes the Armijo test vacuous (nothing can beat it), so the
        # guard accepts the seed instead of grinding down to t_min.
        fF = lambda t: t**2  # fF(0) = 0, and every trial is worse
        t, F, n = _armijo_line_search(fF, -1.0)
        assert jnp.isclose(t, -1.0)
        assert int(n) == 2

    def test_jittable(self):
        fF = lambda t: 1.0 + 2.0 * t
        t, F, n = jax.jit(lambda: _armijo_line_search(fF, -1.0))()
        assert bool(jnp.isfinite(t)) and bool(jnp.isfinite(F))
