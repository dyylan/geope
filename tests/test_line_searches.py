"""
Tests for geope/line_searches.py.

Tested items:
  Functions:
    - _golden_section_search
    - _adam_line_search
    - _quadratic_armijo_line_search
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geope.utils import golden_section_search_np

from geope.line_searches import (
    _golden_section_search,
    _adam_line_search,
    _quadratic_armijo_line_search,
)

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

    def test_agrees_with_numpy_version(self):
        f_np = lambda x: (x - 1.5) ** 2
        f_jax = lambda x: (x - 1.5) ** 2
        x_np, _ = golden_section_search_np(f_np, 0, 3, tol=1e-6)
        x_jax, _, _ = _golden_section_search(f_jax, 0.0, 3.0, tol=1e-6)
        assert jnp.abs(x_np - x_jax) < 1e-3

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
        # q <= q_floor: no trustworthy minimiser -> full step t0 = a, accepted on
        # a purely linear (descent) objective.
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
