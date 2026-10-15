"""
Tests for geope/line_searches.py.

Tested items:
  Functions:
    - _golden_section_search
    - _quadratic_armijo_line_search
    - _armijo_line_search
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from geope.line_searches import (
    _armijo_line_search,
    _golden_section_search,
    _quadratic_armijo_line_search,
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
        x, _, n = _golden_section_search(f, 0.0, 5.0, tol=1e-6)
        assert 0.0 <= x <= 5.0
        # At least the two initial f1/f2 probes were spent.
        assert int(n) >= 2

    def test_f_matches_x(self):
        f = lambda x: (x + 1.0) ** 2
        x, fx, _ = _golden_section_search(f, -3.0, 1.0, tol=1e-6)
        assert jnp.isclose(fx, f(x), atol=1e-8)


# ===================================================================
# Tests — _quadratic_armijo_line_search
# ===================================================================
#
# The routine consumes the exact slope s = psi'(0) and curvature q = psi''(0)
# supplied by the caller, on the one-sided descent bracket [0, alpha_max] with a
# descent direction (s < 0, so the model minimiser -s/q is positive).


class TestQuadraticArmijoLineSearch:
    def test_returns_triple(self):
        fF = lambda t: 1.0 - 1.0 * t + 0.5 * 4.0 * t**2
        result = _quadratic_armijo_line_search(fF, 1.0, -1.0, 4.0, 1.0)
        assert len(result) == 3

    def test_exact_seed_on_parabola(self):
        # For a parabola, the quadratic seed -s/q is the exact minimiser; when it
        # lies in the bracket it satisfies Armijo immediately (n_eval == 1).
        s, q, F0 = -1.0, 4.0, 1.0
        fF = lambda t: F0 + s * t + 0.5 * q * t**2
        t, F, n = _quadratic_armijo_line_search(fF, 1.0, s, q, F0)
        assert jnp.isclose(t, -s / q)  # = 0.25, in [0, 1]
        assert int(n) == 1
        assert jnp.isclose(F, fF(t))

    def test_seed_clipped_to_bracket(self):
        # -s/q = 2.0 lies outside [0, 1]; the seed is clipped to alpha_max = 1.0.
        s, q, F0 = -1.0, 0.5, 1.0
        fF = lambda t: F0 + s * t + 0.5 * q * t**2
        t, _, n = _quadratic_armijo_line_search(fF, 1.0, s, q, F0)
        assert jnp.isclose(t, 1.0)
        assert int(n) == 1

    def test_nonpositive_curvature_falls_back_to_full_step(self):
        # q <= 0 without the guard: the model is concave, so it has no minimiser
        # -> full step alpha0 = alpha_max, accepted on a purely linear (descent) objective.
        s, F0 = -1.0, 1.0
        fF = lambda t: F0 + s * t  # linear, min on [0, 1] at t = 1
        t, F, n = _quadratic_armijo_line_search(
            fF, 1.0, s, -1.0, F0, concave_guard=False
        )
        assert jnp.isclose(t, 1.0)
        assert int(n) == 1
        assert jnp.isclose(F, fF(1.0))

    def test_concave_guard_seeds_where_the_quadratic_term_takes_over(self):
        # q < 0 with the guard: the seed is min(alpha_max, 2|s|/|q|) = 2 * 1 / 4 = 0.5,
        # where |q| t^2 / 2 = |s| t, instead of the bracket end alpha_max = 1.
        s, q, F0 = -1.0, -4.0, 1.0
        fF = lambda t: F0 + s * t
        t, _, n = _quadratic_armijo_line_search(fF, 1.0, s, q, F0)
        t_off, _, _ = _quadratic_armijo_line_search(
            fF, 1.0, s, q, F0, concave_guard=False
        )
        assert jnp.isclose(t, 0.5) and int(n) == 1
        assert jnp.isclose(t_off, 1.0)

    def test_concave_guard_never_seeds_past_the_bracket(self):
        # 2|s|/|q| = 2 > alpha_max = 1, so the guard leaves the full step alone.
        s, q, F0 = -1.0, -1.0, 1.0
        fF = lambda t: F0 + s * t
        t, _, n = _quadratic_armijo_line_search(fF, 1.0, s, q, F0)
        assert jnp.isclose(t, 1.0) and int(n) == 1

    def test_concave_guard_bounds_an_unbounded_bracket(self):
        # With alpha_max = inf a concave step no longer seeds at inf: the guard gives a
        # finite seed, so the search is an ordinary one-evaluation accept.
        s, q, F0 = -1.0, -4.0, 1.0
        fF = lambda t: F0 + s * t
        t, F, n = jax.jit(
            lambda: _quadratic_armijo_line_search(fF, jnp.inf, s, q, F0)
        )()
        assert jnp.isclose(t, 0.5) and int(n) == 1
        assert bool(jnp.isfinite(F))

    def test_zero_curvature_takes_the_full_step_either_way(self):
        # q == 0 gives no scale (2|s|/|q| = inf), so both settings seed at alpha_max.
        s, F0 = -1.0, 1.0
        fF = lambda t: F0 + s * t
        for guard in (True, False):
            t, _, n = _quadratic_armijo_line_search(
                fF, 1.0, s, 0.0, F0, concave_guard=guard
            )
            assert jnp.isclose(t, 1.0) and int(n) == 1

    def test_backtracks_when_seed_overshoots(self):
        # True objective is far steeper than the model (q=4) used for the seed, so
        # the seed fails sufficient decrease and the search backtracks (n_eval > 1),
        # returning a step that does satisfy Armijo.
        s, q_model, F0 = -1.0, 4.0, 1.0
        c1 = 1e-4
        fF = lambda t: F0 + s * t + 0.5 * 100.0 * t**2  # steep true curvature
        t, F, n = _quadratic_armijo_line_search(fF, 1.0, s, q_model, F0, c1=c1)
        assert int(n) > 1
        assert float(F) <= F0 + c1 * float(t) * s + 1e-9  # Armijo holds at return

    def test_jittable(self):
        s, q, F0 = -1.0, 4.0, 1.0
        fF = lambda t: F0 + s * t + 0.5 * q * t**2
        t, F, _ = jax.jit(lambda: _quadratic_armijo_line_search(fF, 1.0, s, q, F0))()
        assert bool(jnp.isfinite(t)) and bool(jnp.isfinite(F))

    def test_unbounded_bracket_terminates(self):
        # alpha_max = inf with q == 0 (which the concave guard cannot bound) seeds
        # alpha0 = inf, a fixed point of t <- beta * t: fF is NaN there (as the
        # propagator is), the Armijo bound is -inf, and |beta * t| >= t_min never
        # fails. Only the default budget ends the loop, after the 52 contractions
        # that take beta = 0.5 to machine precision.
        fF = lambda t: 1.0 + 0.0 * t  # NaN at t = inf
        t, _, n = jax.jit(
            lambda: _quadratic_armijo_line_search(fF, jnp.inf, -1.0, 0.0, 1.0)
        )()
        assert int(n) == 1 + 52
        assert bool(jnp.isinf(t))

    def test_max_iter_caps_the_contractions(self):
        # A non-decreasing objective fails Armijo at every trial; an explicit
        # max_iter stops it after that many contractions, far above t_min.
        fF = lambda t: 1.0 + t
        t, F, n = _quadratic_armijo_line_search(fF, 1.0, -1.0, -1.0, 1.0, max_iter=3)
        assert int(n) == 1 + 3
        assert jnp.isclose(t, 0.5**3)
        assert jnp.isclose(F, fF(t))

    def test_default_budget_rejects_beta_outside_unit_interval(self):
        fF = lambda t: 1.0 - t
        with pytest.raises(ValueError, match="beta"):
            _quadratic_armijo_line_search(fF, 1.0, -1.0, 4.0, 1.0, beta=1.0)


# ===================================================================
# Tests — _armijo_line_search
# ===================================================================
#
# The non-quadratic sibling: same one-sided bracket [0, alpha_max] with s < 0,
# but the seed is the full bracket step alpha0 = alpha_max and no curvature is
# consumed. With s omitted it defaults to the radial slope s = -2 * F0, so the
# objectives below use F0 = 1.0 and a true slope of -2.0 to be consistent with
# that.


class TestArmijoLineSearch:
    def test_returns_triple(self):
        fF = lambda t: 1.0 - 2.0 * t
        result = _armijo_line_search(fF, 1.0)
        assert len(result) == 3

    def test_takes_full_step_when_accepted(self):
        # Linear descent objective: the full bracket step satisfies Armijo
        # outright, so no backtracking happens. n_eval == 2: the F0 probe plus
        # the seed evaluation.
        fF = lambda t: 1.0 - 2.0 * t
        t, F, n = _armijo_line_search(fF, 1.0)
        assert jnp.isclose(t, 1.0)
        assert int(n) == 2
        assert jnp.isclose(F, fF(1.0))

    def test_backtracks_when_full_step_overshoots(self):
        # A steep upward curvature makes the full step *increase* the objective,
        # so the search must contract until sufficient decrease holds.
        c1 = 1e-4
        F0 = 1.0
        s = -2.0 * F0  # the radial slope the routine will infer
        fF = lambda t: F0 + s * t + 0.5 * 100.0 * t**2
        t, F, n = _armijo_line_search(fF, 1.0, c1=c1)
        assert int(n) > 2
        assert 0.0 <= float(t) <= 1.0
        assert float(F) <= F0 + c1 * float(t) * s + 1e-9  # Armijo holds at return

    def test_explicit_F0_and_s_skip_the_probe(self):
        # Supplying F0 saves its evaluation (n_eval == 1), and passing the same
        # values the routine would have inferred reproduces the same step.
        fF = lambda t: 1.0 - 2.0 * t
        t_auto, F_auto, n_auto = _armijo_line_search(fF, 1.0)
        t, F, n = _armijo_line_search(fF, 1.0, F0=1.0, s=-2.0)
        assert int(n) == 1
        assert int(n_auto) == 2
        assert jnp.isclose(t, t_auto) and jnp.isclose(F, F_auto)

    def test_explicit_slope_changes_the_threshold(self):
        # The Armijo bound is F0 + c1 * t * s with t > 0, so a *more negative* s
        # is a *stricter* demand and buys a shorter step. c1 is raised here
        # because at the default 1e-4 the slope barely moves the threshold at all.
        F0, c1 = 1.0, 0.4
        fF = lambda t: F0 - 2.0 * t + 0.5 * 6.0 * t**2  # fF(1) = 2.0 > F0
        t_radial, _, _ = _armijo_line_search(fF, 1.0, c1=c1)  # radial s = -2 * F0
        t_steep, F_steep, _ = _armijo_line_search(fF, 1.0, F0=F0, s=-4.0, c1=c1)
        assert abs(float(t_steep)) < abs(float(t_radial))
        assert float(F_steep) <= F0 + c1 * float(t_steep) * -4.0 + 1e-9

    def test_respects_t_min(self):
        # No descent anywhere on the bracket: Armijo can never pass, so the search
        # contracts to the floor and gives up there rather than looping forever.
        t_min = 1e-2
        fF = lambda t: 1.0 + 2.0 * t  # increases as t goes positive
        t, F, _ = _armijo_line_search(fF, 1.0, t_min=t_min)
        assert t_min <= abs(float(t))
        assert 0.0 <= float(t) <= 1.0
        assert jnp.isclose(F, fF(t))

    def test_converged_objective_accepts_seed(self):
        # F0 <= 0 makes the Armijo test vacuous (nothing can beat it), so the
        # guard accepts the seed instead of grinding down to t_min.
        fF = lambda t: t**2  # fF(0) = 0, and every trial is worse
        t, _, n = _armijo_line_search(fF, 1.0)
        assert jnp.isclose(t, 1.0)
        assert int(n) == 2

    def test_jittable(self):
        fF = lambda t: 1.0 - 2.0 * t
        t, F, _ = jax.jit(lambda: _armijo_line_search(fF, 1.0))()
        assert bool(jnp.isfinite(t)) and bool(jnp.isfinite(F))

    def test_unbounded_bracket_terminates(self):
        # The seed is alpha_max itself, so alpha_max = inf is the fixed point outright;
        # see the quadratic sibling. F0 and s are supplied, so there is no probe.
        fF = lambda t: 1.0 + 0.0 * t  # NaN at t = inf
        t, _, n = jax.jit(lambda: _armijo_line_search(fF, jnp.inf, F0=1.0, s=-2.0))()
        assert int(n) == 1 + 52
        assert bool(jnp.isinf(t))

    def test_max_iter_caps_the_contractions(self):
        # The F0 probe and the seed, then exactly max_iter contractions.
        fF = lambda t: 1.0 + 2.0 * t
        t, F, n = _armijo_line_search(fF, 1.0, max_iter=3)
        assert int(n) == 2 + 3
        assert jnp.isclose(t, 0.5**3)
        assert jnp.isclose(F, fF(t))

    def test_default_budget_rejects_beta_outside_unit_interval(self):
        fF = lambda t: 1.0 - 2.0 * t
        with pytest.raises(ValueError, match="beta"):
            _armijo_line_search(fF, 1.0, beta=1.0)
