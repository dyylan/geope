"""
Tests for geope/optimizers.py.

Tested items:
  Classes:
    - Optimizer (base contract)
    - GradientDescent
    - Adam
    - LBFGS
    - NewtonTRM
    - NewtonRFO
    - NewtonSaddleFree
  Functions:
    - newton_trm_step
    - newton_rfo_step
    - newton_saddle_free_step
    - the strong-Wolfe path shared by the three Newton rules (``wolfe=True``)

The update rules are pinned against closed forms rather than against the
``optax`` transforms they replaced: that dependency is gone, so a comparison
would either import a removed package or freeze magic constants. A quadratic
has an exact minimiser and Adam has an exact first step, which is stronger
evidence anyway.
"""

import dataclasses
from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from geope.optimizers import (
    LBFGS,
    Adam,
    GradientDescent,
    NewtonRFO,
    NewtonSaddleFree,
    NewtonTRM,
    Optimizer,
    OptimizerResult,
    newton_rfo_step,
    newton_saddle_free_step,
    newton_trm_step,
)

# ===================================================================
# A quadratic stand-in for the context
# ===================================================================


def _spd(n, seed=0, shift=1.0):
    """A symmetric positive-definite ``(n, n)`` matrix with known spectrum."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigenvalues = np.linspace(shift, shift + 2.0, n)
    return q @ np.diag(eigenvalues) @ q.T, eigenvalues


class _QuadraticContext:
    r"""A `GeometricContext` stand-in over $C(x)=\tfrac12 x^\top A x - b^\top x$.

    Exposes exactly the members `geope.optimizers` is allowed to read, and
    **raises** on the tier-0 members it is forbidden to touch — so a rule that
    reaches for ``point`` or ``A`` fails here rather than silently costing a
    propagator in the real pipeline.
    """

    def __init__(self, matrix, offset, x, shape=None):
        self._matrix = jnp.asarray(matrix)
        self._offset = jnp.asarray(offset)
        self._shape = shape or (1, offset.size)
        self.free_params = jnp.asarray(x).reshape(self._shape)
        self.coeffs = None

    def _cost(self, x_flat):
        return 0.5 * x_flat @ self._matrix @ x_flat - self._offset @ x_flat

    @property
    def value_and_grad(self):
        flat = self.free_params.flatten()
        grad = self._matrix @ flat - self._offset
        return self._cost(flat), grad.reshape(self._shape)

    @property
    def gradient(self):
        return self.value_and_grad[1]

    @property
    def cost_hessian(self):
        return self._matrix

    @property
    def slope(self):
        return jnp.sum(jnp.real(self.gradient) * jnp.real(self.coeffs))

    def set_direction(self, coeffs):
        if self.coeffs is not None:
            raise ValueError("direction already set")
        self.coeffs = coeffs

    def infidelity_at(self, t):
        return self._cost((self.free_params + t * self.coeffs).flatten())

    # The members an update rule must never read.
    def _forbidden(name):
        @property
        def raiser(self):
            raise AssertionError(f"an update rule must not read ctx.{name}")

        return raiser

    point = _forbidden("point")
    jacobian = _forbidden("jacobian")
    A = _forbidden("A")
    infidelity = _forbidden("infidelity")
    fidelity = _forbidden("fidelity")
    omegas = _forbidden("omegas")
    del _forbidden


def _run(optimizer, ctx, state=None):
    """One step; returns ``(result, new_x_flat)``."""
    state = optimizer.init(ctx.free_params) if state is None else state
    result = optimizer(ctx, state)
    new_x = ctx.free_params + result.dt * result.coeffs
    return result, np.asarray(new_x).flatten()


# ===================================================================
# Tests — the Newton directions, against closed forms
# ===================================================================


class TestNewtonDirections:
    def test_trm_is_the_exact_newton_step_when_delta_is_below_the_spectrum(self):
        # delta <= lambda_min means no shift, so the direction is exactly A^-1 g.
        matrix, eigenvalues = _spd(6, seed=1, shift=2.0)
        gradient = np.arange(1.0, 7.0)
        got = newton_trm_step(jnp.asarray(matrix), jnp.asarray(gradient), 0.5)
        assert eigenvalues.min() > 0.5  # the branch this test is aiming at
        assert np.allclose(got, np.linalg.solve(matrix, gradient))

    def test_trm_shifts_the_spectrum_when_delta_is_above_it(self):
        matrix, eigenvalues = _spd(5, seed=2, shift=0.1)
        gradient = np.arange(1.0, 6.0)
        delta = 1.5
        assert eigenvalues.min() < delta
        shifted = matrix + (delta - eigenvalues.min()) * np.eye(5)
        got = newton_trm_step(jnp.asarray(matrix), jnp.asarray(gradient), delta)
        assert np.allclose(got, np.linalg.solve(shifted, gradient))

    def test_trm_regularises_an_indefinite_hessian_to_positive_definite(self):
        # The point of the shift: a saddle must still give a descent direction.
        rng = np.random.default_rng(3)
        q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        matrix = q @ np.diag([-2.0, -0.5, 1.0, 3.0]) @ q.T
        gradient = rng.normal(size=4)
        direction = np.asarray(
            newton_trm_step(jnp.asarray(matrix), jnp.asarray(gradient), 0.1)
        )
        # `direction` is uphill, so <g, d> must be strictly positive.
        assert float(gradient @ direction) > 0

    def test_rfo_gives_an_uphill_direction(self):
        matrix, _ = _spd(4, seed=4, shift=0.05)
        gradient = np.arange(1.0, 5.0)
        direction = np.asarray(
            newton_rfo_step(jnp.asarray(matrix), jnp.asarray(gradient), 100.0)
        )
        assert float(gradient @ direction) > 0

    def test_both_directions_are_jittable(self):
        matrix, _ = _spd(4, seed=5)
        gradient = np.ones(4)

        def _jitted(fn, arg):
            return jax.jit(lambda m, g: fn(m, g, arg))

        for fn, arg in (
            (newton_trm_step, 0.1),
            (newton_rfo_step, 100.0),
            (newton_saddle_free_step, 1e-8),
        ):
            out = _jitted(fn, arg)(jnp.asarray(matrix), jnp.asarray(gradient))
            assert np.all(np.isfinite(out))


class TestSaddleFreeDirection:
    """`newton_saddle_free_step` — inverts |lambda|, and drops the null space."""

    @staticmethod
    def _symmetric(eigenvalues, seed=0):
        """A symmetric matrix with a prescribed spectrum, and its eigenbasis."""
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.normal(size=(len(eigenvalues),) * 2))
        return q @ np.diag(eigenvalues) @ q.T, q

    def test_it_is_the_exact_newton_step_on_a_well_conditioned_spd_hessian(self):
        # No eigenvalue falls below the cutoff and none is negative, so the
        # magnitude and the truncation are both no-ops.
        matrix, eigenvalues = _spd(6, seed=1, shift=2.0)
        gradient = np.arange(1.0, 7.0)
        assert eigenvalues.min() > 0  # the branch this test is aiming at
        got = newton_saddle_free_step(jnp.asarray(matrix), jnp.asarray(gradient))
        assert np.allclose(got, np.linalg.solve(matrix, gradient))

    def test_it_is_uphill_on_an_indefinite_hessian_where_a_signed_pinv_is_not(self):
        # The whole reason the *magnitudes* are inverted rather than the signed
        # eigenvalues. With the gradient weighted onto the negative-curvature
        # directions, <g, pinv(H) g> = sum g_i^2 / lambda_i goes negative, which
        # on GEOPE's convention is a non-descent direction the line search can
        # never satisfy. Inverting |lambda| gives sum g_i^2 / |lambda_i| > 0
        # unconditionally.
        matrix, q = self._symmetric(np.array([-2.0, -1.0, 1.0, 3.0]), seed=0)
        gradient = q @ np.array([1.0, 1.0, 0.2, 0.2])

        saddle_free = np.asarray(
            newton_saddle_free_step(jnp.asarray(matrix), jnp.asarray(gradient))
        )
        signed = np.asarray(jnp.linalg.pinv(jnp.asarray(matrix)) @ gradient)

        assert float(gradient @ saddle_free) > 0
        assert float(gradient @ signed) < 0  # the failure being guarded against

    def test_it_discards_the_null_space(self):
        # A rank-deficient Hessian is the case that matters: the solutions of a
        # gate-synthesis problem form a manifold, so ker(H) is its tangent space.
        matrix, q = self._symmetric(np.array([3.0, 1.0, 0.0, 0.0]), seed=2)
        rng = np.random.default_rng(3)
        gradient = rng.normal(size=4)

        direction = np.asarray(
            newton_saddle_free_step(jnp.asarray(matrix), jnp.asarray(gradient))
        )

        assert np.all(np.isfinite(direction))  # not inf, despite the 1/0
        # No component along either null eigenvector...
        assert np.allclose(q[:, 2:].T @ direction, 0.0, atol=1e-12)
        # ...and the exact Newton step on the retained subspace.
        retained = q[:, :2]
        expected = retained @ ((retained.T @ gradient) / np.array([3.0, 1.0]))
        assert np.allclose(direction, expected)
        assert float(gradient @ direction) > 0

    def test_rcond_controls_what_is_discarded(self):
        # An eigenvalue at 1e-6 relative to the largest survives the default
        # cutoff and is discarded by a looser one.
        matrix, q = self._symmetric(np.array([1.0, 1e-6]), seed=4)
        gradient = q @ np.array([1.0, 1.0])

        kept = np.asarray(
            newton_saddle_free_step(jnp.asarray(matrix), jnp.asarray(gradient), 1e-8)
        )
        dropped = np.asarray(
            newton_saddle_free_step(jnp.asarray(matrix), jnp.asarray(gradient), 1e-4)
        )
        assert np.abs(q[:, 1] @ kept) > 1e5  # inverted: 1 / 1e-6
        assert np.allclose(q[:, 1] @ dropped, 0.0, atol=1e-12)


# ===================================================================
# Tests — GradientDescent
# ===================================================================


class TestGradientDescent:
    def test_one_step_is_exactly_minus_lr_times_the_gradient(self):
        matrix, _ = _spd(4, seed=6)
        offset = np.arange(1.0, 5.0)
        x = np.full(4, 0.3)
        ctx = _QuadraticContext(matrix, offset, x)
        _, new_x = _run(GradientDescent(learning_rate=0.05), ctx)
        assert np.allclose(new_x, x - 0.05 * (matrix @ x - offset))

    def test_value_is_the_cost_at_the_landing_point_not_the_start(self):
        # The off-by-one this replaced: `value` must describe where the step
        # *lands*, so Grape's params.fidelity matches params.parameters.
        matrix, _ = _spd(4, seed=7)
        offset = np.arange(1.0, 5.0)
        x = np.full(4, 0.3)
        ctx = _QuadraticContext(matrix, offset, x)
        result, new_x = _run(GradientDescent(learning_rate=0.05), ctx)
        assert np.isclose(float(result.value), float(ctx._cost(jnp.asarray(new_x))))
        assert not np.isclose(float(result.value), float(ctx._cost(jnp.asarray(x))))

    def test_state_reports_one_evaluation(self):
        matrix, _ = _spd(3, seed=8)
        ctx = _QuadraticContext(matrix, np.ones(3), np.zeros(3))
        result, _ = _run(GradientDescent(0.1), ctx)
        assert set(result.state) == {"n_eval"}
        assert int(result.state["n_eval"]) == 1


# ===================================================================
# Tests — Adam
# ===================================================================


def _adam_reference(grads, lr, b1=0.9, b2=0.999, eps=1e-8):
    """An independent numpy Adam, to pin the bias correction and eps placement."""
    m = np.zeros_like(grads[0])
    v = np.zeros_like(grads[0])
    steps = []
    for t, g in enumerate(grads, start=1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        steps.append(-lr * m_hat / (np.sqrt(v_hat) + eps))
    return steps


class TestAdam:
    def test_first_step_is_minus_lr_times_the_sign_of_the_gradient(self):
        # From zeroed moments the bias correction cancels exactly:
        #   m_hat = g, v_hat = g^2  =>  step = -lr * g/(|g| + eps).
        # This pins the correction without any golden constant.
        matrix, _ = _spd(4, seed=9)
        offset = np.arange(1.0, 5.0)
        x = np.full(4, 0.3)
        ctx = _QuadraticContext(matrix, offset, x)
        lr = 0.02
        _, new_x = _run(Adam(learning_rate=lr), ctx)
        grad = matrix @ x - offset
        assert np.allclose(new_x - x, -lr * np.sign(grad), atol=1e-9)

    def test_matches_an_independent_numpy_reference_over_many_steps(self):
        matrix, _ = _spd(5, seed=10)
        offset = np.arange(1.0, 6.0)
        lr = 0.05
        opt = Adam(learning_rate=lr)

        x = np.full(5, 0.2)
        state = opt.init(jnp.asarray(x).reshape(1, 5))
        grads, produced = [], []
        for _ in range(20):
            ctx = _QuadraticContext(matrix, offset, x)
            grads.append(np.asarray(matrix @ x - offset))
            result, new_x = _run(opt, ctx, state)
            produced.append(new_x - x)
            state, x = result.state, new_x

        for got, want in zip(produced, _adam_reference(grads, lr)):
            assert np.allclose(got, want, rtol=0, atol=1e-14)

    def test_moments_thread_and_accumulate_across_steps(self):
        matrix, _ = _spd(3, seed=11)
        opt = Adam(0.05)
        x = np.full(3, 0.4)
        ctx = _QuadraticContext(matrix, np.ones(3), x)
        state = opt.init(ctx.free_params)
        assert int(state["count"]) == 0
        result, _ = _run(opt, ctx, state)
        assert int(result.state["count"]) == 1
        assert set(result.state) == {"m", "v", "count", "n_eval"}
        assert not np.allclose(result.state["m"], 0.0)

    def test_moments_are_real_even_on_a_complex_gradient(self):
        # The dtype hardening: value_and_grad's autodiff fallback can carry a
        # spurious imaginary part, which would inflate v through g*g.
        matrix, _ = _spd(3, seed=12)
        ctx = _QuadraticContext(matrix, np.ones(3), np.full(3, 0.4))
        opt = Adam(0.05)
        result, _ = _run(opt, ctx)
        assert jnp.isrealobj(result.state["m"])
        assert jnp.isrealobj(result.state["v"])
        assert np.all(np.asarray(result.state["v"]) >= 0.0)


# ===================================================================
# Tests — the Newton rules end to end on a quadratic
# ===================================================================


class TestNewtonRules:
    def test_a_full_newton_step_lands_on_the_exact_minimiser(self):
        # With delta below the spectrum the direction is A^-1 g, and the Armijo
        # test accepts the full bracket step dt = -1 on the first trial, so one
        # step is exact. That it accepts immediately is itself the proof that the
        # slope is a genuine descent slope.
        matrix, eigenvalues = _spd(5, seed=13, shift=2.0)
        offset = np.arange(1.0, 6.0)
        x = np.full(5, 0.25)
        ctx = _QuadraticContext(matrix, offset, x)
        assert eigenvalues.min() > 0.5
        result, new_x = _run(NewtonTRM(delta=0.5), ctx)
        assert np.allclose(new_x, np.linalg.solve(matrix, offset))
        assert float(result.dt) == -1.0
        assert int(result.state["n_eval"]) == 1

    def test_the_slope_is_the_gradient_pairing_not_the_direction_norm(self):
        # The regression that matters. The bug this replaced paired the direction
        # with *itself*, giving ||A^-1 g||^2 — which is also positive, so a sign
        # check alone would not have caught it. Assert the value.
        matrix, _ = _spd(5, seed=14, shift=2.0)
        offset = np.arange(1.0, 6.0)
        ctx = _QuadraticContext(matrix, offset, np.full(5, 0.25))
        grad = np.asarray(ctx.gradient).flatten()
        direction = np.asarray(
            newton_trm_step(jnp.asarray(matrix), jnp.asarray(grad), 0.5)
        )
        ctx.set_direction(direction.reshape(ctx.free_params.shape))

        assert np.isclose(float(ctx.slope), float(grad @ direction))
        assert not np.isclose(float(ctx.slope), float(direction @ direction))
        assert float(ctx.slope) > 0  # descent, on GEOPE's uphill-coeffs convention

    def test_backtracks_and_decreases_when_the_full_step_overshoots(self):
        # A quartic: the Newton direction is right but the full step overshoots,
        # so the Armijo test must actually reject and contract. Under the old
        # positive-offset test it would have accepted the overshoot.
        class _Quartic(_QuadraticContext):
            def _cost(self, x_flat):
                return jnp.sum(x_flat**4)

            @property
            def value_and_grad(self):
                flat = self.free_params.flatten()
                return self._cost(flat), (4.0 * flat**3).reshape(self._shape)

            @property
            def cost_hessian(self):
                return jnp.diag(12.0 * self.free_params.flatten() ** 2)

        ctx = _Quartic(np.eye(3), np.zeros(3), np.full(3, 3.0))
        f0 = float(ctx.value_and_grad[0])
        result, _ = _run(NewtonTRM(delta=1e-6), ctx)
        assert float(result.value) < f0
        assert int(result.state["n_eval"]) >= 1

    def test_rfo_decreases_the_cost(self):
        matrix, _ = _spd(4, seed=15, shift=0.05)
        offset = np.arange(1.0, 5.0)
        ctx = _QuadraticContext(matrix, offset, np.full(4, 0.25))
        f0 = float(ctx.value_and_grad[0])
        result, _ = _run(NewtonRFO(kappa=100.0), ctx)
        assert float(result.value) < f0

    def test_warm_start_threads_the_previous_step(self):
        matrix, _ = _spd(4, seed=16, shift=2.0)
        opt = NewtonTRM(delta=0.5)
        state = opt.init(jnp.zeros((1, 4)))
        assert float(state["dt"]) == -opt.max_step
        ctx = _QuadraticContext(matrix, np.arange(1.0, 5.0), np.full(4, 0.25))
        result, _ = _run(opt, ctx, state)
        assert float(result.state["dt"]) == float(result.dt)


# ===================================================================
# Tests — L-BFGS
# ===================================================================


class _NoHessianContext(_QuadraticContext):
    """`_QuadraticContext` with the tier-2 Hessian made forbidden too.

    The stock harness *permits* ``cost_hessian``, since the Newton rules need
    it. Not reading it is the whole claim of :class:`LBFGS`, so it needs a
    harness where reaching for it fails.
    """

    class _Manifold:
        def __init__(self, matrix, offset, shape):
            self._matrix, self._offset, self._shape = matrix, offset, shape

        def value_and_grad(self, x):
            flat = x.flatten()
            value = 0.5 * flat @ self._matrix @ flat - self._offset @ flat
            return value, (self._matrix @ flat - self._offset).reshape(self._shape)

    def __init__(self, matrix, offset, x, shape=None):
        super().__init__(matrix, offset, x, shape)
        self.manifold = self._Manifold(self._matrix, self._offset, self._shape)

    @property
    def cost_hessian(self):
        raise AssertionError("LBFGS must not read ctx.cost_hessian")


def _descend(optimizer, matrix, offset, x0, steps, context=_NoHessianContext):
    """Run ``steps`` L-BFGS steps from ``x0``; returns ``(x, state, per_step)``."""
    x = np.asarray(x0, float)
    state = optimizer.init(jnp.zeros((1, x.size)))
    per_step = []
    for _ in range(steps):
        ctx = context(matrix, offset, x)
        result = optimizer(ctx, state)
        x = np.asarray(ctx.free_params + result.dt * result.coeffs).flatten()
        state = result.state
        per_step.append(result)
    return x, state, per_step


class TestLBFGS:
    def test_it_never_reads_the_hessian(self):
        # The entire point of the rule: O(mP) vectors, no (P, P) matrix and no
        # eigh. `_NoHessianContext` raises if it reaches for one.
        matrix, _ = _spd(5, seed=30, shift=1.0)
        _descend(LBFGS(4), matrix, np.arange(1.0, 6.0), np.full(5, 0.3), steps=6)

    def test_a_cold_start_is_exactly_steepest_descent(self):
        # An empty buffer means every rho is 0, so both loops are no-ops and
        # gamma falls back to 1: the direction must be the gradient itself.
        matrix, _ = _spd(4, seed=31, shift=1.0)
        ctx = _NoHessianContext(matrix, np.ones(4), np.full(4, 0.3))
        opt = LBFGS(5)
        result = opt(ctx, opt.init(ctx.free_params))
        assert np.allclose(np.asarray(result.coeffs), np.asarray(ctx.gradient))

    def test_state_keys_and_shapes_are_pinned(self):
        # The state round-trips a jit boundary every step, so its structure must
        # not drift between init() and __call__.
        matrix, _ = _spd(4, seed=32, shift=1.0)
        opt = LBFGS(3)
        ctx = _NoHessianContext(matrix, np.ones(4), np.full(4, 0.3))
        state = opt.init(ctx.free_params)
        expected = {"s", "y", "rho", "prev_x", "prev_g", "count", "dt", "n_eval"}
        assert set(state) == expected
        result = opt(ctx, state)
        assert set(result.state) == expected
        for key in expected:
            assert jnp.shape(result.state[key]) == jnp.shape(state[key])
            assert jnp.result_type(result.state[key]) == jnp.result_type(state[key])
        assert int(result.state["count"]) == 1

    @pytest.mark.parametrize("wolfe", [True, False], ids=["wolfe", "armijo"])
    def test_it_reaches_the_exact_minimiser_of_a_quadratic(self, wolfe):
        # With memory >= n the recursion reconstructs the exact inverse Hessian
        # once n independent pairs are stored, so it must converge outright.
        n = 6
        matrix, _ = _spd(n, seed=33, shift=1.0)
        offset = np.arange(1.0, n + 1.0)
        opt = LBFGS(n, wolfe=wolfe)
        x, _, _ = _descend(opt, matrix, offset, np.zeros(n), steps=3 * n)
        assert np.allclose(x, np.linalg.solve(matrix, offset), atol=1e-8)

    def test_the_buffer_fills_and_then_saturates(self):
        matrix, _ = _spd(5, seed=34, shift=1.0)
        opt = LBFGS(3)
        counts = []
        x = np.full(5, 0.3)
        state = opt.init(jnp.zeros((1, 5)))
        for _ in range(6):
            ctx = _NoHessianContext(matrix, np.arange(1.0, 6.0), x)
            result = opt(ctx, state)
            x = np.asarray(ctx.free_params + result.dt * result.coeffs).flatten()
            state = result.state
            counts.append(int(np.count_nonzero(np.asarray(state["rho"]))))
        # One pair per completed step, capped at `memory`; the newest is last.
        assert counts == [0, 1, 2, 3, 3, 3]

    def test_a_non_positive_curvature_pair_is_skipped_not_stored(self):
        # s.y <= 0 would make rho negative and the approximation indefinite.
        # The guard must drop the pair, leaving rho at exactly zero.
        opt = LBFGS(2)
        state = opt.init(jnp.zeros((1, 3)))
        s_new = jnp.asarray([[1.0, 0.0, 0.0]])
        pushed = opt._push(state, s_new, -s_new, jnp.asarray(True))
        assert float(pushed["rho"][-1]) == 0.0
        assert np.allclose(np.asarray(pushed["s"][-1]), 0.0)
        # ... and a genuine pair is kept, with rho = 1 / (s.y).
        kept = opt._push(state, s_new, 2.0 * s_new, jnp.asarray(True))
        assert np.isclose(float(kept["rho"][-1]), 0.5)

    def test_the_direction_is_uphill_and_the_step_negative(self):
        matrix, _ = _spd(5, seed=35, shift=1.0)
        _, _, per_step = _descend(
            LBFGS(4), matrix, np.arange(1.0, 6.0), np.full(5, 0.4), steps=5
        )
        for result in per_step:
            assert float(result.dt) <= 0.0

    def test_the_state_stays_real_on_a_complex_gradient(self):
        matrix, _ = _spd(3, seed=36, shift=1.0)
        ctx = _NoHessianContext(matrix, np.ones(3), np.full(3, 0.4))
        opt = LBFGS(3)
        result = opt(ctx, opt.init(ctx.free_params))
        for key in ("s", "y", "rho", "prev_x", "prev_g"):
            assert jnp.isrealobj(result.state[key])

    def test_both_wolfe_conditions_hold_at_the_accepted_step(self):
        # Why `wolfe=True` is the default here: only the curvature condition
        # guarantees s.y > 0, which is what keeps the implicit H positive
        # definite. It is not covered by `TestStrongWolfe`'s exactness test,
        # whose premise -- that the direction is the exact Newton step -- is
        # false for L-BFGS's steepest-descent first step.
        matrix, _ = _spd(5, seed=38, shift=1.0)
        offset = np.arange(1.0, 6.0)
        opt = LBFGS(4, wolfe=True)
        ctx = _NoHessianContext(matrix, offset, np.full(5, 0.3))
        result = opt(ctx, opt.init(ctx.free_params))

        alpha = -float(result.dt)
        phi0 = float(ctx.value_and_grad[0])
        d0 = -float(ctx.slope)  # phi'(0) < 0 at a descent direction
        phi_a = float(result.value)
        flat = np.asarray(ctx.free_params + result.dt * result.coeffs).flatten()
        grad_a = matrix @ flat - offset
        d_a = -float(grad_a @ np.asarray(result.coeffs).flatten())

        assert phi_a <= phi0 + opt.c1 * alpha * d0 + 1e-12  # sufficient decrease
        assert abs(d_a) <= opt.c2 * abs(d0) + 1e-12  # curvature

    def test_it_stalls_gracefully_at_the_floating_point_floor(self):
        # Past the point where the achievable decrease falls below the
        # resolution of the cost itself, the strong-Wolfe search is testing
        # rounding noise and returns dt = 0. It must stop *moving*, not
        # diverge, and the zero-length pair must be rejected rather than
        # poisoning the memory with rho = inf.
        n = 6
        matrix, _ = _spd(n, seed=39, shift=1.0)
        offset = np.arange(1.0, n + 1.0)
        opt = LBFGS(n, wolfe=True)
        x, state, per_step = _descend(opt, matrix, offset, np.zeros(n), steps=30)

        assert np.allclose(x, np.linalg.solve(matrix, offset), atol=1e-7)
        assert np.all(np.isfinite(np.asarray(state["rho"])))
        assert np.all(np.isfinite(np.asarray(state["s"])))
        values = [float(r.value) for r in per_step]
        assert np.all(np.diff(values) <= 1e-12)  # never uphill
        # Whatever the tail does, it does not move.
        assert float(per_step[-1].dt) == 0.0

    def test_memory_participates_in_the_compile_memo(self):
        assert LBFGS(5) != LBFGS(10)
        assert LBFGS(5) == LBFGS(5)

    def test_it_is_jittable_end_to_end(self):
        # The real pipeline calls the rule inside `jax.jit`, so the ring buffer
        # and both loops must be traceable with no Python branch on a value.
        matrix, _ = _spd(4, seed=37, shift=1.0)
        opt = LBFGS(3)
        ctx = _NoHessianContext(matrix, np.ones(4), np.full(4, 0.3))
        state = opt.init(ctx.free_params)

        def one_step(st):
            return opt(_NoHessianContext(matrix, np.ones(4), np.full(4, 0.3)), st).dt

        assert float(jax.jit(one_step)(state)) <= 0.0


# ===================================================================
# Tests — the strong-Wolfe line search
# ===================================================================


class _WolfeContext(_QuadraticContext):
    """`_QuadraticContext` plus the one thing strong Wolfe needs beyond it.

    The curvature condition wants $\\phi'(\\alpha)$ at trial points, which in the
    real pipeline comes from ``ctx.manifold.value_and_grad`` — a jitted function
    of an arbitrary pulse. Here it is the quadratic's own gradient.
    """

    class _Manifold:
        def __init__(self, matrix, offset, shape):
            self._matrix, self._offset, self._shape = matrix, offset, shape

        def value_and_grad(self, x):
            flat = x.flatten()
            value = 0.5 * flat @ self._matrix @ flat - self._offset @ flat
            return value, (self._matrix @ flat - self._offset).reshape(self._shape)

    def __init__(self, matrix, offset, x, shape=None):
        super().__init__(matrix, offset, x, shape)
        self.manifold = self._Manifold(self._matrix, self._offset, self._shape)


class TestStrongWolfe:
    @pytest.mark.parametrize(
        "make",
        [
            lambda: NewtonTRM(0.0, wolfe=True),
            lambda: NewtonSaddleFree(wolfe=True),
        ],
        ids=["nr-trm", "nr-saddle-free"],
    )
    def test_it_lands_on_the_exact_minimiser_of_a_quadratic(self, make):
        # On a quadratic the model is exact, so the seed alpha_1 = -phi'(0)/q is
        # exactly 1 and phi'(1) = 0 satisfies the curvature condition outright:
        # the search accepts its first trial and one step is exact.
        matrix, eigenvalues = _spd(6, seed=1, shift=2.0)
        offset = np.arange(1.0, 7.0)
        assert eigenvalues.min() > 0  # so the direction is the exact Newton step
        ctx = _WolfeContext(matrix, offset, np.zeros(6))

        result, new_x = _run(make(), ctx)

        assert np.allclose(new_x, np.linalg.solve(matrix, offset))
        assert np.isclose(float(result.dt), -1.0)
        # One value and one gradient: the first trial is accepted.
        assert int(result.state["n_eval"]) == 2

    def test_both_wolfe_conditions_hold_at_the_accepted_step(self):
        # A quartic along the ray, so the quadratic model is wrong and the search
        # actually has to bracket and zoom rather than accept its seed.
        class _QuarticContext(_WolfeContext):
            def _cost(self, flat):
                return 0.0 + jnp.sum(flat**4) - self._offset @ flat

            @property
            def value_and_grad(self):
                flat = self.free_params.flatten()
                grad = 4.0 * flat**3 - self._offset
                return self._cost(flat), grad.reshape(self._shape)

            @property
            def cost_hessian(self):
                flat = self.free_params.flatten()
                return jnp.diag(12.0 * flat**2) + 1e-6 * jnp.eye(flat.size)

        offset = np.arange(1.0, 5.0)
        ctx = _QuarticContext(np.eye(4), offset, np.full(4, 0.6))
        f0, _ = ctx.value_and_grad

        opt = NewtonSaddleFree(wolfe=True)
        result, _ = _run(opt, ctx)

        alpha = -float(result.dt)
        assert alpha > 0.0
        coeffs = ctx.coeffs
        slope0 = float(jnp.sum(jnp.real(ctx.gradient) * jnp.real(coeffs)))
        d0 = -slope0  # phi'(0) < 0 for a descent direction

        phi_alpha = float(result.value)
        _f, grad_alpha = ctx.manifold.value_and_grad(ctx.free_params - alpha * coeffs)
        d_alpha = -float(jnp.sum(jnp.real(grad_alpha) * jnp.real(coeffs)))

        # Armijo (sufficient decrease) and the curvature condition.
        assert phi_alpha <= float(f0) + opt.c1 * alpha * d0
        assert abs(d_alpha) <= -opt.c2 * d0

    def test_it_is_a_separate_program_from_the_armijo_path(self):
        # `wolfe` is a plain Python bool, so the branch is resolved at trace time
        # and the two paths differ in the state they thread and the work they do.
        matrix, _ = _spd(5, seed=7, shift=1.0)
        offset = np.arange(1.0, 6.0)

        armijo, _ = _run(NewtonSaddleFree(), _WolfeContext(matrix, offset, np.zeros(5)))
        wolfe, _ = _run(
            NewtonSaddleFree(wolfe=True), _WolfeContext(matrix, offset, np.zeros(5))
        )

        # Both keep "dt" in the state, which the warm start depends on.
        assert "dt" in armijo.state and "dt" in wolfe.state
        # Wolfe spends a gradient as well as a value on its accepted trial.
        assert int(wolfe.state["n_eval"]) > int(armijo.state["n_eval"])

    def test_the_flag_participates_in_the_compile_memo(self):
        assert NewtonTRM(0.1, wolfe=True) != NewtonTRM(0.1)
        assert NewtonTRM(0.1, wolfe=True) == NewtonTRM(0.1, wolfe=True)
        assert NewtonSaddleFree(c2=0.9) != NewtonSaddleFree(c2=0.5)


# ===================================================================
# Tests — the frozen-dataclass contract
# ===================================================================


class TestOptimizerValueSemantics:
    @pytest.mark.parametrize(
        "make",
        [
            lambda: GradientDescent(0.05),
            lambda: Adam(0.05),
            lambda: LBFGS(10),
            lambda: NewtonTRM(0.1),
            lambda: NewtonRFO(100.0),
            lambda: NewtonSaddleFree(1e-8),
        ],
        ids=["gd", "adam", "lbfgs", "nr-trm", "nr-rfo", "nr-saddle-free"],
    )
    def test_value_equality_drives_the_compile_memo(self, make):
        assert make() == make()
        assert hash(make()) == hash(make())
        assert len({make(), make()}) == 1

    def test_the_primary_hyperparameter_is_first_positionally(self):
        # Dataclass inheritance puts a base's fields ahead of a subclass's, so
        # without kw_only on the shared line-search fields NewtonTRM(0.1) would
        # silently set c1 instead of delta.
        assert NewtonTRM(0.1).delta == 0.1
        assert NewtonTRM(0.1).c1 == 1e-4
        assert NewtonRFO(50.0).kappa == 50.0
        assert NewtonRFO(50.0).c1 == 1e-4
        assert NewtonSaddleFree(1e-6).rcond == 1e-6
        assert NewtonSaddleFree(1e-6).c1 == 1e-4
        assert Adam(0.05).learning_rate == 0.05
        assert GradientDescent(0.05).learning_rate == 0.05
        assert LBFGS(7).memory == 7
        assert LBFGS(7).c1 == 1e-4

    def test_line_search_fields_are_keyword_only(self):
        with pytest.raises(TypeError):
            NewtonTRM(0.1, 1e-3)  # c1 must be passed by keyword
        with pytest.raises(TypeError):
            LBFGS(10, 1e-3)

    def test_differing_hyperparameters_compare_unequal(self):
        assert NewtonTRM(0.1) != NewtonTRM(0.2)
        assert NewtonTRM(0.1) != NewtonTRM(0.1, c1=1e-3)
        assert Adam(0.05) != Adam(0.05, b1=0.5)
        assert GradientDescent(0.05) != Adam(0.05)

    def test_replace_and_immutability(self):
        assert dataclasses.replace(NewtonTRM(0.1), delta=0.2) == NewtonTRM(0.2)
        with pytest.raises(FrozenInstanceError):
            NewtonTRM(0.1).delta = 0.2

    def test_names_are_stable(self):
        assert GradientDescent().name == "gradient_descent"
        assert Adam().name == "adam"
        assert NewtonTRM().name == "newton_trm"
        assert NewtonRFO().name == "newton_rfo"
        assert NewtonSaddleFree().name == "newton_saddle_free"
        assert LBFGS().name == "lbfgs"

    def test_every_rule_is_an_optimizer_and_returns_an_optimizer_result(self):
        matrix, _ = _spd(3, seed=17, shift=2.0)
        for opt in (
            GradientDescent(0.05),
            Adam(0.05),
            # Armijo here: `_QuadraticContext` deliberately lacks the `manifold`
            # that LBFGS's default strong-Wolfe search needs for trial gradients.
            LBFGS(4, wolfe=False),
            NewtonTRM(0.5),
            NewtonRFO(),
            NewtonSaddleFree(),
        ):
            assert isinstance(opt, Optimizer)
            ctx = _QuadraticContext(matrix, np.ones(3), np.full(3, 0.2))
            result, _ = _run(opt, ctx)
            assert isinstance(result, OptimizerResult)
            assert "n_eval" in result.state
            # Uphill direction, negative step — GEOPE's convention.
            assert float(result.dt) < 0

    def test_base_optimizer_declines(self):
        with pytest.raises(NotImplementedError):
            Optimizer()(None, {})
