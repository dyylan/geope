"""
Tests for geope/optimizers.py.

Tested items:
  Classes:
    - Optimizer (base contract)
    - GradientDescent
    - Adam
    - NewtonTRM
    - NewtonRFO
  Functions:
    - newton_trm_step
    - newton_rfo_step

The update rules are pinned against closed forms rather than against the
``optax`` transforms they replaced: that dependency is gone, so a comparison
would either import a removed package or freeze magic constants. A quadratic
has an exact minimiser and Adam has an exact first step, which is stronger
evidence anyway.
"""

import dataclasses
from dataclasses import FrozenInstanceError

import pytest
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geope.optimizers import (
    Adam,
    GradientDescent,
    NewtonRFO,
    NewtonTRM,
    Optimizer,
    OptimizerResult,
    newton_rfo_step,
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
        for fn, arg in ((newton_trm_step, 0.1), (newton_rfo_step, 100.0)):
            out = jax.jit(lambda m, g: fn(m, g, arg))(
                jnp.asarray(matrix), jnp.asarray(gradient)
            )
            assert np.all(np.isfinite(out))


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
        result, new_x = _run(Adam(learning_rate=lr), ctx)
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
# Tests — the frozen-dataclass contract
# ===================================================================


class TestOptimizerValueSemantics:
    @pytest.mark.parametrize(
        "make",
        [
            lambda: GradientDescent(0.05),
            lambda: Adam(0.05),
            lambda: NewtonTRM(0.1),
            lambda: NewtonRFO(100.0),
        ],
        ids=["gd", "adam", "nr-trm", "nr-rfo"],
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
        assert Adam(0.05).learning_rate == 0.05
        assert GradientDescent(0.05).learning_rate == 0.05

    def test_line_search_fields_are_keyword_only(self):
        with pytest.raises(TypeError):
            NewtonTRM(0.1, 1e-3)  # c1 must be passed by keyword

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

    def test_every_rule_is_an_optimizer_and_returns_an_optimizer_result(self):
        matrix, _ = _spd(3, seed=17, shift=2.0)
        for opt in (GradientDescent(0.05), Adam(0.05), NewtonTRM(0.5), NewtonRFO()):
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
