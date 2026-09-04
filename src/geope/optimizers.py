r"""Pluggable update rules for :class:`geope.Grape`.

An optimiser here *produces* the step GRAPE takes: it reads the infidelity and
its derivatives off the step's `geope.geometry.GeometricContext` and returns a
direction together with a step along it. That is what separates this family from
`geope.line_searches`, whose members only tune a **scalar** step size along a
direction GEOPE has already solved for. The two are deliberately not one
hierarchy — but they are the same *shape*, and everything the line-search
contract says applies here verbatim:

Each optimiser is a ``@dataclass(frozen=True)`` — immutable config that gets
value-based ``__eq__``/``__hash__``/``__repr__`` for free. The value ``__eq__``
drives GRAPE's compile memo (``NewtonTRM(0.1) == NewtonTRM(0.1)`` ⇒ no recompile)
and the immutability keeps hyperparameter sweeps correct (a config cannot be
mutated in place and silently reuse a stale compiled function).

**The call contract.** `geope.Grape` builds one context per step (inside the
jitted update) and calls ``optimizer(ctx, state)``, which returns an
:class:`OptimizerResult` ``(dt, coeffs, value, state)``; GRAPE then forms
``free_params + dt * coeffs``. ``value`` is the infidelity **at the point that
lands on** — not at the base point — so `geope.Grape` can report a fidelity that
actually describes the parameters it stores.

**All four rules have one shape:** an *uphill* direction and a *negative* step.

| rule | $p$ | $\mathrm dt$ |
|---|---|---|
| `GradientDescent` | $\nabla C$ | $-\eta$ |
| `Adam` | $\hat m/(\sqrt{\hat v}+\varepsilon)$ | $-\eta$ |
| `NewtonTRM` / `NewtonRFO` | $H_{\text{reg}}^{-1}\nabla C$ | Armijo backtrack on $[-t_{\max}, 0]$ |

Adam fits because its per-coordinate rescaling is a *preconditioner on the
direction*, not a step size. That this is GEOPE's convention — ``coeffs`` uphill,
the accepted step negative, `GeometricContext.slope` positive at a descent
direction — is what lets `geope.line_searches._armijo_line_search` serve here
**unchanged**, and it is why `newton_trm_step` returns $H^{-1}g$ with no minus
sign: on this convention that already *is* the direction.

**What an optimiser may read off the context.** Tier 0′ (``value_and_grad``,
``gradient``), tier 1 ``slope``, tier 2 ``cost_hessian``, and tier 3 along the ray
after ``set_direction``. It must **not** touch ``point``, ``infidelity``,
``fidelity``, ``jacobian`` or ``A``: the first three re-exponentiate the pulse
that ``value_and_grad`` already propagated, and ``A`` would trace a matrix
logarithm no gradient method needs. Because every context quantity is lazy,
obeying that is all it takes to keep the logarithm and the Jacobian out of a
GRAPE run entirely — the same dividend `Gecko` gets by reading only ``omegas``.

The three orders of information cost different things per step:
:class:`GradientDescent` and :class:`Adam` pay one pullback pass plus a single
propagator to score the landing point; :class:`NewtonTRM` and :class:`NewtonRFO`
additionally pay the dense ``(P, P)`` Hessian and one propagator per backtracking
trial.

Cross-step state is a JAX pytree threaded through the jitted update (never a
mutated attribute — a jitted closure traces once). Every optimiser carries
``{"n_eval"}``, the count of infidelity evaluations it spent on the step;
:class:`Adam` adds its moments and the Newton pair the warm-started trial step.
``Grape.optimize`` re-``init()``s the state at the start of every run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from .line_searches import _armijo_line_search


class OptimizerResult(NamedTuple):
    """What an :class:`Optimizer` returns.

    Attributes:
        dt: The accepted step along `coeffs` — **negative**, on GEOPE's
            convention. `geope.Grape` forms ``free_params + dt * coeffs`` and
            reports this as ``step_size``.
        coeffs: The *uphill* direction, the same shape as ``ctx.free_params``.
        value: The infidelity **at** ``free_params + dt * coeffs``. Every
            optimiser here evaluates it anyway — the Newton pair get it from the
            last accepted backtracking trial — and reporting it is what keeps
            ``params.fidelity`` describing ``params.parameters``.
        state: The new optimiser-owned state pytree (always carries ``"n_eval"``).
    """

    dt: Array
    coeffs: Array
    value: Array
    state: dict


class Optimizer:
    """Base update rule: turns a step's geometry into a direction and a step.

    Subclasses are frozen dataclasses (immutable config) owning an opaque JAX
    pytree state. The base state carries only ``{"n_eval"}`` — the per-step count
    of infidelity evaluations spent — which every optimiser reports; stateful ones
    extend it.

    Unlike `geope.line_searches.LineSearch.init`, :meth:`init` takes the free
    parameters: a first-order method needs moment buffers shaped like them. That
    is the one deliberate divergence between the two contracts.
    """

    name = "optimizer"

    def init(self, free_params: Array) -> dict:
        """Return a fresh state pytree (called once per ``optimize()`` run).

        Args:
            free_params: The initial pulse, used for the shape and dtype of any
                per-parameter buffers.
        """
        del free_params
        return {"n_eval": jnp.asarray(0, jnp.int32)}

    def __call__(self, ctx, state: dict) -> OptimizerResult:
        """Produce this step's direction and step size.

        Args:
            ctx: The step's `geope.geometry.GeometricContext`, with no direction
                set — the optimiser sets it, since the direction *is* what it
                computes.
            state: This optimiser's threaded state from the previous step.

        Returns:
            An :class:`OptimizerResult`.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class _FixedStep(Optimizer):
    r"""Shared tail of the first-order rules: an uphill direction, a fixed $-\eta$.

    Both subclasses differ only in how they precondition the gradient. Neither
    line-searches, so the step is whatever ``learning_rate`` says and a too-large
    rate diverges; reach for :class:`NewtonTRM` when you want the step chosen for
    you.

    The one propagator spent here scores the landing point. `set_direction` may be
    called once, so the direction goes in and the ray is read at the accepted
    ``dt`` — which is what makes ``value`` the infidelity of the parameters GRAPE
    is about to store, rather than of the ones it just left.
    """

    learning_rate: float = 0.01

    def direction(self, gradient: Array, state: dict) -> tuple[Array, dict]:
        """The uphill direction, and whatever state carried it."""
        raise NotImplementedError

    def __call__(self, ctx, state):
        grad = ctx.gradient
        # Realify once: the pulse is complex128 with an identically-zero imaginary
        # part, and every rule below is real arithmetic.
        coeffs, new_state = self.direction(jnp.real(grad), state)
        ctx.set_direction(coeffs.astype(grad.dtype))
        dt = jnp.asarray(-self.learning_rate, jnp.float64)
        value = ctx.infidelity_at(dt)
        new_state["n_eval"] = jnp.asarray(1, jnp.int32)
        return OptimizerResult(dt, coeffs.astype(grad.dtype), value, new_state)


@dataclass(frozen=True)
class GradientDescent(_FixedStep):
    r"""Plain gradient descent: $\phi \leftarrow \phi - \eta\,\nabla C$.

    Carries only the base ``{"n_eval"}`` state.

    Args:
        learning_rate: The step size $\eta$. Defaults to 0.01.
    """

    name = "gradient_descent"

    def direction(self, gradient, state):
        return gradient, {}


@dataclass(frozen=True)
class Adam(_FixedStep):
    r"""Adam — first-order with per-parameter adaptive step sizes.

    The standard bias-corrected rule,

    $$m \leftarrow \beta_1 m + (1-\beta_1)g,\qquad
      v \leftarrow \beta_2 v + (1-\beta_2)g^2,\qquad
      p = \frac{\hat m}{\sqrt{\hat v} + \varepsilon},$$

    with $\hat m = m/(1-\beta_1^t)$, $\hat v = v/(1-\beta_2^t)$ and the step
    $-\eta p$. The defaults match the reference implementation, so runs stay
    comparable with the ``optax.adam`` this replaced.

    **The moments are accumulated on the real part of the gradient.**
    `geope.geometry.Manifold.value_and_grad`'s analytic path returns a gradient
    that is real by construction, but its autodiff fallback does not, and a
    spurious imaginary part silently inflates $v$ through $g^2$. Realifying in
    `_FixedStep` makes that unmakeable on either path.

    Args:
        learning_rate: The step size $\eta$. Defaults to 0.01.
        b1: First-moment decay $\beta_1$. Defaults to 0.9.
        b2: Second-moment decay $\beta_2$. Defaults to 0.999.
        eps: Numerical-stability term $\varepsilon$. Defaults to 1e-8.
    """

    name = "adam"
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8

    def init(self, free_params):
        zeros = jnp.zeros(free_params.shape, jnp.float64)
        return {
            "m": zeros,
            "v": zeros,
            "count": jnp.asarray(0, jnp.int32),
            "n_eval": jnp.asarray(0, jnp.int32),
        }

    def direction(self, gradient, state):
        m = self.b1 * state["m"] + (1.0 - self.b1) * gradient
        v = self.b2 * state["v"] + (1.0 - self.b2) * gradient * gradient
        count = state["count"] + 1
        t = jnp.asarray(count, jnp.float64)
        m_hat = m / (1.0 - self.b1**t)
        v_hat = v / (1.0 - self.b2**t)
        return m_hat / (jnp.sqrt(v_hat) + self.eps), {"m": m, "v": v, "count": count}


@dataclass(frozen=True)
class _BacktrackingNewton(Optimizer):
    r"""Shared machinery for the two regularised-Newton rules.

    They differ only in how they turn the raw ``(P, P)`` Hessian into an uphill
    direction; the step sizing is identical, and is
    `geope.line_searches._armijo_line_search` — the same backtracking the Armijo
    line searches use, reused unchanged because GRAPE speaks GEOPE's convention:
    the bracket is one-sided $[-t_{\max}, 0]$ and
    $s = \langle\nabla C, H_{\text{reg}}^{-1}\nabla C\rangle > 0$ is exactly its
    documented descent slope.

    The trial step is warm-started at ``clip(increase * dt_prev, a, 0)`` and shrunk
    by ``beta`` until sufficient decrease holds, reproducing the growth and cap of
    the transform this replaced. ``max_step = 1.0`` means the first trial of the
    first step is the **full Newton step**, which is the right default: the Newton
    direction already carries its own scale, so there is no $1/G$ renormalisation
    of the kind `geope.Geope` applies to its own bracket.

    Args:
        c1: Armijo sufficient-decrease constant. Defaults to 1e-4.
        beta: Backtracking contraction factor in ``(0, 1)``. Defaults to 0.8.
        t_min: Minimum step magnitude before the search gives up. Defaults to
            1e-10, which allows ~103 contractions at ``beta=0.8`` from a full step.
        max_step: Magnitude cap on the trial step. Defaults to 1.0.
        increase: Growth factor applied to the previous step before capping.
            Defaults to 1.5; ``1.0`` keeps the previous step as the guess.

    All five are **keyword-only**. Dataclass inheritance places a base's fields
    ahead of a subclass's, so without that ``NewtonTRM(0.1)`` would set ``c1``
    rather than ``delta`` — the same trap
    `geope.geometry.stiefel.stiefel.Stiefel.projective` documents.
    """

    c1: float = field(default=1e-4, kw_only=True)
    beta: float = field(default=0.8, kw_only=True)
    t_min: float = field(default=1e-10, kw_only=True)
    max_step: float = field(default=1.0, kw_only=True)
    increase: float = field(default=1.5, kw_only=True)

    def direction(self, hessian: Array, gradient: Array) -> Array:
        """The uphill direction from the flattened Hessian and gradient."""
        raise NotImplementedError

    def init(self, free_params):
        del free_params
        return {
            "dt": jnp.asarray(-self.max_step, jnp.float64),
            "n_eval": jnp.asarray(0, jnp.int32),
        }

    def __call__(self, ctx, state):
        value, grad = ctx.value_and_grad
        # Solve in the reals: the Hessian is real and the gradient real-valued, so
        # carrying the pulse's complex dtype into the Cholesky buys nothing.
        flat = self.direction(ctx.cost_hessian, jnp.real(grad).flatten())
        coeffs = flat.reshape(grad.shape).astype(grad.dtype)
        ctx.set_direction(coeffs)

        a = jnp.asarray(-self.max_step, jnp.float64)
        # Warm start, as the transform this replaced did. A previous step of
        # exactly 0 (a search that gave up) would otherwise pin the bracket shut
        # for the rest of the run, so fall back to the full bracket there.
        warm = jnp.clip(self.increase * state["dt"], a, 0.0)
        a_eff = jnp.where(warm == 0.0, a, warm)

        # ctx.slope is <grad, coeffs> = g^T H_reg^-1 g > 0 for the positive
        # definite regularised Hessian, so this is a genuine sufficient-decrease
        # test rather than the relaxation a same-signed slope would give.
        dt, value_at_dt, n_eval = _armijo_line_search(
            ctx.infidelity_at,
            a_eff,
            F0=value,
            s=ctx.slope,
            c1=self.c1,
            beta=self.beta,
            t_min=self.t_min,
        )
        # A non-finite trial means the step is unusable; stand still rather than
        # propagate a nan into the parameters.
        finite = jnp.isfinite(value_at_dt)
        dt = jnp.where(finite, dt, 0.0)
        value_at_dt = jnp.where(finite, value_at_dt, value)
        return OptimizerResult(dt, coeffs, value_at_dt, {"dt": dt, "n_eval": n_eval})


@dataclass(frozen=True)
class NewtonTRM(_BacktrackingNewton):
    r"""Trust-region Newton — the default: spectrum-shifted Hessian plus backtracking.

    Shifts the Hessian's spectrum by $\sigma = \max(0,\ \delta - \lambda_{\min})$
    so it is positive definite with smallest eigenvalue at least $\delta$, then
    solves $H_{\text{reg}}p = \nabla C$ by Cholesky. A larger ``delta``
    regularises harder and shortens the step toward gradient descent; a very small
    one takes near-pure Newton steps, which bounce on an indefinite landscape.

    Args:
        delta: Trust-region floor on the regularised spectrum. Defaults to 0.1.
        c1: Armijo sufficient-decrease constant. Defaults to 1e-4.
        beta: Backtracking contraction factor. Defaults to 0.8.
        t_min: Minimum step magnitude. Defaults to 1e-10.
        max_step: Magnitude cap on the trial step. Defaults to 1.0.
        increase: Growth factor on the previous step. Defaults to 1.5.
    """

    name = "newton_trm"
    delta: float = 0.1

    def direction(self, hessian, gradient):
        return newton_trm_step(hessian, gradient, self.delta)


@dataclass(frozen=True)
class NewtonRFO(_BacktrackingNewton):
    r"""Rational-function-optimisation Newton — conditioning-driven regularisation.

    Instead of a fixed spectral floor, scales the augmented Hessian
    $\begin{pmatrix}\alpha^2 H & \alpha g\\ \alpha g^\intercal & 0\end{pmatrix}$
    down by $\alpha \leftarrow 0.9\,\alpha$ until the recovered Hessian's condition
    number falls below ``kappa``, shifting away any negative eigenvalue at each
    round. The direction is then Cholesky-solved as in :class:`NewtonTRM`.

    Args:
        kappa: Target condition number. Defaults to 100.
        c1: Armijo sufficient-decrease constant. Defaults to 1e-4.
        beta: Backtracking contraction factor. Defaults to 0.8.
        t_min: Minimum step magnitude. Defaults to 1e-10.
        max_step: Magnitude cap on the trial step. Defaults to 1.0.
        increase: Growth factor on the previous step. Defaults to 1.5.
    """

    name = "newton_rfo"
    kappa: float = 100.0

    def direction(self, hessian, gradient):
        return newton_rfo_step(hessian, gradient, self.kappa)


# ---------------------------------------------------------------------------
# Newton directions — the regularised solves the two second-order rules use.
# Un-jitted, so they fuse into `Grape`'s enclosing update-step trace.
# ---------------------------------------------------------------------------


def newton_trm_step(hessian: Array, gradient: Array, delta: float | Array) -> Array:
    r"""Solve $H_{\text{reg}}p = g$ with the spectrum floored at ``delta``.

    Args:
        hessian: The real ``(P, P)`` objective Hessian.
        gradient: The flattened ``(P,)`` gradient.
        delta: Floor on the regularised spectrum.

    Returns:
        The ``(P,)`` uphill direction $p$; GRAPE steps along $-p$.
    """
    eigenvalues, u = jnp.linalg.eigh(hessian)
    # Shift only if the spectrum reaches below delta.
    shift = jnp.maximum(0.0, delta - jnp.min(eigenvalues))
    regularised = eigenvalues + shift
    cfac = jax.scipy.linalg.cho_factor(u @ (jnp.diag(regularised) @ u.conj().T))
    return jax.scipy.linalg.cho_solve(cfac, gradient)


def condition_loop(hessian: Array, g: Array, kappa: float | Array):
    """Scale the augmented Hessian down until its condition number is below ``kappa``.

    Args:
        hessian: The real ``(P, P)`` objective Hessian.
        g: The flattened ``(P,)`` gradient.
        kappa: Target condition number.

    Returns:
        The loop carry ``(cond, iters, alpha, hessian)``, of which only the last
        element is used.
    """
    nparams = hessian.shape[0]
    phi = 0.9  # 0.9 seems to work well
    max_cond = kappa  # 1e4 is from Spinach Settings
    max_iter = 300  # 0.9**300 = 1e-14
    g = jnp.expand_dims(g, axis=1)

    def body_fn(val):
        k, i, a, H = val
        H_aug = jnp.block([[H * a**2, g * a], [g.T * a, 0.0]])
        # Regularize
        sigma = jnp.min(jnp.array([0.0, jnp.min(jnp.linalg.eigvalsh(H_aug))]))
        H_aug = H_aug - jnp.eye(H_aug.shape[0]) * sigma
        # Grab original Hamiltonian
        H = H_aug[:nparams, :nparams] / a**2
        return jnp.linalg.cond(H), i + 1, a * phi, H

    def cond_fn(val):
        # If kappa is larger than our target condition number, stop
        cond1 = val[0] > max_cond
        # Stop at max iterations
        cond2 = val[1] < max_iter
        return jax.lax.bitwise_and(cond1, cond2)

    # set initial alpha
    alpha_0 = 1.0  # Other choices are possible but this seems to work well.
    return jax.lax.while_loop(cond_fn, body_fn, (jnp.inf, 0, alpha_0, hessian))


def newton_rfo_step(hessian: Array, gradient: Array, kappa: float | Array) -> Array:
    r"""Solve $H_{\text{reg}}p = g$ with RFO conditioning-driven regularisation.

    Args:
        hessian: The real ``(P, P)`` objective Hessian.
        gradient: The flattened ``(P,)`` gradient.
        kappa: Target condition number.

    Returns:
        The ``(P,)`` uphill direction $p$; GRAPE steps along $-p$.
    """
    # Regularize in loop
    _, _, _, hessian = condition_loop(hessian, gradient, kappa)
    # Symmetrize
    hessian = jnp.real(hessian + hessian.T) / 2
    # Cholesky solve
    cfac = jax.scipy.linalg.cho_factor(hessian)
    return jax.scipy.linalg.cho_solve(cfac, gradient)
