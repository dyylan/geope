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

**All six rules have one shape:** an *uphill* direction and a *negative* step.

| rule | $p$ | $\mathrm dt$ |
|---|---|---|
| `GradientDescent` | $\nabla C$ | $-\eta$ |
| `Adam` | $\hat m/(\sqrt{\hat v}+\varepsilon)$ | $-\eta$ |
| `LBFGS` | $H_m\nabla C$, $H_m$ implicit in $m$ stored pairs | Armijo or strong-Wolfe backtrack on $[-t_{\max}, 0]$ |
| `NewtonTRM` / `NewtonRFO` | $H_{\text{reg}}^{-1}\nabla C$ | Armijo or strong-Wolfe backtrack on $[-t_{\max}, 0]$ |
| `NewtonSaddleFree` | $\lvert H\rvert^{+}\nabla C$ | Armijo or strong-Wolfe backtrack on $[-t_{\max}, 0]$ |

Adam fits because its per-coordinate rescaling is a *preconditioner on the
direction*, not a step size. That this is GEOPE's convention — ``coeffs`` uphill,
the accepted step negative, `GeometricContext.slope` positive at a descent
direction — is what lets `geope.line_searches._armijo_line_search` serve here
**unchanged**, and it is why `newton_trm_step` returns $H^{-1}g$ with no minus
sign: on this convention that already *is* the direction.

**The three Newton rules differ only in how they regularise $H$**, and the choice
matters because the cost Hessian is *singular* near a solution: the solutions form
a manifold whose tangent directions lie in $\ker\nabla^2C$, so
$\mathrm{cond}(H)$ runs to $10^{12}$ and beyond. :class:`NewtonTRM` and
:class:`NewtonRFO` restore invertibility by **shifting**, which on such a spectrum
means the shift dominates whatever curvature survives and the direction collapses
onto $\nabla C$ — second-order behaviour degenerating to first-order.
:class:`NewtonSaddleFree` **discards** the null space instead, which is what keeps
the curvature that is actually there.

**What an optimiser may read off the context.** Tier 0′ (``value_and_grad``,
``gradient``), tier 1 ``slope``, tier 2 ``cost_hessian``, and tier 3 along the ray
after ``set_direction``. It must **not** touch ``point``, ``infidelity``,
``fidelity``, ``jacobian`` or ``A``: the first three re-exponentiate the pulse
that ``value_and_grad`` already propagated, and ``A`` would trace a matrix
logarithm no gradient method needs. Because every context quantity is lazy,
obeying that is all it takes to keep the logarithm and the Jacobian out of a
GRAPE run entirely — the same dividend `Gecko` gets by reading only ``omegas``.

The orders of information cost different things per step:
:class:`GradientDescent` and :class:`Adam` pay one pullback pass plus a single
propagator to score the landing point; the three Newton rules additionally pay the
dense ``(P, P)`` Hessian, an ``eigh`` of it, and one propagator per backtracking
trial. Setting ``wolfe=True`` adds one *gradient* pullback per trial that gets past
sufficient decrease, since the curvature condition needs $\phi'(\alpha)$ and not
just $\phi(\alpha)$ — roughly twice the per-trial cost, in exchange for a step that
cannot be arbitrarily short.

:class:`LBFGS` sits between the two: it is second-order in *behaviour* but pays no
Hessian at all. Its direction costs $O(mP)$ in vector operations — the two-loop
recursion over $m$ stored $(s, y)$ pairs forms no matrix and decomposes nothing —
so the whole step is one pullback plus the line search. That is the one rule for
which the $O(P^3)$ ``eigh`` argument against the Newton family does not apply; it
gives up the ability to *exploit* negative curvature instead, since the curvature
condition keeps its implicit $H_m$ positive definite.

Cross-step state is a JAX pytree threaded through the jitted update (never a
mutated attribute — a jitted closure traces once). Every optimiser carries
``{"n_eval"}``, the count of infidelity evaluations it spent on the step;
:class:`Adam` adds its moments, :class:`LBFGS` its ring buffer of curvature pairs
and the previous iterate and gradient, and the Newton rules the warm-started trial
step. ``Grape.optimize`` re-``init()``s the state at the start of every run.
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
    r"""Shared machinery for the three regularised-Newton rules.

    They differ only in how they turn the raw ``(P, P)`` Hessian into an uphill
    direction; the step sizing is shared. By default it is
    `geope.line_searches._armijo_line_search` — the same backtracking the Armijo
    line searches use, reused unchanged because GRAPE speaks GEOPE's convention:
    the bracket is one-sided $[-t_{\max}, 0]$ and
    $s = \langle\nabla C, H_{\text{reg}}^{-1}\nabla C\rangle > 0$ is exactly its
    documented descent slope. Setting ``wolfe=True`` swaps in
    :func:`_strong_wolfe_line_search` instead.

    The trial step is warm-started at ``clip(increase * dt_prev, a, 0)`` and shrunk
    by ``beta`` until sufficient decrease holds, reproducing the growth and cap of
    the transform this replaced. ``max_step = 1.0`` means the first trial of the
    first step is the **full Newton step**, which is the right default: the Newton
    direction already carries its own scale, so there is no $1/G$ renormalisation
    of the kind `geope.Geope` applies to its own bracket.

    Args:
        c1: Sufficient-decrease constant, for both searches. Defaults to 1e-4.
        beta: Backtracking contraction factor in ``(0, 1)``; Armijo only.
            Defaults to 0.8.
        t_min: Minimum step magnitude before the search gives up. Defaults to
            1e-10, which allows ~103 contractions at ``beta=0.8`` from a full step.
        max_step: Magnitude cap on the trial step. Defaults to 1.0.
        increase: Growth factor applied to the previous step before capping.
            Defaults to 1.5; ``1.0`` keeps the previous step as the guess.
        wolfe: Use the strong-Wolfe search rather than backtracking Armijo.
            Defaults to ``False``. See :func:`_strong_wolfe_line_search` for what
            it buys and what it costs.
        c2: Strong-Wolfe curvature constant, unused when ``wolfe`` is ``False``.
            Defaults to 0.9, the standard Newton value.

    All seven are **keyword-only**. Dataclass inheritance places a base's fields
    ahead of a subclass's, so without that ``NewtonTRM(0.1)`` would set ``c1``
    rather than ``delta`` — the same trap
    `geope.geometry.stiefel.stiefel.Stiefel.projective` documents.
    """

    c1: float = field(default=1e-4, kw_only=True)
    beta: float = field(default=0.8, kw_only=True)
    t_min: float = field(default=1e-10, kw_only=True)
    max_step: float = field(default=1.0, kw_only=True)
    increase: float = field(default=1.5, kw_only=True)
    wolfe: bool = field(default=False, kw_only=True)
    c2: float = field(default=0.9, kw_only=True)

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
        hessian = ctx.cost_hessian
        # Solve in the reals: the Hessian is real and the gradient real-valued, so
        # carrying the pulse's complex dtype into the solve buys nothing.
        flat_grad = jnp.real(grad).flatten()
        flat = self.direction(hessian, flat_grad)
        coeffs = flat.reshape(grad.shape).astype(grad.dtype)
        ctx.set_direction(coeffs)

        # Read every cached_property *before* entering `lax` control flow: a
        # raising cached_property memoises nothing, so a first read inside a loop
        # body is a trap.
        slope = ctx.slope

        a = jnp.asarray(-self.max_step, jnp.float64)
        # Warm start, as the transform this replaced did. A previous step of
        # exactly 0 (a search that gave up) would otherwise pin the bracket shut
        # for the rest of the run, so fall back to the full bracket there.
        warm = jnp.clip(self.increase * state["dt"], a, 0.0)
        a_eff = jnp.where(warm == 0.0, a, warm)

        if self.wolfe:
            # `self.wolfe` is a Python bool, so this branch is resolved at trace
            # time and neither search is traced into the other's program.
            #
            # The exact directional curvature of the model is free here: the
            # Hessian is already formed, so p^T H p costs one matvec and gives
            # the quadratic model's minimiser as the first trial.
            curvature = flat @ (hessian @ flat)
            dt, value_at_dt, n_eval = _strong_wolfe_line_search(
                ctx,
                coeffs,
                value,
                slope,
                curvature,
                max_step=self.max_step,
                c1=self.c1,
                c2=self.c2,
            )
        else:
            # ctx.slope is <grad, coeffs> = g^T H_reg^-1 g > 0 for the positive
            # definite regularised Hessian, so this is a genuine
            # sufficient-decrease test rather than the relaxation a same-signed
            # slope would give.
            dt, value_at_dt, n_eval = _armijo_line_search(
                ctx.infidelity_at,
                a_eff,
                F0=value,
                s=slope,
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


@dataclass(frozen=True)
class NewtonSaddleFree(_BacktrackingNewton):
    r"""Saddle-free Newton — regularise by *discarding* the null space, not shifting.

    Solves $p = \lvert H\rvert^{+}\nabla C$: diagonalise, invert the eigenvalue
    **magnitudes**, and drop the directions whose magnitude falls below ``rcond``
    times the largest. Prefer this to :class:`NewtonTRM` and :class:`NewtonRFO`
    whenever the Hessian is rank-deficient, which for gate synthesis it is — the
    solutions form a manifold, so $\nabla^2C$ inherits its tangent directions as a
    null space and $\mathrm{cond}(H)$ reaches $10^{12}$ near a solution. A shift
    large enough to make that invertible is large enough to swamp the surviving
    curvature; truncation is not.

    Args:
        rcond: Relative eigenvalue cutoff. Defaults to 1e-8.
        c1: Sufficient-decrease constant. Defaults to 1e-4.
        beta: Backtracking contraction factor. Defaults to 0.8.
        t_min: Minimum step magnitude. Defaults to 1e-10.
        max_step: Magnitude cap on the trial step. Defaults to 1.0.
        increase: Growth factor on the previous step. Defaults to 1.5.
        wolfe: Use the strong-Wolfe search. Defaults to ``False``.
        c2: Strong-Wolfe curvature constant. Defaults to 0.9.
    """

    name = "newton_saddle_free"
    rcond: float = 1e-8

    def direction(self, hessian, gradient):
        return newton_saddle_free_step(hessian, gradient, self.rcond)


@dataclass(frozen=True)
class LBFGS(Optimizer):
    r"""Limited-memory BFGS — second-order behaviour with no Hessian.

    The direction is $p = H_m\nabla C$, where $H_m$ approximates the inverse
    Hessian from the last ``memory`` curvature pairs
    $s_k=\phi_{k+1}-\phi_k$, $y_k=\nabla C_{k+1}-\nabla C_k$. It is never
    formed: the two-loop recursion (N&W Alg. 7.4) applies it to the gradient
    with inner products and ``axpy`` updates alone, $O(mP)$ per step. **That is
    the whole point of this rule** — the three Newton rules each pay a dense
    $(P,P)$ Hessian and an $O(P^3)$ ``eigh``, and this one pays neither, so it
    never reads ``ctx.cost_hessian``.

    What it gives up is negative curvature. The curvature condition below keeps
    $H_m$ positive definite, so unlike :class:`NewtonSaddleFree` it cannot
    follow an indefinite direction downhill; near a saddle it degrades to a
    well-scaled descent method rather than escaping along $\lambda<0$.

    **``wolfe`` defaults to ``True`` here, unlike the Newton rules.** Only the
    Wolfe curvature condition guarantees $s^\intercal y>0$, which is what keeps
    $H_m$ positive definite in exact arithmetic. The guard below runs on both
    paths regardless — a pair failing the curvature test is *skipped*, not
    damped — so ``wolfe=False`` is safe, merely weaker.

    **What the Wolfe path does at the floating-point floor.** Once the
    achievable decrease $\sim\lVert g\rVert^2/\lambda$ drops below the
    resolution of $\phi$ itself, sufficient decrease is testing rounding noise:
    the search exhausts its bracket and zoom budget and returns $\mathrm dt=0$.
    That is a *graceful* stall — the iterate stops moving, the zero-length pair
    is rejected by the curvature guard so the memory is not corrupted, and the
    accuracy reached matches what ``scipy``'s L-BFGS-B stops at on the same
    problem. It is not free, though: each such step spends the full evaluation
    budget. `geope.Grape`'s loop exits on its precision test, so this only
    costs anything when the run plateaus *above* the requested precision.

    Two consequences of the fixed-shape state, both deliberate:

    - ``memory`` is a Python ``int``, so it fixes the buffer shapes at trace
      time and participates in the value ``__eq__`` compile memo. Changing it
      re-traces `geope.Grape`'s update step; that is correct, not a leak.
    - The two loops are unrolled into the jaxpr, ~6 equations per slot. At the
      default 10 that is negligible beside the chart's ``lax.scan``; a memory in
      the hundreds would not be.

    Args:
        memory: Number of curvature pairs $m$ to retain. Defaults to 10;
            5--20 is the usual range and the cost is linear in it.
        c1: Sufficient-decrease constant, for both searches. Defaults to 1e-4.
        beta: Backtracking contraction factor in ``(0, 1)``; Armijo only.
            Defaults to 0.8.
        t_min: Minimum step magnitude before the search gives up. Defaults to
            1e-10.
        max_step: Magnitude cap on the trial step. Defaults to 1.0, so the
            first trial is the **unit step** — the one L-BFGS's scaling is built
            to make acceptable.
        increase: Growth factor applied to the previous step before capping.
            Defaults to 1.5; Armijo only.
        wolfe: Use the strong-Wolfe search rather than backtracking Armijo.
            Defaults to ``True``; see above.
        c2: Strong-Wolfe curvature constant. Defaults to 0.9.

    All but ``memory`` are keyword-only, as in `_BacktrackingNewton`.
    """

    name = "lbfgs"
    memory: int = 10
    c1: float = field(default=1e-4, kw_only=True)
    beta: float = field(default=0.8, kw_only=True)
    t_min: float = field(default=1e-10, kw_only=True)
    max_step: float = field(default=1.0, kw_only=True)
    increase: float = field(default=1.5, kw_only=True)
    wolfe: bool = field(default=True, kw_only=True)
    c2: float = field(default=0.9, kw_only=True)

    #: Relative floor on $s^\intercal y$ for a pair to be kept.
    _curvature_rtol = 1e-12

    def init(self, free_params):
        shape = free_params.shape
        # float64 throughout: the pulse is complex128 with an identically-zero
        # imaginary part, and every recursion below is real arithmetic.
        return {
            "s": jnp.zeros((self.memory, *shape), jnp.float64),
            "y": jnp.zeros((self.memory, *shape), jnp.float64),
            "rho": jnp.zeros(self.memory, jnp.float64),
            "prev_x": jnp.zeros(shape, jnp.float64),
            "prev_g": jnp.zeros(shape, jnp.float64),
            "count": jnp.asarray(0, jnp.int32),
            "dt": jnp.asarray(-self.max_step, jnp.float64),
            "n_eval": jnp.asarray(0, jnp.int32),
        }

    def _push(self, state, s_new, y_new, fresh):
        r"""Append a curvature pair, newest last, dropping the oldest.

        ``jnp.roll`` by a static amount keeps the shapes fixed and avoids a
        traced index entirely. A rejected or not-yet-formed pair is stored with
        $\rho=0$, which makes it an **exact** no-op in both loops rather than
        something the recursion has to branch around.
        """
        sy = jnp.vdot(s_new, y_new)
        scale = jnp.linalg.norm(s_new) * jnp.linalg.norm(y_new)
        keep = jnp.logical_and(fresh, sy > self._curvature_rtol * scale)
        # Both arms of a `where` are evaluated, so the reciprocal needs a safe
        # denominator or a rejected pair poisons `rho` with inf before the mask
        # can drop it.
        rho_new = jnp.where(keep, 1.0 / jnp.where(keep, sy, 1.0), 0.0)
        zero = jnp.zeros_like(s_new)
        return {
            "s": jnp.roll(state["s"], -1, axis=0)
            .at[-1]
            .set(jnp.where(keep, s_new, zero)),
            "y": jnp.roll(state["y"], -1, axis=0)
            .at[-1]
            .set(jnp.where(keep, y_new, zero)),
            "rho": jnp.roll(state["rho"], -1).at[-1].set(rho_new),
        }

    def _two_loop(self, gradient, s, y, rho):
        r"""$H_m g$ by the two-loop recursion, newest pair last.

        Returns the **uphill** direction: the textbook's leading minus is
        dropped, because on GEOPE's convention ``coeffs`` points uphill and the
        accepted step is negative.
        """
        # gamma = s.y / y.y from the newest retained pair; 1.0 on a cold start,
        # which makes the first direction exactly steepest descent.
        yy = jnp.vdot(y[-1], y[-1])
        usable = jnp.logical_and(rho[-1] != 0.0, yy > 0.0)
        gamma = jnp.where(
            usable, jnp.vdot(s[-1], y[-1]) / jnp.where(usable, yy, 1.0), 1.0
        )

        q = gradient
        alpha = [None] * self.memory
        for i in range(self.memory - 1, -1, -1):  # newest -> oldest
            alpha[i] = rho[i] * jnp.vdot(s[i], q)
            q = q - alpha[i] * y[i]
        r = gamma * q
        for i in range(self.memory):  # oldest -> newest
            beta = rho[i] * jnp.vdot(y[i], r)
            r = r + s[i] * (alpha[i] - beta)
        return r

    def __call__(self, ctx, state):
        value, grad = ctx.value_and_grad
        g = jnp.real(grad)
        x = jnp.real(ctx.free_params)

        # The pair closing the *previous* step. `prev_x` rather than the last
        # `dt * coeffs`, so it measures the change GRAPE actually realised.
        fresh = state["count"] > 0
        memory = self._push(state, x - state["prev_x"], g - state["prev_g"], fresh)

        direction = self._two_loop(g, memory["s"], memory["y"], memory["rho"])
        coeffs = direction.astype(grad.dtype)
        ctx.set_direction(coeffs)

        # Read every cached_property before entering `lax` control flow: a
        # raising cached_property memoises nothing.
        slope = ctx.slope

        if self.wolfe:
            # Resolved at trace time, so neither search is traced into the
            # other's program. `curvature=0` is not a fudge: with no Hessian
            # there is no model curvature to seed with, and a non-positive one
            # makes the search start at `alpha_max` — the unit step, which is
            # exactly L-BFGS's intended first trial.
            dt, value_at_dt, n_eval = _strong_wolfe_line_search(
                ctx,
                coeffs,
                value,
                slope,
                jnp.asarray(0.0, jnp.float64),
                max_step=self.max_step,
                c1=self.c1,
                c2=self.c2,
            )
        else:
            a = jnp.asarray(-self.max_step, jnp.float64)
            warm = jnp.clip(self.increase * state["dt"], a, 0.0)
            a_eff = jnp.where(warm == 0.0, a, warm)
            dt, value_at_dt, n_eval = _armijo_line_search(
                ctx.infidelity_at,
                a_eff,
                F0=value,
                s=slope,
                c1=self.c1,
                beta=self.beta,
                t_min=self.t_min,
            )
        # A non-finite trial means the step is unusable; stand still rather than
        # propagate a nan into the parameters.
        finite = jnp.isfinite(value_at_dt)
        dt = jnp.where(finite, dt, 0.0)
        value_at_dt = jnp.where(finite, value_at_dt, value)
        return OptimizerResult(
            dt,
            coeffs,
            value_at_dt,
            {
                **memory,
                "prev_x": x,
                "prev_g": g,
                "count": state["count"] + 1,
                "dt": dt,
                "n_eval": n_eval,
            },
        )


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


def newton_saddle_free_step(
    hessian: Array, gradient: Array, rcond: float | Array = 1e-8
) -> Array:
    r"""Solve $p = \lvert H\rvert^{+}g$ — invert $|\lambda|$, discard the null space.

    Two things here are easy to get wrong, and both are load-bearing.

    **Invert the magnitudes, not the eigenvalues.** A plain pseudo-inverse of an
    indefinite $H$ inverts $\lambda$ *with its sign*, so the negative-curvature
    directions are climbed rather than descended and $p$ need not be an uphill
    direction at all — on GEOPE's convention that makes ``ctx.slope`` negative and
    the sufficient-decrease test unsatisfiable, so the search collapses to
    ``t_min`` and the run stalls. Inverting $|\lambda|$ instead (saddle-free
    Newton, Dauphin *et al.*) follows negative curvature *downhill* and gives

    $$\langle g, p\rangle = \sum_i \tilde g_i^2/\lvert\lambda_i\rvert > 0$$

    unconditionally, which is exactly the descent guarantee
    `_BacktrackingNewton`'s line search assumes.

    **Truncate the raw spectrum, not a shifted one.** Shifting first and
    truncating afterwards is worse than either alone: the shift lifts the
    near-null eigenvalues to just *above* the cutoff, so instead of being
    discarded they survive and are inverted, amplifying numerical noise in the
    directions that carry the least information.

    Args:
        hessian: The real ``(P, P)`` objective Hessian.
        gradient: The flattened ``(P,)`` gradient.
        rcond: Relative cutoff; eigenvalues with
            $|\lambda| \le \texttt{rcond}\cdot\max_j|\lambda_j|$ are discarded.

    Returns:
        The ``(P,)`` uphill direction $p$; GRAPE steps along $-p$.
    """
    eigenvalues, u = jnp.linalg.eigh(hessian)
    magnitude = jnp.abs(eigenvalues)
    keep = magnitude > rcond * jnp.max(magnitude)
    inverse = jnp.where(keep, 1.0 / jnp.where(keep, magnitude, 1.0), 0.0)
    return u @ (inverse * (u.T @ gradient))


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
        _, i, a, H = val
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


# ---------------------------------------------------------------------------
# The strong-Wolfe line search — Nocedal & Wright, *Numerical Optimization* 2nd
# ed., Algorithms 3.5 (bracket) and 3.6 (zoom), with the safeguarded interpolant
# of their §3.5.
#
# This is the one 1-D minimiser that does *not* live in `geope.line_searches`,
# and deliberately: the curvature condition needs the directional derivative
# phi'(alpha) at trial points, i.e. `Manifold.value_and_grad` at an arbitrary
# pulse, which is a GRAPE-side object the geodesic line searches never touch.
#
# Everything below runs on the *positive* step length alpha, so the book reads
# verbatim; GEOPE's negative-step convention is restored on the way out.
#
#     phi(alpha)  = C(phi - alpha * p)
#     phi'(alpha) = -<grad(phi - alpha * p), p>,  so phi'(0) = -ctx.slope < 0.
#
# Every `lax` carry element is float64 or int32 with a fixed structure, which is
# what `while_loop` requires.
# ---------------------------------------------------------------------------

_WOLFE_MAX_BRACKET = 12
_WOLFE_MAX_ZOOM = 20
_WOLFE_BIG = 1e300


def _cubicmin(a, fa, da, b, fb, c, fc):
    """Minimiser of the cubic through ``(a, fa, da)``, ``(b, fb)`` and ``(c, fc)``.

    Returns ``nan`` when the system is singular or the cubic has no interior
    minimum, which the caller treats as "fall through to the quadratic". Every
    division is guarded, since both arms of a ``where`` are evaluated.
    """
    db, dc = b - a, c - a
    denom = (db * dc) ** 2 * (db - dc)
    safe_denom = jnp.where(jnp.abs(denom) < 1e-300, 1.0, denom)

    r0 = fb - fa - da * db
    r1 = fc - fa - da * dc
    aa = (dc**2 * r0 - db**2 * r1) / safe_denom
    bb = (-(dc**3) * r0 + db**3 * r1) / safe_denom

    radicand = bb * bb - 3.0 * aa * da
    safe_aa = jnp.where(jnp.abs(aa) < 1e-300, 1.0, aa)
    root = a + (-bb + jnp.sqrt(jnp.maximum(radicand, 0.0))) / (3.0 * safe_aa)
    ok = (
        (jnp.abs(denom) >= 1e-300)
        & (jnp.abs(aa) >= 1e-300)
        & (radicand >= 0.0)
        & jnp.isfinite(root)
    )
    return jnp.where(ok, root, jnp.nan)


def _quadmin(a, fa, da, b, fb):
    """Minimiser of the quadratic through ``(a, fa, da)`` and ``(b, fb)``.

    ``nan`` when the fitted curvature is non-positive, i.e. when the model has no
    minimum to offer.
    """
    db = b - a
    curvature = fb - fa - da * db
    safe = jnp.where(jnp.abs(curvature) < 1e-300, 1.0, curvature)
    root = a - 0.5 * da * db * db / safe
    return jnp.where((curvature > 0.0) & jnp.isfinite(root), root, jnp.nan)


def _wolfe_interpolate(lo, f_lo, d_lo, hi, f_hi, rec, f_rec):
    """A safeguarded trial point strictly inside ``[lo, hi]``.

    Cubic through the three most recent points, falling back to the quadratic and
    then to bisection, and clipped away from both endpoints by 10% of the
    interval — N&W's safeguard, and what stops the zoom stalling against an end.
    """
    left, right = jnp.minimum(lo, hi), jnp.maximum(lo, hi)
    width = right - left
    inner_lo, inner_hi = left + 0.1 * width, right - 0.1 * width

    candidate = _cubicmin(lo, f_lo, d_lo, hi, f_hi, rec, f_rec)
    candidate = jnp.where(
        jnp.isnan(candidate), _quadmin(lo, f_lo, d_lo, hi, f_hi), candidate
    )
    bisect = 0.5 * (lo + hi)
    candidate = jnp.where(jnp.isnan(candidate), bisect, candidate)
    outside = (candidate < inner_lo) | (candidate > inner_hi)
    return jnp.where(outside, bisect, candidate)


def _strong_wolfe_line_search(
    ctx,
    coeffs: Array,
    value: Array,
    slope: Array,
    curvature: Array,
    max_step: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
) -> tuple[Array, Array, Array]:
    r"""Strong-Wolfe step along ``coeffs``; N&W Alg. 3.5 + 3.6.

    Enforces **both** Wolfe conditions,

    $$\phi(\alpha)\le\phi(0)+c_1\alpha\phi'(0),\qquad
      \lvert\phi'(\alpha)\rvert\le c_2\lvert\phi'(0)\rvert,$$

    where backtracking Armijo tests only the first and can therefore accept an
    arbitrarily short step. The curvature condition is what rules that out, and it
    is the reason this search needs a gradient per trial.

    **The first trial is the model minimiser** $-\phi'(0)/(p^\intercal Hp)$, which
    costs nothing extra: a Newton rule has already formed $H$, so the exact
    directional curvature is one matvec away.

    Args:
        ctx: The step's `geope.geometry.GeometricContext`, with the direction
            already set. Its ``manifold.value_and_grad`` supplies $\phi'$.
        coeffs: The uphill direction $p$, shaped like ``ctx.free_params``.
        value: $\phi(0)$, the infidelity at the base point.
        slope: ``ctx.slope`` $=\langle\nabla C,p\rangle>0$, so $\phi'(0)=-$``slope``.
        curvature: $p^\intercal Hp$, used only to seed the first trial.
        max_step: Cap on the step length $\alpha$.
        c1: Sufficient-decrease constant.
        c2: Curvature constant.

    Returns:
        ``(dt, value, n_eval)`` with ``dt = -alpha`` **negative**, matching
        :func:`geope.line_searches._armijo_line_search`'s contract. On exhaustion
        it returns the best point it bracketed.
    """
    f64 = lambda x: jnp.asarray(x, dtype=jnp.float64)
    i32 = lambda x: jnp.asarray(x, dtype=jnp.int32)

    f0 = f64(value)
    d0 = -f64(slope)  # phi'(0) < 0 for a descent direction
    alpha_max = f64(max_step)
    free = ctx.free_params
    # Hoisted: `value_and_grad` is a cached_property returning a jitted callable,
    # and a first read inside a `lax` loop body would be a trap.
    value_and_grad = ctx.manifold.value_and_grad

    def phi(alpha):
        return f64(ctx.infidelity_at(-alpha))

    def dphi(alpha):
        f_a, g_a = value_and_grad(free - alpha * coeffs)
        return f64(f_a), -f64(jnp.sum(jnp.real(g_a) * jnp.real(coeffs)))

    # A non-positive model curvature has no minimum to offer; take the full step.
    alpha_1 = jnp.where(
        curvature > 0.0, jnp.clip(-d0 / curvature, 1e-30, alpha_max), alpha_max
    )

    # --- phase 1: bracket ---------------------------------------------------
    # status 0 running, 1 accepted, 2 hand (lo, hi) to the zoom.
    bracket_init = (
        i32(0),
        i32(0),
        f64(0.0),
        f0,
        d0,  # i, status, a_prev, f_prev, d_prev
        alpha_1,
        f64(0.0),
        f0,  # a, a_star, f_star
        f64(0.0),
        f0,
        d0,
        alpha_1,
        f64(_WOLFE_BIG),  # lo, f_lo, d_lo, hi, f_hi
        i32(0),  # n_eval
    )

    def bracket_cond(carry):
        return (carry[1] == 0) & (carry[0] < _WOLFE_MAX_BRACKET)

    def bracket_body(carry):
        i, _s, a_prev, f_prev, d_prev, a, _as, _fs, _lo, _flo, _dlo, _hi, _fhi, n = (
            carry
        )
        f_a = phi(a)
        n = n + 1
        # Armijo failed, or we went uphill relative to the previous trial: the
        # minimum lies between a_prev and a.
        bad = (f_a > f0 + c1 * a * d0) | ((f_a >= f_prev) & (i > 0))

        _f, d_a = dphi(a)
        n = n + 1
        curvature_ok = jnp.abs(d_a) <= -c2 * d0
        went_up = d_a >= 0.0

        status = jnp.where(
            bad,
            i32(2),
            jnp.where(curvature_ok, i32(1), jnp.where(went_up, i32(2), i32(0))),
        )
        # zoom(lo, hi) is (a_prev, a) when Armijo failed and (a, a_prev) when the
        # slope turned positive.
        new_lo = jnp.where(bad, a_prev, jnp.where(went_up, a, a_prev))
        new_f_lo = jnp.where(bad, f_prev, jnp.where(went_up, f_a, f_prev))
        new_d_lo = jnp.where(bad, d_prev, jnp.where(went_up, d_a, d_prev))
        new_hi = jnp.where(bad, a, jnp.where(went_up, a_prev, a))
        new_f_hi = jnp.where(bad, f_a, jnp.where(went_up, f_prev, f_a))

        a_next = jnp.minimum(alpha_max, 2.0 * a)
        # A trial already at the cap cannot expand; accept rather than spin.
        status = jnp.where((status == 0) & (a >= alpha_max), i32(1), status)

        return (
            i + 1,
            status,
            a,
            f_a,
            d_a,
            a_next,
            a,
            f_a,
            new_lo,
            new_f_lo,
            new_d_lo,
            new_hi,
            new_f_hi,
            n,
        )

    carry = jax.lax.while_loop(bracket_cond, bracket_body, bracket_init)
    status, a_star, f_star = carry[1], carry[6], carry[7]
    lo, f_lo, d_lo, hi, f_hi, n_eval = carry[8:14]
    # Bracketing ran out of iterations without deciding: zoom on what it holds.
    status = jnp.where(status == 0, i32(2), status)

    # --- phase 2: zoom ------------------------------------------------------
    zoom_init = (
        i32(0),
        status == 1,  # already accepted -> skip the zoom entirely
        lo,
        f_lo,
        d_lo,
        hi,
        f_hi,
        hi,
        f_hi,  # rec, f_rec
        jnp.where(status == 1, a_star, lo),
        jnp.where(status == 1, f_star, f_lo),
        n_eval,
    )

    def zoom_cond(carry):
        return (~carry[1]) & (carry[0] < _WOLFE_MAX_ZOOM)

    def zoom_body(carry):
        j, _d, lo, f_lo, d_lo, hi, f_hi, rec, f_rec, a_star, f_star, n = carry
        a_j = _wolfe_interpolate(lo, f_lo, d_lo, hi, f_hi, rec, f_rec)
        f_j = phi(a_j)
        n = n + 1

        # Sufficient decrease fails, or we are above the current low end: shrink
        # from the high side and do not spend a gradient.
        shrink_hi = (f_j > f0 + c1 * a_j * d0) | (f_j >= f_lo)
        _f, d_j = dphi(a_j)
        n = n + jnp.where(shrink_hi, 0, 1)

        accept = (~shrink_hi) & (jnp.abs(d_j) <= -c2 * d0)
        # phi'(a_j) pointing back across the bracket: the far end moves to lo.
        flip = (~shrink_hi) & (~accept) & (d_j * (hi - lo) >= 0.0)

        new_rec = jnp.where(shrink_hi, hi, lo)
        new_f_rec = jnp.where(shrink_hi, f_hi, f_lo)
        new_hi = jnp.where(shrink_hi, a_j, jnp.where(flip, lo, hi))
        new_f_hi = jnp.where(shrink_hi, f_j, jnp.where(flip, f_lo, f_hi))
        new_lo = jnp.where(shrink_hi, lo, a_j)
        new_f_lo = jnp.where(shrink_hi, f_lo, f_j)
        new_d_lo = jnp.where(shrink_hi, d_lo, d_j)

        # Keep the best point seen either way, so an exhausted zoom still returns
        # a step that satisfied sufficient decrease.
        better = f_j < f_star
        return (
            j + 1,
            accept,
            new_lo,
            new_f_lo,
            new_d_lo,
            new_hi,
            new_f_hi,
            new_rec,
            new_f_rec,
            jnp.where(accept | better, a_j, a_star),
            jnp.where(accept | better, f_j, f_star),
            n,
        )

    zoomed = jax.lax.while_loop(zoom_cond, zoom_body, zoom_init)
    return -zoomed[9], zoomed[10], zoomed[11]
