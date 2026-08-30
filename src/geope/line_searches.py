"""Pluggable line-search objects for :class:`geope.Geope`.

A line search tunes the scalar geodesic **step size ``t``** along the search
direction GEOPE computes each step (it is not a full-parameter optimiser — that
is the separate :class:`Grape` class). The active object is passed to
``Geope.optimize(line_search=...)``; when omitted it defaults to
:class:`GoldenSection`.

Each line search is a ``@dataclass(frozen=True)`` — immutable config that gets
value-based ``__eq__``/``__hash__``/``__repr__`` for free. The value ``__eq__``
drives GEOPE's compile memo (``Adam(1e-2) == Adam(1e-2)`` ⇒ no recompile) and the
immutability keeps hyperparameter sweeps correct (a config cannot be mutated in
place and silently reuse a stale compiled function).

**The call contract.** ``Geope`` builds one
:class:`~geope.geometry.GeometricContext` per step (inside the jitted update) and
calls ``line_search(ctx, a, b, state)``, which returns a :class:`LineSearchResult`
``(dt, value, state)``. The context carries **only geometry**; the bracket
``[a, b]`` and the threaded state are the search's own bookkeeping and travel
alongside it, which is what lets a consumer with no bracket (``Gecko``) share the
same context type. Every quantity on the context is lazy, and the line search is
traced *inside* the jitted update, so a quantity a method never reads is never
traced: zeroth-order methods pay nothing for the ``logm``/HVP the geometry needs.

Each search declares which scalar objective it minimises as the class attribute
``objective`` — ``"infidelity"`` (``ctx.infidelity_at``) or ``"distance"``
(``ctx.distance_at``, the squared geodesic distance). It is deliberately *not* a
dataclass field, so it stays out of the value ``__eq__`` that drives the compile
memo. ``value`` in the result is that objective at the accepted step — every 1-D
minimiser already computes it — and ``Geope.optimize`` decides whether the step
made progress by comparing it against the same objective at ``t = 0``.

The three orders of information cost different things per GEOPE step:
:class:`GoldenSection` and :class:`Adam` are zeroth-order and evaluate only the
cheap ``ctx.infidelity_at``; :class:`Armijo` is first-order and evaluates the
``logm``-bearing ``ctx.distance_at`` a few times but forms no derivative;
:class:`QuadraticArmijo` is second-order and additionally pays one directional
HVP to seed its step from the exact curvature. None of them pays for a second
matrix logarithm: the context's single ``ctx.A`` serves the geodesic step and the
line search alike.

Cross-step state is a JAX pytree threaded through the jitted update (mirroring
``Grape.optimizer_state``): a jitted closure traces once, so persistent state
must enter/leave as an argument/result rather than as a mutated attribute. The
state is line-search-owned and opaque to GEOPE — every search carries
``{"n_eval"}`` (the per-step count of 1-D-objective evaluations it spent), and
:class:`Adam` additionally carries ``{"t_prev"}`` (warm-start).
``Geope.optimize`` re-``init()``s the state at the start of every run.
"""

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class LineSearchResult(NamedTuple):
    """What a :class:`LineSearch` returns.

    Attributes:
        dt: The accepted step size along the search direction.
        value: The search's own ``objective`` at ``dt`` — every 1-D minimiser
            computes this anyway, and ``Geope.optimize`` tests progress against
            it rather than against a quantity the search never minimised.
        state: The new line-search-owned state pytree (always carries
            ``"n_eval"``).
    """

    dt: Array
    value: Array
    state: dict


class LineSearch:
    """Base line search: tunes the scalar geodesic step size ``t``.

    Subclasses are frozen dataclasses (immutable config) that own an opaque
    JAX-pytree state. The base state carries only ``{"n_eval"}`` — the per-step
    count of 1-D-objective evaluations the search spent — which every line
    search reports; stateful searches (e.g. :class:`Adam`) extend it.

    Attributes:
        objective: Which scalar of the context this search minimises,
            ``"infidelity"`` or ``"distance"``. A class attribute, not a
            dataclass field, so it stays out of the compile memo; ``Geope`` reads
            it to interpret :attr:`LineSearchResult.value`.
    """

    name = "line_search"
    objective: str = "infidelity"

    def init(self):
        """Return a fresh state pytree (called once per ``optimize()`` run)."""
        return {"n_eval": jnp.asarray(0, jnp.int32)}

    def __call__(self, ctx, a: Array, b: Array, state: dict) -> LineSearchResult:
        """Choose a step on the bracket ``[a, b]``.

        Args:
            ctx: The step's :class:`~geope.geometry.GeometricContext`, with a
                direction already set.
            a: Bracket endpoint (``Geope`` passes ``-max_step_size / G``).
            b: Bracket endpoint (``Geope`` passes ``0.0``).
            state: This search's threaded state from the previous step.

        Returns:
            A :class:`LineSearchResult`.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class GoldenSection(LineSearch):
    """Golden-section search (the default) — zeroth-order.

    Minimises ``ctx.infidelity_at`` on the bracket, and reads nothing else off
    the context: no logarithm, no Jacobian, no HVP is traced on its account.
    Carries only the base ``{"n_eval"}`` state (the per-step evaluation count).

    Args:
        tol: Convergence tolerance for the search interval. Defaults to 1e-5.
    """

    name = "golden_section"
    tol: float = 1e-5

    def __call__(self, ctx, a, b, state):
        dt, value, n_eval = _golden_section_search(
            ctx.infidelity_at, a, b, tol=self.tol
        )
        return LineSearchResult(dt, value, {"n_eval": n_eval})


@dataclass(frozen=True)
class Adam(LineSearch):
    """1-D Adam line search — zeroth-order (minimises ``ctx.infidelity_at``).

    Args:
        lr: Adam learning rate. Defaults to 0.05.
        num_steps: Number of Adam iterations. Defaults to 30.
        finite_difference: If ``True`` (default), estimate the gradient with a
            finite-difference secant; otherwise use ``jax.value_and_grad``.
        warm_start: If ``True``, seed each step's search from the previous
            step's ``t`` (carried across GEOPE steps via the threaded state).
            Defaults to ``False``.
        fd_step: Probe size for the finite-difference bootstrap. Defaults to 1e-3.
        beta1: First-moment decay. Defaults to 0.9.
        beta2: Second-moment decay. Defaults to 0.999.
        eps: Numerical-stability term. Defaults to 1e-8.
    """

    name = "adam"
    lr: float = 0.05
    num_steps: int = 30
    finite_difference: bool = True
    warm_start: bool = False
    fd_step: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def init(self):
        # t_prev seeds the warm-start within a run; reset fresh each run.
        return {
            "t_prev": jnp.asarray(0.0, jnp.float64),
            "n_eval": jnp.asarray(0, jnp.int32),
        }

    def __call__(self, ctx, a, b, state):
        t0 = state["t_prev"] if self.warm_start else 0.0
        dt, value, n_eval = _adam_line_search(
            ctx.infidelity_at,
            a,
            b,
            lr=self.lr,
            num_steps=self.num_steps,
            finite_difference=self.finite_difference,
            t_init=t0,
            fd_step=self.fd_step,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
        )
        return LineSearchResult(dt, value, {"t_prev": dt, "n_eval": n_eval})


@dataclass(frozen=True)
class Armijo(LineSearch):
    r"""Backtracking Armijo line search — first-order, no curvature.

    The non-quadratic sibling of :class:`QuadraticArmijo`: it seeds the trial
    step at the full bracket step $t_0=a=-t_{\max}$ instead of at the quadratic
    model minimiser $-s/q$, and then enforces sufficient decrease by Armijo
    backtracking on ``ctx.distance_at``. The note's §15 shows this loses nothing
    in correctness — termination follows from the descent slope alone, and the
    curvature only rescales the *first* trial.

    Both quantities the Armijo test needs are free: ``ctx.F0`` is the objective
    at $t=0$, which tier 0 of the context has already computed from its single
    logarithm, and ``ctx.s`` is the exact slope $\langle A,\Omega\rangle$, one
    contraction of tier 0's Jacobian. So this search spends no propagator and no
    logarithm of its own before its first trial, and it uses the exact slope
    rather than the radial $s=2F_0$ of the note's §11 — which held only under
    perfect tangent matching, and not once ``coeffs`` has been renormalised.

    Because it never reads the *curvature*, this line search still works under
    ``param_transform`` (where the chart has no analytic second differential and
    :class:`QuadraticArmijo` raises): the slope comes from the Jacobian, not the
    HVP.

    Args:
        c1: Armijo sufficient-decrease constant. Defaults to 1e-4.
        beta: Backtracking contraction factor in ``(0, 1)``. Defaults to 0.5.
        t_min: Minimum step magnitude before the search gives up. Defaults to 1e-8.
    """

    name = "armijo"
    objective = "distance"
    c1: float = 1e-4
    beta: float = 0.5
    t_min: float = 1e-8

    def __call__(self, ctx, a, b, state):
        dt, value, n_eval = _armijo_line_search(
            ctx.distance_at,
            a,
            F0=ctx.F0,
            s=ctx.s,
            c1=self.c1,
            beta=self.beta,
            t_min=self.t_min,
        )
        return LineSearchResult(dt, value, {"n_eval": n_eval})


@dataclass(frozen=True)
class QuadraticArmijo(LineSearch):
    r"""Quadratic-seeded Armijo line search — second-order (geometry-aware).

    Implements the note *Quadratic-Seeded Armijo Line Search on
    $\mathrm{SU}(N)$*: it reads the exact slope $s=\psi'(0)$ and curvature
    $q=\psi''(0)$ of the squared-geodesic-distance objective off the context
    (``ctx.s`` and ``ctx.q``), seeds the trial step at the local quadratic minimiser
    $-s/q$ (clipped to the bracket, with a full-bracket fallback when the
    curvature is non-positive), and enforces sufficient decrease with Armijo
    backtracking on ``ctx.distance_at``. The per-step evaluation count is recorded
    in the state as ``n_eval``.

    Requires the standard mode; ``ctx.q`` raises under ``param_transform``, where
    the chart has no exponential-product structure for the HVP to exploit. See
    :class:`Armijo` for the first-order variant that drops the curvature (and
    with it the HVP and that restriction).

    Args:
        c1: Armijo sufficient-decrease constant. Defaults to 1e-4.
        beta: Backtracking contraction factor in ``(0, 1)``. Defaults to 0.5.
        t_min: Minimum step magnitude before the search gives up. Defaults to 1e-8.
    """

    name = "quadratic_armijo"
    objective = "distance"
    c1: float = 1e-4
    beta: float = 0.5
    t_min: float = 1e-8

    def __call__(self, ctx, a, b, state):
        dt, value, n_eval = _quadratic_armijo_line_search(
            ctx.distance_at,
            a,
            ctx.s,
            ctx.q,
            ctx.F0,
            c1=self.c1,
            beta=self.beta,
            t_min=self.t_min,
        )
        return LineSearchResult(dt, value, {"n_eval": n_eval})


@dataclass(frozen=True)
class ApproximateQuadraticArmijo(LineSearch):
    r"""Residual-aware quadratic-seeded Armijo line search — second-order.

    Identical to :class:`QuadraticArmijo` except in *which* curvature seeds the
    step: it uses ``ctx.q_exact`` rather than ``ctx.q``. The difference is
    the intrinsic term of $\psi''(0)$,

    $$\psi''(0)=\underbrace{\langle\Omega,\mathcal K_A\Omega\rangle_F}
      _{\text{exact}}+\langle A,\dot\Omega(0)\rangle_F,
      \qquad
      \mathcal K_A=\frac{\operatorname{ad}_A}{2}
        \coth\!\left(\frac{\operatorname{ad}_A}{2}\right),$$

    which :class:`QuadraticArmijo` replaces by $\|\Omega\|_F^2$. That replacement
    is exact only when the achieved tangent $\Omega$ is parallel to the geodesic
    tangent $A$ — i.e. only when GEOPE's least-squares solve for the search
    direction leaves **no residual**. When it does leave one, writing
    $\Xi=\Omega-A$ and using $\mathcal K_AA=A$ gives

    $$\langle\Omega,\mathcal K_A\Omega\rangle_F
      =\|A\|_F^2+2\langle A,\Xi\rangle_F+\langle\Xi,\mathcal K_A\Xi\rangle_F,$$

    so the residual couples into the curvature through the Riemannian Hessian.
    Evaluating the form directly captures all three terms without ever forming
    $\Xi$: the residual is already inside the $\Omega$ that the directional HVP
    returns.

    Because $\mathcal K_A\preceq I$, ``q_exact <= q`` always, so this seeds a
    **longer** step than :class:`QuadraticArmijo`. The extra cost over it is one
    ``eigh`` on a group (see :func:`geope.jax.su_hessian_quadratic_form`; on
    `geope.Stiefel` it is instead one small operator exponential, see
    :func:`geope.jax.stiefel_hessian_quadratic_form`), negligible beside
    the ``logm`` and HVP both already pay. Note that the slope ``s`` needs no
    correction — it is exact for any $\Omega$.

    Requires the standard mode; ``ctx.q_exact`` raises under ``param_transform``.

    Args:
        c1: Armijo sufficient-decrease constant. Defaults to 1e-4.
        beta: Backtracking contraction factor in ``(0, 1)``. Defaults to 0.5.
        t_min: Minimum step magnitude before the search gives up. Defaults to 1e-8.
    """

    name = "approximate_quadratic_armijo"
    objective = "distance"
    c1: float = 1e-4
    beta: float = 0.5
    t_min: float = 1e-8

    def __call__(self, ctx, a, b, state):
        dt, value, n_eval = _quadratic_armijo_line_search(
            ctx.distance_at,
            a,
            ctx.s,
            ctx.q_exact,
            ctx.F0,
            c1=self.c1,
            beta=self.beta,
            t_min=self.t_min,
        )
        return LineSearchResult(dt, value, {"n_eval": n_eval})


# ---------------------------------------------------------------------------
# Raw 1-D minimisers — private, JIT-compatible backends for the objects above.
# Not part of the public API; imported directly by tests.
# ---------------------------------------------------------------------------


def _golden_section_search(
    f: Callable[[Array], Array],
    a_init: float | Array,
    b_init: float | Array,
    tol: float = 1e-5,
) -> tuple[Array, Array, Array]:
    """JIT-compatible golden-section search using JAX.

    Finds the minimum of a unimodal function `f` on the interval
    $[a, b]$ using ``jax.lax.while_loop``, making it compatible
    with JIT compilation.

    Args:
        f: Scalar-valued unimodal callable.
        a_init: Left endpoint of the search interval.
        b_init: Right endpoint of the search interval.
        tol: Convergence tolerance. Defaults to 1e-5.

    Returns:
        A tuple ``(x_min, f_min, n_eval)`` of the approximate minimiser, its
        function value, and the number of ``f`` evaluations spent.

    Example:
        ```python
        f = lambda x: (x - 2) ** 2
        x_min, f_min, n_eval = _golden_section_search(f, 1.0, 5.0)
        ```

    References:
        [Golden-section search](https://en.wikipedia.org/wiki/Golden-section_search)
    """
    phi = (jnp.sqrt(5.0) - 1.0) / 2.0
    resphi = 1.0 - phi
    max_iter = jnp.array(
        (jnp.ceil(jnp.log(tol / (b_init - a_init)) / jnp.log(phi))), int
    )

    a = a_init
    b = b_init

    x1 = a + resphi * (b - a)
    x2 = a + phi * (b - a)
    f1 = f(x1)
    f2 = f(x2)

    state0 = (a, b, x1, x2, f1, f2, jnp.array(0, dtype=jnp.int32))

    def cond_fun(state):
        a, b, x1, x2, f1, f2, i = state
        interval_check = (b - a) > tol
        iter_check = i < max_iter
        return jnp.logical_and(interval_check, iter_check)

    def body_fun(state):
        a, b, x1, x2, f1, f2, i = state

        def left_branch(s):
            # Minimum is in [a, x2]: discard the right portion (b <- x2).
            a, b, x1, x2, f1, f2, i = s
            b_new = x2
            x2_new = x1
            f2_new = f1
            x1_new = a + resphi * (b_new - a)
            f1_new = f(x1_new)
            return (a, b_new, x1_new, x2_new, f1_new, f2_new, i + 1)

        def right_branch(s):
            # Minimum is in [x1, b]: discard the left portion (a <- x1).
            a, b, x1, x2, f1, f2, i = s
            a_new = x1
            x1_new = x2
            f1_new = f2
            x2_new = a_new + phi * (b - a_new)
            f2_new = f(x2_new)
            return (a_new, b, x1_new, x2_new, f1_new, f2_new, i + 1)

        return jax.lax.cond(f1 < f2, left_branch, right_branch, state)

    a, b, x1, x2, f1, f2, i = jax.lax.while_loop(cond_fun, body_fun, state0)

    t_best = jnp.where(f1 < f2, x1, x2)
    f_best = jnp.where(f1 < f2, f1, f2)
    # Each loop iteration spends one new ``f`` evaluation; the two initial
    # ``f1``/``f2`` probes bring the total to ``i + 2``.
    n_eval = i + jnp.array(2, dtype=jnp.int32)
    return t_best, f_best, n_eval


# TODO can we remove the finite differences here?
def _adam_line_search(
    f: Callable[[Array], Array],
    a_init: float | Array,
    b_init: float | Array,
    lr: float = 0.05,
    num_steps: int = 30,
    finite_difference: bool = True,
    fd_step: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    t_init: float | Array = 0.0,
) -> tuple[Array, Array, Array]:
    """JIT-compatible 1-D Adam line search using JAX.

    Minimises a scalar function `f` on the interval $[a, b]$ by running
    a fixed number of Adam steps on the scalar variable ``t``, clipping
    ``t`` back into the interval after every step. Uses
    ``jax.lax.fori_loop`` (fixed step count), making it compatible with
    JIT compilation.

    The gradient ``df/dt`` is obtained either by a finite-difference
    secant from successive evaluations (``finite_difference=True``;
    derivative-free, one ``f`` evaluation per step) or by
    ``jax.value_and_grad`` (``finite_difference=False``; exact, but
    differentiates through ``f``). ``f`` must map a real scalar to a
    real scalar.

    Adam is not monotone, so the best iterate visited is tracked and
    returned — the result is never worse than ``f(t_init)``.

    Args:
        f: Scalar-valued callable (real -> real).
        a_init: Left endpoint of the search interval.
        b_init: Right endpoint of the search interval.
        lr: Adam learning rate. Defaults to 0.05.
        num_steps: Number of Adam iterations. Defaults to 30.
        finite_difference: If ``True`` (default), estimate the gradient
            with a finite-difference secant; otherwise use
            ``jax.value_and_grad``.
        fd_step: Probe size for the finite-difference bootstrap.
            Defaults to 1e-3.
        beta1: First-moment decay. Defaults to 0.9.
        beta2: Second-moment decay. Defaults to 0.999.
        eps: Numerical-stability term. Defaults to 1e-8.
        t_init: Starting point for ``t``. Defaults to 0.0.

    Returns:
        A tuple ``(t_best, f_best, n_eval)`` of the best minimiser found, its
        function value, and the number of ``f`` evaluations spent (the fixed
        ``num_steps + 2``), matching the ``(x_min, f_min, n_eval)`` contract of
        :func:`_golden_section_search`.

    Example:
        ```python
        f = lambda x: (x - 2.0) ** 2
        x_min, f_min, n_eval = _adam_line_search(f, 0.0, 5.0, lr=0.1, num_steps=200)
        ```

    References:
        [Adam](https://arxiv.org/abs/1412.6980)
    """
    lo = jnp.minimum(a_init, b_init)
    hi = jnp.maximum(a_init, b_init)
    f64 = lambda x: jnp.asarray(x, dtype=jnp.float64)

    def adam_update(i, t, m, v, g):
        # Shared Adam moment update + bias correction + interval clip.
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        step = f64(i) + 1.0
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        t_new = jnp.clip(t - lr * m_hat / (jnp.sqrt(v_hat) + eps), lo, hi)
        return t_new, m, v

    t0 = jnp.clip(f64(t_init), lo, hi)
    f0 = f64(f(t0))

    if finite_difference:
        # Bootstrap: one inward probe to seed (t_prev, f_prev).
        direction = jnp.sign((lo + hi) / 2.0 - t0)
        direction = jnp.where(direction == 0, -1.0, direction)
        t_start = jnp.clip(t0 + direction * fd_step, lo, hi)
        # state: (t, m, v, t_prev, f_prev, t_best, f_best)
        state0 = (t_start, f64(0.0), f64(0.0), t0, f0, t0, f0)

        def body_fun(i, state):
            t, m, v, t_prev, f_prev, t_best, f_best = state
            ft = f64(f(t))
            improved = ft < f_best
            t_best = jnp.where(improved, t, t_best)
            f_best = jnp.where(improved, ft, f_best)
            dt_ = t - t_prev
            dt_safe = jnp.where(dt_ == 0, fd_step, dt_)  # guard exact-zero
            g = (ft - f_prev) / dt_safe  # secant slope
            t_new, m, v = adam_update(i, t, m, v, g)
            return (t_new, m, v, t, ft, t_best, f_best)

        t, m, v, t_prev, f_prev, t_best, f_best = jax.lax.fori_loop(
            0, num_steps, body_fun, state0
        )
    else:
        # When ``f`` maps the real ``t`` through complex intermediates (e.g.
        # unitaries), JAX may emit a benign ComplexWarning while forming the
        # real cotangent of ``t``; the gradient is correct (verified against
        # finite differences).
        value_and_grad = jax.value_and_grad(f)
        # state: (t, m, v, t_best, f_best)
        state0 = (t0, f64(0.0), f64(0.0), t0, f0)

        def body_fun(i, state):
            t, m, v, t_best, f_best = state
            ft, g = value_and_grad(t)
            ft = f64(ft)
            improved = ft < f_best
            t_best = jnp.where(improved, t, t_best)
            f_best = jnp.where(improved, ft, f_best)
            t_new, m, v = adam_update(i, t, m, v, g)
            return (t_new, m, v, t_best, f_best)

        t, m, v, t_best, f_best = jax.lax.fori_loop(0, num_steps, body_fun, state0)

    # Also consider the final iterate (evaluated once after the loop).
    f_last = f64(f(t))
    take_last = f_last < f_best
    t_best = jnp.where(take_last, t, t_best)
    f_best = jnp.where(take_last, f_last, f_best)
    # Fixed schedule: ``f(t0)`` + ``num_steps`` body evals + one final ``f(t)``
    # (the grad path counts each ``value_and_grad`` as one evaluation).
    n_eval = jnp.asarray(num_steps + 2, dtype=jnp.int32)
    return t_best, f_best, n_eval


def _quadratic_armijo_line_search(
    fF: Callable[[Array], Array],
    a: float | Array,
    s: float | Array,
    q: float | Array,
    F0: float | Array,
    c1: float = 1e-4,
    beta: float = 0.5,
    t_min: float = 1e-8,
) -> tuple[Array, Array, Array]:
    r"""JIT-compatible quadratic-seeded Armijo line search using JAX.

    Implements the line search of the note *Quadratic-Seeded Armijo Line Search
    on $\mathrm{SU}(N)$*. Given the exact slope $s=\psi'(0)$ and curvature
    $q=\psi''(0)$ of the objective along the search ray (supplied by the caller,
    typically from the SU(N) geometry rather than autodiff), it seeds a trial
    step at the minimiser of the local quadratic model and then enforces
    sufficient decrease with Armijo backtracking.

    Unlike :func:`_golden_section_search` / :func:`_adam_line_search`, which see
    only the scalar objective, this routine consumes the second-order
    information ``(s, q)`` directly. The bracket is one-sided: the descent side
    ``[a, 0]`` with ``a`` the maximum-magnitude step (``a < 0`` in GEOPE, where a
    useful step is negative and ``t = 0`` is "no move"). It expects ``s > 0`` on
    that convention, so the model minimiser $-s/q$ is negative.

    Method (note §§6–8):

    - **Seed.** If ``q > 0`` (the local model has a minimiser), start at
      ``t0 = clip(-s / q, a, 0)`` — the model minimiser, clipped to the bracket.
      Otherwise (``q <= 0``: the model is concave, so it has no minimiser at all)
      fall back to the full step ``t0 = a``. A non-positive ``q`` never
      invalidates the *direction* — descent is guaranteed by ``s > 0`` on GEOPE's
      sign convention — only its *scale*, which the backtracking then fixes.
    - **Armijo.** Accept ``t`` when ``fF(t) <= F0 + c1 * t * s`` (sufficient
      decrease; the right-hand side is below ``F0`` since ``t * s < 0``).
    - **Backtrack.** Otherwise ``t <- beta * t`` (shrinking the magnitude toward
      0) until the test passes or the next step would fall below ``t_min`` in
      magnitude.

    Args:
        fF: Scalar-valued objective along the ray, ``fF(t) -> value``. This is
            the objective whose derivatives ``s`` and ``q`` describe (e.g. the
            squared-geodesic-distance pullback), not necessarily the fidelity.
        a: Maximum-magnitude (bracket) step; ``a < 0`` on GEOPE's convention.
        s: Exact slope $\psi'(0)$ of ``fF`` along the ray (expected ``< 0`` for a
            descent direction, i.e. ``> 0`` before the sign of ``a``; see above).
        q: Exact curvature $\psi''(0)$ of ``fF`` along the ray.
        F0: The objective value ``fF(0)`` at the current point.
        c1: Armijo sufficient-decrease constant. Defaults to 1e-4.
        beta: Backtracking contraction factor in ``(0, 1)``. Defaults to 0.5.
        t_min: Minimum allowed step magnitude before the search gives up.
            Defaults to 1e-8.

    Returns:
        A tuple ``(t_best, F_best, n_eval)``: the accepted step, ``fF`` at that
        step, and the number of ``fF`` evaluations spent.
    """
    f64 = lambda x: jnp.asarray(x, dtype=jnp.float64)
    a = f64(a)
    s = f64(s)
    q = f64(q)
    F0 = f64(F0)
    # TODO: cut locus boundary.
    # With s > 0, a concave model (q < 0) puts -s/q on the *positive* side, which
    # the clip would collapse to t = 0 (a stalled step); take the full bracket
    # step instead. q == 0 reaches the same place via -inf, so branch on q > 0.
    t0 = jnp.where(q > 0.0, jnp.clip(-s / q, a, 0.0), a)

    state0 = (t0, f64(fF(t0)), jnp.array(1, dtype=jnp.int32))

    def cond_fun(state):
        t, Ft, i = state
        armijo_ok = Ft <= F0 + c1 * t * s
        step_ok = jnp.abs(beta * t) >= t_min
        return jnp.logical_and(jnp.logical_not(armijo_ok), step_ok)

    def body_fun(state):
        t, Ft, i = state
        t_new = beta * t
        return (t_new, f64(fF(t_new)), i + 1)

    t_best, F_best, n_eval = jax.lax.while_loop(cond_fun, body_fun, state0)
    return t_best, F_best, n_eval


def _armijo_line_search(
    fF: Callable[[Array], Array],
    a: float | Array,
    F0: float | Array | None = None,
    s: float | Array | None = None,
    c1: float = 1e-4,
    beta: float = 0.5,
    t_min: float = 1e-8,
) -> tuple[Array, Array, Array]:
    r"""JIT-compatible backtracking Armijo line search using JAX.

    The non-quadratic counterpart of :func:`_quadratic_armijo_line_search`: it
    seeds the trial step at the full bracket step ``t0 = a`` (i.e. clipped to
    $t_{\max}$) rather than at the quadratic model minimiser $-s/q$, so it needs
    no curvature and no second derivative of the objective. §15 of the note
    *Quadratic-Seeded Armijo Line Search on $\mathrm{SU}(N)$* shows this loses
    nothing in correctness — termination follows from the descent slope alone, and
    $q$ serves only to scale the *first* trial.

    The bracket convention matches the quadratic version: one-sided ``[a, 0]``
    with ``a < 0``, a useful step negative, ``t = 0`` meaning "don't move", and a
    descent direction giving ``s > 0``.

    Method:

    - **Slope.** When ``s`` is not supplied it is taken from the objective value
      as $s=2F_0$. This is the note's §11 *exact radial specialization*: under the
      tangent matching $\Omega=-A$ that GEOPE's geodesic step targets,
      $s=\|A\|_F^2=2F_0$, and the Armijo test collapses to
      $F(t)\le F_0\,(1-2c_1 t)$ — current and trial objective values only.
      It is an approximation once ``coeffs`` has been renormalised to
      $\|p\|_F=\sqrt{G}$ (so $\|\Omega\|_F\neq\|A\|_F$), but the test is scaled by
      ``c1``, which is tiny by default. Pass ``s`` explicitly (e.g. the exact
      $\langle A,\Omega\rangle_F$) to override.
    - **Seed.** ``t0 = a``, the full bracket step.
    - **Armijo.** Accept ``t`` when ``fF(t) <= F0 + c1 * t * s`` (the right-hand
      side is below ``F0`` since ``t * s < 0``). A non-positive ``F0`` — a
      converged iterate, where the test is vacuous — also accepts, so the loop
      cannot grind through $\log_\beta(t_{\min}/|a|)$ evaluations at the end of a
      run.
    - **Backtrack.** Otherwise ``t <- beta * t`` until the test passes or the next
      step would fall below ``t_min`` in magnitude.

    Args:
        fF: Scalar-valued objective along the ray, ``fF(t) -> value`` (e.g. the
            squared-geodesic-distance pullback ``ctx.distance_at``).
        a: Maximum-magnitude (bracket) step; ``a < 0`` on GEOPE's convention.
        F0: The objective value ``fF(0)``. Evaluated here when omitted, which
            costs one ``fF`` evaluation.
        s: Slope $\psi'(0)$ of ``fF`` along the ray (``> 0`` for a descent
            direction on this convention). Defaults to the radial $2F_0$.
        c1: Armijo sufficient-decrease constant. Defaults to 1e-4.
        beta: Backtracking contraction factor in ``(0, 1)``. Defaults to 0.5.
        t_min: Minimum allowed step magnitude before the search gives up.
            Defaults to 1e-8.

    Returns:
        A tuple ``(t_best, F_best, n_eval)``: the accepted step, ``fF`` at that
        step, and the number of ``fF`` evaluations spent — matching the contract
        of :func:`_golden_section_search` and
        :func:`_quadratic_armijo_line_search`.
    """
    f64 = lambda x: jnp.asarray(x, dtype=jnp.float64)
    a = f64(a)
    # An omitted F0 costs one probe; an omitted s is free (the radial 2 * F0).
    n_probe = 0
    if F0 is None:
        F0 = fF(0.0)
        n_probe = 1
    F0 = f64(F0)
    s = f64(2.0 * F0 if s is None else s)

    t0 = a  # clip to t_max: the full bracket step, no curvature involved
    state0 = (t0, f64(fF(t0)), jnp.array(1 + n_probe, dtype=jnp.int32))

    def cond_fun(state):
        t, Ft, i = state
        armijo_ok = Ft <= F0 + c1 * t * s
        # F0 <= 0: converged, so the test can never pass — accept and stop.
        armijo_ok = jnp.logical_or(armijo_ok, F0 <= 0.0)
        step_ok = jnp.abs(beta * t) >= t_min
        return jnp.logical_and(jnp.logical_not(armijo_ok), step_ok)

    def body_fun(state):
        t, Ft, i = state
        t_new = beta * t
        return (t_new, f64(fF(t_new)), i + 1)

    t_best, F_best, n_eval = jax.lax.while_loop(cond_fun, body_fun, state0)
    return t_best, F_best, n_eval
