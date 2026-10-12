"""Pluggable line-search objects for :class:`geope.Geope`.

A line search tunes the scalar geodesic **step length ``alpha``** along the search
direction GEOPE computes each step (it is not a full-parameter optimiser — that
is the separate :class:`Grape` class). The active object is passed to
``Geope.optimize(line_search=...)``; when omitted it defaults to
:class:`Armijo`.

Each line search is a ``@dataclass(frozen=True)`` — immutable config that gets
value-based ``__eq__``/``__hash__``/``__repr__`` for free. The value ``__eq__``
drives GEOPE's compile memo (``Armijo() == Armijo()`` ⇒ no recompile) and the
immutability keeps hyperparameter sweeps correct (a config cannot be mutated in
place and silently reuse a stale compiled function).

**The call contract.** ``Geope`` builds one
:class:`~geope.geometry.GeometricContext` per step (inside the jitted update) and
calls ``line_search(ctx, a, b, state)``, which returns a :class:`LineSearchResult`
``(step_size, value, state)``. The context carries **only geometry**; the bracket
``[a, b]`` and the threaded state are the search's own bookkeeping and travel
alongside it, which is what lets a consumer with no bracket (``Gecko``) share the
same context type. The direction on the context is a **descent** direction and
the bracket is ``[a, b] = [0, alpha_max]``: ``alpha = 0`` is "don't move", a
useful step is positive, and the context's slopes (``ctx.velocity``,
``ctx.slope``) are negative there — the textbook (Nocedal & Wright) convention.
Every quantity on the context is lazy, and the line search is
traced *inside* the jitted update, so a quantity a method never reads is never
traced: zeroth-order methods pay nothing for the ``logm``/HVP the geometry needs.

Each search declares which scalar objective it minimises as the class attribute
``objective`` — ``"infidelity"`` (``ctx.infidelity_at``) or ``"distance"``
(``ctx.distance_at``, the squared geodesic distance). It is deliberately *not* a
dataclass field, so it stays out of the value ``__eq__`` that drives the compile
memo. ``value`` in the result is that objective at the accepted step — every 1-D
minimiser already computes it — and ``Geope.optimize`` decides whether the step
made progress by comparing it against the same objective at ``alpha = 0``.

The three orders of information cost different things per GEOPE step:
:class:`GoldenSection` is zeroth-order and evaluates only the
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
``{"n_eval"}`` (the per-step count of 1-D-objective evaluations it spent).
``Geope.optimize`` re-``init()``s the state at the start of every run.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class LineSearchResult(NamedTuple):
    """What a :class:`LineSearch` returns.

    Attributes:
        step_size: The accepted step size along the search direction.
        value: The search's own ``objective`` at ``step_size`` — every 1-D minimiser
            computes this anyway, and ``Geope.optimize`` tests progress against
            it rather than against a quantity the search never minimised.
        state: The new line-search-owned state pytree (always carries
            ``"n_eval"``).
    """

    step_size: Array
    value: Array
    state: dict


class LineSearch:
    """Base line search: tunes the scalar geodesic step length ``alpha``.

    Subclasses are frozen dataclasses (immutable config) that own an opaque
    JAX-pytree state. The base state carries only ``{"n_eval"}`` — the per-step
    count of 1-D-objective evaluations the search spent — which every line
    search reports; stateful searches extend it.

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
            a: Near bracket endpoint, "don't move" (``Geope`` passes ``0.0``).
            b: Far bracket endpoint, the largest step (``Geope`` passes
                ``max_step_size / (sqrt(G) * ||coeffs|| * delta_t)``, or
                ``max_step_size`` itself when its least squares is not
                normalised).
            state: This search's threaded state from the previous step.

        Returns:
            A :class:`LineSearchResult`.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class GoldenSection(LineSearch):
    """Golden-section search — zeroth-order.

    Minimises ``ctx.infidelity_at`` on the bracket, and reads nothing else off
    the context: no logarithm, no Jacobian, no HVP is traced on its account.
    Carries only the base ``{"n_eval"}`` state (the per-step evaluation count).

    Args:
        tol: Convergence tolerance for the search interval. Defaults to 1e-5.
    """

    name = "golden_section"
    tol: float = 1e-5

    def __call__(self, ctx, a, b, state):
        step_size, value, n_eval = _golden_section_search(
            ctx.infidelity_at, a, b, tol=self.tol
        )
        return LineSearchResult(step_size, value, {"n_eval": n_eval})


@dataclass(frozen=True)
class Armijo(LineSearch):
    r"""Backtracking Armijo line search (the default) — first-order, no curvature.

    The non-quadratic sibling of :class:`QuadraticArmijo`: it seeds the trial
    step at the full bracket step $\alpha_0=b=\alpha_{\max}$ instead of at the quadratic
    model minimiser $-s/q$, and then enforces sufficient decrease by Armijo
    backtracking on ``ctx.distance_at``. The note's §15 shows this loses nothing
    in correctness — termination follows from the descent slope alone, and the
    curvature only rescales the *first* trial.

    Both quantities the Armijo test needs are free: ``ctx.F0`` is the objective
    at $t=0$, which tier 0 of the context has already computed from its single
    logarithm, and ``ctx.velocity`` is the exact slope $\langle A,\Omega\rangle$, one
    contraction of tier 0's Jacobian. So this search spends no propagator and no
    logarithm of its own before its first trial, and it uses the exact slope
    rather than the radial $s=-2F_0$ of the note's §11 — which holds only under
    perfect tangent matching, i.e. not once the least-squares solve leaves a
    residual.

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
        step_size, value, n_eval = _armijo_line_search(
            ctx.distance_at,
            b,
            F0=ctx.F0,
            s=ctx.velocity,
            c1=self.c1,
            beta=self.beta,
            t_min=self.t_min,
        )
        return LineSearchResult(step_size, value, {"n_eval": n_eval})


@dataclass(frozen=True)
class QuadraticArmijo(LineSearch):
    r"""Quadratic-seeded Armijo line search — second-order (geometry-aware).

    Implements the note *Quadratic-Seeded Armijo Line Search on
    $\mathrm{SU}(N)$*: it reads the exact slope $s=\psi'(0)$ and curvature
    $q=\psi''(0)$ of the squared-geodesic-distance objective off the context
    (``ctx.velocity`` and ``ctx.q``), seeds the trial step at the local quadratic minimiser
    $-s/q$ (clipped to the bracket; when the curvature is negative, at the step
    $2|s|/|q|$ where the quadratic term overtakes the linear one, see
    ``concave_guard``), and enforces sufficient decrease with Armijo
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
        concave_guard: When the curvature is negative, seed at
            $\min(\alpha_{\max}, 2|s|/|q|)$ instead of at $\alpha_{\max}$.
            Defaults to ``True``. Without it a concave step starts at the end of
            the bracket, which is a sensible scale only when that end is a step
            of fixed length, as `geope.Geope`'s normalised bracket is; see
            :func:`_quadratic_armijo_line_search`.
    """

    name = "quadratic_armijo"
    objective = "distance"
    c1: float = 1e-4
    beta: float = 0.5
    t_min: float = 1e-8
    concave_guard: bool = True

    def __call__(self, ctx, a, b, state):
        step_size, value, n_eval = _quadratic_armijo_line_search(
            ctx.distance_at,
            b,
            ctx.velocity,
            ctx.q,
            ctx.F0,
            c1=self.c1,
            beta=self.beta,
            t_min=self.t_min,
            concave_guard=self.concave_guard,
        )
        return LineSearchResult(step_size, value, {"n_eval": n_eval})


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
        concave_guard: When the curvature is negative, seed at
            $\min(\alpha_{\max}, 2|s|/|q|)$ instead of at $\alpha_{\max}$, as in
            :class:`QuadraticArmijo`. Defaults to ``True``.
    """

    name = "approximate_quadratic_armijo"
    objective = "distance"
    c1: float = 1e-4
    beta: float = 0.5
    t_min: float = 1e-8
    concave_guard: bool = True

    def __call__(self, ctx, a, b, state):
        step_size, value, n_eval = _quadratic_armijo_line_search(
            ctx.distance_at,
            b,
            ctx.velocity,
            ctx.q_exact,
            ctx.F0,
            c1=self.c1,
            beta=self.beta,
            t_min=self.t_min,
            concave_guard=self.concave_guard,
        )
        return LineSearchResult(step_size, value, {"n_eval": n_eval})


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
        a, b, _, _, _, _, i = state
        interval_check = (b - a) > tol
        iter_check = i < max_iter
        return jnp.logical_and(interval_check, iter_check)

    def body_fun(state):
        _, _, _, _, f1, f2, _ = state

        def left_branch(s):
            # Minimum is in [a, x2]: discard the right portion (b <- x2).
            a, _, x1, x2, f1, _, i = s
            b_new = x2
            x2_new = x1
            f2_new = f1
            x1_new = a + resphi * (b_new - a)
            f1_new = f(x1_new)
            return (a, b_new, x1_new, x2_new, f1_new, f2_new, i + 1)

        def right_branch(s):
            # Minimum is in [x1, b]: discard the left portion (a <- x1).
            _, b, x1, x2, _, f2, i = s
            a_new = x1
            x1_new = x2
            f1_new = f2
            x2_new = a_new + phi * (b - a_new)
            f2_new = f(x2_new)
            return (a_new, b, x1_new, x2_new, f1_new, f2_new, i + 1)

        # Ties keep the left portion, i.e. drift toward ``a`` — which on the
        # descent bracket ``[0, alpha_max]`` is "don't move".
        return jax.lax.cond(f1 <= f2, left_branch, right_branch, state)

    a, b, x1, x2, f1, f2, i = jax.lax.while_loop(cond_fun, body_fun, state0)

    t_best = jnp.where(f1 <= f2, x1, x2)
    f_best = jnp.where(f1 <= f2, f1, f2)
    # Each loop iteration spends one new ``f`` evaluation; the two initial
    # ``f1``/``f2`` probes bring the total to ``i + 2``.
    n_eval = i + jnp.array(2, dtype=jnp.int32)
    return t_best, f_best, n_eval


def _backtracking_max_iter(beta: float, max_iter: int | None) -> int:
    r"""The contraction budget of the two Armijo backtrackers.

    ``max_iter`` when given; otherwise the fewest contractions that shrink the
    seed by a factor of float64 machine epsilon,
    $\lceil\log\varepsilon/\log\beta\rceil$ (52 at ``beta = 0.5``). For any seed
    below ``t_min / eps`` the ``t_min`` floor is reached first, so this default
    never binds on a finite seed of sensible size; it exists for the seeds that
    do not shrink at all.

    Raises:
        ValueError: If the default is requested and ``beta`` is not in
            ``(0, 1)``, where the formula has no meaning.
    """
    if max_iter is not None:
        return max_iter
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    return math.ceil(math.log(jnp.finfo(jnp.float64).eps) / math.log(beta))


def _quadratic_armijo_line_search(
    fF: Callable[[Array], Array],
    alpha_max: float | Array,
    s: float | Array,
    q: float | Array,
    F0: float | Array,
    c1: float = 1e-4,
    beta: float = 0.5,
    t_min: float = 1e-8,
    max_iter: int | None = None,
    concave_guard: bool = True,
) -> tuple[Array, Array, Array]:
    r"""JIT-compatible quadratic-seeded Armijo line search using JAX.

    Implements the line search of the note *Quadratic-Seeded Armijo Line Search
    on $\mathrm{SU}(N)$*. Given the exact slope $s=\psi'(0)$ and curvature
    $q=\psi''(0)$ of the objective along the search ray (supplied by the caller,
    typically from the SU(N) geometry rather than autodiff), it seeds a trial
    step at the minimiser of the local quadratic model and then enforces
    sufficient decrease with Armijo backtracking.

    Unlike :func:`_golden_section_search`, which sees
    only the scalar objective, this routine consumes the second-order
    information ``(s, q)`` directly. The bracket is one-sided: ``[0, alpha_max]``,
    with ``alpha = 0`` "no move" and a useful step positive. It expects a descent
    direction, ``s < 0``, so the model minimiser $-s/q$ is positive.

    Method (note §§6–8):

    - **Seed.** If ``q > 0`` (the local model has a minimiser), start at
      ``alpha0 = clip(-s / q, 0, alpha_max)`` — the model minimiser, clipped to
      the bracket. If ``q < 0`` the model is concave and has no minimiser, but
      $|q|$ still sets the scale at which the model stops being linear: the
      quadratic term $\tfrac12|q|\alpha^2$ overtakes the linear one $|s|\alpha$
      at $\alpha=2|s|/|q|$. With ``concave_guard`` (the default) the seed is
      ``alpha0 = min(alpha_max, 2|s|/|q|)``; without it, the full step
      ``alpha0 = alpha_max``. ``q == 0`` gives no scale either way and seeds at
      ``alpha_max``. A non-positive ``q`` never invalidates the *direction* —
      descent is guaranteed by ``s < 0`` — only its *scale*. Leaving that scale
      to the backtracking alone is safe only when ``alpha_max`` is itself
      sensible: on an unnormalised direction such as the raw Gauss–Newton step
      $-J^+\gamma$, whose norm grows with the conditioning of $J$,
      ``alpha_max`` can lie orders of magnitude past where the model holds, and
      the weak Armijo test accepts the first lower point it finds out there.
    - **Armijo.** Accept ``alpha`` when ``fF(alpha) <= F0 + c1 * alpha * s``
      (sufficient decrease; the right-hand side is below ``F0`` since
      ``alpha * s < 0``).
    - **Backtrack.** Otherwise ``alpha <- beta * alpha`` (shrinking toward 0)
      until the test passes, the next step would fall below ``t_min``, or
      ``max_iter`` contractions have been spent. The last exit is the only one
      that is guaranteed: ``alpha0 = inf`` (an unbounded ``alpha_max`` with
      ``q == 0``, or with ``q < 0`` and no guard) is a fixed point of the
      contraction, and there the other two stay shut — the Armijo bound is
      ``-inf`` and ``|beta * alpha| >= t_min`` holds forever.

    Args:
        fF: Scalar-valued objective along the ray, ``fF(alpha) -> value``. This is
            the objective whose derivatives ``s`` and ``q`` describe (e.g. the
            squared-geodesic-distance pullback), not necessarily the fidelity.
        alpha_max: The largest (bracket) step, ``> 0``.
        s: Exact slope $\psi'(0)$ of ``fF`` along the ray, ``< 0`` for a
            descent direction.
        q: Exact curvature $\psi''(0)$ of ``fF`` along the ray.
        F0: The objective value ``fF(0)`` at the current point.
        c1: Armijo sufficient-decrease constant. Defaults to 1e-4.
        beta: Backtracking contraction factor in ``(0, 1)``. Defaults to 0.5.
        t_min: Minimum allowed step magnitude before the search gives up.
            Defaults to 1e-8.
        max_iter: Maximum number of contractions after the seed. Defaults to
            ``None``: enough to shrink the seed to machine precision,
            $\lceil\log\varepsilon/\log\beta\rceil$ (52 at ``beta = 0.5``).
        concave_guard: When ``q < 0``, seed at ``min(alpha_max, 2|s|/|q|)``
            rather than at ``alpha_max``. Defaults to ``True``.

    Returns:
        A tuple ``(alpha_best, F_best, n_eval)``: the accepted step, ``fF`` at
        that step, and the number of ``fF`` evaluations spent. When a cap stops
        the search instead of the Armijo test, it is the last trial.

    Raises:
        ValueError: If ``max_iter`` is ``None`` and ``beta`` is not in ``(0, 1)``.
    """
    f64 = lambda x: jnp.asarray(x, dtype=jnp.float64)
    alpha_max = f64(alpha_max)
    s = f64(s)
    q = f64(q)
    F0 = f64(F0)
    # TODO: cut locus boundary.
    # With s < 0, a concave model (q < 0) puts -s/q on the *negative* side, which
    # the clip would collapse to alpha = 0. concave_guard prevents stalls.
    if concave_guard:
        alpha_concave = jnp.where(
            q < 0.0, jnp.minimum(alpha_max, 2.0 * jnp.abs(s) / jnp.abs(q)), alpha_max
        )
    else:
        alpha_concave = alpha_max
    alpha0 = jnp.where(q > 0.0, jnp.clip(-s / q, 0.0, alpha_max), alpha_concave)
    max_iter = _backtracking_max_iter(beta, max_iter)

    state0 = (alpha0, f64(fF(alpha0)), jnp.array(1, dtype=jnp.int32))

    def cond_fun(state):
        alpha, F_alpha, n_eval = state
        armijo_ok = F_alpha <= F0 + c1 * alpha * s
        step_ok = jnp.abs(beta * alpha) >= t_min
        # n_eval counts the seed, so n_eval - 1 contractions have been spent.
        budget_ok = n_eval - 1 < max_iter
        return jnp.logical_not(armijo_ok) & step_ok & budget_ok

    def body_fun(state):
        alpha, _, i = state
        alpha_new = beta * alpha
        return (alpha_new, f64(fF(alpha_new)), i + 1)

    alpha_best, F_best, n_eval = jax.lax.while_loop(cond_fun, body_fun, state0)
    return alpha_best, F_best, n_eval


def _armijo_line_search(
    fF: Callable[[Array], Array],
    alpha_max: float | Array,
    F0: float | Array | None = None,
    s: float | Array | None = None,
    c1: float = 1e-4,
    beta: float = 0.5,
    t_min: float = 1e-8,
    max_iter: int | None = None,
) -> tuple[Array, Array, Array]:
    r"""JIT-compatible backtracking Armijo line search using JAX.

    The non-quadratic counterpart of :func:`_quadratic_armijo_line_search`: it
    seeds the trial step at the full bracket step ``alpha0 = alpha_max`` rather
    than at the quadratic model minimiser $-s/q$, so it needs
    no curvature and no second derivative of the objective. §15 of the note
    *Quadratic-Seeded Armijo Line Search on $\mathrm{SU}(N)$* shows this loses
    nothing in correctness — termination follows from the descent slope alone, and
    $q$ serves only to scale the *first* trial.

    The bracket convention matches the quadratic version: one-sided
    ``[0, alpha_max]``, a useful step positive, ``alpha = 0`` meaning "don't
    move", and a descent direction giving ``s < 0``.

    Method:

    - **Slope.** When ``s`` is not supplied it is taken from the objective value
      as $s=-2F_0$. This is the note's §11 *exact radial specialization*: under
      the tangent matching $\Omega=-A$ that GEOPE's geodesic step targets,
      $s=-\|A\|_F^2=-2F_0$, and the Armijo test collapses to
      $F(\alpha)\le F_0\,(1-2c_1\alpha)$ — current and trial objective values only.
      It is an approximation once $\Omega\neq-A$ — a least-squares residual, or
      a direction rescaled away from its least-squares scale — but the test is
      scaled by ``c1``, which is tiny by default. Pass ``s`` explicitly (e.g. the exact
      $\langle A,\Omega\rangle_F$) to override.
    - **Seed.** ``alpha0 = alpha_max``, the full bracket step.
    - **Armijo.** Accept ``alpha`` when ``fF(alpha) <= F0 + c1 * alpha * s`` (the
      right-hand side is below ``F0`` since ``alpha * s < 0``). A non-positive
      ``F0`` — a converged iterate, where the test is vacuous — also accepts, so
      the loop cannot grind through $\log_\beta(t_{\min}/\alpha_{\max})$
      evaluations at the end of a run.
    - **Backtrack.** Otherwise ``alpha <- beta * alpha`` until the test passes,
      the next step would fall below ``t_min``, or ``max_iter`` contractions
      have been spent. As in :func:`_quadratic_armijo_line_search`, only the
      last exit is guaranteed: an unbounded ``alpha_max = inf`` is a fixed point
      of the contraction that neither of the other two ever closes.

    Args:
        fF: Scalar-valued objective along the ray, ``fF(alpha) -> value`` (e.g.
            the squared-geodesic-distance pullback ``ctx.distance_at``).
        alpha_max: The largest (bracket) step, ``> 0``.
        F0: The objective value ``fF(0)``. Evaluated here when omitted, which
            costs one ``fF`` evaluation.
        s: Slope $\psi'(0)$ of ``fF`` along the ray, ``< 0`` for a descent
            direction. Defaults to the radial $-2F_0$.
        c1: Armijo sufficient-decrease constant. Defaults to 1e-4.
        beta: Backtracking contraction factor in ``(0, 1)``. Defaults to 0.5.
        t_min: Minimum allowed step magnitude before the search gives up.
            Defaults to 1e-8.
        max_iter: Maximum number of contractions after the seed. Defaults to
            ``None``: enough to shrink the seed to machine precision,
            $\lceil\log\varepsilon/\log\beta\rceil$ (52 at ``beta = 0.5``).

    Returns:
        A tuple ``(alpha_best, F_best, n_eval)``: the accepted step, ``fF`` at
        that step, and the number of ``fF`` evaluations spent — matching the contract
        of :func:`_golden_section_search` and
        :func:`_quadratic_armijo_line_search`. When a cap stops the search
        instead of the Armijo test, it is the last trial.

    Raises:
        ValueError: If ``max_iter`` is ``None`` and ``beta`` is not in ``(0, 1)``.
    """
    f64 = lambda x: jnp.asarray(x, dtype=jnp.float64)
    alpha_max = f64(alpha_max)
    # An omitted F0 costs one probe; an omitted s is free (the radial -2 * F0).
    n_probe = 0
    if F0 is None:
        F0 = fF(0.0)
        n_probe = 1
    F0 = f64(F0)
    s = f64(-2.0 * F0 if s is None else s)

    alpha0 = alpha_max  # the full bracket step, no curvature involved
    max_iter = _backtracking_max_iter(beta, max_iter)
    state0 = (alpha0, f64(fF(alpha0)), jnp.array(1 + n_probe, dtype=jnp.int32))

    def cond_fun(state):
        alpha, F_alpha, n_eval = state
        armijo_ok = F_alpha <= F0 + c1 * alpha * s
        # F0 <= 0: converged, so the test can never pass — accept and stop.
        armijo_ok = jnp.logical_or(armijo_ok, F0 <= 0.0)
        step_ok = jnp.abs(beta * alpha) >= t_min
        # n_eval counts the probe and the seed; the rest are contractions.
        budget_ok = n_eval - (1 + n_probe) < max_iter
        return jnp.logical_not(armijo_ok) & step_ok & budget_ok

    def body_fun(state):
        alpha, _, i = state
        alpha_new = beta * alpha
        return (alpha_new, f64(fF(alpha_new)), i + 1)

    alpha_best, F_best, n_eval = jax.lax.while_loop(cond_fun, body_fun, state0)
    return alpha_best, F_best, n_eval
