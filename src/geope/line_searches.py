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

**The call contract.** ``Geope`` builds a :class:`LineSearchContext` once per step
(inside the jitted ``update_linesearch``) and calls ``line_search(ctx)``, which
returns ``(dt, new_state)``. The context always carries the cheap scalar
objective ``ctx.f`` (infidelity along the line), the bracket ``ctx.a``/``ctx.b``,
and the threaded ``ctx.state``. It *also* offers, behind lazy accessors, the
squared-geodesic-distance objective ``ctx.distance_f`` and the exact SU(N)
second-order geometry ``ctx.geometry()`` — because the line search is traced
*inside* the jitted update, a lazy accessor a method never calls is never traced,
so zeroth-order methods pay nothing for the ``logm``/HVP the geometry needs.

Cross-step state is a JAX pytree threaded through the jitted update (mirroring
``Grape.optimizer_state``): a jitted closure traces once, so persistent state
must enter/leave as an argument/result rather than as a mutated attribute. The
state is line-search-owned and opaque to GEOPE — :class:`Adam` carries
``{"t_prev"}`` (warm-start), :class:`QuadraticArmijo` carries ``{"n_eval"}``,
:class:`GoldenSection` is stateless (``{}``). ``Geope.optimize`` re-``init()``s the
state at the start of every run.
"""

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class LineSearchGeometry(NamedTuple):
    """Exact second-order geometry of the objective along the search ray.

    All fields are in the *coefficient* units of the search direction ``coeffs``
    (i.e. derivatives of ``t -> F(theta + t * coeffs)``), so the model minimiser
    ``-s / q`` is directly the step GEOPE adds. Assembled by ``Geope`` from the
    SU(N) log and the directional HVP (see :meth:`Geope.get_update_linesearch`).

    Attributes:
        F0: The squared-geodesic-distance objective at the current point.
        s: Slope $\\psi'(0) = \\langle A, \\Omega\\rangle_F$.
        q: Curvature $\\psi''(0) = \\langle\\Omega,\\Omega\\rangle_F +
            \\langle A, V^\\dagger V + x^\\dagger W\\rangle_F$.
        chi: The dimensionless radial bending coefficient $\\chi_\\phi$.
        A_norm2: $\\|A\\|_F^2 = 2 F0$.
    """

    F0: Array
    s: Array
    q: Array
    chi: Array
    A_norm2: Array


@dataclass(frozen=True, eq=False)
class LineSearchContext:
    """Per-step information handed to a :class:`LineSearch`.

    Built once per GEOPE step inside the jitted ``update_linesearch`` and
    consumed immediately, so it is a plain (non-pytree) container: its array
    fields are traced values and its callables are closures over the current
    parameters/direction. The two lazy accessors are the point of the design —
    they are only traced if a line search actually calls them.

    Attributes:
        f: The scalar infidelity along the line, ``f(t) -> infidelity``. Cheap;
            always present.
        a: Bracket endpoint (GEOPE passes ``-max_step_size / piecewise_steps``).
        b: Bracket endpoint (GEOPE passes ``0.0``).
        state: The threaded, line-search-owned state pytree.
        geometry: Lazy, memoised accessor ``() -> LineSearchGeometry`` giving the
            exact SU(N) slope/curvature along the direction. Calling it traces a
            ``logm`` and a directional HVP.
        distance_f: The squared-geodesic-distance objective along the line,
            ``distance_f(t) -> ½‖log_min(y† U(θ+t·coeffs))‖²``, whose derivatives
            ``geometry()`` reports (used for the Armijo sufficient-decrease test).
    """

    f: Callable[[Array], Array]
    a: Array
    b: Array
    state: dict
    geometry: Callable[[], LineSearchGeometry]
    distance_f: Callable[[Array], Array]


class LineSearch:
    """Base line search: tunes the scalar geodesic step size ``t``.

    Subclasses are frozen dataclasses (immutable config) that own an opaque
    JAX-pytree state. The base state is empty (``{}``) — stateless searches need
    nothing more.
    """

    name = "line_search"

    def init(self):
        """Return a fresh state pytree (called once per ``optimize()`` run)."""
        return {}

    def __call__(self, ctx: LineSearchContext):
        """Choose a step from ``ctx``; return ``(dt, new_state)``."""
        raise NotImplementedError


@dataclass(frozen=True)
class GoldenSection(LineSearch):
    """Golden-section search (the default) — stateless, zeroth-order.

    Minimises the infidelity ``ctx.f`` on the bracket; never touches the
    geometry, so it incurs no ``logm``/HVP cost.

    Args:
        tol: Convergence tolerance for the search interval. Defaults to 1e-5.
    """

    name = "golden_section"
    tol: float = 1e-5

    def __call__(self, ctx: LineSearchContext):
        dt, _ = _golden_section_search(ctx.f, ctx.a, ctx.b, tol=self.tol)
        return dt, ctx.state  # passthrough: no cross-step state


@dataclass(frozen=True)
class Adam(LineSearch):
    """1-D Adam line search — zeroth-order (minimises ``ctx.f``).

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
        return {"t_prev": jnp.asarray(0.0, jnp.float64)}

    def __call__(self, ctx: LineSearchContext):
        t0 = ctx.state["t_prev"] if self.warm_start else 0.0
        dt, _ = _adam_line_search(
            ctx.f,
            ctx.a,
            ctx.b,
            lr=self.lr,
            num_steps=self.num_steps,
            finite_difference=self.finite_difference,
            t_init=t0,
            fd_step=self.fd_step,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
        )
        return dt, {"t_prev": dt}


@dataclass(frozen=True)
class QuadraticArmijo(LineSearch):
    r"""Quadratic-seeded Armijo line search — second-order (geometry-aware).

    Implements the note *Quadratic-Seeded Armijo Line Search on
    $\mathrm{SU}(N)$*: it reads the exact slope $s=\psi'(0)$ and curvature
    $q=\psi''(0)$ of the squared-geodesic-distance objective from
    ``ctx.geometry()``, seeds the trial step at the local quadratic minimiser
    $-s/q$ (safety-scaled by ``gamma``, clipped to the bracket, with a
    full-step fallback when the curvature is not trustworthy), and enforces
    sufficient decrease with Armijo backtracking on ``ctx.distance_f``. The
    per-step evaluation count is recorded in the state as ``n_eval``.

    Requires the standard (projective) mode; ``ctx.geometry()`` raises under
    ``param_transform``.

    Args:
        c1: Armijo sufficient-decrease constant. Defaults to 1e-4.
        beta: Backtracking contraction factor in ``(0, 1)``. Defaults to 0.5.
        gamma: Initial-step safety factor ``>= 1``. Defaults to 1.0.
        t_min: Minimum step magnitude before the search gives up. Defaults to 1e-8.
        q_floor: Relative curvature floor for the quadratic seed. Defaults to 1e-12.
    """

    name = "quadratic_armijo"
    c1: float = 1e-4
    beta: float = 0.5
    gamma: float = 1.0
    t_min: float = 1e-8
    q_floor: float = 1e-12

    def init(self):
        return {"n_eval": jnp.asarray(0, jnp.int32)}

    def __call__(self, ctx: LineSearchContext):
        g = ctx.geometry()
        dt, _, n_eval = _quadratic_armijo_line_search(
            ctx.distance_f,
            ctx.a,
            g.s,
            g.q,
            g.F0,
            c1=self.c1,
            beta=self.beta,
            gamma=self.gamma,
            t_min=self.t_min,
            q_floor=self.q_floor,
        )
        return dt, {"n_eval": n_eval}


# ---------------------------------------------------------------------------
# Raw 1-D minimisers — private, JIT-compatible backends for the objects above.
# Not part of the public API; imported directly by tests.
# ---------------------------------------------------------------------------


def _golden_section_search(
    f: Callable[[Array], Array],
    a_init: float | Array,
    b_init: float | Array,
    tol: float = 1e-5,
) -> tuple[Array, Array]:
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
        A tuple ``(x_min, f_min)`` of the approximate minimiser
        and its function value.

    Example:
        ```python
        f = lambda x: (x - 2) ** 2
        x_min, f_min = _golden_section_search(f, 1.0, 5.0)
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
    return t_best, f_best


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
) -> tuple[Array, Array]:
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
        A tuple ``(t_best, f_best)`` of the best minimiser found and its
        function value, matching the ``(x_min, f_min)`` contract of
        :func:`_golden_section_search`.

    Example:
        ```python
        f = lambda x: (x - 2.0) ** 2
        x_min, f_min = _adam_line_search(f, 0.0, 5.0, lr=0.1, num_steps=200)
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
    return t_best, f_best


def _quadratic_armijo_line_search(
    fF: Callable[[Array], Array],
    a: float | Array,
    s: float | Array,
    q: float | Array,
    F0: float | Array,
    c1: float = 1e-4,
    beta: float = 0.5,
    gamma: float = 1.0,
    t_min: float = 1e-8,
    q_floor: float = 1e-12,
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

    - **Seed.** If ``q > q_floor * max(1, |s|)`` (a trustworthy positive
      curvature), start at ``t0 = clip(-s / (gamma * q), a, 0)`` — the local
      quadratic minimiser, safety-scaled by ``gamma`` and clipped to the
      bracket. Otherwise (no reliable minimiser) fall back to the full step
      ``t0 = a``.
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
        gamma: Initial-step safety factor ``>= 1`` shortening the seed without
            changing its geometry-informed scaling. Defaults to 1.0.
        t_min: Minimum allowed step magnitude before the search gives up.
            Defaults to 1e-8.
        q_floor: Relative curvature floor; ``q`` below ``q_floor * max(1, |s|)``
            triggers the full-step fallback. Defaults to 1e-12.

    Returns:
        A tuple ``(t_best, F_best, n_eval)``: the accepted step, ``fF`` at that
        step, and the number of ``fF`` evaluations spent.
    """
    f64 = lambda x: jnp.asarray(x, dtype=jnp.float64)
    a = f64(a)
    s = f64(s)
    q = f64(q)
    F0 = f64(F0)

    # TODO: Do we need this safety?
    q_scale = jnp.maximum(1.0, jnp.abs(s))
    use_quad = q > q_floor * q_scale
    t0 = jnp.where(use_quad, jnp.clip(-s / (gamma * q), a, 0.0), a)

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
