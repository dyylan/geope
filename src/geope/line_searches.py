"""Pluggable line-search objects for :class:`geope.Geope`.

A line search tunes the scalar geodesic **step size ``t``** along the search
direction GEOPE computes each step (it is not a full-parameter optimiser — that
is the separate :class:`Grape` class). The active object is passed to
``Geope.optimize(line_search=...)``; when omitted it defaults to
:class:`GoldenSection`.

Each line search is a ``@dataclass(frozen=True)`` — immutable config that gets
value-based ``__eq__``/``__hash__``/``__repr__`` for free. The value ``__eq__``
drives GEOPE's compile memo (``adam(1e-2) == adam(1e-2)`` ⇒ no recompile) and the
immutability keeps hyperparameter sweeps correct (a config cannot be mutated in
place and silently reuse a stale compiled function).

Cross-step state is a JAX pytree threaded through the jitted update (mirroring
``Grape.optimizer_state``): a jitted closure traces once, so persistent state
must enter/leave as an argument/result rather than as a mutated attribute. The
state is line-search-owned and opaque to GEOPE — :class:`Adam` carries
``{"t_prev"}`` (warm-start), :class:`GoldenSection` is stateless (``{}``).
``Geope.optimize`` re-``init()``s the state at the start of every run.
"""

from dataclasses import dataclass

import jax.numpy as jnp

from .utils import golden_section_search, adam_line_search


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

    def __call__(self, f, a, b, state):
        """Minimise ``f`` on ``[a, b]``; return ``(t_best, f_best, new_state)``."""
        raise NotImplementedError


@dataclass(frozen=True)
class GoldenSection(LineSearch):
    """Golden-section search (the default) — stateless.

    Args:
        tol: Convergence tolerance for the search interval. Defaults to 1e-5.
    """

    name = "golden_section"
    tol: float = 1e-5

    def __call__(self, f, a, b, state):
        dt, infid = golden_section_search(f, a, b, tol=self.tol)
        return dt, infid, state  # passthrough: no cross-step state


@dataclass(frozen=True)
class Adam(LineSearch):
    """1-D Adam line search.

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

    def __call__(self, f, a, b, state):
        t0 = state["t_prev"] if self.warm_start else 0.0
        dt, infid = adam_line_search(
            f,
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
        return dt, infid, {"t_prev": dt}


# lowercase aliases so ``line_search=adam(1e-2)`` reads naturally
adam = Adam
golden_section = GoldenSection
