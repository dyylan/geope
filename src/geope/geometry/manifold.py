r"""The manifold interface: what the geodesic algorithm needs of a space.

A `Manifold` here is a **submanifold of the ambient space**
$\mathcal A = \mathbb C^{N\times m}$ that `geope.geometry.chart` describes, and
the division of labour between the two is the organising idea of this package:

* everything valued in $\mathcal A$ — the chart $\Phi(\phi) = U(\phi)\,x_0$ and
  its two differentials — is `geope.geometry.chart`'s, composed once by
  `Manifold.bind`, which adds no mathematics of its own;
* everything that happens in $T_x\mathcal M$ — the metric, the coefficient
  frame, the logarithm, the fidelity — is the *manifold's*, and lives on the
  hooks below.

`to_tangent` is the one bridge between them. This module is deliberately free of
any Lie-group assumption — `geope.geometry.lie.groups.MatrixLieGroup` is where
left-trivialisation and a global generator basis enter, and
`geope.geometry.stiefel` holds spaces that have neither.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Callable, ClassVar, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

jax.config.update("jax_enable_x64", True)

from ..jax.hessian import get_hessian_fn
from .chart import (
    get_chart_fn,
    get_chart_hvp_fn,
    get_jacobian_fn,
    get_split_jacobian_fn,
)
from .context import GeometricContext
from .tangent import TangentBundle

if TYPE_CHECKING:
    # Annotation only: a real import would pull in `lie/__init__`, which imports
    # `.groups`, which imports this module.
    from .lie.basis import Basis


@dataclass(frozen=True, eq=False)
class Manifold(ABC):
    r"""A Riemannian submanifold of the ambient space $\mathcal A$.

    Constructed with only its dimensions this is the pure space, and its
    geometric primitives already work — `log`, `distance2`, `inner`, `norm2`,
    `fidelity`, `hessian_quadratic_form`. `bind` attaches the run's chart,
    coefficient frame and target, which is what a per-step `GeometricContext`
    needs. (`coefficients` is the one exception: a coordinate choice *is* problem
    data, so it needs the frame that `bind` brings.)

    Attributes:
        base_point: $\Phi(0) \in \mathcal A$ — the point the chart's orbit starts
            from, and the whole of what a manifold contributes to its own chart.
            ``None`` means the pulse propagator *is* the point, which is the case
            on any matrix Lie group; a homogeneous space supplies the state or
            frame the pulse drives. Unlike the other three this is part of the
            *space*, set at construction rather than by `bind`.
        target: The target point; ``None`` when unbound.
        compute_point: The chart $\Phi$, ``phi -> point``; ``None`` when unbound.
        tangent: The `TangentBundle`; ``None`` when unbound.

    All four are keyword-only and defaulted, so a subclass can declare positional
    fields of its own.
    """

    base_point: Array | None = field(default=None, kw_only=True)
    target: Array | None = field(default=None, kw_only=True)
    compute_point: Callable[[Array], Array] | None = field(default=None, kw_only=True)
    tangent: TangentBundle | None = field(default=None, kw_only=True)

    #: ``True`` when the geometry quotients out a global phase.
    projective: ClassVar[bool]
    #: Short human-readable name, used in error messages and logs.
    name: ClassVar[str]

    # --- the hooks ------------------------------------------------------

    @property
    @abstractmethod
    def ambient_shape(self) -> tuple[int, ...]:
        """The shape of one point in the embedding space."""

    @property
    @abstractmethod
    def manifold_dim(self) -> int:
        """The intrinsic (real) dimension of the manifold."""

    @abstractmethod
    def to_tangent(self, point: Array, ambient: Array) -> Array:
        r"""Map an ambient velocity at ``point`` into this manifold's tangent representation.

        The representation is whatever `inner`, `coefficients` and `log` all
        speak: left-trivialised algebra elements on a Lie group, ambient
        projections on a homogeneous space. Batches over leading axes.

        Args:
            point: The base point, of shape ``ambient_shape``.
            ambient: Ambient velocities, shape ``(..., *ambient_shape)``.
        """

    @abstractmethod
    def inner(self, point: Array, x: Array, y: Array) -> Array:
        r"""The Riemannian metric $\langle x, y\rangle_{\text{point}}$.

        Takes the base point because in general the metric depends on it (a
        bi-invariant group metric simply ignores it). Batches over leading axes.
        """

    @abstractmethod
    def coefficients(self, point: Array, tangent: Array) -> Array:
        r"""Linear coordinates for tangent vectors at ``point``.

        Must be faithful and metric-consistent up to a fixed constant $c > 0$:
        $\sum_k c_k(u)\,c_k(v) = c\,\langle u, v\rangle_{\text{point}}$. These are
        the coordinates the geodesic least-squares problem is posed in, so the
        constant cancels between its two operands.
        """

    @abstractmethod
    def log(self, x: Array, y: Array, key: Array | None = None) -> Array:
        r"""$\mathrm{Log}_x(y)$: the minimal-geodesic tangent **at $x$**, pointing to $y$.

        Taking it at $x$ rather than at $y$ is what lets it be compared with a
        velocity at $x$: on a general manifold $T_x$ and $T_y$ are unrelated
        spaces, and only a group's trivialisation makes the distinction vacuous.

        Args:
            x: The base point.
            y: The end point.
            key: Optional JAX key, for implementations whose logarithm is
                randomised (the unitary-specialised one is not).
        """

    @abstractmethod
    def tangent_acceleration(self, point: Array, v: Array, w: Array) -> Array:
        r"""$\dot\Omega(0)$: the derivative of the tangent representation along the chart.

        Given the ambient first and second directional derivatives $V, W$ of a
        curve through ``point``, return
        $\frac{\mathrm d}{\mathrm dt}\,\texttt{to\_tangent}(c(t), \dot c(t))$ at
        $t = 0$. This is the term that carries the *chart's* bending into the
        curvature of the objective, as distinct from the manifold's own curvature
        (`hessian_quadratic_form`). Finite-differencing `to_tangent` along the
        chart is the definitive check on an implementation.
        """

    @abstractmethod
    def hessian_quadratic_form(
        self, point: Array, a: Array, omega: Array
    ) -> tuple[Array, Array]:
        r"""$\bigl(\langle\Omega,\mathcal K_A\Omega\rangle,\ \text{spread}\bigr)$.

        The intrinsic term of the second derivative of
        $\tfrac12 d_g(\cdot, y)^2$ along a curve with tangent $\Omega$, where
        $\mathcal K_A$ is the Riemannian Hessian of the squared distance at
        ``point``, plus a scalar diagnostic of how close ``a`` is to the cut
        locus. Must be exact for an arbitrary $\Omega$, and must reduce to
        $\lVert A\rVert^2$ when $\Omega = A$ (the radial direction).

        **``a`` is the context's $A = -\mathrm{Log}_x(y)$**, pointing *away* from
        the target, so an implementation that cares about the sign must negate
        it. Most do not — on a bi-invariant group and on the sphere the kernel is
        even in $A$ — but `geope.geometry.stiefel.stiefel.Stiefel` genuinely is
        not: $-A$ addresses the geodesic reflection of the target, which has a
        different Hessian. Do not read the group's evenness as the contract.

        A manifold may decline: raising `NotImplementedError` is the documented
        way to withdraw `GeometricContext.q_exact` and `GeometricContext.rho`
        (and with them `geope.line_searches.ApproximateQuadraticArmijo`) rather
        than return an inexact form, and `tests/test_manifolds.py` probes for the
        capability instead of assuming it.
        """

    @abstractmethod
    def fidelity(self, x: Array, y: Array) -> Array:
        """The convergence score between two points: ``1`` when they agree.

        The optimisers' stopping test and every logged trajectory read this, so
        it should be normalised to $1$ at the target whatever the manifold.
        "Fidelity" is GEOPE's name for that score; a manifold is free to define
        it however its geometry suggests.
        """

    @abstractmethod
    def infidelity(self, x: Array, y: Array) -> Array:
        """The cost the optimisers minimise: ``0`` when the two points agree.

        Declared separately from `fidelity` rather than derived from it as
        ``1 - fidelity``: which cost a space is optimised against is the
        manifold's own business, and tying the two together here would impose a
        relation that only happens to hold for the trace fidelities.
        """

    # --- derived from the hooks --------------------------------------------

    @property
    def ambient_ndim(self) -> int:
        """The number of array axes one point has."""
        return len(self.ambient_shape)

    def ambient_inner(self, x: Array, y: Array) -> Array:
        r"""$\mathrm{Re}\sum \bar x\,y$ over the ambient axes — the metric of $\mathcal A$.

        The default building block for `inner`, and correct as it stands for any
        manifold whose metric is the one *induced by the embedding* — which
        includes a bi-invariant group metric, where it is
        $\mathrm{Re}\,\mathrm{Tr}(x^\dagger y)$, and the round metric on the
        state sphere. A manifold whose metric is **not** the ambient one must say
        so: `geope.geometry.stiefel.stiefel.Stiefel` carries the canonical metric
        $\mathrm{Re}\,\mathrm{Tr}(x^\dagger W_Q y)$ instead, which is what gives
        it different geodesics from the embedding's.
        """
        axes = tuple(range(-self.ambient_ndim, 0))
        return jnp.real(jnp.sum(jnp.conj(x) * y, axis=axes))

    def norm2(self, point: Array, x: Array) -> Array:
        r"""$\lVert x\rVert^2_{\text{point}}$."""
        return self.inner(point, x, x)

    def distance2(self, x: Array, y: Array, key: Array | None = None) -> Array:
        r"""Half the squared geodesic distance, $\tfrac12 d_g(x, y)^2$.

        The objective the first- and second-order line searches minimise, and the
        quantity whose derivatives a `GeometricContext` reports. Costs one `log`.
        """
        a = self.log(x, y, key)
        return 0.5 * self.norm2(x, a)

    # --- binding -----------------------------------------------------------

    @property
    def is_bound(self) -> bool:
        """Whether a chart, tangent bundle and target are attached."""
        return (
            self.target is not None
            and self.compute_point is not None
            and self.tangent is not None
        )

    def bind(
        self,
        *,
        target: Array | np.ndarray | None,
        generators: Basis,
        frame: Basis | None = None,
        columns: np.ndarray | None = None,
        wrap_chart: Callable[[Callable], Callable] | None = None,
    ) -> Manifold:
        r"""Attach this problem's chart, coefficient frame and target.

        The seam between the two layers this package is built on, and it holds no
        mathematics of its own: `geope.geometry.chart` builds the ambient jet
        $(\Phi,\ \mathrm D\Phi,\ \mathrm D^2\Phi)$ from ``generators`` and this
        manifold's `base_point`, and everything else here is assembly.

        Returns a *new* manifold — this one is frozen — so the unbound
        mathematical object stays reusable.

        Args:
            target: The point being synthesised, of shape ``ambient_shape``, or
                ``None`` for a problem that names no target yet. The chart and
                the tangent bundle still attach, so the pulse stays computable
                and differentiable, but `is_bound` stays ``False`` and every
                target-dependent quantity raises.
            generators: The chart's generator sub-`geope.geometry.lie.Basis` —
                the proj+drift basis the pulse is expressed in.
            frame: The **ambient** coefficient frame `coefficients` resolves a
                tangent vector against, if this manifold coordinatises through
                one. ``None`` is not a degraded mode: it is the right answer for
                a manifold whose fibre coordinates are a real/imaginary split of
                the ambient array (see
                `geope.geometry.stiefel.sphere.StateSphere`), and it means no
                projector is ever built.
            columns: Boolean mask over the chart's coefficient columns selecting
                those the geodesic solve may move. ``None`` means every column.
            wrap_chart: An opaque **reparametrisation of the chart's input**, or
                ``None``. Its presence is the single ``param_transform`` signal:
                it disables the analytic HVP (and with it the curvature tier and
                the manual propagator Hessian) and frees every column. The
                manifold never inspects it, which is what keeps
                `geope.parameters.Parameters` out of this layer.

                That it reparametrises the *input* is what lets it wrap the
                landed chart rather than the bare propagator:
                $\mathrm{wrap}(\Phi)(\phi) = U(\tau(\phi))\,x_0$ either way, so
                the base point never has to be threaded through it.

        Returns:
            A bound `Manifold` of the same class.

        Raises:
            ValueError: If this manifold is already bound, or if ``target`` is
                not of shape ``ambient_shape``.
        """
        if self.is_bound:
            raise ValueError(
                f"{self.name} is already bound to a chart and a target. Pass the "
                "pure space and let `Parameters` bind it; re-binding would "
                "silently replace what you attached."
            )
        if target is not None:
            target = jnp.asarray(target)
            if target.shape != tuple(self.ambient_shape):
                raise ValueError(
                    f"{self.name} expects a {tuple(self.ambient_shape)} target, "
                    f"got {tuple(target.shape)}."
                )

        compute_point = get_chart_fn(generators.basis, self.base_point)
        if wrap_chart is None:
            hvp = get_chart_hvp_fn(generators.basis, self.base_point)
            jacobian_fn = get_jacobian_fn
        else:
            compute_point = wrap_chart(compute_point)
            # Holomorphic autodiff through a real-valued user transform would
            # drop the imaginary part of the intermediates; and there is no
            # exponential-product structure left to exploit, so no analytic HVP,
            # no manual propagator Hessian, and every column is free.
            jacobian_fn = get_split_jacobian_fn
            generators = hvp = columns = None

        return replace(
            self,
            target=target,
            compute_point=compute_point,
            tangent=TangentBundle(
                frame=frame,
                jacobian=jacobian_fn(compute_point),
                hvp=hvp,
                generators=generators,
                columns=columns,
            ),
        )

    def validate_point(self, point: np.ndarray, what: str = "point") -> None:
        """Raise if ``point`` does not lie on this manifold.

        The *membership* check — unitarity for a matrix group, unit norm on the
        sphere — as opposed to the shape check `bind` does.

        **Host-side only, and deliberately numpy.** Membership of the target is a
        *configuration-time* question — `Parameters.__init__` calls this once,
        before the chart exists, so that a bad target fails at construction
        rather than as a plausible-looking fidelity. Keeping it numpy is what
        makes the alternative unmakeable: inside a `jax.jit` even
        ``jnp.asarray(numpy_array)`` yields a tracer, so a jnp-based membership
        check raises `ConcretizationTypeError` instead of validating. That is
        the trap that killed ``validate_basis``; keep membership checks
        host-side and numpy.

        The base implementation accepts everything, so a manifold that cannot
        cheaply decide membership simply does not override it.

        Args:
            point: The candidate point, of shape ``ambient_shape``.
            what: Name of the offending argument, used in the error message.

        Raises:
            ValueError: If ``point`` is not on the manifold.
        """
        return None

    def _require_bound(self, what: str) -> None:
        """Raise a pointed error when a bound-only quantity is asked for."""
        if not self.is_bound:
            raise ValueError(
                f"{what} needs a bound manifold: call `bind(target=..., "
                "generators=...)`, or read the manifold off "
                "`Parameters.manifold`, which binds it for you. A manifold bound "
                "with `target=None` counts as unbound here — the chart works, "
                "but nothing that needs a target does."
            )

    # --- the chart's objectives, in parameter space -------------------------

    @cached_property
    def fidelity_at(self) -> Callable[[Array], Array]:
        """``phi ->`` the fidelity of the pulse ``phi`` against the target.

        Compiled and memoised rather than evaluated eagerly, because the
        host-side loops call it one trial pulse at a time: uncompiled, each
        call re-lowers the chart's `jax.lax.scan`, which cost ~100 ms per
        call on a 3-qubit, 20-step pulse and made `geope.Geope`'s
        Gram-Schmidt fallback more expensive than the whole geodesic step it
        was replacing.
        """
        self._require_bound("fidelity_at")
        return jax.jit(lambda phi: self.fidelity(self.compute_point(phi), self.target))

    @cached_property
    def infidelity_at(self) -> Callable[[Array], Array]:
        """``phi ->`` the infidelity of the pulse ``phi`` — what GRAPE minimises.

        Compiled and memoised, for the same reason as `fidelity_at`.
        """
        self._require_bound("infidelity_at")
        return jax.jit(
            lambda phi: self.infidelity(self.compute_point(phi), self.target)
        )

    # TODO: For GRAPE we will need to use the Jacobian propagators for efficient
    # calculation of Grad F
    @cached_property
    def value_and_grad(self) -> Callable[[Array], tuple[Array, Array]]:
        """``phi -> (infidelity, dinfidelity/dphi)`` (used by GRAPE)."""
        self._require_bound("value_and_grad")
        return jax.value_and_grad(self.infidelity_at)

    # TODO: For GRAPE we will need to use the Hessian propagators for efficient
    # calculation of Grad F. Use autodiff only to check in tests.
    @cached_property
    def hessian_autodiff(self) -> Callable[[Array], Array]:
        """The infidelity Hessian by forward-over-reverse HVPs."""
        self._require_bound("hessian_autodiff")
        return get_hessian_fn(self.infidelity_at)

    @cached_property
    def hessian(self) -> Callable[[Array], Array]:
        """The infidelity Hessian (used by GRAPE's Newton methods).

        Autodiff by default; a manifold whose chart has exploitable structure may
        override with an analytic one (see
        `geope.geometry.lie.groups.MatrixLieGroup`).
        """
        return self.hessian_autodiff

    # --- the per-step geometry ---------------------------------------------

    def context(self, free_params: Array) -> GeometricContext:
        """Open a `GeometricContext` at the pulse ``free_params``.

        Cheap: every quantity on the returned context is lazy, so this on its own
        evaluates nothing. Build one per optimisation step, *inside* the jitted
        update — see `GeometricContext` for why it must not leave that trace.
        """
        self._require_bound("context")
        return GeometricContext(self, free_params)
