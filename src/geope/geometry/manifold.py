"""The manifold interface: what the geodesic algorithm needs of a space.

See `geope.geometry` for the ownership chain. This module is deliberately free of
any Lie-group assumption — `geope.geometry.lie.groups.MatrixLieGroup` is where
left-trivialisation and a global generator basis enter, and
`geope.geometry.stiefel` holds spaces that have neither.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Callable, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

jax.config.update("jax_enable_x64", True)

from ..jax.hessian import get_hessian_fn
from .context import GeometricContext
from .tangent import TangentBundle


@dataclass(frozen=True, eq=False)
class Manifold(ABC):
    r"""A Riemannian manifold embedded in a space of arrays.

    Constructed with only its dimensions this is the pure space, and every
    geometric primitive already works — `log`, `distance2`, `fidelity`,
    `hessian_quadratic_form`. `bind` attaches the run's chart, tangent bundle and
    target, which is what a per-step `GeometricContext` needs.

    Attributes:
        target: The target point being synthesised; ``None`` when unbound.
        compute_point: The chart $\Phi$, ``phi -> point``; ``None`` when unbound.
        tangent: The `TangentBundle`; ``None`` when unbound. Keyword-only, and
            defaulted, so a subclass can declare positional fields of its own.
    """

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

    @abstractmethod
    def chart(self, generators) -> Callable[[Array], Array]:
        r"""Build the chart $\Phi:\mathbb R^{G\times K}\to M$ from a generator basis.

        Args:
            generators: The `geope.geometry.lie.Basis` of control (plus drift)
                generators the pulse is expressed in.

        Returns:
            An un-jitted ``phi -> point`` callable, to be composed into the
            optimiser's jitted update.
        """

    def chart_hvp(
        self, generators
    ) -> Callable[[Array, Array], tuple[Array, Array, Array]] | None:
        r"""The chart's second differential, ``(phi, p) -> (point, V, W)``.

        $V = \mathrm D\Phi_\phi[p]$ and $W = \mathrm D^2\Phi_\phi[p, p]$. Returning
        ``None`` — the default — simply means this manifold offers no analytic
        second differential, which disables the curvature tier of a
        `GeometricContext` (and with it the second-order line searches) and
        nothing else.

        Args:
            generators: The same generator basis `chart` was given.
        """
        return None

    # --- derived from the hooks --------------------------------------------

    @property
    def ambient_ndim(self) -> int:
        """The number of array axes one point has."""
        return len(self.ambient_shape)

    def _euclidean_inner(self, x: Array, y: Array) -> Array:
        r"""$\mathrm{Re}\sum \bar x\,y$ over the ambient axes — the embedded metric.

        The default building block for `inner`: correct for any manifold whose
        metric is the one induced by the embedding (which includes a
        bi-invariant group metric, where it is $\mathrm{Re}\,\mathrm{Tr}(x^\dagger y)$).
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
        target: Array,
        compute_point: Callable[[Array], Array],
        tangent: TangentBundle,
    ) -> Manifold:
        """Attach the run's chart, tangent bundle and target.

        Returns a *new* manifold — this one is frozen — so the unbound
        mathematical object stays reusable.

        Args:
            target: The target point, of shape ``ambient_shape``.
            compute_point: The chart ``phi -> point``.
            tangent: The `TangentBundle`.

        Returns:
            A bound `Manifold` of the same class.

        Raises:
            ValueError: If ``target`` is not of shape ``ambient_shape``.
        """
        target = jnp.asarray(target)
        if target.shape != tuple(self.ambient_shape):
            raise ValueError(
                f"{self.name} expects a {tuple(self.ambient_shape)} target, got "
                f"{tuple(target.shape)}."
            )
        return replace(
            self, target=target, compute_point=compute_point, tangent=tangent
        )

    def validate_point(self, point: np.ndarray, what: str = "point") -> None:
        """Raise if ``point`` does not lie on this manifold.

        The *membership* check — unitarity for a matrix group, unit norm on the
        sphere — as opposed to the shape check `bind` does.

        **Host-side only, and deliberately numpy.** `Parameters.__init__` calls
        it once on the target, at configuration time. It must never run inside a
        trace: `bind` is reached lazily through `Parameters.manifold`, which may
        first be touched inside a `jax.jit`, and there even
        ``jnp.asarray(numpy_array)`` yields a tracer — so a jnp-based check
        placed on that path raises `ConcretizationTypeError` instead of doing
        its job. Using numpy keeps that mistake impossible to make quietly.

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
                "compute_point=..., tangent=...)`, or read the manifold off "
                "`Parameters.manifold`, which binds it for you."
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

    @cached_property
    def value_and_grad(self) -> Callable[[Array], tuple[Array, Array]]:
        """``phi -> (infidelity, dinfidelity/dphi)`` (used by GRAPE)."""
        self._require_bound("value_and_grad")
        return jax.value_and_grad(self.infidelity_at)

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
