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
    get_chart_hessian_fn,
    get_chart_hvp_fn,
    get_chart_jacobian_fn,
    get_chart_vjp_fn,
    get_split_jacobian_fn,
    get_split_vjp_fn,
)
from .context import GeometricContext
from .tangent import TangentBundle

if TYPE_CHECKING:
    from .basis import Basis


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

    # --- the declinable cost-derivative hooks -------------------------------

    def cost_gradient(self, x: Array, y: Array) -> Array:
        r"""$\partial C/\partial\bar x$: the ambient gradient of `infidelity` at ``x``.

        The Wirtinger derivative with respect to the *conjugate* point, so that
        the chain rule out of the ambient space reads

        $$\frac{\partial\,C(\Phi(\phi))}{\partial\phi_a}
          = 2\,\mathrm{Re}\bigl\langle \hat G,\ \mathrm D\Phi_a \bigr\rangle,
          \qquad \hat G = \texttt{cost\_gradient}(x, y),$$

        which is what makes this — rather than the full Jacobian — the object a
        gradient is built from: contracted through
        `geope.geometry.tangent.TangentBundle.vjp` it costs one pullback, and no
        $(*\text{ambient}, G, K)$ tensor is ever formed.

        **Declinable.** The base implementation raises `NotImplementedError`,
        which withdraws the analytic `value_and_grad` in favour of autodiff — the
        same way `hessian_quadratic_form` may withdraw `GeometricContext.q_exact`.
        A manifold that declines still optimises; it just differentiates the slow
        way. Every manifold GEOPE ships implements it, in two lines, by
        delegating to `geope.geometry.cost.trace_cost_gradient`.

        Args:
            x: The current point, of shape ``ambient_shape``.
            y: The target, of shape ``ambient_shape``.

        Returns:
            An ``Array`` of shape ``ambient_shape``.

        Raises:
            NotImplementedError: If this manifold declines to supply one.
        """
        raise NotImplementedError(
            f"{self.name} does not supply an ambient cost gradient; "
            "`value_and_grad` falls back to autodiff."
        )

    def cost_hessian_form(self, x: Array, y: Array, u: Array) -> Array:
        r"""The cost's **own** second-derivative form, $\mathrm{Hess}_C[u_a, u_b]$.

        Only the term that comes from the curvature of $C$ in the ambient space.
        The other term — the *chart's* bending, $2\,\mathrm{Re}\langle\hat G,
        \mathrm D^2\Phi_{ab}\rangle$ — is the chain rule applied to `cost_gradient`
        and is assembled generically by `hessian`, so an implementation here must
        **not** include it. A cost that is affine in the point (any phase-sensitive
        trace fidelity) returns zeros.

        Args:
            x: The current point, of shape ``ambient_shape``.
            y: The target, of shape ``ambient_shape``.
            u: The chart's Jacobian columns, of shape ``(P, *ambient_shape)``
                with ``P = G * K``.

        Returns:
            A real ``Array`` of shape ``(P, P)``.

        Raises:
            NotImplementedError: If this manifold declines to supply one, which
                withdraws the analytic `hessian` in favour of `hessian_autodiff`.
        """
        raise NotImplementedError(
            f"{self.name} does not supply an ambient cost Hessian form; "
            "`hessian` falls back to autodiff."
        )

    @property
    def has_cost_gradient(self) -> bool:
        """Whether this manifold overrides `cost_gradient`.

        A host-side capability probe — an identity check on the bound method's
        function, so nothing is traced and no exception has to be caught inside a
        `jax.jit`. `value_and_grad` reads it to pick its path.
        """
        return type(self).cost_gradient is not Manifold.cost_gradient

    @property
    def has_cost_hessian_form(self) -> bool:
        """Whether this manifold overrides `cost_hessian_form`. See `has_cost_gradient`."""
        return type(self).cost_hessian_form is not Manifold.cost_hessian_form

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
                it drops both second differentials (the HVP, and with it the
                curvature tier, and the dense chart Hessian), turns the first one
                into its autodiff form in both directions, and frees every
                column. The manifold never inspects it, which is what keeps
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
            # The whole jet from the propagator recursions: no autodiff anywhere
            # on this path.
            jacobian = get_chart_jacobian_fn(generators.basis, self.base_point)
            vjp = get_chart_vjp_fn(generators.basis, self.base_point)
            hvp = get_chart_hvp_fn(generators.basis, self.base_point)
            hessian = get_chart_hessian_fn(generators.basis, self.base_point)
        else:
            compute_point = wrap_chart(compute_point)
            # Holomorphic autodiff through a real-valued user transform would
            # drop the imaginary part of the intermediates; and there is no
            # exponential-product structure left to exploit, so the first
            # differential falls back to autodiff in both directions, there is no
            # second differential at all, and every column is free.
            jacobian = get_split_jacobian_fn(compute_point)
            vjp = get_split_vjp_fn(compute_point)
            generators = hvp = hessian = columns = None

        return replace(
            self,
            target=target,
            compute_point=compute_point,
            tangent=TangentBundle(
                frame=frame,
                jacobian=jacobian,
                vjp=vjp,
                hvp=hvp,
                hessian=hessian,
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

    @cached_property
    def value_and_grad_autodiff(self) -> Callable[[Array], tuple[Array, Array]]:
        """``phi -> (infidelity, dinfidelity/dphi)`` by reverse-mode autodiff.

        The reference the analytic path is checked against, and the fallback for a
        manifold that declines `cost_gradient`.

        Note that differentiating `infidelity_at` holomorphically with respect to
        a ``complex128`` array whose imaginary part is identically zero gives a
        gradient with a spurious imaginary part. `value_and_grad` does not.
        """
        self._require_bound("value_and_grad_autodiff")
        return jax.value_and_grad(self.infidelity_at)

    @cached_property
    def value_and_grad(self) -> Callable[[Array], tuple[Array, Array]]:
        r"""``phi -> (infidelity, dinfidelity/dphi)`` (used by GRAPE).

        Analytic when this manifold supplies `cost_gradient`: **one** pass over
        the pulse through `geope.geometry.tangent.TangentBundle.vjp`, which
        returns the point and a pullback sharing the gate exponentials and their
        partial products, then one ambient cost gradient through it,

        $$\frac{\partial C}{\partial\phi_{g,k}}
          = 2\,\mathrm{Re}\,\mathrm{Tr}\bigl(\hat G^\dagger\,\partial_{g,k}\Phi\bigr),$$

        which never forms the Jacobian. That the value comes back *with* the
        pullback is what keeps it to one pass: the covector $\hat G$ is a function
        of the point, so a separate `compute_point` call would re-exponentiate
        every gate. Falls back to `value_and_grad_autodiff` otherwise.

        The gradient is **real by construction** and returned in the parameter's
        own dtype, so it stays a drop-in for optax on the ``complex128`` pulse
        arrays `geope.parameters.Parameters.free` hands out.
        """
        self._require_bound("value_and_grad")
        if not self.has_cost_gradient:
            return self.value_and_grad_autodiff

        target = self.target
        vjp = self.tangent.vjp

        @jax.jit
        def value_and_grad_fn(phi: Array) -> tuple[Array, Array]:
            point, pullback = vjp(phi)
            grad = 2.0 * jnp.real(pullback(self.cost_gradient(point, target)))
            return self.infidelity(point, target), grad.astype(phi.dtype)

        return value_and_grad_fn

    @cached_property
    def hessian_autodiff(self) -> Callable[[Array], Array]:
        """The infidelity Hessian ``(P, P)`` by forward-over-reverse HVPs.

        The reference the analytic path is checked against, and the fallback
        wherever the analytic one is unavailable — a manifold that declines
        `cost_hessian_form`, or the ``param_transform`` chart, which has no
        second differential to build one from.

        Carries the same spurious imaginary part as `value_and_grad_autodiff` on
        a ``complex128`` pulse whose imaginary part is identically zero — it
        differentiates the same holomorphic gradient. Its **real part** is right
        in every case, and it is exact on the ``float64`` pulses of the
        ``param_transform`` path where it is actually used. Compare against it on
        real input.
        """
        self._require_bound("hessian_autodiff")
        raw = get_hessian_fn(self.infidelity_at)
        # `get_hessian_fn` returns (P, *phi.shape); flatten so both paths share
        # the one (P, P) contract their callers rely on.
        return lambda phi: jnp.reshape(raw(phi), (phi.size, phi.size))

    @cached_property
    def hessian(self) -> Callable[[Array], Array]:
        r"""The infidelity Hessian ``(P, P)`` (used by GRAPE's Newton methods).

        Analytic when this manifold supplies `cost_hessian_form` *and* the chart
        has a second differential to offer
        (`geope.geometry.tangent.TangentBundle.hessian`). The assembly is the
        chain rule in two terms — the cost's own curvature and the chart's
        bending —

        $$\partial_a\partial_b\,C(\Phi(\phi))
          = \mathrm{Hess}_C[\mathrm D\Phi_a, \mathrm D\Phi_b]
          + 2\,\mathrm{Re}\bigl\langle \hat G,\ \mathrm D^2\Phi_{ab}\bigr\rangle,$$

        the first from the hook and the second generic, since $\hat G$ is just
        `cost_gradient`. Falls back to `hessian_autodiff` otherwise.

        Dense: $O(G^2 d^2 K^2)$, which is what a Newton step costs on this chart
        by either route.
        """
        self._require_bound("hessian")
        if not (self.has_cost_hessian_form and self.tangent.hessian is not None):
            return self.hessian_autodiff

        target = self.target
        compute_point = self.compute_point
        jacobian_fn = self.tangent.jacobian
        hessian_fn = self.tangent.hessian

        @jax.jit
        def hessian_fn_of_phi(phi: Array) -> Array:
            p = phi.size
            point = compute_point(phi)

            # (*ambient, G, K) -> (P, *ambient): one Jacobian column per parameter.
            columns = jnp.moveaxis(jacobian_fn(phi), (-2, -1), (0, 1))
            columns = columns.reshape((p, *self.ambient_shape))

            # Contract the ambient axes of D^2 Phi first, in its native
            # (G, G, *ambient, K, K) layout, so only a (G, G, K, K) block is
            # ever transposed; then reorder to the row-major (gate, coeff)
            # flattening that `phi.flatten()` and the gradient both use.
            grad = self.cost_gradient(point, target)
            grad_axes = tuple(range(2, 2 + self.ambient_ndim))
            chart = 2.0 * jnp.real(
                jnp.sum(
                    jnp.conj(jnp.expand_dims(grad, (0, 1, -2, -1))) * hessian_fn(phi),
                    axis=grad_axes,
                )
            )
            chart = jnp.transpose(chart, (0, 2, 1, 3)).reshape(p, p)
            return self.cost_hessian_form(point, target, columns) + chart

        return hessian_fn_of_phi

    # --- the per-step geometry ---------------------------------------------

    def context(self, free_params: Array) -> GeometricContext:
        """Open a `GeometricContext` at the pulse ``free_params``.

        Cheap: every quantity on the returned context is lazy, so this on its own
        evaluates nothing. Build one per optimisation step, *inside* the jitted
        update — see `GeometricContext` for why it must not leave that trace.
        """
        self._require_bound("context")
        return GeometricContext(self, free_params)
