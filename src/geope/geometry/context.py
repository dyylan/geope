"""The per-step geometric context: every quantity a GEOPE step needs, in tiers."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_enable_x64", True)

if TYPE_CHECKING:
    from .manifold import Manifold
    from .tangent import TangentBundle


class GeometricContext:
    r"""Every per-step geometric quantity, derived lazily in cost tiers.

    Built by `Manifold.context` once per optimisation step, **inside** the jitted
    update. Every quantity is a `cached_property`, tier 0 included, so the tiers
    are a description of what a given consumer *pays*, not of what is computed:
    `Gecko` reads only `omegas` and therefore never traces the matrix logarithm.

    | tier | quantities | cost |
    |---|---|---|
    | **0** base point | `point`, `jacobian`, `A`, `A_norm2`, `F0`, `gammas`, `omegas`, `infidelity`, `fidelity` | one propagator, one Jacobian, one logarithm |
    | **1** direction | `V`, `Omega`, `omega_norm2`, `s`, `xi_rel` | free: one contraction of tier 0's Jacobian |
    | **2** curvature | `W`, `accel`, `chi`, `q`, `q_exact`, `rho` | one directional HVP plus the manifold's Riemannian Hessian |
    | **3** ray | `point_at`, `infidelity_at`, `fidelity_at`, `distance_at` | one propagator per trial point |

    Only tier 0 is direction-free. Tiers 1–3 need the search direction, tier 3
    because the ray *is* ``free_params + t * coeffs``; it arrives through
    `set_direction` after the least-squares solve (which needs this context's own
    `gammas` and `omegas`), and may be set **once**. That single rule replaces any
    list of what to invalidate: every direction-dependent property raises while
    `coeffs` is ``None``, and a raising `cached_property` memoises nothing, so no
    value computed for one direction can survive into another.

    Tier 2 additionally needs `TangentBundle.hvp`, which is ``None`` under
    ``param_transform``; tiers 0, 1 and 3 work there.

    **The context is trace-time only.** Never register it as a pytree, return it
    from a jitted function, or put it in a ``scan``/``while_loop`` carry: its memo
    is valid only inside the trace that built it. A property must also be first
    read *outside* any ``lax`` control flow you intend to read it from.

    Attributes:
        manifold: The bound `Manifold` this context is opened on.
        free_params: The pulse $\phi$ at the base point, shape ``(G, K_free)``.
    """

    def __init__(self, manifold: Manifold, free_params: Array) -> None:
        self.manifold = manifold
        self.free_params = free_params
        self._coeffs: Array | None = None

    # --- the direction ------------------------------------------------------

    @property
    def tangent(self) -> TangentBundle:
        """The manifold's `TangentBundle`."""
        return self.manifold.tangent

    @property
    def target(self) -> Array:
        """The target point being synthesised."""
        return self.manifold.target

    @property
    def coeffs(self) -> Array | None:
        """The search direction, or ``None`` before `set_direction`."""
        return self._coeffs

    def set_direction(self, coeffs: Array) -> None:
        """Attach the search direction, once.

        Args:
            coeffs: The direction ``(G, K_free)``, in the same parameter space as
                ``free_params``.

        Raises:
            ValueError: If a direction has already been set. Open a new context
                instead — the memoised tier-0 quantities are direction-free and
                cost nothing to keep, but a second direction would silently reuse
                tier-1/2 values computed for the first.
        """
        if self._coeffs is not None:
            raise ValueError(
                "This context already has a direction; a GeometricContext takes "
                "one. Open a new context for a new direction."
            )
        self._coeffs = coeffs

    def _require_direction(self, what: str) -> None:
        if self._coeffs is None:
            raise ValueError(
                f"`{what}` is direction-dependent: call `set_direction(coeffs)` "
                "first (the solve for `coeffs` reads `gammas`/`omegas`, which "
                "are direction-free)."
            )

    def _require_curvature(self, what: str) -> None:
        if self.tangent.hvp is None:
            raise NotImplementedError(
                f"`{what}` needs the chart's second differential, which is "
                "unavailable under `param_transform` (the manual propagator "
                "derivatives assume a plain product of exponentials). Use a "
                "line search that does not read the curvature — GoldenSection, "
                "Adam or Armijo."
            )

    # --- tier 0: the base point ---------------------------------------------

    @cached_property
    def point(self) -> Array:
        r"""The pulse's point on the manifold, $\Phi(\phi)$. One propagator."""
        return self.manifold.compute_point(self.free_params)

    @cached_property
    def jacobian(self) -> Array:
        r"""$\partial\Phi/\partial\phi$, shape ``(*ambient, G, K_free)``. One Jacobian."""
        return jnp.asarray(self.tangent.jacobian(self.free_params))

    @cached_property
    def A(self) -> Array:
        r"""The geodesic tangent $A = -\mathrm{Log}_U(V)$. **The only logarithm.**

        The minimal-geodesic tangent **at the base point**, negated so that it
        points *away* from the target: the slope `s` is then positive at a descent
        direction and the line-search bracket is $[-t_{\max}, 0]$.

        Taking it at $U$ rather than at the target is what makes it comparable
        with `Omega` on a general manifold — $T_U$ and $T_V$ are unrelated spaces
        unless a group's trivialisation identifies them.
        """
        return -self.manifold.log(self.point, self.target)

    @cached_property
    def A_norm2(self) -> Array:
        r"""$\lVert A\rVert_F^2 = 2 F_0$."""
        return self.manifold.norm2(self.point, self.A)

    @cached_property
    def F0(self) -> Array:
        r"""The squared-geodesic-distance objective $F_0 = \tfrac12\lVert A\rVert_F^2$."""
        return 0.5 * self.A_norm2

    @cached_property
    def gammas(self) -> Array:
        r"""The geodesic tangent's coefficients — the least-squares target.

        $\gamma = \mathrm{coeff}(A)$, in whatever linear coordinates the manifold
        puts on $T_U$. `omegas` goes through the same map, so the arbitrary
        constant in `Manifold.coefficients` cancels between the two operands and
        the solve is the metric-orthogonal projection it is defined to be.
        """
        return self.manifold.coefficients(self.point, self.A)

    @cached_property
    def omegas(self) -> Array:
        r"""The left-trivialised Jacobian's coefficients — the least-squares matrix.

        $\omega_{g,k} = \mathrm{coeff}\bigl(\texttt{to\_tangent}(U, \partial_{g,k}U)\bigr)$,
        restricted to the solvable columns. Shape ``(G, K_solvable, K)``.
        """
        # (*ambient, G, K_free) -> (G, K_free, *ambient): one Jacobian column
        # per gate and parameter, however many axes a point has.
        columns = jnp.moveaxis(self.jacobian, (-2, -1), (0, 1))
        tangents = self.manifold.to_tangent(self.point, columns)
        # Resolved gate by gate rather than as one flat batch: above 5 qubits
        # the on-the-fly Pauli projector is memory-bound in its batch size.
        per_gate = jnp.stack(
            [self.manifold.coefficients(self.point, t) for t in tangents]
        )
        return self.tangent.restrict(per_gate)

    @cached_property
    def infidelity(self) -> Array:
        """The infidelity at the base point. Free, given `point`."""
        return self.manifold.infidelity(self.point, self.target)

    @cached_property
    def fidelity(self) -> Array:
        """The fidelity at the base point. Free, given `point`."""
        return self.manifold.fidelity(self.point, self.target)

    # --- tier 1: the direction ----------------------------------------------

    @cached_property
    def V(self) -> Array:
        r"""The ambient velocity $V = \mathrm D\Phi_\phi[p]$.

        A contraction of tier 0's Jacobian, so it costs nothing beyond it — which
        is what makes the slope available even where the HVP is not.
        """
        self._require_direction("V")
        # Contract the trailing (G, K) axes, leaving the ambient ones.
        return jnp.tensordot(self.jacobian, self.coeffs, axes=[[-2, -1], [0, 1]])

    @cached_property
    def Omega(self) -> Array:
        r"""The velocity in the manifold's tangent representation, $\Omega$."""
        return self.manifold.to_tangent(self.point, self.V)

    @cached_property
    def omega_norm2(self) -> Array:
        r"""$\lVert\Omega\rVert_F^2$."""
        return self.manifold.norm2(self.point, self.Omega)

    @cached_property
    def s(self) -> Array:
        r"""The slope $\psi'(0) = \langle A, \Omega\rangle_F$ of `distance_at`.

        Exact for an arbitrary $\Omega$ — it makes no tangent-matching
        assumption — and positive on a descent direction, where the accepted step
        is negative.
        """
        return self.manifold.inner(self.point, self.A, self.Omega)

    @cached_property
    def xi_rel(self) -> Array:
        r"""The relative tangent-matching error: $\sin\angle(\Omega, A)$.

        The fraction of the geodesic direction the controls cannot reproduce —
        the least-squares residual made dimensionless — and ``0`` exactly when
        the geodesic direction is reachable. Deliberately *scale-invariant*:
        ``coeffs`` is renormalised to a fixed norm, so $\lVert\Omega\rVert_F$
        carries an arbitrary factor that a plain
        $\lVert\Omega - A\rVert_F/\lVert A\rVert_F$ would misreport as error,
        whereas ``q_exact == q`` holds for any scale multiple of $A$.
        """
        denom = self.A_norm2 * self.omega_norm2
        positive = denom > 0
        # At a converged iterate both norms vanish, so define the directions as
        # aligned there (xi_rel = 0) rather than 0/0.
        cos2 = jnp.where(positive, self.s**2 / jnp.where(positive, denom, 1.0), 1.0)
        return jnp.sqrt(jnp.clip(1.0 - cos2, 0.0, 1.0))

    # --- tier 2: the curvature ----------------------------------------------

    @cached_property
    def W(self) -> Array:
        r"""The ambient acceleration $W = \mathrm D^2\Phi_\phi[p, p]$. One HVP."""
        self._require_direction("W")
        self._require_curvature("W")
        # The HVP also returns the propagator and V; both are already known from
        # tier 0/1, and recomputing them is inherent to its O(G) recursion.
        return self.tangent.hvp(jnp.real(self.free_params), self.coeffs)[2]

    @cached_property
    def accel(self) -> Array:
        r"""The extrinsic term $\langle A, V^\dagger V + U^\dagger W\rangle_F$.

        The part of the curvature that comes from the *chart* bending, as opposed
        to the manifold's own curvature.
        """
        bend = self.manifold.tangent_acceleration(self.point, self.V, self.W)
        return self.manifold.inner(self.point, self.A, bend)

    @cached_property
    def chi(self) -> Array:
        r"""The dimensionless radial bending coefficient
        $\chi_\phi = \mathrm{accel}/\lVert\Omega\rVert_F^2$."""
        return self.accel / self.omega_norm2

    @cached_property
    def q(self) -> Array:
        r"""The curvature $\psi''(0)$ with the **radial surrogate**
        $\lVert\Omega\rVert_F^2$ for the intrinsic term.

        Exact only when $\Omega\parallel A$, i.e. when the geodesic direction is
        exactly reachable (`xi_rel` ``== 0``). See `q_exact`.
        """
        return self.omega_norm2 + self.accel

    @cached_property
    def _hessian_form(self) -> tuple[Array, Array]:
        """``(intrinsic curvature, cut-locus spread)`` from one hook call.

        One ``eigh`` on a group or the state sphere; one small operator
        exponential on `geope.geometry.stiefel.stiefel.Stiefel`, whose Hessian
        is not a scalar function of a single adjoint.
        """
        return self.manifold.hessian_quadratic_form(self.point, self.A, self.Omega)

    @cached_property
    def q_exact(self) -> Array:
        r"""The curvature $\psi''(0)$ with the **exact** intrinsic term
        $\langle\Omega,\mathcal K_A\Omega\rangle_F$.

        Since $\mathcal K_A\preceq I$, ``q_exact <= q`` always, with equality iff
        `xi_rel` vanishes.
        """
        return self._hessian_form[0] + self.accel

    @cached_property
    def rho(self) -> Array:
        r"""The eigenphase spread $\rho = \Delta(A)$.

        Below $\pi$ the intrinsic Hessian is positive definite; $2\pi$ is the cut
        locus, where the minimal geodesic stops being unique.
        """
        return self._hessian_form[1]

    # --- tier 3: the ray ----------------------------------------------------

    def point_at(self, t: Array) -> Array:
        r"""The point at $\phi + t\,p$ — the single gate for all of tier 3."""
        self._require_direction("point_at")
        return self.manifold.compute_point(self.free_params + t * self.coeffs)

    def infidelity_at(self, t: Array) -> Array:
        """The infidelity along the ray. One propagator."""
        return self.manifold.infidelity(self.point_at(t), self.target)

    def fidelity_at(self, t: Array) -> Array:
        """The fidelity along the ray. One propagator."""
        return self.manifold.fidelity(self.point_at(t), self.target)

    def distance_at(self, t: Array) -> Array:
        r"""The squared-geodesic-distance objective $\tfrac12 d_g(\cdot, V)^2$
        along the ray, whose derivatives `s`, `q` and `q_exact` describe.

        One propagator plus one logarithm.
        """
        return self.manifold.distance2(self.point_at(t), self.target)
