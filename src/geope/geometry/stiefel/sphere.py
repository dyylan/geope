r"""The state sphere: $\mathrm{St}(n, 1)$, i.e. state preparation.

The first non-group manifold in the library, and the $p = 1$ case of the Stiefel
family. Its value is twofold: it is a genuinely useful problem (drive
$\lvert\psi_0\rangle$ to a target state with the same pulse machinery GEOPE uses
for gates), and it is the proof that
`geope.geometry.manifold.Manifold` is a real abstraction — the optimiser, the
line searches and `Gecko`'s null space run on it unchanged, with no group
structure anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar

import jax.numpy as jnp
import numpy as np
from jax import Array

from ...jax.hessian import get_hvp_propagator
from ..chart import get_compute_matrices_params_list_fn
from ..manifold import Manifold

# Below this the geodesic quantities are taken at their (finite) limits: the
# minimal geodesic degenerates as two states coincide.
_TINY = 1e-12

# Tolerance on a point's norm, matching the group's unitarity tolerance.
_UNIT_ATOL = 1e-8


@dataclass(frozen=True, eq=False)
class StateSphere(Manifold):
    r"""$\mathbb{CP}^{n-1}$: unit states in $\mathbb C^n$, up to a global phase.

    A point is a normalised state vector; the chart is the pulse's unitary acting
    on a fixed initial state, $\Phi(\phi) = U(\phi)\lvert\psi_0\rangle$, so the
    whole product-of-exponentials machinery carries over unchanged and the
    Jacobian and HVP are the group's composed with one matrix-vector product.

    **Not a Lie group.** $\mathrm{St}(n,1) = \mathrm U(n)/\mathrm U(n-1)$ is a
    homogeneous space: no identity, no left translation, and therefore no
    trivialisation identifying the tangent spaces at different points. Everything
    here is genuinely point-dependent — `to_tangent` is a projection rather than a
    translation, and `log` is a great-circle formula rather than a matrix
    logarithm.

    **Phase invariance.** The geometry is the Fubini–Study one: `log` first rotates
    the target's phase so that $\langle x, y\rangle \ge 0$, which is the
    horizontal lift of the Hopf fibration, and the fidelity is
    $\lvert\langle x, y\rangle\rvert$ — matching
    `geope.geometry.lie.groups.SpecialUnitaryGroup`'s convention that a global
    phase is not physical. A phase-sensitive $S^{2n-1}$ variant is a subclass
    overriding `log` and `fidelity` to drop that alignment.

    The cut locus is the antipode, $\lvert\langle x,y\rangle\rvert = 0$: there the
    minimal geodesic is not unique and `log` returns the degenerate ``0``, the
    same measure-zero caveat the matrix logarithm has at $\theta = 2\pi$.

    Attributes:
        dim: The Hilbert-space dimension $n$; a point is an ``(n,)`` state.
        base_point: The initial state $\lvert\psi_0\rangle$ the chart acts on;
            must be a unit vector of shape ``(n,)``.
    """

    dim: int
    base_point: Array

    projective: ClassVar[bool] = True
    name: ClassVar[str] = "CP(n-1)"

    def __post_init__(self) -> None:
        base = jnp.asarray(self.base_point)
        if base.shape != (self.dim,):
            raise ValueError(
                f"{self.name} with dim={self.dim} expects a ({self.dim},) "
                f"base_point, got {tuple(base.shape)}."
            )
        self.validate_point(base, "base_point")

    def validate_point(self, point: np.ndarray, what: str = "point") -> None:
        r"""Check membership of the sphere: $\lVert x\rVert = 1$.

        Every geodesic formula here assumes it — `log` reads $\arccos\lvert
        \langle x,y\rangle\rvert$ as an angle, which is only a distance on the
        unit sphere.

        Host-side and numpy; see `geope.geometry.manifold.Manifold.validate_point`.
        """
        norm = float(np.linalg.norm(np.asarray(point)))
        if abs(norm - 1.0) > _UNIT_ATOL:
            raise ValueError(
                f"{self.name} needs a unit-norm {what} (a unit vector); "
                f"its norm is {norm:.6g}."
            )

    # --- the interface ------------------------------------------------------

    @property
    def ambient_shape(self) -> tuple[int, ...]:
        return (self.dim,)

    @property
    def manifold_dim(self) -> int:
        r"""$\dim_{\mathbb R}\mathbb{CP}^{n-1} = 2n - 2$."""
        return 2 * self.dim - 2

    @staticmethod
    def _braket(x: Array, y: Array) -> Array:
        r"""$\langle x, y\rangle = \sum_i \bar x_i y_i$, batched over ``y``'s leading axes."""
        return jnp.einsum("i,...i->...", jnp.conj(x), y)

    def to_tangent(self, point: Array, ambient: Array) -> Array:
        r"""Project onto $T_x = \{z : \langle x, z\rangle = 0\}$: $z - x\langle x,z\rangle$.

        Removes the whole *complex* component along $x$, i.e. both the radial
        direction and the phase direction — which is exactly the horizontal space
        of the Hopf fibration, and so the right tangent space for the projective
        geometry.
        """
        overlap = self._braket(point, ambient)
        return ambient - point * overlap[..., None]

    def inner(self, point: Array, x: Array, y: Array) -> Array:
        r"""$\mathrm{Re}\langle x, y\rangle$ — the round metric, inherited from $\mathbb C^n$."""
        return self._euclidean_inner(x, y)

    def coefficients(self, point: Array, tangent: Array) -> Array:
        r"""Split into real and imaginary parts: $u \mapsto (\mathrm{Re}\,u, \mathrm{Im}\,u)$.

        There is no global frame to resolve against on a non-parallelisable
        manifold, and none is needed: this is a faithful real-linear map with
        $\sum_k c_k(u)c_k(v) = \mathrm{Re}\langle u,v\rangle$ exactly, so the
        `Manifold.coefficients` constant is $c = 1$. Its image is a hyperplane
        (tangency is one real constraint), which the contract explicitly allows —
        the least-squares problem needs faithfulness, not surjectivity.
        """
        return jnp.concatenate([jnp.real(tangent), jnp.imag(tangent)], axis=-1)

    def log(self, x: Array, y: Array, key: Array | None = None) -> Array:
        r"""$\mathrm{Log}_x(y) = \theta\,\hat u$ along the great circle through $x$ and $y$.

        With the target phase-aligned so $c = \langle x,y\rangle \ge 0$,
        $\theta = \arccos c$ is the Fubini–Study distance and
        $\hat u = (y - xc)/\lVert y - xc\rVert$ the unit tangent toward it. Since
        $\lVert y - xc\rVert = \sin\theta$, the returned $\theta/\sin\theta$ scale
        tends to $1$ as the states coincide, which is the limit taken below.

        Args:
            x: The base state, unit ``(n,)``.
            y: The end state, unit ``(n,)``.
            key: Unused; accepted for interface parity.
        """
        overlap = self._braket(x, y)
        magnitude = jnp.abs(overlap)
        # Rotate y's phase onto the horizontal lift, so <x, y'> = |<x, y>| >= 0.
        # Both branches of a `where` are evaluated, so keep the division safe.
        safe = jnp.where(magnitude > _TINY, magnitude, 1.0)
        phase = jnp.where(magnitude > _TINY, overlap / safe, 1.0)
        aligned = y * jnp.conj(phase)

        theta = jnp.arccos(jnp.clip(magnitude, -1.0, 1.0))
        u = self.to_tangent(x, aligned)
        norm_u = jnp.sqrt(self.norm2(x, u))
        far = norm_u > _TINY
        scale = jnp.where(far, theta / jnp.where(far, norm_u, 1.0), 1.0)
        return scale * u

    def tangent_acceleration(self, point: Array, v: Array, w: Array) -> Array:
        r"""$\dot\Omega(0) = W - V\langle x,V\rangle - x(\langle V,V\rangle + \langle x,W\rangle)$.

        The product-rule derivative of $\Omega(t) = \dot c - c\langle c,\dot c\rangle$
        along the chart's curve — the projection's own bending, which on a group
        would instead be $V^\dagger V + U^\dagger W$.
        """
        return (
            w
            - v * self._braket(point, v)
            - point * (self._braket(v, v) + self._braket(point, w))
        )

    def hessian_quadratic_form(
        self, point: Array, a: Array, omega: Array
    ) -> tuple[Array, Array]:
        r"""$\lVert\Omega_\parallel\rVert^2 + \theta\cot\theta\,\lVert\Omega_\perp\rVert^2$.

        On a sphere the Riemannian Hessian of $\tfrac12 d^2$ is the identity along
        the radial direction $A$ and $\theta\cot\theta$ on its orthogonal
        complement — the same shape as the group's
        $\frac{\operatorname{ad}}{2}\coth\frac{\operatorname{ad}}{2}$ spectrum, and
        likewise $\preceq I$ for $\theta < \pi$. Returns $\theta$ as the
        cut-locus diagnostic: it diverges at $\theta = \pi$, the antipode.
        """
        theta = jnp.sqrt(self.norm2(point, a))
        far = theta > _TINY
        safe_theta = jnp.where(far, theta, 1.0)
        # Component of Omega along A, in units of length.
        parallel = jnp.where(far, self.inner(point, a, omega) / safe_theta, 0.0)
        # theta * cot(theta), continuously extended to 1 at theta = 0. Feed tan a
        # safe argument: both branches of a `where` are evaluated.
        h = jnp.where(far, safe_theta / jnp.tan(safe_theta), 1.0)
        perpendicular = self.norm2(point, omega) - parallel**2
        return parallel**2 + h * perpendicular, theta

    def fidelity(self, x: Array, y: Array) -> Array:
        r"""$\lvert\langle x, y\rangle\rvert \in [0, 1]$ — the state fidelity."""
        return jnp.abs(self._braket(x, y))

    def infidelity(self, x: Array, y: Array) -> Array:
        r"""$1 - \lvert\langle x, y\rangle\rvert \in [0, 1]$."""
        return 1.0 - jnp.abs(self._braket(x, y))

    # --- the chart: the group action on the base state ----------------------

    def chart(self, generators) -> Callable[[Array], Array]:
        r"""$\Phi(\phi) = U(\phi)\lvert\psi_0\rangle$.

        The pulse's unitary — the same product of piecewise-constant exponentials
        `geope.geometry.lie.groups.MatrixLieGroup.chart` builds — applied to the
        base state. This is what makes a homogeneous space reuse the group's chart
        machinery wholesale.
        """
        compute_U = get_compute_matrices_params_list_fn(generators.basis)
        base = jnp.asarray(self.base_point, dtype=jnp.complex128)

        def chart(free_params: Array) -> Array:
            return compute_U(free_params) @ base

        return chart

    def chart_hvp(
        self, generators
    ) -> Callable[[Array, Array], tuple[Array, Array, Array]]:
        """The group's propagator HVP, carried through the action by linearity."""
        group_hvp = get_hvp_propagator(jnp.asarray(generators.basis))
        base = jnp.asarray(self.base_point, dtype=jnp.complex128)

        def hvp(free_params: Array, direction: Array):
            x, v, w = group_hvp(free_params, direction)
            return x @ base, v @ base, w @ base

        return hvp
