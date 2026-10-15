r"""Rigid motions: $\mathrm{SE}(3)$ under the product metric of $\mathrm{SO}(3)\times\mathbb R^3$.

The first manifold in the library that is not a $\mathrm U(N)$-orbit. A point is a
homogeneous transform $X = \begin{psmallmatrix}R & p\\ 0 & 1\end{psmallmatrix}$,
and the chart is still the pulse's product of exponentials acting on a base point,
$\Phi(\phi) = U(\phi)\,X_0$ — now with *real* $\mathfrak{se}(3)$ generators, so
that a serial arm's product-of-exponentials forward kinematics
$e^{[S_1]\theta_1}\cdots e^{[S_n]\theta_n}M$ is literally a GEOPE chart with base
point $M$ (see `geope.robotics`).

**The metric.** $\mathrm{SE}(3)$ has no positive-definite bi-invariant metric, and a
length scale $\ell$ is needed to compare a rotation with a translation. This module
fixes the left-invariant product metric

$$\bigl\lVert(R\hat\omega,\ u)\bigr\rVert^2 = \ell^2\lVert\omega\rVert^2 + \lVert u\rVert^2,$$

under which $\mathrm{SE}(3)\cong\mathrm{SO}(3)\times\mathbb R^3$ isometrically: a
geodesic rotates along an $\mathrm{SO}(3)$ geodesic and translates along a straight
line, and the Riemannian logarithm is
$\mathrm{Log}_X(Y) = \bigl(\log(R_X^\intercal R_Y),\ p_Y - p_X\bigr)$ — **not** the
screw-motion logarithm $\log(X^{-1}Y)$ of the group, which is not the gradient of
$\tfrac12 d^2$ for this metric.

**The chart's derivatives.** The analytic jet in `geope.geometry.chart` assumes
skew-Hermitian generators (an ``eigh`` and $U^\dagger = U^{-1}$), which real
$\mathfrak{se}(3)$ elements are not, so this manifold is driven through the
``param_transform`` (autodiff) chart, whose Jacobian is exact for any generator. The
pulse map itself is a plain ``expm`` and is exact either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
from jax import Array

from ..manifold import Manifold

#: Below this squared rotation-vector norm a rotation is treated as the identity.
_TINY2 = 1e-24
#: Tolerance of the host-side membership check.
_MEMBER_ATOL = 1e-8


def hat(w: Array) -> Array:
    r"""$\hat\omega$: the skew matrix with $\hat\omega x = \omega\times x$, batched.

    Args:
        w: Rotation vectors of shape ``(..., 3)``.

    Returns:
        Skew-symmetric matrices of shape ``(..., 3, 3)``.
    """
    w = jnp.asarray(w)
    z = jnp.zeros_like(w[..., 0])
    return jnp.stack(
        [
            jnp.stack([z, -w[..., 2], w[..., 1]], axis=-1),
            jnp.stack([w[..., 2], z, -w[..., 0]], axis=-1),
            jnp.stack([-w[..., 1], w[..., 0], z], axis=-1),
        ],
        axis=-2,
    )


def vee(m: Array) -> Array:
    r"""The inverse of `hat`, applied to the skew part of ``m``, batched.

    Antisymmetrising first makes it a left inverse of `hat` that is also the
    orthogonal projection of $\mathfrak{gl}(3)$ onto $\mathfrak{so}(3)$, which is
    what a linear coefficient map needs.

    Args:
        m: Matrices of shape ``(..., 3, 3)``.

    Returns:
        Vectors of shape ``(..., 3)``.
    """
    m = jnp.asarray(m)
    return 0.5 * jnp.stack(
        [
            m[..., 2, 1] - m[..., 1, 2],
            m[..., 0, 2] - m[..., 2, 0],
            m[..., 1, 0] - m[..., 0, 1],
        ],
        axis=-1,
    )


def so3_log_vee(r: Array) -> Array:
    r"""$\operatorname{vee}\log R$: the minimising rotation vector of $R \in \mathrm{SO}(3)$.

    Closed form: with $s = \operatorname{vee}\tfrac12(R - R^\intercal) =
    \sin\theta\,\hat n$ and $c = \tfrac12(\mathrm{Tr}R - 1) = \cos\theta$,
    $\theta = \operatorname{atan2}(\lVert s\rVert, c)$ and $\log R = \theta\,\hat n$.
    The ``atan2`` keeps $\theta$ accurate at both ends of $[0, \pi]$, where an
    ``arccos`` loses half its digits. The factor $\theta/\sin\theta$ tends to $1$ at
    the identity, which is taken through a double ``where`` so that autodiff through
    the identity is finite too.

    The cut locus is $\theta = \pi$, where the minimising logarithm is not unique
    and $s$ vanishes; there the direction is numerically undetermined, the same
    measure-zero caveat the matrix logarithm has on its branch cut.

    Args:
        r: Rotation matrices of shape ``(..., 3, 3)``.

    Returns:
        Rotation vectors $\theta\hat n$ of shape ``(..., 3)``.
    """
    s = vee(r)
    c = 0.5 * (jnp.trace(r, axis1=-2, axis2=-1) - 1.0)
    s2 = jnp.sum(s * s, axis=-1)
    small = s2 < _TINY2
    sin = jnp.sqrt(jnp.where(small, 1.0, s2))
    theta = jnp.arctan2(sin, c)
    scale = jnp.where(small, 1.0, theta / sin)
    return scale[..., None] * s


def _theta_cot(theta: Array) -> Array:
    r"""$h(\theta) = \tfrac\theta2\cot\tfrac\theta2$, with $h(0) = 1$."""
    small = theta < 1e-8
    half = 0.5 * jnp.where(small, 1.0, theta)
    return jnp.where(small, 1.0, half * jnp.cos(half) / jnp.sin(half))


@dataclass(frozen=True, eq=False)
class SpecialEuclideanGroup(Manifold):
    r"""$\mathrm{SE}(3)$ with the product metric $\ell^2\lVert\omega\rVert^2 + \lVert u\rVert^2$.

    **Tangent representation.** A tangent vector at $X = (R, p)$ is stored as the
    $4\times4$ array $\begin{psmallmatrix}\hat\omega & u\\ 0 & 0\end{psmallmatrix}$
    with $\hat\omega = R^\intercal\dot R$ the *body* angular velocity and
    $u = \dot p$ the *spatial* linear velocity. This mixed form is chosen because in
    it the plain time derivative is the Levi-Civita one: the rotation factor is a
    left-trivialised bi-invariant group, where $\nabla_\Omega\Omega = \dot\Omega$,
    and the translation factor is flat. Left-trivialising the whole of
    $\mathrm{SE}(3)$ instead ($X^{-1}\dot X$, which rotates $u$ into the body frame)
    would make `tangent_acceleration` miss a connection term $\omega\times v$.

    **Coefficients** are $(\ell\,\omega,\ u) \in \mathbb R^6$, frame-free like the
    Stiefel family's, so $\sum_k c_k(a)c_k(b) = \langle a, b\rangle$ exactly
    ($c = 1$): the geodesic least squares is then posed as
    $\min\lVert W^{1/2}(Jv - e)\rVert$ with $W = \mathrm{diag}(\ell^2 I_3, I_3)$.

    **Objective.** `infidelity` is $\tfrac12 d_g^2$ itself, so `velocity` and
    `slope` describe one function, and `fidelity` is $1 - \tfrac12 d_g^2$: $1$ at
    the target as the interface asks, but unbounded below, and it rounds to exactly
    ``1.0`` once $\tfrac12 d_g^2 \lesssim 10^{-16}$ — read `infidelity` to resolve
    convergence below that.

    The chart's points arrive as ``complex128`` (the pipeline's dtype) with an
    identically zero imaginary part; every hook reads the real part.

    Attributes:
        length_scale: $\ell > 0$, the length that one radian of rotation is worth.
            Choosing $\ell = \varepsilon_p/\varepsilon_R$ makes a position error
            $\varepsilon_p$ and an angular error $\varepsilon_R$ equally costly.
        base_point: $\Phi(0)$, the pose the chart starts from — a serial arm's home
            configuration $M$. ``None`` means the identity, so the propagator is the
            point.
    """

    length_scale: float = 1.0

    projective: ClassVar[bool] = False
    name: ClassVar[str] = "SE(3)"

    def __post_init__(self) -> None:
        if not self.length_scale > 0:
            raise ValueError(
                f"{self.name} needs a positive `length_scale`, got {self.length_scale}."
            )
        if self.base_point is not None:
            base = np.asarray(self.base_point)
            if base.shape != (4, 4):
                raise ValueError(
                    f"{self.name} expects a (4, 4) base_point, got {base.shape}."
                )
            self.validate_point(base, "base_point")

    def validate_point(self, point: np.ndarray, what: str = "point") -> None:
        r"""Check membership: $R^\intercal R = I$, $\det R = 1$, last row $(0,0,0,1)$, real.

        Host-side and numpy; see `geope.geometry.manifold.Manifold.validate_point`.
        """
        point = np.asarray(point)
        if point.shape != (4, 4):
            raise ValueError(f"{self.name} needs a (4, 4) {what}, got {point.shape}.")
        imag = float(np.max(np.abs(np.imag(point))))
        real = np.real(point)
        rot = real[:3, :3]
        orth = float(np.max(np.abs(rot.T @ rot - np.eye(3))))
        det = float(np.linalg.det(rot))
        row = float(np.max(np.abs(real[3] - np.array([0.0, 0.0, 0.0, 1.0]))))
        if max(imag, orth, abs(det - 1.0), row) > _MEMBER_ATOL:
            raise ValueError(
                f"{self.name} needs a rigid transform as {what}: "
                f"max|Im| = {imag:.3g}, max|RᵀR - I| = {orth:.3g}, det R = {det:.6g}, "
                f"max|last row - (0,0,0,1)| = {row:.3g}."
            )

    @property
    def ambient_shape(self) -> tuple[int, ...]:
        return (4, 4)

    @property
    def manifold_dim(self) -> int:
        return 6

    @staticmethod
    def _assemble(rot: Array, trans: Array) -> Array:
        """The ``(..., 4, 4)`` tangent array with blocks ``rot`` and ``trans``."""
        out = jnp.zeros((*rot.shape[:-2], 4, 4), dtype=rot.dtype)
        return out.at[..., :3, :3].set(rot).at[..., :3, 3].set(trans)

    def to_tangent(self, point: Array, ambient: Array) -> Array:
        r"""$\dot X \mapsto \begin{psmallmatrix}R^\intercal\dot R & \dot p\\ 0 & 0\end{psmallmatrix}$.

        Body angular velocity, spatial linear velocity; batches over the leading axes
        of ``ambient``.
        """
        rot = jnp.real(point[:3, :3])
        z = jnp.real(ambient)
        return self._assemble(
            jnp.einsum("ji,...jk->...ik", rot, z[..., :3, :3]), z[..., :3, 3]
        )

    def coefficients(self, point: Array, tangent: Array) -> Array:
        r"""$(\ell\,\omega,\ u)$ — exactly metric-consistent, with constant $c = 1$."""
        t = jnp.real(tangent)
        return jnp.concatenate(
            [self.length_scale * vee(t[..., :3, :3]), t[..., :3, 3]], axis=-1
        )

    def inner(self, point: Array, x: Array, y: Array) -> Array:
        r"""$\ell^2\,\omega_x\cdot\omega_y + u_x\cdot u_y$ — left-invariant, not bi-invariant."""
        return jnp.sum(
            self.coefficients(point, x) * self.coefficients(point, y), axis=-1
        )

    def log(self, x: Array, y: Array, key: Array | None = None) -> Array:
        r"""$\mathrm{Log}_X(Y) = \bigl(\log(R_X^\intercal R_Y),\ p_Y - p_X\bigr)$.

        The product metric's logarithm, in the mixed representation of
        `to_tangent`. Its norm is the geodesic distance
        $d^2 = \ell^2\theta^2 + \lVert p_Y - p_X\rVert^2$. See `so3_log_vee` for the
        cut locus at a relative rotation of $\pi$.

        Args:
            x: The base pose.
            y: The end pose.
            key: Unused; accepted for interface parity.
        """
        x, y = jnp.real(x), jnp.real(y)
        rel = jnp.einsum("...ji,...jk->...ik", x[..., :3, :3], y[..., :3, :3])
        return self._assemble(hat(so3_log_vee(rel)), y[..., :3, 3] - x[..., :3, 3])

    def tangent_acceleration(self, point: Array, v: Array, w: Array) -> Array:
        r"""$\dot\Omega(0) = \bigl(\dot R^\intercal\dot R + R^\intercal\ddot R,\ \ddot p\bigr)$.

        The product-rule derivative of `to_tangent` along the chart's curve. The
        rotation block is the unitary group's $V^\dagger V + U^\dagger W$ (a rotation
        *is* a real unitary), and the translation block is flat.
        """
        rot = jnp.real(point[:3, :3])
        v, w = jnp.real(v), jnp.real(w)
        block = jnp.einsum("...ji,...jk->...ik", v[..., :3, :3], v[..., :3, :3])
        block = block + jnp.einsum("ji,...jk->...ik", rot, w[..., :3, :3])
        return self._assemble(block, w[..., :3, 3])

    def hessian_quadratic_form(
        self, point: Array, a: Array, omega: Array
    ) -> tuple[Array, Array]:
        r"""$\ell^2\bigl[(\omega\cdot\hat n)^2 + h(\theta)\,\lVert\omega_\perp\rVert^2\bigr] + \lVert u\rVert^2$.

        The Riemannian Hessian of $\tfrac12 d^2$ splits over the product. The
        translation factor is flat, so it contributes $\lVert u\rVert^2$. On
        $\mathrm{SO}(3)$ the bi-invariant Hessian is
        $\tfrac{\mathrm{ad}_A}2\cot\tfrac{\mathrm{ad}_A}2$ (the unitary group's
        kernel), and for $A = \theta\hat n$ the operator $\mathrm{ad}_A$ is
        $\theta\,\hat n\times$: eigenvalue $0$ along the axis, $\pm i\theta$ across
        it. So the radial component keeps weight $1$ and the transverse one gets
        $h(\theta) = \tfrac\theta2\cot\tfrac\theta2$, which falls from $1$ to $0$
        on $[0, \pi)$ — hence $\mathcal K_A \preceq I$ there. The form is even in
        ``a``, so the context's sign convention is immaterial.

        Returns ``(value, theta)``: the relative rotation angle is the cut-locus
        diagnostic, the minimising logarithm being unique only for $\theta < \pi$.
        """
        rot_a = vee(jnp.real(a)[..., :3, :3])
        theta = jnp.sqrt(jnp.sum(rot_a * rot_a, axis=-1))
        safe = jnp.where(theta > 0, theta, 1.0)
        axis = rot_a / safe
        om = jnp.real(omega)
        w = vee(om[..., :3, :3])
        u = om[..., :3, 3]
        radial = jnp.sum(w * axis, axis=-1)
        transverse = jnp.sum(w * w, axis=-1) - radial**2
        value = self.length_scale**2 * (
            radial**2 + _theta_cot(theta) * transverse
        ) + jnp.sum(u * u, axis=-1)
        return value, theta

    def infidelity(self, x: Array, y: Array) -> Array:
        r"""$\tfrac12 d_g(x, y)^2$ — the squared-distance objective itself."""
        return self.distance2(x, y)

    def fidelity(self, x: Array, y: Array) -> Array:
        r"""$1 - \tfrac12 d_g(x, y)^2$: ``1`` at the target, unbounded below."""
        return 1.0 - self.infidelity(x, y)
