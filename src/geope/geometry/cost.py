r"""The derivatives of a trace cost, shared by every manifold in the library.

`geope.geometry.manifold.Manifold` deliberately makes no assumption about what a
space is optimised against — `Manifold.infidelity` is a hook, and declared
separately from `Manifold.fidelity` for exactly that reason. But *all four*
manifolds GEOPE ships happen to score the same way: through the single complex
ambient overlap

$$z = \langle y, x\rangle_{\mathcal A} = \sum \bar y\,x
\qquad\text{(summed over the ambient axes)},$$

with a cost that is either **phase-sensitive**, $C = 1 - \mathrm{Re}(z)/\kappa$,
or **projective**, $C = 1 - \lvert z\rvert/\kappa$:

| manifold | $\kappa$ | projective |
| --- | --- | --- |
| `geope.geometry.lie.groups.SpecialUnitaryGroup` | $d$ | yes |
| `geope.geometry.lie.groups.UnitaryGroup` | $d$ | no |
| `geope.geometry.stiefel.sphere.StateSphere` | $1$ | yes |
| `geope.geometry.stiefel.stiefel.Stiefel` | $m$ | either |

So one pair of formulas serves all of them, and each manifold's
`Manifold.cost_gradient` / `Manifold.cost_hessian_form` is a two-line delegation
to what is here with its own $(\kappa, \texttt{projective})$. That is a *shared
implementation*, not a promise: a manifold whose cost is not of this shape simply
does not call these, and declines the hooks instead.

Both functions are rank-generic in the ambient shape — a point that is a matrix,
a frame or a state all take the same path — and this module imports nothing but
JAX, so it sits alongside `geope.geometry.chart` below the manifolds.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def ambient_overlap(x: Array, y: Array, ambient_ndim: int) -> Array:
    r"""$z = \sum \bar y\,x$, contracted over the trailing ``ambient_ndim`` axes.

    The Hermitian inner product every trace cost is a function of, batched over
    any leading axes so a stack of Jacobian columns resolves in one call.

    Args:
        x: The point, or a batch of ambient vectors, of shape
            ``(..., *ambient_shape)``.
        y: The target, of shape ``ambient_shape``.
        ambient_ndim: How many trailing axes one ambient point has.

    Returns:
        A complex ``Array`` of the leading batch shape.
    """
    axes = tuple(range(-ambient_ndim, 0))
    return jnp.sum(jnp.conj(y) * x, axis=axes)


def trace_cost_gradient(
    x: Array, y: Array, ambient_ndim: int, scale: Array | float, projective: bool
) -> Array:
    r"""$\partial C/\partial\bar x$: the ambient Wirtinger gradient of the infidelity.

    For a real-valued $C$ of a complex point, the chain rule out of the ambient
    space reads $\partial_a C = 2\,\mathrm{Re}\langle \hat G, \partial_a x\rangle$
    with $\hat G = \partial C/\partial\bar x$, which is what makes this — and not
    the full Jacobian — the object a gradient needs. With
    $z = \langle y, x\rangle$,

    $$\hat G = -\frac{y}{2\kappa}\quad\text{(phase-sensitive)},\qquad
      \hat G = -\frac{z\,y}{2\kappa\lvert z\rvert}\quad\text{(projective)}.$$

    The projective form is singular at $z = 0$ — the near-identity /
    traceless-target gotcha the autodiff gradient has too, documented in
    ``docs/user_guide.md``. It is guarded here only against a hard ``nan``: at
    exactly $z = 0$ the phase factor is taken to be $1$, which is as arbitrary as
    the geometry is there.

    Args:
        x: The current point, of shape ``ambient_shape``.
        y: The target, of shape ``ambient_shape``.
        ambient_ndim: How many axes one ambient point has.
        scale: The cost's normalisation $\kappa$.
        projective: Whether the cost quotients out a global phase.

    Returns:
        An ``Array`` of shape ``ambient_shape``.
    """
    if not projective:
        return -y / (2.0 * scale)
    z = ambient_overlap(x, y, ambient_ndim)
    r = jnp.abs(z)
    # z / |z| is the phase of the overlap; at z = 0 the cost is not differentiable
    # at all, so pick 1 rather than let a nan propagate out of the whole gradient.
    positive = r > 0
    phase = jnp.where(positive, z / jnp.where(positive, r, 1.0), 1.0)
    return -(phase * y) / (2.0 * scale)


def trace_cost_hessian_form(
    x: Array,
    y: Array,
    u: Array,
    ambient_ndim: int,
    scale: Array | float,
    projective: bool,
) -> Array:
    r"""The intrinsic second-derivative form $\mathrm{Hess}_C[u_a, u_b]$ of the cost.

    The part of $\partial_a\partial_b\,C(\Phi(\phi))$ that comes from the *cost's*
    curvature in the ambient space, as opposed to the chart's bending — the caller
    adds $2\,\mathrm{Re}\langle\hat G, \mathrm D^2\Phi_{ab}\rangle$ for that.

    A phase-sensitive cost is affine in the point, so its form **vanishes
    identically**. The projective one factors entirely through the scalars
    $\zeta_a = \langle y, u_a\rangle$: writing $r = \lvert z\rvert$,

    $$\mathrm{Hess}_C[u_a, u_b] = -\frac1\kappa\left[
        \frac{\mathrm{Re}(\bar\zeta_a\zeta_b) + \mathrm{Re}(\bar z\,\eta_{ab})}{r}
        - \frac{\mathrm{Re}(\bar z\zeta_a)\,\mathrm{Re}(\bar z\zeta_b)}{r^3}\right]$$

    where the $\eta_{ab}$ term is the chart's, supplied by the caller. Only the
    $\zeta$-dependent terms are returned here, so the whole $P\times P$ form comes
    from $P$ scalars: rank-structured, and never a $P^2$-sized ambient object.

    Args:
        x: The current point, of shape ``ambient_shape``.
        y: The target, of shape ``ambient_shape``.
        u: The chart's Jacobian columns, of shape ``(P, *ambient_shape)``.
        ambient_ndim: How many axes one ambient point has.
        scale: The cost's normalisation $\kappa$.
        projective: Whether the cost quotients out a global phase.

    Returns:
        A real ``Array`` of shape ``(P, P)``.
    """
    p = u.shape[0]
    if not projective:
        return jnp.zeros((p, p), dtype=jnp.result_type(float))

    z = ambient_overlap(x, y, ambient_ndim)
    zeta = ambient_overlap(u, y, ambient_ndim)  # (P,)
    r = jnp.abs(z)
    z_bar = jnp.conj(z)
    re_zz = jnp.real(z_bar * zeta)  # (P,)
    first = jnp.real(jnp.outer(jnp.conj(zeta), zeta)) / r
    second = jnp.outer(re_zz, re_zz) / r**3
    return -(first - second) / scale
