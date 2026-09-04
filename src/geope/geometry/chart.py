r"""The ambient layer: the pulse model, where it lands, and how it differentiates.

Every manifold in the library is a submanifold of one **ambient space**
$\mathcal A = \mathbb C^{N\times m}$, and the pulse acts on all of them the same
way — by left multiplication with a product of piecewise-constant exponentials,

$$U(\phi) = \prod_g \exp\Bigl(i\sum_k \phi_{g,k} G_k\Bigr),
\qquad \Phi(\phi) = U(\phi)\,x_0 .$$

The chart is therefore the **orbit map** of that one ambient action through a
base point $x_0 = \Phi(0)$: the identity on a matrix group (where the propagator
*is* the point), a state on `geope.geometry.stiefel.sphere.StateSphere`, a frame
on `geope.geometry.stiefel.stiefel.Stiefel`. Only $x_0$ varies, which is why no
manifold writes chart code of its own — `geope.geometry.manifold.Manifold.bind`
composes what is here, once.

This module owns the whole **jet** $(\Phi,\ \mathrm D\Phi,\ \mathrm D^2\Phi)$,
i.e. everything valued in $\mathcal A$. What happens in the tangent space
$T_x\mathcal M$ — the metric, the coefficient frame, the logarithm, the fidelity
— is the `Manifold`'s, so the two never mix:

```
ambient  A = C^{N×m}    Φ = U(φ)·x₀ ,  DΦ ,  D²Φ          <- here
     │
     │  to_tangent :  A → T_x M
     ▼
submanifold  M ⊂ A      inner, coefficients, log, ...     <- Manifold hooks
```

Like `geope.geometry.tangent`, this module imports nothing but JAX and
`geope.jax`: it sits *below* the manifolds rather than beside them.

Every factory here returns an **un-jitted** callable, so it fuses into the
enclosing ``@jax.jit`` update step that `geope.Geope.optimize` traces once.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax import Array

from functools import partial
from typing import Callable

from ..jax.hessian import get_hessian_propagator, get_hvp_propagator
from ..jax.jacobian import get_jacobian_propagator, get_vjp_propagator


def compute_matrices_params_list_fn(params_list: Array, basis: Array) -> Array:
    """Compute the product unitary from a list of parameter vectors.

    For each parameter vector in `params_list`, constructs a Hamiltonian
    as a linear combination of the `basis` elements, exponentiates it,
    and accumulates the product unitary via `jax.lax.scan`.

    Args:
        params_list: ``Array`` of shape ``(piecewise_steps, K)`` where each row
            contains the Lie-algebra coefficients for one gate segment.
        basis: ``Array`` of shape ``(K, d, d)`` of Hermitian basis matrices.

    Returns:
        The product unitary ``Array`` of shape ``(d, d)``.
    """

    def step(U, params):
        A = jnp.tensordot(params, basis, axes=[[-1], [0]])
        Ui = jax.scipy.linalg.expm(1j * A)
        U_new = jnp.matmul(Ui, U)
        return U_new, None

    U0 = jnp.eye(basis.shape[1], dtype=basis.dtype)
    U_final, _ = jax.lax.scan(step, U0, jnp.stack(params_list))
    return U_final


def get_compute_matrices_params_list_fn(basis: np.ndarray) -> Callable[[Array], Array]:
    """Create a partial unitary-computation function with a fixed basis.

    Args:
        basis: Array of shape ``(K, d, d)`` of Hermitian basis matrices.

    Returns:
        A ``Callable[[Array], Array]`` that accepts a parameter list
        and returns the product unitary.
    """
    return partial(compute_matrices_params_list_fn, basis=basis)


def get_chart_fn(
    generators: np.ndarray, base_point: Array | None = None
) -> Callable[[Array], Array]:
    r"""Build the chart: the pulse's orbit through ``base_point``.

    $\Phi(\phi) = U(\phi)\,x_0$, with $U$ the propagator above. Since that
    propagator's `jax.lax.scan` is seeded at the ambient identity,
    $\Phi(0) = x_0$ — the base point *is* where the chart starts.

    Args:
        generators: The chart's generator basis ``(K, d, d)``, Hermitian.
        base_point: The point $x_0 \in \mathcal A$ the pulse drives, of shape
            ``ambient_shape``. ``None`` means the propagator *is* the point,
            which is the case on any matrix Lie group; the underlying callable is
            then returned **unchanged**, so no multiplication by the identity is
            introduced on that hot path. The branch is on a Python value and is
            resolved at trace time.

    Returns:
        A ``Callable[[Array], Array]`` mapping ``(G, K)`` parameters to a point
        of shape ``ambient_shape``.
    """
    propagator = get_compute_matrices_params_list_fn(generators)
    if base_point is None:
        return propagator
    base = jnp.asarray(base_point, dtype=jnp.complex128)

    def chart(params_list: Array) -> Array:
        return propagator(params_list) @ base

    return chart


def get_chart_hvp_fn(
    generators: np.ndarray, base_point: Array | None = None
) -> Callable[[Array, Array], tuple[Array, Array, Array]]:
    r"""Build the chart's second differential: the jet $(\Phi,\ \mathrm D\Phi[p],\ \mathrm D^2\Phi[p, p])$.

    The same landing as `get_chart_fn`, applied termwise. That it *can* be
    applied termwise is the whole reason a homogeneous space reuses the group's
    propagator machinery wholesale: right multiplication by a constant is linear,
    so it commutes with differentiation and carries the jet with no
    manifold-specific code and no linearity contract to uphold.

    Args:
        generators: The chart's generator basis ``(K, d, d)``, Hermitian.
        base_point: As `get_chart_fn` — ``None`` returns the propagator's own HVP
            unchanged.

    Returns:
        A ``Callable[[Array, Array], tuple[Array, Array, Array]]`` accepting
        parameters and a direction, both ``(G, K)``, and returning the triple
        landed on ``base_point``.
    """
    # The bare two-argument form: `method="eig", hermitian=True`, the defaults
    # the whole pipeline has always run on.
    propagator_hvp = get_hvp_propagator(jnp.asarray(generators))
    if base_point is None:
        return propagator_hvp
    base = jnp.asarray(base_point, dtype=jnp.complex128)

    def chart_hvp(params_list: Array, direction: Array) -> tuple[Array, Array, Array]:
        point, velocity, acceleration = propagator_hvp(params_list, direction)
        return point @ base, velocity @ base, acceleration @ base

    return chart_hvp


def get_chart_jacobian_fn(
    generators: np.ndarray, base_point: Array | None = None
) -> Callable[[Array], Array]:
    r"""Build the chart's first differential $\mathrm D\Phi$, from the propagator.

    The manual `geope.jax.get_jacobian_propagator` — two ``jax.lax.scan`` partial
    products and one per-gate spectral derivative — landed on ``base_point`` and
    transposed into the layout `geope.geometry.tangent.TangentBundle` declares.

    This is the live Jacobian path. `get_jacobian_fn` is the autodiff equivalent,
    kept as the reference the tests and benchmarks compare against, and still
    used through `get_split_jacobian_fn` on the ``param_transform`` path where no
    exponential-product structure survives.

    Args:
        generators: The chart's generator basis ``(K, d, d)``, Hermitian.
        base_point: As `get_chart_fn` — ``None`` means the propagator *is* the
            point, and only the transpose is applied.

    Returns:
        A ``Callable[[Array], Array]`` mapping ``(G, K)`` parameters to the
        Jacobian of shape ``(*ambient_shape, G, K)``.

    Note:
        Built with ``hermitian=True``: the per-gate derivative diagonalises with
        ``eigh``, which assumes the pulse coefficients are real. That holds
        throughout the pipeline — `geope.parameters.Parameters.free` promotes to
        ``complex128`` with an exactly-zero imaginary part — but it is an
        assumption the holomorphic autodiff path did not make.
    """
    jac = get_jacobian_propagator(jnp.asarray(generators))
    if base_point is None:
        # (G, d, d, K) -> (d, d, G, K).
        return lambda params_list: jnp.moveaxis(jac(params_list), 0, -2)
    base = jnp.asarray(base_point, dtype=jnp.complex128)

    def chart_jacobian(params_list: Array) -> Array:
        # (G, d, d, K) -> (G, d, K, *base_trailing) -> (G, *ambient, K).
        landed = jnp.moveaxis(
            jnp.tensordot(jac(params_list), base, axes=[[2], [0]]), 2, -1
        )
        return jnp.moveaxis(landed, 0, -2)

    return chart_jacobian


def get_chart_vjp_fn(
    generators: np.ndarray, base_point: Array | None = None
) -> Callable[[Array], tuple[Array, Callable[[Array], Array]]]:
    r"""Build the chart's value and **pullback** $(\Phi,\ \mathrm D\Phi^\intercal)$.

    Returns ``phi -> (point, pullback)``, in the shape of `jax.vjp`. The pullback
    contracts an ambient covector against every column of $\mathrm D\Phi$ without
    forming it — see `geope.jax.vjp_propagator` for why that is
    $O(G(d^3 + d^2K))$ rather than $O(G d^3 K)$, and why the value comes back
    with it rather than from a second `get_chart_fn` call that would recompute
    every gate exponential.

    This is what an objective's gradient in parameter space is made of, and the
    reason `geope.geometry.manifold.Manifold.value_and_grad` needs no autodiff.

    The base point lands on the *covector* rather than on the differential, using

    $$\langle \hat G,\ \mathrm dU\,x_0\rangle = \langle \hat G\,x_0^\dagger,\ \mathrm dU\rangle,$$

    which is the same "right multiplication is linear" fact that lets
    `get_chart_hvp_fn` land termwise — just read in the opposite direction.

    Args:
        generators: The chart's generator basis ``(K, d, d)``, Hermitian.
        base_point: As `get_chart_fn`.

    Returns:
        A ``Callable[[Array], tuple[Array, Callable]]`` taking ``(G, K)``
        parameters to the point of shape ``ambient_shape`` and a callable
        mapping an ambient covector of that shape to the complex overlaps of
        shape ``(G, K)``. See `geope.jax.vjp_propagator` for why those are
        complex rather than already realified.
    """
    vjp = get_vjp_propagator(jnp.asarray(generators))
    if base_point is None:
        return vjp
    base = jnp.asarray(base_point, dtype=jnp.complex128)
    # Reshape a state (d,) to the (d, 1) frame it is, so one expression serves both.
    base_2d = base.reshape(base.shape[0], -1)

    def chart_vjp(params_list: Array) -> tuple[Array, Callable[[Array], Array]]:
        propagator, pullback = vjp(params_list)

        def landed_pullback(cotangent: Array) -> Array:
            return pullback(cotangent.reshape(base_2d.shape) @ jnp.conj(base_2d).T)

        return propagator @ base, landed_pullback

    return chart_vjp


def get_chart_hessian_fn(
    generators: np.ndarray, base_point: Array | None = None
) -> Callable[[Array], Array]:
    r"""Build the chart's dense second differential $\mathrm D^2\Phi$, from the propagator.

    `geope.jax.get_hessian_propagator` landed on ``base_point``. Unlike
    `get_chart_hvp_fn`, which takes one direction in $O(G)$, this materialises
    every pair — $O(G^2 d^2 K^2)$ in both flops and memory — so it is for the
    small systems where a Newton step is worth taking.

    Args:
        generators: The chart's generator basis ``(K, d, d)``, Hermitian.
        base_point: As `get_chart_fn`.

    Returns:
        A ``Callable[[Array], Array]`` mapping ``(G, K)`` parameters to the
        Hessian of shape ``(G, G, *ambient_shape, K, K)``.

    Note:
        `geope.jax.hessian_propagator` builds its off-diagonal blocks *using
        unitarity*, so this is valid only for real pulse coefficients — which is
        what the pipeline always has, but see the note on
        `get_chart_jacobian_fn`.
    """
    hess = get_hessian_propagator(jnp.asarray(generators))
    if base_point is None:
        return hess
    base = jnp.asarray(base_point, dtype=jnp.complex128)

    def chart_hessian(params_list: Array) -> Array:
        # (G, G, d, d, K, K) -> (G, G, d, K, K, *trail) -> (G, G, *ambient, K, K).
        landed = jnp.tensordot(hess(params_list), base, axes=[[3], [0]])
        return jnp.moveaxis(landed, (3, 4), (-2, -1))

    return chart_hessian


def get_jacobian_fn(
    compute_point_fn: Callable[[Array], Array]
) -> Callable[[Array], Array]:
    """Build the autodiff Jacobian of the chart w.r.t. parameters.

    Returns the holomorphic ``jax.jacobian`` of ``compute_point_fn``. The live
    path is now the manual `get_chart_jacobian_fn`; this is kept as the reference
    the tests and benchmarks check it against, and as the basis of
    `get_split_jacobian_fn`, which the ``param_transform`` path still needs.

    Args:
        compute_point_fn: Callable mapping a parameter list to the chart's point.

    Returns:
        A ``Callable[[Array], Array]`` returning the Jacobian of the point.
    """
    return jax.jacobian(compute_point_fn, argnums=0, holomorphic=True)


def get_split_jacobian_fn(
    compute_point_fn: Callable[[Array], Array],
) -> Callable[[Array], Array]:
    """Build a real/imag-split Jacobian of ``compute_point_fn``.

    Used on the ``param_transform`` path: differentiating through the
    real-valued user transform with a holomorphic Jacobian would discard the
    imaginary part of intermediates, so the point is split into real and
    imaginary parts, each differentiated, then recombined.

    Args:
        compute_point_fn: The (wrapped) experimental-space chart.

    Returns:
        A ``Callable[[Array], Array]`` returning the complex Jacobian.
    """

    def _split_U(x):
        U = compute_point_fn(x)
        return jnp.stack([jnp.real(U), jnp.imag(U)])

    _raw_jac_split = jax.jacobian(_split_U, argnums=0)

    def _jac_fn(x):
        jac_split = _raw_jac_split(x)
        return jac_split[0] + 1j * jac_split[1]

    return _jac_fn


def get_split_vjp_fn(
    compute_point_fn: Callable[[Array], Array],
) -> Callable[[Array], tuple[Array, Callable[[Array], Array]]]:
    r"""Build a real/imag-split value-and-pullback of ``compute_point_fn``.

    The autodiff counterpart of `get_chart_vjp_fn`, and the ``param_transform``
    path's only option: there is no exponential-product structure left to pull
    back through, and the user's transform must be differentiated as it stands.
    `jax.vjp` already has the ``(value, pullback)`` shape, and for the same
    reason — so the value is shared rather than recomputed.

    Split for the same reason `get_split_jacobian_fn` is — real intermediates in
    the transform would drop the imaginary part under a holomorphic pullback —
    and returning the same complex $\mathrm{Tr}(C^\dagger \partial\Phi)$ so that
    `geope.geometry.manifold.Manifold.value_and_grad` is one expression on both
    paths.

    Args:
        compute_point_fn: The (wrapped) experimental-space chart.

    Returns:
        A ``Callable[[Array], tuple[Array, Callable]]``, as `get_chart_vjp_fn`.
    """

    def _split_U(x):
        U = compute_point_fn(x)
        return jnp.stack([jnp.real(U), jnp.imag(U)])

    def _vjp_fn(x: Array) -> tuple[Array, Callable[[Array], Array]]:
        split, pullback = jax.vjp(_split_U, x)

        def _pull(cotangent: Array) -> Array:
            # <C, dPhi> = <Re C, dRe Phi> + <Im C, dIm Phi>
            #             + i(<Re C, dIm Phi> - <Im C, dRe Phi>)
            re_c, im_c = jnp.real(cotangent), jnp.imag(cotangent)
            real_part = pullback(jnp.stack([re_c, im_c]))[0]
            imag_part = pullback(jnp.stack([-im_c, re_c]))[0]
            return real_part + 1j * imag_part

        return split[0] + 1j * split[1], _pull

    return _vjp_fn
