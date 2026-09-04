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

from ..jax.hessian import get_hvp_propagator


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


def get_jacobian_fn(
    compute_point_fn: Callable[[Array], Array]
) -> Callable[[Array], Array]:
    """Build the autodiff Jacobian of the chart w.r.t. parameters.

    Returns the holomorphic ``jax.jacobian`` of ``compute_point_fn``. This is the
    live Jacobian path for *all* system sizes: the manual Jacobian
    (`geope.jax.jacobian.get_jacobian_propagator`) exists and is independently
    tested, but is not currently wired into the optimisation pipeline (the
    autodiff path historically overwrote it for the >5-qubit branch — see
    issue #4).

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
