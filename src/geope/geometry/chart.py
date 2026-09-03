r"""The pulse model: a product of piecewise-constant exponentials, and its Jacobians.

This is the *chart* every manifold in the library is coordinatised by,

$$\Phi(\phi) = \prod_g \exp\Bigl(i\sum_k \phi_{g,k} G_k\Bigr),$$

and it belongs to no single manifold: `geope.geometry.lie.groups.MatrixLieGroup`
takes it as its chart directly, and
`geope.geometry.stiefel.sphere.StateSphere` composes it with a base state. What
*is* manifold-specific — the metric, the logarithm, the fidelity — lives on the
`geope.geometry.manifold.Manifold` hooks instead.

Like `geope.geometry.tangent`, this module imports nothing but JAX: it sits below
the manifolds rather than beside them.

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
