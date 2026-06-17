from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from functools import partial
from typing import Callable


def Ui(x: Array, basis: Array) -> Array:
    """Compute a unitary from a linear combination of Hermitian basis matrices.

    Constructs $U = \\exp(i \\sum_k x_k B_k)$.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        basis: Array of Hermitian matrices of shape ``(K, d, d)``.

    Returns:
        A unitary matrix of shape ``(d, d)``.
    """
    A = jnp.tensordot(x, basis, axes=[[-1], [0]])
    return jax.scipy.linalg.expm(1j * A)


def get_Ui_fn(basis: Array) -> Callable[[Array], Array]:
    """Create a partial unitary function with a fixed basis.

    Args:
        basis: Array of Hermitian matrices of shape ``(K, d, d)``.

    Returns:
        A callable that accepts a coefficient vector and returns
        the corresponding unitary matrix.
    """
    return partial(Ui, basis=basis)


@jax.jit
def dexpm_block(A: Array, x: Array) -> Array:
    """Compute the derivative of the matrix exponential via the block method.

    Implements the block-matrix approach of
    `Al-Mohy & Higham (2009) <https://arxiv.org/pdf/1506.00628>`_,
    Eq. (31), extracting $d\\exp(iA)/dA \\cdot x$ from the upper-right
    block of $\\exp(i[[A, x], [0, A]])$.

    Args:
        A: The Hamiltonian matrix of shape ``(d, d)``.
        x: The direction matrix of shape ``(d, d)``.

    Returns:
        The directional derivative matrix of shape ``(d, d)``.
    """
    dim = A.shape[0]
    # Create block matrix
    block_mat = jnp.block([[A, x], [jnp.zeros_like(A), A]])
    # Take matrix exponential
    dblock_mat = jax.scipy.linalg.expm(1j * block_mat)
    # Upper right block contains derivative
    return dblock_mat[:dim, dim:]


def dexpm(x: Array, basis: Array) -> Array:
    """Compute the derivative of the exponential map for all basis directions.

    For each basis element $B_k$, computes
    $\\partial \\exp(i \\sum_j x_j B_j) / \\partial x_k$.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        basis: Array of Hermitian matrices of shape ``(K, d, d)``.

    Returns:
        An array of shape ``(d, d, K)`` whose last axis indexes the
        partial derivatives with respect to each coefficient.
    """
    # Construct argument of exponential
    print(x.shape)
    A = jnp.tensordot(x, basis, axes=[[-1], [0]])
    # For each element in the basis, get the derivative. Stack in last axis.
    return jax.vmap(lambda b: dexpm_block(A, b), out_axes=2)(basis)


def dexpm_batched(x: Array, basis: Array, batch_size: int) -> Array:
    """Batched derivative of the exponential map.

    Same as `dexpm` but uses ``jax.lax.map`` with a configurable
    `batch_size` to limit peak memory usage.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        basis: Array of Hermitian matrices of shape ``(K, d, d)``.
        batch_size: Number of basis elements to process per batch.

    Returns:
        An array of shape ``(d, d, K)``.
    """
    # Construct argument of exponential
    A = jnp.tensordot(x, basis, axes=[[-1], [0]])
    # For each element in the basis, get the derivative. Stack in last axis.
    return jnp.transpose(
        jax.lax.map(lambda b: dexpm_block(A, b), basis, batch_size=batch_size),
        axes=(1, 2, 0),
    )


def get_dexpm(basis: Array, batch_size: int | None = None) -> Callable[[Array], Array]:
    """Create a JIT-compiled exponential-map derivative function.

    Args:
        basis: Array of Hermitian matrices of shape ``(K, d, d)``.
        batch_size: Optional batch size. If ``None``, the full vmap
            variant is used; otherwise the batched variant.

    Returns:
        A callable that accepts a coefficient vector and returns
        the derivative array of shape ``(d, d, K)``.
    """
    if batch_size is None:
        return jax.jit(partial(dexpm, basis=basis))
    else:
        return partial(dexpm_batched, basis=basis, batch_size=batch_size)
