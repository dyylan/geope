import jax
import jax.numpy as jnp

from functools import partial


def Ui(x, basis):
    """Create a unitary from a basis of Hermitian matrices"""
    A = jnp.tensordot(x, basis, axes=[[-1], [0]])
    return jax.scipy.linalg.expm(1j * A)


def get_Ui_fn(basis):
    """Return partial function that only takes coefficients"""
    return partial(Ui, basis=basis)


@jax.jit
def dexpm_block(A, x):
    """Perform block approach of https://arxiv.org/pdf/1506.00628 Eq. (31)"""
    dim = A.shape[0]
    # Create block matrix
    block_mat = jnp.block([[A, x],
                           [jnp.zeros_like(A), A]])
    # Take matrix exponential
    dblock_mat = jax.scipy.linalg.expm(1j * block_mat)
    # Upper right block contains derivative
    return dblock_mat[:dim, dim:]


def dexpm(x, basis):
    """Derivative of exponential map"""
    # Construct argument of exponential
    print(x.shape)
    A = jnp.tensordot(x, basis, axes=[[-1], [0]])
    # For each element in the basis, get the derivative. Stack in last axis.
    return jax.vmap(lambda b: dexpm_block(A, b), out_axes=2)(basis)


def dexpm_batched(x, basis, batch_size):
    """Derivative of exponential map"""
    # Construct argument of exponential
    A = jnp.tensordot(x, basis, axes=[[-1], [0]])
    # For each element in the basis, get the derivative. Stack in last axis.
    return jnp.transpose(jax.lax.map(lambda b: dexpm_block(A, b), basis, batch_size=batch_size), axes=(1, 2, 0))


def get_dexpm(basis, batch_size=None):
    if batch_size is None:
        return jax.jit(partial(dexpm, basis=basis))
    else:
        return partial(dexpm_batched, basis=basis, batch_size=batch_size)
