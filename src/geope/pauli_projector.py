from functools import partial
import itertools as it
import numpy as np

import jax

from jax import numpy as jnp
from .lie import Basis


def project_omegas(x, basis, dim):
    return jnp.real(jnp.einsum("ijk, nkj->ni", basis, x)) / dim


def get_project_omegas_fn(basis:Basis):
    return partial(project_omegas, basis=basis.basis, dim=basis.dim)


def get_kron_chain(n):
    paulis = jnp.stack([jnp.eye(2).astype(complex),
                        jnp.array([[0, 1], [1, 0]], complex),
                        jnp.array([[0, -1j], [1j, 0]], complex),
                        jnp.array([[1, 0], [0, -1]], complex)])

    @jax.jit
    def kron_chain(comb):
        p = paulis[comb[0]]
        for i in range(1, n):
            p = jnp.kron(p, paulis[comb[i]])
        return p

    return kron_chain


def get_project_omegas_fn_otf(basis:Basis, batch_size: int = None):
    n = basis.n
    combs = jnp.array(list(it.product([0, 1, 2, 3], repeat=n))[1:], dtype=jnp.int32)
    kron_chain = get_kron_chain(n)

    @jax.jit
    def projector(c, x):
        pauli = kron_chain(c)
        return jnp.real(jnp.einsum('ij,ji->', pauli, x)) / x.shape[0]
    # vmap over combinations
    vmap_projector = jax.vmap(projector, in_axes=(0, None))
    if batch_size is None:
        # vmap over input axes
        return jax.vmap(lambda x: vmap_projector(combs, x))
    else:
        total = combs.shape[0]
        remainder = (-total) % batch_size
        # Pad so it's divisible by batch size
        if remainder != 0:
            padding = np.tile(combs[-1:], (remainder, 1))
            combs = np.concatenate([combs, padding], axis=0)
        combs = combs.reshape((batch_size, combs.shape[0] // batch_size, -1))
        # scan over combs
        def batched_vmap_projector(c, x):
            @jax.jit
            def scan_fn(carry, batch):
                return carry, vmap_projector(batch, x)

            results = jax.lax.scan(scan_fn, None, c)[1]
            return jnp.concatenate(results, axis=0)[:total]

        # Do not jit, otherwise whole loop gets allocated and memory explodes.
        # vmap over input axes
        return jax.vmap(lambda x: batched_vmap_projector(combs, x))
