import sys
from pathlib import Path

# Ensure the src/ layout is importable without pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import jax

# float64/complex128 throughout, matching every test/source module.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from geope.utils import construct_full_pauli_basis


def make_basis(n: int) -> jnp.ndarray:
    """Full ``n``-qubit Pauli basis as a ``(K, d, d)`` array.

    ``K = 4**n - 1`` (identity excluded) and ``d = 2**n``.
    """
    return jnp.asarray(construct_full_pauli_basis(n).basis)


def make_params(n_steps: int, K: int, key: jax.Array) -> jnp.ndarray:
    """Random ``(n_steps, K)`` parameter array in complex128.

    Inputs are complex because the autodiff Jacobians use
    ``holomorphic=True`` and the manual path also expects complex
    coefficients.
    """
    return jax.random.normal(key, (n_steps, K)).astype(jnp.complex128)


def warm(fn, *args):
    """Force compilation and return a device-ready result.

    Used both to warm up a function before timing and to materialise a
    result for the correctness guards.
    """
    return jax.block_until_ready(fn(*args))
