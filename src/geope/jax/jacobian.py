from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from functools import partial
from typing import Callable

from .dexpm import get_dexpm_eig


def Ui(x: Array, basis: Array) -> Array:
    """Compute a unitary from a linear combination of Hermitian basis matrices.

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
        A callable that takes a coefficient vector and returns
        the corresponding unitary.
    """
    return partial(Ui, basis=basis)


def manual_jacobian(
    params: Array, Ui_fn: Callable[[Array], Array], jac_fn: Callable[[Array], Array]
) -> Array:
    r"""Compute the full Jacobian of the product unitary manually.

    The product unitary follows the convention of
    :func:`geope.engine.compute_matrices_params_list_fn`, where each gate is
    left-multiplied onto the accumulator,

    $$U = U_{G-1} \cdots U_1 U_0, \qquad U_i = \exp\!\Big(i \sum_k x_{i,k} G_k\Big).$$

    The derivative with respect to a parameter of gate $i$ leaves every other
    gate untouched, so it is a product with a single factor replaced by the
    per-gate derivative:

    $$\frac{\partial U}{\partial x_{i,k}}
        = \underbrace{U_{G-1} \cdots U_{i+1}}_{L_i}\,
          \frac{\partial U_i}{\partial x_{i,k}}\,
          \underbrace{U_{i-1} \cdots U_0}_{R_i}.$$

    Both the left ($L_i$, exclusive suffix product) and right ($R_i$, exclusive
    prefix product) partial products are obtained in $O(G)$ matrix
    multiplications with two ``jax.lax.scan`` passes, after which the per-gate
    derivative blocks are combined with a single vectorised ``einsum``. This is
    the equivalent of differentiating the whole sequence with autodiff, but
    built explicitly from the per-gate derivative ``jac_fn``.

    Args:
        params: Parameter ``Array`` of shape ``(G, K)``.
        Ui_fn: Callable mapping a coefficient ``Array`` to a unitary ``Array``.
        jac_fn: Callable computing the per-gate Jacobian ``Array`` of shape
            ``(d, d, K)`` (e.g. :func:`geope.jax.dexpm`).

    Returns:
        An ``Array`` of shape ``(G, d, d, K)`` containing the full Jacobian.
    """
    # Per-gate unitaries (G, d, d) and per-gate derivatives (G, d, d, K).
    gates = jax.vmap(Ui_fn)(params)
    jacs = jax.vmap(jac_fn)(params)

    eye = jnp.eye(gates.shape[1], dtype=gates.dtype)

    # Exclusive prefix products: R[i] = gates[i-1] @ ... @ gates[0], R[0] = I.
    # Emit the running product *before* folding in the current gate.
    def step_right(R, g):
        return g @ R, R

    Rs = jax.lax.scan(step_right, eye, gates)[1]

    # Exclusive suffix products: L[i] = gates[G-1] @ ... @ gates[i+1], L[G-1] = I.
    # Scan in reverse so the running product holds the gates processed so far.
    def step_left(L, g):
        return L @ g, L

    Ls = jax.lax.scan(step_left, eye, gates, reverse=True)[1]

    # Block_i[a, c, k] = L_i[a, b] jac_i[b, e, k] R_i[e, c].
    return jax.vmap(lambda L, J, R: jnp.einsum("ab,bek,ec->ack", L, J, R))(
        Ls, jacs, Rs
    )


def get_jacobian_manual(
    gate_basis: Array, hermitian: bool = True
) -> Callable[[Array], Array]:
    """Create a JIT-compiled manual Jacobian function for a given gate basis.

    The per-gate derivative uses the spectral method (`geope.jax.dexpm_eig`),
    and the returned function is wrapped in ``jax.jit`` so it is compiled once
    and reused across calls (rather than retracing on every invocation).

    Args:
        gate_basis: ``Array`` of Hermitian basis matrices of shape ``(K, d, d)``.
        hermitian: Assume real parameters (skew-Hermitian generators) and use
            the faster ``eigh``-based per-gate derivative. Set ``False`` for
            complex-valued parameters.

    Returns:
        A ``Callable[[Array], Array]`` that accepts a parameter array
        of shape ``(G, K)`` and returns the Jacobian of shape
        ``(G, d, d, K)``.
    """
    Ui_fn = get_Ui_fn(gate_basis)
    jac_fn = get_dexpm_eig(gate_basis, hermitian=hermitian)
    return jax.jit(partial(manual_jacobian, Ui_fn=Ui_fn, jac_fn=jac_fn))
