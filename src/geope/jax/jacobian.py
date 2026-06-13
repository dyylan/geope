from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from functools import partial
from typing import Callable

from .dexpm import get_dexpm


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


def scan_single_switch_matmul(
    carry: tuple[Array, Array], x: tuple[Array, Array]
) -> tuple[tuple[Array, Array], None]:
    """Scan body that conditionally applies a Jacobian or a gate matrix.

    Used inside ``jax.lax.scan`` to build the product unitary while
    inserting a Jacobian slice at the position indicated by `idx`.

    Args:
        carry: Tuple of ``(U, jacobian)`` — the running product
            ``Array`` and the Jacobian ``Array`` to insert.
        x: Tuple of ``(idx, gate)`` where ``idx`` is a boolean
            ``Array`` switch and ``gate`` is the gate ``Array``.

    Returns:
        Updated carry tuple and ``None`` (no stacked output).
    """
    # Use idx to switch branches
    U, jacobian = carry
    idx, gate = x
    # If true, apply jacobian, else apply gate
    U = jax.lax.cond(idx,
                     lambda op: jnp.einsum("ij,jk->ik", jacobian, op),
                     lambda op: jnp.einsum("ij,jk->ik", gate, op), U)
    return (U, jacobian), None


def get_apply_branch(gates: Array) -> Callable[[Array, Array], tuple[Array, Array]]:
    """Build a JIT-compiled branch-application function.

    Given pre-computed gate unitaries, returns a function that
    computes the product unitary with one gate replaced by a
    Jacobian slice.

    Args:
        gates: ``Array`` of gate unitaries of shape ``(G, d, d)``.

    Returns:
        A JIT-compiled callable ``(idx, jac) -> tuple[Array, Array]``
        where ``idx`` is a boolean array indicating which gate to
        replace.
    """
    # initialize U0
    U0 = jnp.eye(gates.shape[1], dtype=complex)
    # Apply the branch based on the idx
    return jax.jit(lambda idx, jac: jax.lax.scan(scan_single_switch_matmul, (U0, jac), (idx, gates))[0])


def scan_branch(
    jac: Array,
    indices_i: Array,
    branch_fn: Callable[[Array, Array], tuple[Array, Array]],
) -> Array:
    """Scan over Jacobian columns and apply the branch function.

    Args:
        jac: Jacobian ``Array`` with shape ``(d, d, K)``.
        indices_i: Boolean index ``Array`` selecting the gate to replace.
        branch_fn: Branch-application callable from `get_apply_branch`.

    Returns:
        Stacked output ``Array`` of shape ``(d, d, K)``.
    """
    def body(carry, j):  # carry is unused (None)
        out = branch_fn(indices_i, jac[..., j])[0]
        return carry, out  # carry unchanged, out collected

    _, stacked = jax.lax.scan(body, None, jnp.arange(jac.shape[-1]))
    # scan puts the scan dimension first; move it to the end to match your stack
    return jnp.moveaxis(stacked, 0, -1)


def get_scan_branch(
    branch_fn: Callable[[Array, Array], tuple[Array, Array]],
) -> Callable[[Array, Array], Array]:
    """Create a partial scan-branch function.

    Args:
        branch_fn: Branch-application function from `get_apply_branch`.

    Returns:
        A callable ``(jac, indices_i)``.
    """
    return partial(scan_branch, branch_fn=branch_fn)


def manual_jacobian(
    params: Array, Ui_fn: Callable[[Array], Array], jac_fn: Callable[[Array], Array]
) -> Array:
    """Compute the full Jacobian of the product unitary manually.

    For each gate segment, evaluates the derivative of the product
    unitary with respect to each parameter by inserting the
    per-gate Jacobian into the product chain.

    Args:
        params: Parameter ``Array`` of shape ``(G, K)``.
        Ui_fn: Callable mapping a coefficient ``Array`` to a unitary ``Array``.
        jac_fn: Callable computing the per-gate Jacobian ``Array``.

    Returns:
        An ``Array`` of shape ``(G, d, d, K)`` containing the full Jacobian.
    """
    # Get all the gates
    gates = jnp.stack([Ui_fn(p) for p in params])
    # Switches for jacobian calculation
    indices = jnp.eye(gates.shape[0], dtype=bool)
    # We need to pass the parameter at location idx separately so that we can calculate its jacobian
    branch_fn = get_apply_branch(gates)
    scan_branch_fn = get_scan_branch(branch_fn)
    res = []
    for i in range(gates.shape[0]):
        res.append(scan_branch_fn(jac_fn(params[i]), indices[i]))
    return jnp.stack(res)


def get_jacobian_manual(gate_basis: Array) -> Callable[[Array], Array]:
    """Create a manual Jacobian function for a given gate basis.

    Args:
        gate_basis: ``Array`` of Hermitian basis matrices of shape ``(K, d, d)``.

    Returns:
        A ``Callable[[Array], Array]`` that accepts a parameter array
        of shape ``(G, K)`` and returns the Jacobian of shape
        ``(G, d, d, K)``.
    """
    Ui_fn = get_Ui_fn(gate_basis)
    jac_fn = get_dexpm(gate_basis)
    return partial(manual_jacobian, Ui_fn=Ui_fn, jac_fn=jac_fn)
