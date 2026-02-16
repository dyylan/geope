import jax
import jax.numpy as jnp
from functools import partial

from .dexpm import get_dexpm


def Ui(x, basis):
    A = jnp.tensordot(x, basis, axes=[[-1], [0]])
    return jax.scipy.linalg.expm(1j * A)


def get_Ui_fn(basis):
    return partial(Ui, basis=basis)


def scan_single_switch_matmul(carry, x):
    # Use idx to switch branches
    U, jacobian = carry
    idx, gate = x
    # If true, apply jacobian, else apply gate
    U = jax.lax.cond(idx,
                     lambda op: jnp.einsum("ij,jk->ik", jacobian, op),
                     lambda op: jnp.einsum("ij,jk->ik", gate, op), U)
    return (U, jacobian), None


def get_apply_branch(gates):
    # initialize U0
    U0 = jnp.eye(gates.shape[1], dtype=complex)
    # Apply the branch based on the idx
    return jax.jit(lambda idx, jac: jax.lax.scan(scan_single_switch_matmul, (U0, jac), (idx, gates))[0])


def scan_branch(jac, indices_i, branch_fn):
    def body(carry, j):  # carry is unused (None)
        out = branch_fn(indices_i, jac[..., j])[0]
        return carry, out  # carry unchanged, out collected

    _, stacked = jax.lax.scan(body, None, jnp.arange(jac.shape[-1]))
    # scan puts the scan dimension first; move it to the end to match your stack
    return jnp.moveaxis(stacked, 0, -1)


def get_scan_branch(branch_fn):
    return partial(scan_branch, branch_fn=branch_fn)


def manual_jacobian(params, Ui_fn, jac_fn):
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


def get_jacobian_manual(gate_basis):
    Ui_fn = get_Ui_fn(gate_basis)
    jac_fn = get_dexpm(gate_basis)
    return partial(manual_jacobian, Ui_fn=Ui_fn, jac_fn=jac_fn)
