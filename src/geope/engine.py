import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .lie import Basis

from functools import partial


class Engine:
    """
    """

    def __init__(self,
                 target_unitary,
                 full_basis: Basis,
                 projected_basis: Basis,
                 drift_basis: Basis = None,
                 gates: int = 1):
        self.full_basis = full_basis
        self.projected_basis = projected_basis
        self.drift_basis = drift_basis
        self.gates = gates

        # Get the projected indices in the full space
        self.projected_indices = np.array(projected_basis.overlap(full_basis), dtype=bool)
        if drift_basis is None:
            self.drift_indices = np.full(full_basis.lie_algebra_dim, False)
        else:
            self.drift_indices = np.array(drift_basis.overlap(full_basis),
                                          dtype=bool)
        self.proj_drift_indices = self.projected_indices + self.drift_indices
        self.proj_drift_basis = Basis(full_basis.basis[self.projected_indices + self.drift_indices], labels=list(
            np.array(full_basis.labels)[self.projected_indices + self.drift_indices]))

        self.proj_indices_projdrift_basis = np.delete(self.projected_indices, ~self.proj_drift_indices)
        self.drift_indices_projdrift_basis = np.delete(self.drift_indices, ~self.proj_drift_indices)

        # Get the Jax functions from the helper functions
        self.compute_U_fn = jax.jit(get_compute_matrices_params_list_fn(self.proj_drift_basis.basis))
        self.fid_U_fn = jax.jit(get_fidelity_fn(target_unitary))


def fidelity(unitary, target_unitary):
    return jnp.abs(jnp.einsum('ji,ji->', target_unitary.conj(), unitary)) / len(target_unitary[0])


def get_fidelity_fn(target_unitary):
    return partial(fidelity, target_unitary=target_unitary)


def compute_matrices_params_list_fn(params_list, basis):
    def step(U, params):
        A = jnp.tensordot(params, basis, axes=[[-1], [0]])
        Ui = jax.scipy.linalg.expm(1j * A)
        U_new = jnp.matmul(Ui, U)
        return U_new, None

    U0 = jnp.eye(basis.shape[1], dtype=basis.dtype)
    U_final, _ = jax.lax.scan(step, U0, jnp.stack(params_list))
    return U_final


def get_compute_matrices_params_list_fn(basis: np.ndarray):
    return partial(compute_matrices_params_list_fn, basis=basis)
