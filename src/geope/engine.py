from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_enable_x64", True)

from .lie import Basis

from functools import partial
from typing import Callable


class Engine:
    """Base engine for compiling quantum unitaries using Lie-algebraic methods.

    The Engine sets up the algebraic infrastructure for quantum gate synthesis,
    including projected and drift basis indices, and JIT-compiled JAX functions
    for computing unitaries and fidelities.

    Attributes:
        full_basis: The full Lie algebra basis.
        projected_basis: The projected (controllable) subalgebra basis.
        drift_basis: The drift (uncontrollable) subalgebra basis, if any.
        piecewise_steps: Number of piecewise-constant gate segments.
        projected_indices: Boolean mask for projected basis elements in the full basis.
        drift_indices: Boolean mask for drift basis elements in the full basis.
        proj_drift_indices: Combined boolean mask for projected and drift elements.
        proj_drift_basis: Combined projected-and-drift `Basis` object.
        proj_indices_projdrift_basis: Projected indices within the combined basis.
        drift_indices_projdrift_basis: Drift indices within the combined basis.
        compute_U_fn: JIT-compiled function to compute the unitary from parameters.
        fid_U_fn: JIT-compiled function to compute fidelity against the target unitary.
    """

    def __init__(self,
                 target_unitary: np.ndarray,
                 full_basis: Basis,
                 projected_basis: Basis,
                 drift_basis: Basis | None = None,
                 piecewise_steps: int = 1) -> None:
        """Initialise the Engine.

        Args:
            target_unitary: The target unitary matrix as ``np.ndarray``.
            full_basis: The full Lie algebra ``Basis``.
            projected_basis: The projected (controllable) subalgebra ``Basis``.
            drift_basis: The drift (uncontrollable) subalgebra ``Basis``.
                Defaults to ``None``.
            piecewise_steps: Number of piecewise-constant gate segments. Defaults to 1.
        """
        self.full_basis = full_basis
        self.projected_basis = projected_basis
        self.drift_basis = drift_basis
        self.piecewise_steps = piecewise_steps

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


def fidelity(unitary: Array, target_unitary: Array) -> Array:
    """Compute the fidelity between a unitary and a target unitary.

    The fidelity is defined as the normalised absolute value of the
    Hilbert-Schmidt inner product between the two matrices.

    Args:
        unitary: The unitary ``Array`` to evaluate.
        target_unitary: The target unitary ``Array``.

    Returns:
        A scalar fidelity ``Array`` in the range $[0, 1]$.
    """
    return jnp.abs(jnp.einsum('ji,ji->', target_unitary.conj(), unitary)) / len(target_unitary[0])


def get_fidelity_fn(target_unitary: Array) -> Callable[[Array], Array]:
    """Create a partial fidelity function with a fixed target unitary.

    Args:
        target_unitary: The target unitary ``Array`` to bind.

    Returns:
        A ``Callable[[Array], Array]`` that accepts a single unitary
        and returns the fidelity against ``target_unitary``.
    """
    return partial(fidelity, target_unitary=target_unitary)


def infidelity(unitary: Array, target_unitary: Array) -> Array:
    """Projective infidelity $1 - F_{\\mathrm{proj}}(U, U_T)$.

    Args:
        unitary: The unitary ``Array`` to evaluate.
        target_unitary: The target unitary ``Array``.

    Returns:
        A scalar infidelity ``Array`` in $[0, 1]$.
    """
    return 1 - jnp.abs(jnp.einsum('ji,ji->', target_unitary.conj(), unitary)) / len(target_unitary[0])


def get_infidelity_fn(target_unitary: Array) -> Callable[[Array], Array]:
    """Create a partial projective-infidelity function with a fixed target.

    Args:
        target_unitary: The target unitary ``Array`` to bind.

    Returns:
        A ``Callable[[Array], Array]`` returning $1 - F_{\\mathrm{proj}}$.
    """
    return partial(infidelity, target_unitary=target_unitary)


def fidelity_full(unitary: Array, target_unitary: Array) -> Array:
    """Phase-sensitive (non-projective) fidelity.

    $F_{\\mathrm{full}}(U, U_T) = \\mathrm{Re}\\,\\mathrm{Tr}(U_T^\\dagger U) / d$.
    Unlike the projective fidelity, this is sensitive to a global phase
    on $U$ and lies in $[-1, 1]$.

    Args:
        unitary: The unitary ``Array`` to evaluate.
        target_unitary: The target unitary ``Array``.

    Returns:
        A scalar fidelity ``Array`` in $[-1, 1]$.
    """
    return jnp.real(jnp.einsum('ji,ji->', target_unitary.conj(), unitary)) / len(target_unitary[0])


def get_fidelity_full_fn(target_unitary: Array) -> Callable[[Array], Array]:
    """Create a partial phase-sensitive fidelity function with a fixed target.

    Args:
        target_unitary: The target unitary ``Array`` to bind.

    Returns:
        A ``Callable[[Array], Array]`` returning $F_{\\mathrm{full}}$.
    """
    return partial(fidelity_full, target_unitary=target_unitary)


def infidelity_full(unitary: Array, target_unitary: Array) -> Array:
    """Phase-sensitive infidelity $1 - F_{\\mathrm{full}}(U, U_T)$.

    Args:
        unitary: The unitary ``Array`` to evaluate.
        target_unitary: The target unitary ``Array``.

    Returns:
        A scalar infidelity ``Array`` in $[0, 2]$.
    """
    return 1 - jnp.real(jnp.einsum('ji,ji->', target_unitary.conj(), unitary)) / len(target_unitary[0])


def get_infidelity_full_fn(target_unitary: Array) -> Callable[[Array], Array]:
    """Create a partial phase-sensitive infidelity function with a fixed target.

    Args:
        target_unitary: The target unitary ``Array`` to bind.

    Returns:
        A ``Callable[[Array], Array]`` returning $1 - F_{\\mathrm{full}}$.
    """
    return partial(infidelity_full, target_unitary=target_unitary)


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
