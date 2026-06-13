from __future__ import annotations

import numpy as np
import scipy.linalg as spla
import jax
import jax.numpy as jnp
from jax import Array
from typing import TYPE_CHECKING

from .basis import Basis

if TYPE_CHECKING:
    from .hamiltonian import Hamiltonian


class Unitary:
    """A unitary matrix acting on $n$ qubits.

    Validates that the input is indeed unitary ($UU^\\dagger = I$) and
    exposes convenience methods for fidelity and geodesic computation.

    Attributes:
        matrix: The unitary matrix of shape ``(d, d)``.
        n: Number of qubits ($\\log_2 d$).
    """

    def __init__(self, unitary_matrix: np.ndarray) -> None:
        """Initialise a Unitary.

        Args:
            unitary_matrix: A square unitary ``np.ndarray`` of dimension
                $2^n \times 2^n$.

        Raises:
            ValueError: If the matrix is not unitary or not square.
        """
        self.matrix, self.n = Unitary._check_is_unitary(unitary_matrix)

    def parameters(self, basis: Basis) -> np.ndarray:
        """Extract Lie-algebra coefficients via the principal matrix logarithm.

        Args:
            basis: The ``Basis`` to decompose into.

        Returns:
            A real-valued parameter ``np.ndarray`` of length
            ``basis.lie_algebra_dim``.
        """
        return Unitary.parameters_from_unitary(self.matrix, basis)

    def fidelity(self, unitary_matrix: np.ndarray) -> Array:
        """Compute the fidelity to a target unitary.

        Args:
            unitary_matrix: The target unitary ``np.ndarray``.

        Returns:
            Scalar fidelity ``Array`` in $[0, 1]$.
        """
        return Unitary.unitary_fidelity(self.matrix, unitary_matrix)

    def geodesic_hamiltonian(self, basis: Basis, target_unitary: np.ndarray) -> Hamiltonian:
        """Compute the geodesic Hamiltonian towards a target unitary.

        Args:
            basis: The ``Basis`` for the Hamiltonian decomposition.
            target_unitary: The target unitary ``np.ndarray``.

        Returns:
            A ``Hamiltonian`` whose exponentiation yields the geodesic
            rotation from ``self`` to ``target_unitary``.
        """
        from .hamiltonian import Hamiltonian  # deferred — avoids circular import at module load
        g = -1.j * spla.logm(self.matrix.conj().T @ target_unitary)
        params = Hamiltonian.parameters_from_hamiltonian(g, basis)
        return Hamiltonian(basis, params)

    def __matmul__(self, other: Unitary) -> Unitary:
        return Unitary(self.matrix @ other.matrix)

    @staticmethod
    def _check_is_unitary(unitary_matrix: np.ndarray) -> tuple[np.ndarray, int]:
        """Validate that a matrix is unitary and square.

        Args:
            unitary_matrix: The ``np.ndarray`` to validate.

        Returns:
            A tuple ``(matrix, n)`` where ``n`` is the number of qubits.

        Raises:
            ValueError: If the matrix is not unitary or not square.
        """
        if not np.allclose(np.eye(len(unitary_matrix)), unitary_matrix @ unitary_matrix.T.conj()):
            raise ValueError("Matrix given to Unitary must be unitary: U U^dagger = U^dagger U = I")
        if not unitary_matrix.shape[0] == unitary_matrix.shape[1]:
            raise ValueError("Matrix must be square")
        return unitary_matrix, int(np.log2(len(unitary_matrix)))

    @staticmethod
    @jax.jit
    def unitary_fidelity(matrixA: Array, matrixB: Array) -> Array:
        """Compute the fidelity between two unitary matrices.

        Args:
            matrixA: First unitary ``Array``.
            matrixB: Second unitary ``Array``.

        Returns:
            Scalar fidelity ``Array`` in $[0, 1]$.
        """
        return jnp.abs(jnp.trace(matrixA.conj().T @ matrixB)) / len(matrixA[0])

    @staticmethod
    def parameters_from_unitary(unitary_matrix: np.ndarray, basis: Basis) -> np.ndarray:
        """Extract Lie-algebra coefficients from a unitary matrix.

        Uses the principal matrix logarithm to recover the generator,
        then decomposes it in the given basis.

        Args:
            unitary_matrix: The unitary ``np.ndarray`` of shape ``(d, d)``.
            basis: The ``Basis`` to decompose into.

        Returns:
            A real-valued parameter ``np.ndarray`` of length
            ``basis.lie_algebra_dim``.
        """
        from .hamiltonian import Hamiltonian  # deferred — avoids circular import at module load
        return Hamiltonian.parameters_from_hamiltonian(-1.j * spla.logm(unitary_matrix), basis)
