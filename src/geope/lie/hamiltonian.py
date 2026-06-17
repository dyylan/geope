from __future__ import annotations

import numpy as np
import scipy.linalg as spla
import jax
import jax.numpy as jnp
from jax import Array
from typing import TYPE_CHECKING

from .basis import Basis

if TYPE_CHECKING:
    from .unitary import Unitary


class Hamiltonian:
    """A Hamiltonian defined by a Lie-algebra basis and coefficient vector.

    Constructs $H = \\sum_k \\phi_k B_k$ and its matrix exponential
    $U = e^{iH}$.

    Attributes:
        basis: The `Basis` object defining the algebra.
        parameters: Coefficient vector $\\boldsymbol{\\phi}$.
        matrix: The Hamiltonian matrix $H$.
        unitary: A `Unitary` wrapping $U = e^{iH}$.
    """

    def __init__(self, basis: Basis, parameters: np.ndarray) -> None:
        """Initialise a Hamiltonian.

        Args:
            basis: A ``Basis`` instance defining the Lie algebra.
            parameters: Coefficient ``np.ndarray`` of same length as the basis.
        """
        from .unitary import Unitary  # deferred — avoids circular import at module load

        self.basis = basis
        self.parameters = parameters
        self.matrix = self._matrix()
        self.unitary = Unitary(spla.expm(1.0j * self.matrix))

    def geodesic_hamiltonian(self, target_unitary: np.ndarray) -> Hamiltonian:
        """Compute the geodesic Hamiltonian towards a target unitary.

        Args:
            target_unitary: The target unitary ``np.ndarray``.

        Returns:
            A ``Hamiltonian`` whose exponentiation yields the geodesic
            rotation from ``self.unitary`` to ``target_unitary``.
        """
        g = -1.0j * spla.logm(self.unitary.matrix.conj().T @ target_unitary)
        g_params = Hamiltonian.parameters_from_hamiltonian(g, self.basis)
        return Hamiltonian(self.basis, g_params)

    def fidelity(self, unitary_matrix: np.ndarray) -> Array:
        """Compute the fidelity of this Hamiltonian's unitary to a target.

        Args:
            unitary_matrix: The target unitary ``np.ndarray``.

        Returns:
            Scalar fidelity ``Array`` in $[0, 1]$.
        """
        from .unitary import Unitary  # deferred — avoids circular import at module load

        return Unitary.unitary_fidelity(self.unitary.matrix, unitary_matrix)

    @staticmethod
    def parameters_from_hamiltonian(
        hamiltonian: np.ndarray, basis: Basis
    ) -> np.ndarray:
        """Extract Lie-algebra coefficients from a Hamiltonian matrix.

        Args:
            hamiltonian: The Hamiltonian ``np.ndarray`` of shape ``(d, d)``.
            basis: The ``Basis`` to decompose into.

        Returns:
            A real-valued parameter ``np.ndarray`` of length
            ``basis.lie_algebra_dim``.
        """
        return np.real(np.einsum("ijk, kj->i", basis.basis, hamiltonian)) / (
            len(hamiltonian[0])
        )

    def _matrix(self) -> np.ndarray:
        """Build the Hamiltonian matrix via einsum contraction."""
        return np.einsum("ijk,i->jk", self.basis.basis, self.parameters)
