from __future__ import annotations

import numpy as np
import scipy.linalg as spla
import itertools as it
import re
import jax
import jax.numpy as jnp
from jax import Array

from .. import utils


class Basis:
    """A Lie algebra basis for quantum Hamiltonian parameterisation.

    Wraps a rank-3 tensor of Hermitian basis matrices together with
    associated labels, interaction metadata, and convenience utilities
    for building and manipulating Lie-algebraic decompositions.

    Attributes:
        basis: Array of shape ``(K, d, d)`` containing the basis matrices.
        labels: List of Pauli-string labels, e.g. ``['XI', 'ZZ']``.
        plot_labels: LaTeX-formatted labels for plotting.
        interaction_labels: Compact lower-case interaction labels.
        interaction_qubits: Tuple of qubit indices involved in each basis element.
        interaction_graph: List of qubit-pair tuples representing interactions.
        interaction_map: Dictionary mapping qubit tuples to interaction labels.
        n: Number of qubits ($\\log_2 d$).
        local_dim: Local Hilbert-space dimension (default 2).
        dim: Total Hilbert-space dimension $d$.
        lie_algebra_dim: Number of basis elements $K$.
        shape: Shape of the underlying basis tensor ``(K, d, d)``.
    """

    def __init__(
        self,
        basis: np.ndarray,
        labels: list[str] | None = None,
        local_dim: int = 2,
        n_qubits: int | None = None,
        interaction_graph: list[tuple[int, ...]] | None = None,
        interaction_map: dict[tuple[int, ...], list[str]] | None = None,
    ) -> None:
        """Initialise a Basis.

        Args:
            basis: Rank-3 ``np.ndarray`` of shape ``(K, d, d)`` of Hermitian matrices.
            labels: Optional list of string labels for each basis element.
                Defaults to ``None``.
            local_dim: Local Hilbert-space dimension. Defaults to 2.
            n_qubits: Optional override for the number of qubits. Useful
                when the Hilbert-space dimension is not $2^n$ (e.g. a
                direct sum of single-qubit blocks). When ``None`` (the
                default), $n$ is inferred from ``basis.shape[1]`` as
                $\\log_2 d$.
            interaction_graph: Optional list of qubit-index tuples restricting
                which interactions to keep.
            interaction_map: Optional dictionary mapping qubit tuples to lists
                of interaction labels to keep.
        """
        assert basis.ndim == 3, "`basis` must be a rank 3 tensor"
        self._basis = basis
        self._labels = labels if labels is not None else []
        self._n_qubits_override = n_qubits
        self._plot_labels = self._generate_plot_labels()
        self._interaction_labels = self._generate_interaction_labels()
        self._interaction_qubits = self._generate_interaction_qubits()
        self._interaction_graph = (
            self.apply_interaction_graph(interaction_graph)
            if interaction_graph is not None
            else self._generate_interaction_graph()
        )
        self._interaction_map = (
            self.apply_interaction_map(interaction_map)
            if interaction_map is not None
            else self._generate_interaction_map()
        )
        self._local_dim = local_dim
        self._dim = basis.shape[1]
        self._lie_algebra_dim = basis.shape[0]
        if n_qubits is not None:
            self._n = n_qubits
        else:
            self._n = int(np.log2(basis.shape[1]))
        assert self._n

    def linear_span(self, parameters: np.ndarray) -> np.ndarray:
        r"""Compute the linear combination of basis matrices.

        Args:
            parameters: Coefficient ``np.ndarray`` of length ``K``.

        Returns:
            A ``(d, d)`` ``np.ndarray`` equal to $\sum_k \phi_k B_k$.
        """
        parameters = np.reshape(parameters, (-1, 1, 1))
        return np.einsum("nij,nij->ij", parameters, self._basis)

    def overlap(self, other: Basis) -> np.ndarray:
        """Compute the overlap mask between this basis and another.

        Uses trace inner products to determine which elements of
        `other` have non-zero overlap with elements of this basis.

        Args:
            other: Another ``Basis`` instance.

        Returns:
            A boolean ``np.ndarray`` of length ``other.lie_algebra_dim``
            that is ``True`` where an overlap exists.
        """
        out = utils.traces(self.basis, other.basis)
        return ~np.isclose(np.sum(out, axis=0), 0)

    def verify(self) -> bool:
        """Verify that the basis elements are orthogonal under the trace inner product.

        Returns:
            ``True`` if the trace-inner-product Gram matrix is diagonal,
            ``False`` otherwise.
        """
        out = utils.traces(self.basis, self.basis)
        return np.allclose(np.diag(np.diag(out)), out)

    def apply_interaction_graph(
        self, interaction_graph: list[tuple[int, ...]]
    ) -> list[tuple[int, ...]]:
        """Apply an interaction graph to the basis.

        Removes any multi-body basis elements whose qubit indices are not
        present in the supplied interaction graph. Single-body terms are
        always retained.

        Args:
            interaction_graph: A list of tuples or lists, each containing
                the qubit indices of an allowed interaction
                (e.g. ``[(1, 2), (2, 3)]``).

        Returns:
            The applied interaction graph as a list of tuples.
        """
        interaction_graph = [tuple(interaction) for interaction in interaction_graph]
        self._interaction_graph = interaction_graph
        del_indices = []
        for i, interaction in enumerate(self.interaction_qubits):
            if (interaction not in interaction_graph) and (len(interaction) > 1):
                del_indices.append(i)
        self._remove_basis_elements(del_indices)
        return interaction_graph

    def apply_interaction_map(
        self, interaction_map: dict[tuple[int, ...], list[str]]
    ) -> dict[tuple[int, ...], list[str]]:
        """Apply an interaction map to the basis.

        Removes basis elements whose qubit-index tuple is not a key in the
        map, or whose interaction label is not in the corresponding value
        list.

        Args:
            interaction_map: Dictionary mapping qubit-index tuples to lists
                of allowed interaction label strings.

        Returns:
            The applied interaction map dictionary.
        """
        self._interaction_map = interaction_map
        del_indices = []
        for i, interaction in enumerate(self.interaction_qubits):
            if interaction not in interaction_map.keys():
                del_indices.append(i)
            elif self.interaction_labels[i] not in interaction_map[interaction]:
                del_indices.append(i)
        self._remove_basis_elements(del_indices)
        return interaction_map

    def _generate_plot_labels(self) -> list[str] | None:
        """Generate LaTeX-formatted plot labels from string labels."""
        if self.labels:
            new_labels = []
            for label in self.labels:
                new_label = "$"
                for i, c in enumerate(label):
                    new_label += "" if c == "I" else f"{c}_{{{i+1}}}"
                new_label += "$"
                new_labels.append(new_label)
            return new_labels
        else:
            return None

    def _generate_interaction_labels(self) -> list[str] | None:
        """Generate compact lower-case interaction labels from string labels."""
        if self.labels:
            new_labels = []
            for label in self.labels:
                new_label = ""
                for c in label:
                    new_label += "" if c == "I" else f"{c}".lower()
                new_labels.append(new_label)
            return new_labels
        else:
            return None

    def _generate_interaction_qubits(self) -> list[tuple[int, ...]] | None:
        """Extract qubit-index tuples from plot labels."""
        if self.labels:
            interaction_qubits = []
            for label in self.plot_labels:
                qubits = re.findall(r"\d+", label)
                interaction_qubits.append(tuple([int(q) for q in qubits]))
            return interaction_qubits
        else:
            return None

    def _generate_interaction_graph(self) -> list[tuple[int, ...]]:
        """Build the default interaction graph from multi-qubit basis elements."""
        interaction_graph = []
        for interaction in self.interaction_qubits:
            if len(interaction) > 1:
                interaction_graph.append(interaction)
        return interaction_graph

    def _generate_interaction_map(self) -> dict[tuple[int, ...], list[str]]:
        """Build the default interaction map from basis element metadata."""
        interaction_map = {}
        for i, interaction in enumerate(self.interaction_qubits):
            interaction_map.setdefault(interaction, []).append(
                self.interaction_labels[i]
            )
        return interaction_map

    def _remove_basis_elements(self, indices: list[int]) -> bool:
        """Remove basis elements at the given indices.

        Args:
            indices: List of integer indices to remove.

        Returns:
            ``True`` on success.
        """
        for i in sorted(indices, reverse=True):
            self._basis = np.delete(self._basis, i, axis=0)
            del self._labels[i]
            del self._plot_labels[i]
            del self._interaction_qubits[i]
            del self._interaction_labels[i]
        self._lie_algebra_dim = self._basis.shape[0]
        return True

    @property
    def basis(self) -> np.ndarray:
        """The rank-3 array of basis matrices."""
        return self._basis

    @property
    def labels(self) -> list[str] | None:
        """String labels for each basis element, or ``None``."""
        return self._labels if self._labels else None

    @property
    def plot_labels(self) -> list[str] | None:
        """LaTeX-formatted labels for plotting."""
        return self._plot_labels

    @property
    def interaction_labels(self) -> list[str] | None:
        """Compact lower-case interaction labels."""
        return self._interaction_labels

    @property
    def interaction_qubits(self) -> list[tuple[int, ...]] | None:
        """Tuple of qubit indices for each basis element."""
        return self._interaction_qubits

    @property
    def interaction_graph(self) -> list[tuple[int, ...]]:
        """List of qubit-pair tuples representing interactions."""
        return self._interaction_graph

    @property
    def interaction_map(self) -> dict[tuple[int, ...], list[str]]:
        """Dictionary mapping qubit tuples to interaction labels."""
        return self._interaction_map

    @property
    def n(self) -> int:
        """Number of qubits."""
        return self._n

    @property
    def local_dim(self) -> int:
        """Local Hilbert-space dimension."""
        return self._local_dim

    @property
    def dim(self) -> int:
        """Total Hilbert-space dimension."""
        return self._dim

    @property
    def lie_algebra_dim(self) -> int:
        """Number of basis elements."""
        return self._lie_algebra_dim

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the basis tensor ``(K, d, d)``."""
        return self._basis.shape

    def __len__(self) -> int:
        """Return the number of basis elements."""
        return self._basis.shape[0]

    def generate_parameter_list(
        self, parameter_map: dict[int | tuple[int, ...], dict[str, float]]
    ) -> list[float]:
        """Generate a parameter vector from a human-readable parameter map.

        Args:
            parameter_map: Dictionary whose keys are qubit indices (int) or
                qubit-index tuples, and whose values are dictionaries
                mapping interaction labels to parameter values.

        Returns:
            A list of parameter values aligned with the basis ordering.
        """
        parameter_list = []
        for label in self.labels:
            new_label = ""
            qubits = []
            for i, c in enumerate(label):
                if c == "I":
                    new_label += ""
                else:
                    new_label += f"{c}".lower()
                    qubits.append(i + 1)
            qubits = tuple(qubits) if len(qubits) > 1 else qubits[0]

            interactions = parameter_map.get(qubits)
            if interactions is not None:
                param = interactions.get(new_label, 0)
                parameter_list.append(param)
            else:
                parameter_list.append(0)
        return parameter_list

    def generate_bounds(
        self, bounds_map: dict[str, tuple[float, float]], piecewise_steps: int
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Generate lower and upper parameter bounds from a bounds map.

        Args:
            bounds_map: Dictionary whose keys are interaction label strings
                and whose values are ``(min, max)`` tuples.
            piecewise_steps: Number of piecewise gate segments.

        Returns:
            A tuple ``(lower_bounds, upper_bounds)`` where each element is a
            nested list of shape ``(piecewise_steps, K)``.
        """
        upper_bounds = [[] for _ in range(piecewise_steps)]
        lower_bounds = [[] for _ in range(piecewise_steps)]
        for label in self.labels:
            new_label = ""
            qubits = []
            for i, c in enumerate(label):
                if c == "I":
                    new_label += ""
                else:
                    new_label += f"{c}".lower()
                    qubits.append(i + 1)
            qubits = tuple(qubits) if len(qubits) > 1 else qubits[0]

            bounds = bounds_map.get(new_label)
            for gate in range(piecewise_steps):
                if bounds is not None:
                    index = len(lower_bounds[gate])
                    lower_bounds[gate].append(bounds[0])
                    upper_bounds[gate].append(bounds[1])
                else:
                    lower_bounds[gate].append(-jnp.inf)
                    upper_bounds[gate].append(jnp.inf)
        return lower_bounds, upper_bounds
