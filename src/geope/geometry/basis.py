from __future__ import annotations

from typing import Callable

import numpy as np
import itertools as it
import re

import jax
import jax.numpy as jnp
from jax import Array

from functools import partial

import numpy as np


def traces(b_1: np.ndarray, b_2: np.ndarray) -> Array:
    """Compute the trace inner-product Gram matrix between two basis sets.

    Returns a matrix $G_{ij} = \\mathrm{Tr}(B^{(1)}_i B^{(2)}_j)$ for
    all pairs of basis elements.

    Args:
        b_1: First basis tensor ``np.ndarray`` of shape ``(K1, d, d)``.
        b_2: Second basis tensor ``np.ndarray`` of shape ``(K2, d, d)``.

    Returns:
        A complex ``Array`` of shape ``(K1, K2)``.
    """
    # Vectorized Gram matrix calculation
    return jnp.einsum("ikl,jlk->ij", jnp.asarray(b_1), jnp.asarray(b_2))


class Basis:
    r"""A frame for the ambient matrix space, and the pulse's generators.

    Wraps a rank-3 tensor of Hermitian basis matrices together with
    associated labels, interaction metadata, and convenience utilities
    for building and manipulating Lie-algebraic decompositions.

    **Two roles, one class.** A `Basis` is a real-orthogonal frame for the
    Hermitian matrices in the ambient space $\mathbb C^{d\times d}$ — the Paulis
    span that space over $\mathbb C$, the Hermitian matrices in it over
    $\mathbb R$, and, multiplied by $i$, the algebra $\mathfrak u(d)$ over
    $\mathbb R$. The pipeline uses that in two different places and at two
    different sizes:

    * as the **chart's generators** — ``params.proj_drift_basis``, the
      controllable sub-frame the pulse is a product of exponentials in
      (`geope.geometry.chart`);
    * as the **ambient coefficient frame** — ``params.basis``, what
      `geope.geometry.manifold.Manifold.coefficients` resolves a tangent vector
      against, stored on `geope.geometry.TangentBundle.frame`. The single factor
      of $i$ that turns a skew-Hermitian algebra element into something this
      frame can resolve is the one in
      `geope.geometry.lie.groups.MatrixLieGroup.coefficients`.

    Completeness is not required in either role: an incomplete frame simply
    projects onto a subspace, which the spin-boson bases rely on. Nor is a
    manifold obliged to coordinatise through a frame at all — both Stiefel
    manifolds use a real/imaginary split of the ambient array instead.

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
        out = traces(self.basis, other.basis)
        return ~np.isclose(np.sum(out, axis=0), 0)

    def verify(self) -> bool:
        """Verify that the basis elements are orthogonal under the trace inner product.

        Returns:
            ``True`` if the trace-inner-product Gram matrix is diagonal,
            ``False`` otherwise.
        """
        out = traces(self.basis, self.basis)
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


def project_omegas(x: Array, basis: Array, dim: int) -> Array:
    """Project a batch of matrices onto a Lie algebra basis.

    Computes the trace inner product of each matrix in `x` with
    each basis element, normalised by the Hilbert-space dimension.

    Args:
        x: ``Array`` of matrices to project, with shape ``(N, d, d)``.
        basis: Basis tensor ``Array`` of shape ``(K, d, d)``.
        dim: Hilbert-space dimension $d$.

    Returns:
        Real ``Array`` of shape ``(N, K)`` of projected coefficients.
    """
    return jnp.real(jnp.einsum("ijk, nkj->ni", basis, x)) / dim


def get_project_omegas_fn(basis: Basis) -> Callable[[Array], Array]:
    """Create a partial projection function with a fixed basis.

    Args:
        basis: A `Basis` instance whose matrices and dimension are bound.

    Returns:
        A ``Callable[[Array], Array]`` that accepts an array of
        matrices and returns the projected coefficients.
    """
    return partial(project_omegas, basis=basis.basis, dim=basis.dim)


def get_kron_chain(n: int) -> Callable[[Array], Array]:
    """Build a JIT-compiled Kronecker product chain function.

    Constructs the four single-qubit Pauli matrices and returns
    a function that builds an $n$-qubit Pauli string via iterated
    Kronecker products.

    Args:
        n: Number of qubits.

    Returns:
        A JIT-compiled ``Callable[[Array], Array]`` that accepts a
        combination index array of length ``n`` (values 0–3) and
        returns the corresponding ``(2^n, 2^n)`` Pauli matrix.
    """
    paulis = jnp.stack(
        [
            jnp.eye(2).astype(complex),
            jnp.array([[0, 1], [1, 0]], complex),
            jnp.array([[0, -1j], [1j, 0]], complex),
            jnp.array([[1, 0], [0, -1]], complex),
        ]
    )

    @jax.jit
    def kron_chain(comb):
        p = paulis[comb[0]]
        for i in range(1, n):
            p = jnp.kron(p, paulis[comb[i]])
        return p

    return kron_chain


def get_project_omegas_fn_otf(
    basis: Basis, batch_size: int | None = None
) -> Callable[[Array], Array]:
    """Create an on-the-fly omega projection function.

    Instead of storing the full basis in memory, Pauli strings are
    constructed on the fly via Kronecker products. Useful when the
    number of qubits exceeds 5.

    Args:
        basis: A `Basis` instance (only ``basis.n`` is used).
        batch_size: Optional number of batches to split the
            Pauli combinations into. If ``None``, a single vmap
            is used.

    Returns:
        A vmapped ``Callable[[Array], Array]`` that accepts a batch
        of matrices and returns projected coefficients.
    """
    n = basis.n
    combs = jnp.array(list(it.product([0, 1, 2, 3], repeat=n))[1:], dtype=jnp.int32)
    kron_chain = get_kron_chain(n)

    @jax.jit
    def projector(c, x):
        pauli = kron_chain(c)
        return jnp.real(jnp.einsum("ij,ji->", pauli, x)) / x.shape[0]

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
