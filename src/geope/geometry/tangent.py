"""The tangent bundle: fibre coordinates and the chart's differentials.

Split out so that it depends on nothing but JAX: the
metric and the coefficient map are the `geope.geometry.manifold.Manifold`'s (they
are point-dependent in general), and what is left here is data — which basis
coordinatises the fibres, how to differentiate the chart, and which coefficient
columns the geodesic solve may move.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

import numpy as np

import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_enable_x64", True)

if TYPE_CHECKING:
    from .lie.basis import Basis

    from .parameters import Parameters


@dataclass(frozen=True, eq=False)
class TangentBundle:
    r"""The fibre coordinates and the chart's differentials.

    Data, not operations: the metric and the coefficient map are the
    `geope.geometry.manifold.Manifold`'s, because in general both depend on the
    base point. What is left here is what a chart and a coordinate choice are made
    of, which is why this module depends on nothing but JAX.

    ``basis`` is the **coefficient space** — the basis a manifold's
    `Manifold.coefficients` resolves a tangent vector against, i.e.
    ``params.basis``. ``generators`` is a different, smaller set: the chart's own
    generators (the proj+drift sub-basis), from which ``jacobian`` and `hvp` are
    built. Both are optional, because a manifold without a global frame
    coordinatises its fibres without either.

    Attributes:
        basis: The `Basis` spanning (a subspace of) the tangent space, of shape
            ``(K, d, d)`` and Hermitian. ``None`` for a manifold that does not
            coordinatise through a basis.
        project: Batched projection ``(N, d, d) -> (N, K)`` onto ``basis``
            (`geope.geometry.lie.pauli_projector`; the on-the-fly variant above
            5 qubits). ``None`` with ``basis``.
        jacobian: The pushforward of the chart, ``phi -> dPoint/dphi``, of shape
            ``(*ambient_shape, G, K_free)``. ``None`` when unbound.
        hvp: The chart's second differential ``(phi, p) -> (point, V, W)`` with
            $V = \\mathrm D\\Phi_\\phi[p]$ and $W = \\mathrm D^2\\Phi_\\phi[p, p]$, built by
            `Manifold.chart_hvp`. ``None`` disables the curvature tier of a
            `GeometricContext` and nothing else.
        generators: The chart's generator sub-basis, present exactly when the
            chart is a plain product of exponentials in it. ``None`` under
            ``param_transform``, which is the single signal that disables both
            ``hvp`` (and with it every curvature quantity) and the manual
            propagator Hessian.
        columns: Boolean mask over the chart's coefficient columns selecting the
            ones the geodesic solve may move. ``None`` means "every column is
            free", which is what makes both `restrict` and `embed` no-ops under
            ``param_transform``.
    """

    basis: Basis | None = None
    project: Callable[[Array], Array] | None = None
    jacobian: Callable[[Array], Array] | None = None
    hvp: Callable[[Array, Array], tuple[Array, Array, Array]] | None = None
    generators: Basis | None = None
    columns: np.ndarray | None = None

    @property
    def coefficient_dim(self) -> int | None:
        """The number of basis elements $K$, or ``None`` without a ``basis``."""
        return None if self.basis is None else self.basis.lie_algebra_dim

    def restrict(self, coefficients: Array) -> Array:
        """Keep only the solvable coefficient columns (`columns`).

        Args:
            coefficients: ``Array`` of shape ``(G, K_free, K)``.

        Returns:
            ``(G, K_solvable, K)``, or the input unchanged when ``columns`` is
            ``None``.
        """
        if self.columns is None:
            return coefficients
        return coefficients[:, self.columns, :]

    def embed(self, solution: Array) -> Array:
        """Scatter a solved direction back over every chart column — `restrict`'s inverse.

        Args:
            solution: ``Array`` of shape ``(G, K_solvable)``.

        Returns:
            ``(G, K_free)`` with zeros outside ``columns``, or the input
            unchanged when ``columns`` is ``None``.
        """
        if self.columns is None:
            return solution
        out = jnp.zeros((solution.shape[0], self.columns.size), dtype=solution.dtype)
        return out.at[:, self.columns].set(solution)
