"""The tangent bundle: fibre coordinates and the chart's differentials.

Everything here is **ambient-side**: the two frames a run is described by, and
the two differentials of the chart. The *geometry* of $T_x\\mathcal M$ — the
metric, the coefficient map, the logarithm — is the
`geope.geometry.manifold.Manifold`'s, because in general all of it depends on the
base point. That split is why this module depends on nothing but JAX.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

import numpy as np

import jax
import jax.numpy as jnp
from jax import Array

# TODO: can we just do this on __init__ and in conf test?
jax.config.update("jax_enable_x64", True)

if TYPE_CHECKING:
    from .lie.basis import Basis


@dataclass(frozen=True, eq=False)
class TangentBundle:
    r"""The fibre coordinates and the chart's differentials.

    **Data, not operations.** The metric and the coefficient map are the
    `geope.geometry.manifold.Manifold`'s, because in general both depend on the
    base point; what is left here is what a chart and a coordinate choice are
    *made of*.

    The two `geope.geometry.lie.Basis` fields play different roles and are
    different sizes. ``frame`` is the **ambient coefficient frame**: the basis a
    manifold's `Manifold.coefficients` resolves a tangent vector against, i.e.
    ``params.basis``. ``generators`` is the chart's own, smaller sub-basis (the
    proj+drift one), from which ``jacobian`` and ``hvp`` are built. Both are
    optional, and for unrelated reasons — see each below.

    Attributes:
        frame: The ambient coefficient frame, or ``None``. ``None`` is not a
            degraded mode: it is the right answer for a manifold whose fibre
            coordinates are a real/imaginary split of the ambient array rather
            than a resolution against a matrix frame — which is both Stiefel
            manifolds, and which is also *cheaper*, $2Nm$ coefficients against a
            matrix frame's $d^2$. It means no projector is ever built. The
            projector itself, and the >5-qubit on-the-fly switch, belong to the
            manifold that reads this (see
            `geope.geometry.lie.groups.MatrixLieGroup`), because which frame is
            needed is a property of its `Manifold.coefficients`.
        jacobian: The pushforward of the chart, ``phi -> dPoint/dphi``, of shape
            ``(*ambient_shape, G, K_free)``. ``None`` when unbound.
        hvp: The chart's second differential ``(phi, p) -> (point, V, W)`` with
            $V = \\mathrm D\\Phi_\\phi[p]$ and $W = \\mathrm D^2\\Phi_\\phi[p, p]$, built by
            `Manifold.bind`. ``None`` disables the curvature tier of a
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

    frame: Basis | None = None
    jacobian: Callable[[Array], Array] | None = None
    hvp: Callable[[Array, Array], tuple[Array, Array, Array]] | None = None
    generators: Basis | None = None
    columns: np.ndarray | None = None

    def __post_init__(self) -> None:
        """The one co-invariant: the analytic HVP exists exactly when the generators do.

        The HVP *is* the propagator recursion in those generators, so a chart
        built from them has both and a ``param_transform`` chart has neither.
        `Manifold.bind` sets the pair in one branch, but this is reachable — the
        tests build bundles directly — so it keeps a hand-built one honest.
        """
        if (self.generators is None) != (self.hvp is None):
            raise ValueError(
                "`generators` and `hvp` must be given together: the analytic HVP "
                "is the propagator recursion in those generators, so a chart with "
                "them has both and a `param_transform` chart has neither."
            )

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
