"""The geometry layer: the manifold, its tangent bundle, and their primitives.

GEOPE synthesises a target by walking a Riemannian manifold. This package owns
that manifold and its tangent bundle, so that nothing else has to:

```
Manifold  --owns-->  compute_point      (the chart Phi: R^(G x K) -> M)
   |                 target         (the point being synthesised)
   |
   '-----owns-->  TangentBundle --owns-->  Basis      (coordinatises the fibres)
                                          jacobian   (the pushforward D Phi)
                                          hvp        (the second differential D^2 Phi)
```

The chart maps into the manifold; its differentials map into the tangent bundle.
A `Manifold` is usable in two states: constructed with only its dimension
(``SpecialUnitaryGroup(4)``) it is the pure space, and every geometric primitive
already works — `Manifold.log`, `Manifold.distance2`, `Manifold.fidelity`,
`Manifold.hessian_quadratic_form`. `Manifold.bind` attaches the chart, the
tangent bundle and the target, which is what a per-step `GeometricContext` needs.

Modules:

* `manifold` — the `Manifold` interface: what the geodesic algorithm needs of a
  space, with no Lie-group assumption baked in.
* `tangent` — the `TangentBundle`: fibre coordinates and the chart's
  differentials.
* `context` — the `GeometricContext`: every per-step quantity, in cost tiers.
* `binding` — `bind_manifold`, the one place that knows both `Parameters` and the
  geometry layer.
* `lie` — matrix Lie groups: the `Basis` of Hermitian generators, and the
  `MatrixLieGroup` middle layer with its `UnitaryGroup` / `SpecialUnitaryGroup`.
* `stiefel` — the Stiefel family: `Stiefel` (orthonormal $m$-frames under the
  canonical metric, with the iterative Zimmermann–Hüper logarithm) and
  `StateSphere` (state preparation, and
  the proof that this interface needs no group structure), frames later.

**Import invariant: nothing in this package may import `geope.utils` at module
level.** `geope.utils` imports `geometry.lie.basis` for its basis constructors,
so a module-level import back would close the cycle. Defer it into the function
body if it is ever unavoidable — the idiom `lie/hamiltonian.py` uses for its own
deferred imports.
"""

from .context import GeometricContext
from .manifold import Manifold
from .tangent import TangentBundle
from .lie.groups import MatrixLieGroup, SpecialUnitaryGroup, UnitaryGroup
from .stiefel.sphere import StateSphere
from .stiefel.stiefel import Stiefel
