r"""The geometry layer: an ambient space, the submanifolds in it, and their primitives.

GEOPE synthesises a target by walking a Riemannian manifold, and **every manifold
here is a submanifold of one ambient space** $\mathcal A = \mathbb C^{N\times m}$:

| base point $x_0$ | the $\mathrm U(N)$-orbit through it |
| --- | --- |
| $\mathbb 1_N$ | $\mathrm U(N)$, $\mathrm{SU}(N)$ |
| $(\mathbb 1_m;\,0)^\intercal$ | $\mathrm{St}_m(\mathbb C^N)$ |
| $\lvert\psi_0\rangle$ | $S^{2N-1} \to \mathbb{CP}^{N-1}$ |

That is the organising idea, and it draws the one line this package is built
around. The pulse acts on all of $\mathcal A$ the same way — by left
multiplication with a product of exponentials — so everything *valued in*
$\mathcal A$ is shared, and everything that happens *inside a fibre*
$T_x\mathcal M$ is the submanifold's:

```
ambient  A = C^{N×m}     Phi(phi) = U(phi) x_0 ,  D Phi ,  D^2 Phi      chart.py
                         the coefficient frame (Paulis, or the standard one)
     │
     │   to_tangent :  A -> T_x M          <- the only bridge
     ▼
submanifold  M ⊂ A       inner, coefficients, log, tangent_acceleration,
                         hessian_quadratic_form, fidelity, infidelity   Manifold
```

Ownership follows that line exactly:

```
Manifold  --owns-->  base_point     (x_0 = Phi(0): where the orbit starts)
   |                 target         (the point being synthesised)
   |                 compute_point  (the chart Phi: R^(G x K) -> M)
   |
   '-----owns-->  TangentBundle --owns-->  frame      (the ambient coefficient frame)
                                          jacobian   (the pushforward D Phi)
                                          vjp        (the pullback D Phi^T)
                                          hvp        (D^2 Phi along one direction)
                                          hessian    (the dense D^2 Phi)
```

The whole jet comes from the propagator recursions in `geope.jax`, not from
autodiff: `geope.geometry.chart` builds them and lands them on the base point,
and the `Manifold` assembles the *objective's* gradient and Hessian from them
plus its own `Manifold.cost_gradient` and `Manifold.cost_hessian_form`. Autodiff
survives in exactly two places — the ``param_transform`` chart, whose
user-supplied transform must be differentiated as it stands, and the references
the manual paths are tested against.

A `Manifold` is usable in two states: constructed with only its dimension
(``SpecialUnitaryGroup(4)``) it is the pure space, and its geometric primitives
already work — `Manifold.log`, `Manifold.distance2`, `Manifold.inner`,
`Manifold.fidelity`, `Manifold.hessian_quadratic_form`. `Manifold.bind` then
attaches the chart, the frame and the target, which is what a per-step
`GeometricContext` needs. (`Manifold.coefficients` is the one primitive that
needs binding, because a *coordinate choice* is problem data.)

There is no chart hook, and no chart code on any manifold: the chart is
``Phi(phi) = U(phi) x_0`` everywhere, so only ``base_point`` varies — ``None`` on
a matrix group, where the propagator *is* the point. `bind` composes what
`geope.geometry.chart` builds and holds no mathematics of its own.

Modules:

* `chart` — the **ambient layer**: the pulse model and the whole jet
  $(\Phi, \mathrm D\Phi, \mathrm D^2\Phi)$, landed on a base point.
* `manifold` — the `Manifold` interface: what the geodesic algorithm needs of a
  submanifold, with no Lie-group assumption baked in.
* `tangent` — the `TangentBundle`: the ambient coefficient frame and the chart's
  differentials.
* `context` — the `GeometricContext`: every per-step quantity, in cost tiers.
* `basis` — the `Basis` of Hermitian generators and the coefficient projector.
  It is the **ambient** frame, not a Lie-specific one — both Stiefel manifolds
  take one too — which is why it sits here rather than under `lie`.
* `lie` — matrix Lie groups: the `MatrixLieGroup` middle layer with its
  `UnitaryGroup` / `SpecialUnitaryGroup`.
* `stiefel` — the Stiefel family: `Stiefel` (orthonormal $m$-frames under the
  canonical metric, with the iterative Zimmermann–Hüper logarithm) and
  `StateSphere` (state preparation, and
  the proof that this interface needs no group structure), frames later.

**Import invariant: nothing in this package may import `geope.utils` at module
level.** `geope.utils` imports `geometry.basis` for its basis constructors,
so a module-level import back would close the cycle. Defer it into the function
body if it is ever unavoidable — the idiom `lie/hamiltonian.py` uses for its own
deferred imports.
"""

from .basis import Basis, traces, get_project_omegas_fn, get_project_omegas_fn_otf
from .context import GeometricContext
from .manifold import Manifold
from .tangent import TangentBundle
from .lie.groups import MatrixLieGroup, SpecialUnitaryGroup, UnitaryGroup
from .stiefel.sphere import StateSphere
from .stiefel.stiefel import Stiefel
