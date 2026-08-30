# API Reference

The GEOPE library is organised into the following modules:

| Module | Description |
| --- | --- |
| [parameters](parameters.md) | The `Parameters` state object — the central hub of the Basis → Parameters → Optimizer pipeline, and the owner of the problem's (lazily built, cached) `Manifold`. |
| [geope](geope.md) | The top-level `Geope` optimiser for geodesic quantum gate synthesis. |
| [gecko](gecko.md) | The `Gecko` null-space ("auxiliary cost") optimiser for refining GEOPE solutions. |
| [geometry](geometry.md) | The geometry layer: the `Manifold` interface, its `TangentBundle`, and the per-step `GeometricContext` every optimisation step reads. |
| [manifolds](manifolds.md) | The spaces available to synthesise on: the matrix Lie groups, and the Stiefel family (state preparation). |
| [chart](chart.md) | The pulse model: the product-of-exponentials chart every manifold is coordinatised by, and its two Jacobians. |
| [hessian](hessian.md) | Second derivatives of the chart: the manual propagator HVP/Hessian, the autodiff pair, and the Riemannian curvature form. |
| [lie](lie.md) | The Lie-algebraic building block: `Basis`, the rank-3 tensor of Hermitian generators. |
| [utils](utils.md) | Utility functions for constructing Pauli/spin-boson bases, optimisation line searches, and other helpers. |
