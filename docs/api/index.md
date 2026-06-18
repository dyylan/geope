# API Reference

The GEOPE library is organised into the following modules:

| Module | Description |
| --- | --- |
| [parameters](parameters.md) | The `Parameters` state object — the central hub of the Basis → Parameters → Optimizer pipeline, and the source of the (lazily built, cached) optimisation functions. |
| [geope](geope.md) | The top-level `Geope` optimiser for geodesic quantum gate synthesis. |
| [gecko](gecko.md) | The `Gecko` null-space ("auxiliary cost") optimiser for refining GEOPE solutions. |
| [engine](engine.md) | Pure function factories for the optimisation primitives (unitary, fidelity, geodesic, Jacobian, Hessian, gammas/omegas). |
| [lie](lie.md) | Lie-algebraic building blocks: `Basis`, `Hamiltonian`, and `Unitary`. |
| [utils](utils.md) | Utility functions for constructing Pauli/spin-boson bases, optimisation line searches, and other helpers. |
