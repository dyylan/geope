# API Reference

The GEOPE library is organised into the following modules:

| Module | Description |
| --- | --- |
| [parameters](parameters.md) | The `Parameters` state object — the central hub of the Basis → Parameters → Optimizer pipeline. |
| [geope](geope.md) | The top-level `Geope` optimiser and `GeopeEngine` for geodesic quantum gate synthesis. |
| [gecko](gecko.md) | The `Gecko` null-space ("auxiliary cost") optimiser for refining GEOPE solutions. |
| [engine](engine.md) | The base `Engine` for compiling quantum unitaries using Lie-algebraic methods, plus fidelity helpers. |
| [lie](lie.md) | Lie-algebraic building blocks: `Basis`, `Hamiltonian`, and `Unitary`. |
| [utils](utils.md) | Utility functions for constructing Pauli/spin-boson bases, optimisation line searches, and other helpers. |
