# GEOPE

**GEOPE** (Geodesic Pulse Engineering) is a Python library for quantum optimal control and gate synthesis. It implements a new algorithm that uses geodesics on the Riemannian manifold of $SU(2^n)$ and differential programming to design multi-qubit quantum gates with constrained Hamiltonians. Built on [JAX](https://github.com/jax-ml/jax), GEOPE provides JIT-compiled routines for efficient optimisation. 

The package is still in development with many new features and faster implementations on the way soon!

For the full theoretical background, see the paper:

> D. Lewis, R. Wiersema, and S. Bose, *Quantum Optimal Control with Geodesic Pulse Engineering*, [arXiv:2508.16029](https://arxiv.org/abs/2508.16029) (2025).

The package is for non-commercial use, see the licence for details. 
---

## The problem

Designing multi-qubit quantum logic gates under experimental hardware constraints is a central challenge in quantum computing. Given a target unitary $V$ and a set of experimentally accessible Hamiltonian terms $\mathcal{H}$, the goal is to find piecewise-constant control parameters $\mathbf{\Phi} = (\phi_1, \dots, \phi_L)$ such that

$$U_G(\mathbf{\Phi}) = U(\phi_L)\, U(\phi_{L-1})\, \cdots \, U(\phi_1) \approx V,$$

where each $U(\phi_l) = e^{i H(\phi_l)}$ and $H(\phi_l) = \sum_k \phi_{l,k}\, G_k$ is a Hamiltonian restricted to the available interactions $G_k \in \mathcal{H}$.

The standard approach is GRAPE (Gradient Ascent Pulse Engineering), which performs gradient ascent on the fidelity

$$F(\mathbf{\Phi}, V) = \frac{1}{N} \left| \mathrm{Tr}\left\{ U_G^\dagger(\mathbf{\Phi})\, V \right\} \right|.$$

## The GEOPE algorithm

GEOPE takes a fundamentally different approach. Instead of following the gradient of the fidelity, it directly follows the **geodesic** — the shortest path on the $SU(N)$ manifold — from the current unitary $U_G(\mathbf{\Phi})$ to the target $V$. 

The geodesic direction is given by:

$$\Gamma = -i \log\!\left(U_G(\mathbf{\Phi})^\dagger V\right) \in \mathfrak{su}(N).$$

At each iteration, GEOPE solves a **convex least-squares** problem to find the parameter update $\delta\mathbf{\Phi}$ that best aligns the available control directions (the Jacobian) with the geodesic:

$$\mathcal{L}(\delta\mathbf{\Phi}) = \left\| \sum_{l,k} \mathbf{J}_{l,k}(\mathbf{\Phi})\, \delta\phi_{l,k} - i\, U_G(\mathbf{\Phi})\, \Gamma \right\|^2.$$

A golden-section line search then determines the optimal step size along this direction. When the line search fails to improve fidelity (indicating a local minimum), a Gram-Schmidt procedure steps orthogonally to escape. This strategy gives GEOPE two key advantages over GRAPE:

- **Faster convergence**: following the geodesic minimises the distance to the target at each step, rather than merely maximising a local fidelity gradient that may not align with the shortest path.
- **Convex sub-problems**: the update at each step is a linear least-squares problem, avoiding the non-convex landscape traps that can slow or stall GRAPE.

Numerical benchmarks on Rydberg atom platforms show that GEOPE converges to solutions in many times **fewer iterations** than GRAPE across a range of multi-qubit gates (Toffoli, CCZ, QFT), and finds solutions that are out of reach for similar GRAPE implementations.

## Gecko: pulse quality optimisation

A pulse that achieves the target fidelity is rarely unique — there is typically a whole manifold of control parameters $\mathbf{\Phi}$ that realise the same gate. **Gecko** exploits this freedom to refine a fidelity-achieving GEOPE solution without sacrificing fidelity.

Once GEOPE has found a solution, the controllable directions span a tangent subspace whose orthogonal complement — the **Jacobian null space** — is the set of parameter moves that leave the realised unitary (and hence the fidelity) unchanged to first order. Gecko optimises a secondary **auxiliary cost** by moving only within this null space:

$$\delta\mathbf{\Phi} \in \ker \mathbf{J}(\mathbf{\Phi}) \quad\Longrightarrow\quad F(\mathbf{\Phi} + \delta\mathbf{\Phi}, V) \approx F(\mathbf{\Phi}, V).$$

This lets a solution be reshaped to satisfy experimental optimisations that the fidelity alone does not capture:

- **Smoothing** — penalise sharp jumps between piecewise-constant segments (`smooth`, `smooth_frequency`).
- **Pulse length and speed** — shorten the total evolution time or slow the control rate (`length`, `speed`).
- **Robustness** — reduce sensitivity to control errors (`robust`).
- **Bounds** — push parameters into hardware-allowed ranges (`bound`).

Because all of these passes preserve fidelity by construction, they can be composed freely after optimisation. A `Gecko` object either builds its own engine from a `Parameters` object or reuses `Geope` objects engine.

The Gecko method is described in the paper:

> D. Lewis and R. Wiersema, *Pulse Quality Optimisation in Quantum Optimal Control*, [arXiv:2604.25768](https://arxiv.org/abs/2604.25768) (2026).

## Library overview

The library is organised around a few core components:

| Module | Description |
|--------|-------------|
| `Basis`, `Hamiltonian`, `Unitary` | Lie algebraic objects for defining Pauli-string bases, Hamiltonians, and unitaries. |
| `Parameters` | State object bundling basis, control/drift configuration, target, constraints, pulse-shape constraints, and `param_transform`. The single user-facing entry point. |
| `Engine` | Base engine that compiles JAX functions for computing unitaries and fidelities from a given basis. |
| `GeopeEngine` | Extends `Engine` with JIT-compiled Jacobian, geodesic, and projection functions. Built internally by `Geope`. |
| `Geope` | Top-level optimiser that runs the full GEOPE algorithm; requires a `Parameters` object. |
| `Gecko` | Kernel ("auxiliary cost") optimiser that refines a solution — smoothing, pulse length, speed, robustness, bounds — while preserving fidelity. Builds its own engine from a `Parameters`, or reuses a `Geope` engine. |
| `utils` | Utilities for constructing restricted Pauli bases, Heisenberg and 2-local Hamiltonians, line search, and more. |

A typical workflow is:

1. Construct a `Basis` describing the available Hamiltonian interactions.
2. Build a `Parameters` object with that basis, the controllable and drift interactions, and the target unitary.
3. Pass the `Parameters` to `Geope` and call `.optimize()`.

See the [User Guide](user_guide.md) for a complete walkthrough of `Parameters`, pulse-shape constraints, experimental parameters via `param_transform`, and the auxiliary null-space passes (`smooth`, `smooth_frequency`, `speed`, `length`, `robust`, `bound`), which live on `Gecko`. The [Getting Started](examples/getting_started.ipynb) notebook gives a runnable first example.
