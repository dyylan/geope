# GEOPE – Geodesic Pulse Engineering

## Overview

`geope` finds piecewise-constant control pulses that implement a target quantum gate on an $n$-qubit system. Given a target unitary $U_T$, a set of available control generators (the projected basis), and optionally fixed drift generators, the optimiser searches for real-valued parameters $\phi$ such that

$$
U(\phi) \;=\; \prod_{g=1}^{N_g} \exp\!\Bigl(i \sum_{k}\phi_{g,k}\,G_k\Bigr) \;\approx\; U_T,
$$

where each $H_g = \sum_k \phi_{g,k}\,G_k$ is a linear combination of basis generators on segment $g$.

The core algorithm is the **geodesic method**: at each step it computes the shortest path on $U(d)$ from the current unitary to the target, projects that direction onto the controllable subspace, then solves a convex least-squares problem and a one-dimensional line search to take a parameter step. This is distinct from gradient-based methods like GRAPE that follow the fidelity gradient directly.

The entry point is `Parameters` — a state object that bundles every input the optimiser needs (basis, control, drift, target, constraints, pulse constraints, `param_transform`, bounds, init values, seed, projective flag). Pass it to `Geope` and call `.optimize()`. The same `Parameters` object is the destination for the run history (`fidelities`, `parameters`, `best_fidelity`, `to_dict()`, ...).

The lower-level classes `Engine` and `GeopeEngine` are still available for inspection and for advanced users who want to build the JIT-compiled functions directly, but `Geope` itself only accepts a `Parameters`.

## Class hierarchy

```
Basis, Hamiltonian, Unitary      (lie.py)

Engine                           (engine.py)
  └── GeopeEngine                (geope.py)

Parameters                       (parameters.py)
                ↘
                  Geope          (geope.py)
                ↗
GeopeEngine
```

## Lie group classes (`lie.py`)

### `Basis`

Represents a set of Lie algebra generators (e.g. Pauli strings) as a rank-3 tensor of shape $(K, d, d)$.

```python
Basis(basis, labels=None, local_dim=2, n_qubits=None,
      interaction_graph=None, interaction_map=None)
```

| Parameter | Description |
|-----------|-------------|
| `basis` | `np.ndarray` of shape `(K, 2ⁿ, 2ⁿ)` — Hermitian generators |
| `labels` | list of Pauli-string labels, e.g. `["XI", "IX", "ZZ"]` |
| `local_dim` | local Hilbert-space dimension, default 2 |
| `n_qubits` | override for qubit count when $d \neq 2^n$ |
| `interaction_graph` | list of qubit tuples to keep, e.g. `[(1,2), (2,3)]` |
| `interaction_map` | dict of qubit-tuple → allowed interaction labels |

Key properties:

| Property | Description |
|----------|-------------|
| `basis` | the `(K, d, d)` tensor |
| `lie_algebra_dim` | $K$ — number of generators |
| `dim` | $d$ — matrix dimension |
| `n` | number of qubits |
| `labels` | string labels |
| `plot_labels` | LaTeX strings, e.g. `"$X_{1}Z_{2}$"` |
| `interaction_qubits` | tuple of qubit indices for each generator |
| `interaction_graph`, `interaction_map` | as above |

Key methods:

- `overlap(other)` — boolean mask over `other`'s basis, true where there is nonzero trace overlap with `self`. Used by `Engine` to build index masks.
- `verify()` — orthogonality check under the trace inner product.
- `linear_span(parameters)` — $\sum_i \phi_i G_i$.
- `generate_parameter_list(parameter_map)` — converts a dict like `{1: {"x": 0.5}, (1,2): {"zz": 0.3}}` to a flat parameter array.
- `generate_bounds(bounds_map, piecewise_steps)` — converts `{"x": (-1, 1)}` to `(lower, upper)` arrays.
- `apply_interaction_graph(graph)` / `apply_interaction_map(map)` — prune to hardware connectivity.

### `Hamiltonian`

Represents $H = \sum_i \phi_i G_i$ and its unitary $U = e^{iH}$.

```python
Hamiltonian(basis, parameters)
```

| Attribute | Description |
|-----------|-------------|
| `basis` | the `Basis` object |
| `parameters` | coefficient vector $\phi$ |
| `matrix` | $\sum_i \phi_i G_i$ |
| `unitary` | `Unitary(expm(i·matrix))` |

Methods:

- `geodesic_hamiltonian(target_unitary)` — a `Hamiltonian` whose parameters are the geodesic direction $-i\log(U^\dagger U_T)$ decomposed in the basis.
- `fidelity(unitary_matrix)` — $|\mathrm{Tr}(U^\dagger V)|/d$.
- `parameters_from_hamiltonian(H, basis)` (static) — coefficients via $\mathrm{Re}\,\mathrm{Tr}(G_i H)/d$.

### `Unitary`

Wraps a unitary matrix with validation. Validates $UU^\dagger = I$ on construction.

- `parameters(basis)` — Lie-algebra coefficients via the principal `logm`.
- `fidelity(other)`.
- `geodesic_hamiltonian(basis, target)`.

## Basis construction utilities (`utils.py`)

| Function | Description |
|----------|-------------|
| `construct_full_pauli_basis(n)` | all $4^n - 1$ non-identity Pauli strings |
| `construct_two_body_pauli_basis(n)` | 1-body and 2-body terms only |
| `construct_Heisenberg_pauli_basis(n)` | 1-body + same-type 2-body (XX, YY, ZZ) |
| `construct_restricted_pauli_basis(n, restriction)` | custom restriction (list or dict) |
| `construct_full_spin_boson_basis(n_spins, n_bosons, truncation)` | spin-boson hybrid |
| `construct_restricted_spin_boson_basis(...)` | restricted spin-boson |
| `filter_basis_by_control(basis, control)` | filter an existing `Basis` by a control dict (handy when $d \neq 2^n$) |

### Restriction formats

`construct_restricted_pauli_basis` accepts two formats.

**List** — allowed interaction types as lower-case strings:

```python
control = geope.construct_restricted_pauli_basis(2, ['x', 'z'])
control = geope.construct_restricted_pauli_basis(3, ['x', 'y', 'z'])
drift   = geope.construct_restricted_pauli_basis(3, ['zz'])
```

**Dict** — allowed interactions per qubit or qubit pair (1-indexed):

```python
control = geope.construct_restricted_pauli_basis(2, {1: ['x'], 2: ['x'], (1,2): ['zz']})
```

### Drift parameter values

Drift coefficients are specified via `generate_parameter_list` on the drift basis (or passed directly through `Parameters(drift_values=...)`):

```python
drift_basis = geope.construct_restricted_pauli_basis(3, ['zz'])
drift_values = drift_basis.generate_parameter_list({
    (1, 2): {"zz": 1.0},
    (2, 3): {"zz": 1.0},
    (1, 3): {"zz": 1.0},
})
# → [1.0, 1.0, 1.0] matching basis order ["ZZI", "ZIZ", "IZZ"]
```

### Linear equality constraints (global controls)

Constraints enforce that selected projected parameters maintain fixed ratios. Use `generate_parameter_list` to build constraint vectors:

```python
control = geope.construct_restricted_pauli_basis(3, ['x', 'z'])

global_x = control.generate_parameter_list({1: {"x": 1}, 2: {"x": 1}, 3: {"x": 1}})
# → ties X₁ = X₂ = X₃

global_z = control.generate_parameter_list({1: {"z": 1}, 2: {"z": 1}, 3: {"z": 1}})
# → ties Z₁ = Z₂ = Z₃
```

Pass via `Parameters(constraints=[global_x, global_z], ...)`.

### Pulse-shape constraints

Pulse constraints fix the relative values of specified parameters across piecewise steps — the temporal shape is frozen while the overall scale is optimised. This is an alternative to the `drift_basis` + `drift_values` route when you want a drift-like term whose amplitude is still tuned by the optimiser.

```python
projected = geope.construct_restricted_pauli_basis(3, ['x', 'z', 'zz'])
params = geope.Parameters(
    basis=geope.construct_full_pauli_basis(3),
    control={1: ['x', 'z'], 2: ['x', 'z'], 3: ['x', 'z'],
             (1, 2): ['zz'], (2, 3): ['zz'], (1, 3): ['zz']},
    target=U_T,
    piecewise_steps=10,
    pulse_constraints=["ZZI", "ZIZ", "IZZ"],
)
```

| Approach | Use case |
|----------|----------|
| `drift_basis` + `drift_values` | drift is truly fixed and not optimised |
| `pulse_constraints` on projected params | drift-like terms whose amplitude is optimised but whose temporal profile is fixed |

Other utilities:

- `prepare_random_parameters(proj_indices, expander, spread, seed)` — random initial parameters respecting constraints.
- `golden_section_search(f, a, b, tol)` — JIT-compatible 1-D **minimiser** (used internally by the line search).
- `merge_constraints(constraints)` — merges overlapping linear constraints.
- `qft_unitary(n)`, `multicontrol_unitary(U, n_controls)` — common target unitaries.
- `make_per_element_transform(transforms)` — helper to build a `param_transform` from per-element callables.

## The `Parameters` object

`Parameters` is the recommended entry point.

```python
Parameters(basis=None, control=None, drift=None,
           init_values=None, drift_values=None,
           target=None, piecewise_steps=1, fixed_drift=True,
           constraints=None, pulse_constraints=None, bounds=None,
           init_spread=0.1, seed=None,
           param_transform=None, n_experimental_params=None,
           projective=True)
```

| Parameter | Description |
|-----------|-------------|
| `basis` | the full `Basis`; defaults to 2-qubit full Pauli basis if `None` |
| `control` | dict picking the projected (controllable) subset |
| `drift` | dict picking the drift subset |
| `init_values` | dict in `control` format, or `ndarray` of full-basis shape, or `None` (random) |
| `drift_values` | dict, `ndarray`, or `None` (ones) |
| `target` | target unitary as `ndarray` |
| `piecewise_steps` | number of gate segments $N_g$ |
| `fixed_drift` | whether drift is held fixed during optimisation |
| `constraints` | list of constraint vectors / dicts |
| `pulse_constraints` | list (or dict; values ignored) of projected-basis labels whose time-shape is fixed |
| `bounds` | dict `{label: (lo, hi)}` — consumed by `Geope.bound(...)`, not by the main loop |
| `init_spread` | half-width of uniform random init, in units of $\pi$ |
| `seed` | random seed |
| `param_transform` | callable mapping experimental params to basis coefficients |
| `n_experimental_params` | length of the experimental input; defaults to `projected_basis.lie_algebra_dim` |
| `projective` | `True` (default) for projective fidelity, `False` for phase-sensitive |

Attributes populated after construction:

| Attribute | Description |
|-----------|-------------|
| `basis`, `projected_basis`, `drift_basis` | the three `Basis` objects |
| `target` | the target |
| `init_parameters` | initial parameter array, shape `(N_g, K_{full})` |
| `drift_parameters` | drift coefficients (or `None`) |
| `constraint_arrays`, `constraint_expander` | merged constraints and reduced-space mapping |
| `bounds` | pre-built bounds tuple (or `None`) |

Mutable history written back by `Geope`:

| Attribute | Description |
|-----------|-------------|
| `parameters` | list of parameter arrays, shape `(N_g, K_{full})` per entry |
| `fidelities`, `infidelities`, `step_sizes`, `steps` | history scalars |
| `best_fidelity` | `max(fidelities)` |
| `best_parameters` | parameters at the highest-fidelity step |
| `best_basis_coefficients` | best parameters mapped through `param_transform` if set |
| `to_dict()` | best solution as a control-style dict |

## Engine and optimiser

### `Engine` (`engine.py`)

```python
Engine(target_unitary, full_basis, projected_basis,
       drift_basis=None, piecewise_steps=1)
```

Computes index masks between the three bases and JIT-compiles the unitary and fidelity functions.

Index masks (boolean arrays):

- `projected_indices` — shape $(K_{\text{full}},)$, which full-basis elements are controllable (`projected_basis.overlap(full_basis)`).
- `drift_indices` — shape $(K_{\text{full}},)$, which are fixed drift.
- `proj_drift_indices` = `projected_indices | drift_indices`.
- `proj_indices_projdrift_basis` — projected mask within the proj+drift subspace, shape $(K_{\text{pd}},)$.
- `drift_indices_projdrift_basis` — drift mask within the proj+drift subspace.

Derived basis:

- `proj_drift_basis` — `Basis` containing only the projected + drift elements; used for all JIT computations.

JIT functions:

- `compute_U_fn(params_list)` — scans over piecewise steps via `jax.lax.scan`:
  $\,U = \prod_g \exp\!\bigl(i \sum_i \phi_{g,i}\,G_i\bigr).$
  Input shape: $(N_g, K_{\text{pd}})$.
- `fid_U_fn(U)` — $|\mathrm{Tr}(U_T^\dagger U)|/d$ when `projective=True`, $\mathrm{Re}\,\mathrm{Tr}(U_T^\dagger U)/d$ when `projective=False`.

### `GeopeEngine` (`geope.py`)

```python
GeopeEngine(target_unitary, full_basis, projected_basis,
            drift_basis=None, piecewise_steps=1,
            batch_size=None, projective=True)
```

Extends `Engine` with geodesic-specific JIT functions:

- `project_omegas_fn` — projects matrices onto the full Pauli basis via trace inner products $\mathrm{Tr}(G_i M)$. For $n > 5$ uses on-the-fly batched projection (`batch_size`) to manage memory.
- `jac_fn` — Jacobian $\partial U/\partial\phi_{g,k}$. For $n \le 5$, JAX autodiff with `holomorphic=True`. For $n > 5$, manual block-exponential derivative via `dexpm.py`.
- `geo_fn` — geodesic tangent at $U$: $U \cdot \bigl(-i\log(U^\dagger U_T)\bigr)$, with the global-phase generator subtracted when `projective=True`.
- `infid_U_fn` — bound to $1 - F_{\text{proj}}$ or $1 - F_{\text{full}}$ to match `projective`.
- `infid_fn`, `grad_fn` — infidelity of $\phi$ and its gradient (used by fallback methods and by the line search).

### `Geope` (`geope.py`)

```python
Geope(params,
      max_steps=1000, precision=0.9999999,
      max_step_size=0.9, gram_schmidt_step_size=1.3,
      line_search_method="golden_section",
      verbose=False)
```

`Geope` requires a `Parameters` object as its single positional argument. The engine, initial parameters, drift, constraints, pulse constraints, seed, initialisation spread, projective flag and `param_transform` are all read from `params`. Passing a raw `GeopeEngine` raises `TypeError`.

| Parameter | Description |
|-----------|-------------|
| `params` | a `Parameters` instance bundling all inputs |
| `max_steps` | iteration cap |
| `precision` | target fidelity |
| `max_step_size` | maximum line-search step |
| `gram_schmidt_step_size` | step size for the Gram–Schmidt fallback |
| `line_search_method` | `"golden_section"` or `"difference_step"` |
| `verbose` | print per-step progress |
| `seed` | random seed (legacy API) |

State tracked across iterations:

- `parameters` — list of arrays of shape $(N_g, K_{\text{full}})$.
- `fidelities`, `infidelities`, `step_sizes`, `steps` — history lists.

These lists are mirrored onto the `Parameters` object after every `optimize()`, and `optimize()` returns the `Parameters` instance itself — so the user has a single handle for both inputs and outputs.

## Core algorithm: `optimize()`

```
for each step:
    1. Extract free_params = parameters[:, proj_drift_indices]

    2. Compute the geodesic direction:
       U  = compute_U_fn(free_params)
       g  = -i · logm(U† U_T)                       # generator in u(d)
       g  = g - Tr(g)/d · I        if projective    # drop global-phase generator
       Γ  = U · g                                   # geodesic tangent
       γ  = project(Γ) / d                          # coefficients in basis

    3. Compute the Jacobian projections:
       ω[g, k] = project(i · ∂U/∂φ_{g,k})

    4. Solve the constrained least-squares problem:
       sol = argmin ||ω^T · sol - γ||
       (optionally through a constraint+pulse expander E)

    5. Normalise and line-search:
       coeffs = sol · sqrt(N_g) / ||sol||
       dt     = argmin infid(φ + t · coeffs)        # over t ∈ [-t_max, 0]
       φ_new  = φ + dt · coeffs

    6. If fidelity decreased, Gram–Schmidt fallback:
       proj_c = random_direction ⊥ coeffs
       try ±proj_c, keep the side with higher fidelity
```

The line search interval $[-t_{\max}, 0]$ is the toward-target half-line under the algorithm's sign convention: solving $\omega^\top \cdot \mathrm{sol} = \gamma$ orients `coeffs` such that negative `dt` reduces infidelity. The minimiser operates on `infid_U_fn`, which is always non-negative, so the search is well-defined in both `projective=True` and `projective=False` modes. `Geope` reports `fidelity = 1 - infid` at the chosen step.

### Key functions

- **`gammas_and_omegas(free_params)`** — per-iteration core. Computes the unitary, the geodesic Hamiltonian, the projection $\gamma$, the full Jacobian $\partial U/\partial\phi$, and the per-parameter projections $\omega$. Returns $(\gamma, \omega)$.
- **`linear_comb_projected_coeffs_multigate(ω, γ, E)`** — least-squares solve, optionally through a constraint expander $E$.
- **`update_linesearch(params, coeffs, piecewise_steps)`** — golden-section minimisation of $\mathrm{infid}(\phi + t \cdot \mathrm{coeffs})$ over $t \in [-t_{\max}, 0]$.

## Constraints

### Linear equality constraints

`constraints` (or `Parameters.constraints`) takes a list of vectors $c$ of length $K_{\text{proj}}$ enforcing $c \cdot \phi^{\text{proj}} = 0$, or dicts in `control` format that are converted into such vectors. Internally, overlapping constraints are merged via an expander matrix $C$ that maps free parameters to the full projected space, and the least-squares solve becomes $\min \|\omega^\top C \tilde c - \gamma\|$ with $c = C \tilde c$.

### Pulse-shape constraints

`pulse_constraints` fixes the relative shape of selected parameters across piecewise steps. For each constrained label $k$ the time profile $\phi_k(g)$ is constrained to a one-dimensional subspace:

$$
\phi_k(g) \;=\; \alpha_k\, t_k(g), \qquad \|t_k\| = 1, \quad \alpha_k \in \mathbb{R}.
$$

The template $t_k$ is read off the current solution at the moment `optimize()` is called (or the flat template $\mathbf{1}/\sqrt{N_g}$ if the column is empty), and after every iteration $\phi_k$ is re-projected:

$$
\phi_k \;\leftarrow\; \bigl(\phi_k \cdot t_k\bigr) t_k.
$$

Formally, the flat parameter vector $\Phi \in \mathbb{R}^{N_g K_{\text{proj}}}$ is replaced by free parameters $\psi$ via $\Phi = E\,\psi$, where $E$ has $N_g$ identity columns for each unconstrained $k$ and a single template column for each constrained $k$. When combined with a linear-equality expander $C$, the combined expander is $E_{\text{comb}} = (I_{N_g} \otimes C)\,(I_{N_g} \otimes C)^{+}\,E$.

With `param_transform`, pulse constraints reference parameter **indices** in $\phi^{\text{exp}}$ rather than projected-basis labels.

## Experimental parameters (`param_transform`)

GEOPE's native parameters are basis coefficients $\phi^{\text{proj}}_{g,k}$. In practice the **experimentally controllable** quantities are often different — an amplitude–phase pair driving two basis elements through $\cos/\sin$, a small set of pulse-shape coefficients, a calibration map $\phi^{\text{proj}} = f(\text{voltage}, \text{frequency})$. `param_transform` lets you optimise directly over those experimental knobs $\phi^{\text{exp}}$:

$$
\phi^{\text{proj}}_{g,\cdot} \;=\; \tau\bigl(\phi^{\text{exp}}_{g,\cdot}\bigr) \;\;\;\text{or}\;\;\; \tau\bigl(\phi^{\text{exp}}_{g,\cdot},\, g\bigr).
$$

### Contract

`param_transform` must be a JAX-traceable callable. Accepted signatures:

- **Step-independent**: `tau(phi)` with `phi.shape == (n_experimental_params,)`.
- **Step-dependent**: `tau(phi, step_index)` with a scalar `int32` step index.

The output is a 1-D array whose length is either:

- `projected_basis.lie_algebra_dim` — taken as projected-basis coefficients;
- `basis.lie_algebra_dim` — relevant projected entries extracted automatically via `projected_basis.overlap(basis)`.

`Parameters.n_experimental_params` sets the input dimension. When `param_transform` is set, the engine's `compute_U_fn` is wrapped to apply `vmap(τ)` over the gate axis, embed the result into the proj+drift slots, broadcast drift coefficients, and delegate to the unitary-product code. The Jacobian is replaced by a split-real-imaginary version (real intermediates in `τ` would otherwise drop the imaginary part under holomorphic autodiff).

### Helper: `make_per_element_transform`

For element-wise transforms:

```python
import jax.numpy as jnp

tau = geope.make_per_element_transform([
    jnp.cos,                 # phi[0] → cos(phi[0])
    jnp.sin,                 # phi[1] → sin(phi[1])
    lambda x: 0.5 * x,       # phi[2] → 0.5 * phi[2]
    None,                    # phi[3] passes through
])
```

### Worked example: Rabi rotation in $(A, \varphi)$

```python
import numpy as np
import jax.numpy as jnp
import geope

basis = geope.construct_full_pauli_basis(1)

def rabi(phi):                                   # phi = (A, varphi)
    A, varphi = phi[0], phi[1]
    return jnp.array([A * jnp.cos(varphi),       # X coefficient
                      A * jnp.sin(varphi),       # Y coefficient
                      0.0])                      # Z coefficient

theta = np.pi / 3
RX = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
               [-1j*np.sin(theta/2),  np.cos(theta/2)]], dtype=complex)

params = geope.Parameters(
    basis=basis, control={1: ['x', 'y', 'z']}, target=RX,
    piecewise_steps=4,
    param_transform=rabi, n_experimental_params=2,
    init_spread=0.3, seed=0,
)
geope.Geope(params, max_steps=300, precision=1 - 1e-7).optimize()
print(float(params.best_fidelity))
print(params.best_basis_coefficients)
```

### Practical implications

- Null-space methods (`speed`, `length`, `robust`) must use `parameter_indices`, not `parameter_labels`, when `param_transform` is set — labels no longer correspond to optimised parameters. `Geope` raises `ValueError` otherwise.
- Internally `param_transform` mode uses `float64`; basis-coefficient mode uses `complex128`. Tolerances and bounds you supply should match.

## Phase-sensitive vs projective

The two fidelities differ in how the trace is taken:

$$
F_{\text{proj}}(U, U_T) = \frac{|\mathrm{Tr}(U_T^\dagger U)|}{d}, \qquad
F_{\text{full}}(U, U_T) = \frac{\mathrm{Re}\,\mathrm{Tr}(U_T^\dagger U)}{d}.
$$

$F_{\text{proj}} \in [0,1]$ is invariant under $U \mapsto e^{i\theta}U$ (the global phase is unobservable). $F_{\text{full}} \in [-1,1]$ is not. Use `projective=False` only when the absolute phase matters — for example, when the gate is a sub-block of a larger coherent unitary, or when stitching multiple gates whose relative phase enters the composite fidelity.

Two pathologies to keep in mind for phase-sensitive mode:

- **Traceless targets** (Hadamard, single-qubit $X/Y/Z$, etc.) make the gradient of $F_{\text{full}}$ vanish at $U = I$ in every direction; a random init near identity has no descent direction. Use larger `init_spread` or non-zero `init_values`.
- **Stopping criterion**. `precision = 0.9999999` is meaningful for $F_{\text{proj}}$. For $F_{\text{full}}$ the same threshold is valid near the optimum (both fidelities agree as $U \to U_T$), but the optimiser may transit negative-fidelity regions on its way — that's geometry, not a bug.

## Null-space optimisation

After the main GEOPE loop has converged, the null space of the Jacobian $\omega$ represents directions in parameter space that don't change the unitary to first order. Stepping along these lets you optimise secondary objectives while preserving fidelity.

### Available objectives

| Method | Cost minimised | Purpose |
|--------|----------------|---------|
| `smooth(...)` | $\sum_g \|\phi_{g+1} - \phi_g\|^2$ | reduce variation across segments |
| `smooth_frequency(...)` | $\sum_{m \ge 1, k}|\widehat{\phi_k}(m)|^2$ | suppress high-frequency content (DC excluded) |
| `filter_frequency(filter_fn, ...)` | $\|\widehat\phi - \mathcal{F}(\widehat\phi)\|^2$ | drive $\phi$ toward a user-defined filtered version (= $L^2$ distance by Parseval) |
| `speed(parameter_*, ...)` | $\max_{g, k \in P}|\phi_{g,k}|$ | reduce peak control amplitude |
| `length(parameter_*, ...)` | $\sum_g \sqrt{\sum_{k \in P}\phi_{g,k}^2 + \|d_g\|^2}$ | reduce total pulse length (drift contribution included) |
| `robust(parameter_*, delta, num_samples, ...)` | $1 - \min_{\delta \in [-\Delta,+\Delta]^{|P|}} F$ | maximise worst-case fidelity under uniform δ perturbations |
| `bound(bounds, method, ...)` | $\max(\phi - u_b, l_b - \phi)$ | enforce a box constraint via `'projected_gradient'` / `'pg'` or `'mid_point'` / `'mp'` |

Each returns `(success, iters)`. Pass `piecewise_steps_multiplier > 1` to subdivide existing segments before the pass (linear interpolation), giving more null-space degrees of freedom.

### Null-space algorithm: `_null_space_optimisation()`

```
1. Optionally subdivide piecewise steps (piecewise_steps_multiplier)
2. Build the combined expander (pulse × linear-equality)
3. For each iteration:
   a. Compute Jacobian projections ω
   b. SVD of ω → null-space basis N (right-singular vectors below the rank)
   c. Compute the cost gradient ∇C(φ)
   d. Project the negative gradient onto the null space:
        x = lstsq(N, -∇C)
   e. Step: φ ← φ + rate · N·x / ||x||
   f. Enforce pulse templates if applicable
   g. Recompute fidelity (preserved to first order)
```

## Parameter spaces and index mappings

The codebase uses three basis spaces with boolean masks mapping between them:

```
full_basis  (dim K_full)         — all generators
  ├── projected_indices          — shape (K_full,)
  ├── drift_indices              — shape (K_full,)
  └── proj_drift_indices         — projected | drift

proj_drift_basis  (dim K_pd)     — only projected + drift elements
  ├── proj_indices_projdrift_basis    — shape (K_pd,)
  └── drift_indices_projdrift_basis   — shape (K_pd,)

projected_basis  (dim K_proj)    — only the controllable elements
```

Parameters are stored in full-basis space $(N_g, K_{\text{full}})$. JIT functions operate on the proj+drift subspace $(N_g, K_{\text{pd}})$. With `param_transform`, the engine indices are overridden so the optimisation runs uniformly on $\phi^{\text{exp}} \in \mathbb{R}^{N_g \times n_{\text{exp}}}$.

## Usage

```python
import numpy as np
import geope

# Bases
full    = geope.construct_full_pauli_basis(3)
control = {1: ['x', 'z'], 2: ['x', 'z'], 3: ['x', 'z']}
drift   = {(1, 2): ['zz'], (2, 3): ['zz'], (1, 3): ['zz']}

# Target: Toffoli
target = geope.multicontrol_unitary(np.array([[0, 1], [1, 0]]), 2)

# Bundle everything in a Parameters object
params = geope.Parameters(
    basis=full,
    control=control,
    drift=drift,
    drift_values={(1, 2): {"zz": 1.0},
                  (2, 3): {"zz": 1.0},
                  (1, 3): {"zz": 1.0}},
    target=target,
    piecewise_steps=20,
    seed=0,
)

# Run — populates params history in place; returns the same Parameters
result = geope.Geope(params, max_steps=1000, precision=0.9999).optimize()
print(result.best_fidelity)
print(result.to_dict())

# Null-space passes — fidelity preserved
g = geope.Geope(params, max_steps=0, precision=0.9999)
g.smooth(piecewise_steps_multiplier=2, smoothing_rate=0.05, diff_tol=1e-3)
g.smooth_frequency(smoothing_rate=0.05, diff_tol=1e-3)
g.bound({"x": (-1, 1), "z": (-1, 1)}, method='projected_gradient')
g.robust(parameter_labels=["XII", "IXI", "IIX"], delta=0.01)
g.speed(parameter_labels=["XII", "IXI", "IIX"])
g.length()
```

### Building a `Parameters` from pre-built bases

If you've already constructed `Basis` objects (e.g. via `construct_restricted_pauli_basis`) and don't want to re-express them as `control` / `drift` dicts, pass them directly via the `projected_basis` and `drift_basis` kwargs:

```python
projected = geope.construct_restricted_pauli_basis(3, ['x', 'z'])
drift_b   = geope.construct_restricted_pauli_basis(3, ['zz'])

params = geope.Parameters(
    basis=full,
    projected_basis=projected,
    drift_basis=drift_b,
    drift_values=drift_b.generate_parameter_list({
        (1, 2): {"zz": 1.0},
        (2, 3): {"zz": 1.0},
        (1, 3): {"zz": 1.0},
    }),
    target=target,
    piecewise_steps=20,
    seed=0,
)
```

This is the escape hatch for cases where the projected subset can't be expressed as a control dict. `projected_basis` and `control` are mutually exclusive; same for `drift_basis` and `drift`.
