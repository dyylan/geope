# GEOPE – Geodesic Pulse Engineering

## Overview

`geope` finds piecewise-constant control pulses that implement a target quantum gate on an $n$-qubit system. Given a target unitary $U_T$, a set of available control generators (the projected basis), and optionally fixed drift generators, the optimiser searches for real-valued parameters $\phi$ such that

$$
U(\phi) \;=\; \prod_{g=1}^{N_g} \exp\!\Bigl(i \sum_{k}\phi_{g,k}\,G_k\Bigr) \;\approx\; U_T,
$$

where each $H_g = \sum_k \phi_{g,k}\,G_k$ is a linear combination of basis generators on segment $g$.

The core algorithm is the **geodesic method**: at each step it computes the shortest path on $U(d)$ from the current unitary to the target, projects that direction onto the controllable subspace, then solves a convex least-squares problem and a one-dimensional line search to take a parameter step. This is distinct from gradient-based methods like GRAPE that follow the fidelity gradient directly.

The entry point is `Parameters` — a state object that bundles every input the optimiser needs (basis, control, drift, target, constraints, pulse constraints, `param_transform`, bounds, init values, seed, manifold). Pass it to `Geope` and call `.optimize(max_steps=...)`. The returned `Parameters` carries the live/final `parameters` and `fidelity` (and `to_dict()`); the full run trajectory and `best_*` helpers live on an opt-in `History` logger (`geope.history`).

The mathematics lives in the `geometry/` package, split along one line: **every
manifold GEOPE walks is a submanifold of the ambient space
$\mathcal A = \mathbb C^{N\times m}$, and the pulse acts on all of them the same
way.** So `geometry/chart.py` owns everything *valued in* $\mathcal A$ — the
orbit map $\Phi(\phi) = U(\phi)\,x_0$ and its whole jet
$(\Phi, \mathrm D\Phi, \mathrm D^2\Phi)$ — while everything that happens *inside
a tangent space* $T_x\mathcal M$ (the metric, the coefficient frame, the
logarithm, the fidelity) hangs off a `Manifold`, with `to_tangent` the one bridge
between them. `Parameters.manifold` is where the two are composed — once, in
`Parameters.__init__`, by `Manifold.bind`, which holds no mathematics of its own
— so there is no separate engine object: `Geope`, `Grape` and `Gecko` all read
what they need off the (shared) `Parameters`.

Those factories return **un-jitted** callables, so they fuse into the
optimiser's `update_step`, which is JIT-compiled once when `optimize()` first
traces it. The exceptions are the manifold's two host-facing objectives,
`manifold.fidelity_at` and `manifold.infidelity_at`: those are called one trial
pulse at a time from Python loops, so they are compiled once and memoised on
the manifold when first read. Calling them inside a trace still works — a jitted callable
inlines.

## Class hierarchy

```
jax/       — differentiable primitives (logm, dexpm, the propagator
             Jacobian/pullback/Hessian, the autodiff Hessian)
                ↓ used by
geometry/chart.py — the ambient layer: the orbit map Phi(phi) = U(phi)·x_0 and
             its whole jet (Phi, DPhi, D^2Phi). Everything valued in the ambient
             space C^(N x m); manifold-agnostic, imports only JAX + geope.jax.
                ↓ composed by
geometry/  — Manifold  ──owns──▶ base_point (x_0 = Phi(0)), target
             │                   compute_point (the chart), and a
             │                   TangentBundle (frame, jacobian, hvp, columns)
             │                   — `Manifold.bind` attaches all but base_point,
             │                   and holds no mathematics of its own
             └──▶ GeometricContext: every per-step quantity, in cost tiers
                ↓ built once, in __init__, on
Parameters                       (parameters.py)
                ↘
                  Geope / Grape  (geope.py / grape.py)
                  Gecko          (gecko.py)

geometry/lie/      Basis + the UnitaryGroup / SpecialUnitaryGroup manifolds,
                   with the fidelity formulas they own
geometry/stiefel/  Stiefel (orthonormal m-frames, canonical metric) and
                   StateSphere — state preparation; neither is a Lie group
```

The optimisers know nothing about *which* space they are walking: `Geope`, the
line searches and `Gecko` speak only to the `Manifold` interface. Gate synthesis
on $\mathrm{SU}(d)$ and state preparation on $\mathbb{CP}^{n-1}$ are the same
code with a different manifold.

## Lie group classes (`geometry/lie/`)

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

- `overlap(other)` — boolean mask over `other`'s basis, true where there is nonzero trace overlap with `self`. Used by `Parameters` to build its index masks.
- `verify()` — orthogonality check under the trace inner product.
- `linear_span(parameters)` — $\sum_i \phi_i G_i$.
- `generate_parameter_list(parameter_map)` — converts a dict like `{1: {"x": 0.5}, (1,2): {"zz": 0.3}}` to a flat parameter array.
- `generate_bounds(bounds_map, piecewise_steps)` — converts `{"x": (-1, 1)}` to `(lower, upper)` arrays.
- `apply_interaction_graph(graph)` / `apply_interaction_map(map)` — prune to hardware connectivity.

### Removed: `Hamiltonian` and `Unitary`

These two host-side wrapper classes were a second, numpy/scipy implementation of
mathematics the geometry layer now owns, and were removed. Their replacements:

| was | is |
|---|---|
| `Hamiltonian(basis, phi).matrix` | `basis.linear_span(phi)` — the same $\sum_k \phi_k G_k$ |
| `Hamiltonian(basis, phi).unitary.matrix` | `params.manifold.compute_point(phi)` — the chart, for the whole piecewise pulse rather than one gate |
| `Hamiltonian.parameters_from_hamiltonian(H, basis)` | `geope.geometry.basis.project_omegas` — the same $\mathrm{Re}\,\mathrm{Tr}(G_i H)/d$ |
| `h.geodesic_hamiltonian(V)` / `u.geodesic_hamiltonian(basis, V)` | `-params.manifold.log(U, V)`, then `.coefficients(U, ...)` — i.e. `ctx.A` and `ctx.gammas` |
| `Unitary.unitary_fidelity(A, B)` | `params.manifold.fidelity(A, B)` |
| `Unitary(U).parameters(basis)` | `m.coefficients(I, m.log(I, U))` |
| `Unitary(U)`'s $UU^\dagger = I$ check | `manifold.validate_point(U)` — which `Parameters` runs on your `target` at construction, so an off-manifold target now fails loudly instead of yielding meaningless fidelities |

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
    pulse_constraints={(1, 2): ['zz'], (2, 3): ['zz'], (1, 3): ['zz']},
)
```

`pulse_constraints` uses the same `{qubit_index_or_tuple: [interaction]}` dict format as `control` — here it freezes the temporal shape of the three `zz` terms.

| Approach | Use case |
|----------|----------|
| `drift_basis` + `drift_values` | drift is truly fixed and not optimised |
| `pulse_constraints` on projected params | drift-like terms whose amplitude is optimised but whose temporal profile is fixed |

Other utilities:

- `prepare_random_parameters(proj_indices, expander, spread, seed)` — random initial parameters respecting constraints.
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
           manifold=None)
```

| Parameter | Description |
|-----------|-------------|
| `basis` | the full `Basis`; defaults to 2-qubit full Pauli basis if `None` |
| `control` | dict picking the projected (controllable) subset |
| `drift` | dict picking the drift subset; must be disjoint from `control` (see below) |
| `init_values` | dict in `control` format, or `ndarray` of full-basis shape, or `None` (random) |
| `drift_values` | dict, `ndarray`, or `None` (ones) |
| `target` | target unitary as `ndarray` |
| `piecewise_steps` | number of gate segments $N_g$ |
| `fixed_drift` | whether drift is held fixed during optimisation |
| `constraints` | list of constraint vectors / dicts |
| `pulse_constraints` | control-format dict `{site: [ops]}` (same format as `control`) of projected terms whose time-shape is fixed |
| `bounds` | dict `{label: (lo, hi)}` — consumed by `Geope.bound(...)`, not by the main loop |
| `init_spread` | half-width of uniform random init, in units of $\pi$ |
| `seed` | random seed |
| `param_transform` | callable mapping experimental params to basis coefficients |
| `n_experimental_params` | length of the experimental input; defaults to `projected_basis.lie_algebra_dim` |
| `manifold` | the `Manifold` to synthesise on; defaults to `SpecialUnitaryGroup(basis.dim)`. Pass `UnitaryGroup(d)` for the phase-sensitive fidelity, or a `StateSphere` / `Stiefel` for state and subspace problems |

A basis element may not appear in both the control and the drift basis. Drift
coefficients are written after control coefficients on the combined proj+drift array,
so a shared element would have its control value silently overwritten — and under
`param_transform` its gradient zeroed, leaving the parameter dead. `Parameters` raises
a `ValueError` naming the offending elements at construction. To control an element
that also carries a constant offset, leave it out of the drift basis and add the
constant through `param_transform`:

```python
# ZI is controllable *and* sits at a fixed 0.7 offset.
zi = list(full.labels).index("ZI")

def param_transform(x):
    return x.at[zi].add(0.7)
```

Attributes populated after construction:

| Attribute | Description |
|-----------|-------------|
| `basis`, `projected_basis`, `drift_basis` | the three `Basis` objects |
| `target` | the target |
| `drift_parameters` | drift coefficients (or `None`) |
| `constraint_arrays`, `constraint_expander` | merged constraints and reduced-space mapping |
| `bounds` | pre-built bounds tuple (or `None`) |

Live optimisation state — seeded at construction and updated in place by `Geope`:

| Attribute | Description |
|-----------|-------------|
| `parameters` | current parameter array, shape `(N_g, K_{full})`; seeded to the initial guess, holds the final result after `optimize()` |
| `fidelity` | current fidelity (`None` before a run) |
| `infidelity` | `1 - fidelity` (`None` before a run) |
| `basis_coefficients` | current parameters mapped through `param_transform` if set |
| `to_dict()` | current solution as a control-style dict |

The full run trajectory and the `best_*` helpers live on the opt-in [`History`](#history-historypy) logger, not on `Parameters`.

## Metadata, functions, and optimiser

### Algebraic metadata (cached on `Parameters`)

`Parameters` derives the index masks relating the three bases as cached
properties (computed once from `basis` / `projected_basis` / `drift_basis`):

- `projected_indices` — shape $(K_{\text{full}},)$, which full-basis elements are controllable (`projected_basis.overlap(basis)`).
- `drift_indices` — shape $(K_{\text{full}},)$, which are fixed drift.
- `proj_drift_indices` = `projected_indices | drift_indices`.
- `proj_indices_projdrift_basis` — projected mask within the proj+drift subspace, shape $(K_{\text{pd}},)$.
- `drift_indices_projdrift_basis` — drift mask within the proj+drift subspace.
- `proj_drift_basis` — `Basis` containing only the projected + drift elements; used for all JIT computations.

### The manifold (`params.manifold`)

Everything the optimisers need hangs off **one** lazily-built, cached handle:

```python
m = params.manifold          # bound to this problem's chart and target
m.compute_point(phi)             # the chart: parameters -> a point on the manifold
m.fidelity_at(phi)           # the convergence score (compiled); m.infidelity_at(phi) is the cost
m.value_and_grad, m.hessian  # what GRAPE minimises with (analytic, from the propagators)
m.tangent.jacobian           # the chart's pushforward; m.tangent.vjp is its pullback
m.context(phi)               # the per-step geometry (see below)
```

`Parameters` binds the manifold you pass (or the default
`SpecialUnitaryGroup(basis.dim)`) at construction and keeps the one handle, so a
`Geope` and a `Gecko` sharing one `Parameters` share the compiled traces. The
manifold is where the SU-vs-U choice lives — it is consulted nowhere else, and
there is no separate flag to reconcile it against.

**What `bind` actually attaches.** Every manifold here is a submanifold of one
ambient space $\mathcal A = \mathbb C^{N\times m}$, and the pulse acts on all of
them identically, by left multiplication: the chart is the orbit map
$\Phi(\phi) = U(\phi)\,x_0$ through the manifold's `base_point`. So
`geope.geometry.chart` builds the whole ambient jet
$(\Phi, \mathrm D\Phi, \mathrm D^2\Phi)$ and `bind` only composes it, together
with two pieces of problem data:

| passed to `bind` | what it is |
|---|---|
| `generators` | the proj+drift `Basis` the pulse is a product of exponentials in |
| `frame` | the **ambient coefficient frame** `m.coefficients` resolves a tangent against (`params.basis`) |
| `columns` | which coefficient columns the geodesic solve may move |
| `wrap_chart` | the `param_transform` reparametrisation of the chart's *input*, or `None` |

`frame` is optional, and `None` is not a degraded mode. The groups resolve
tangents against a Hermitian matrix frame; `StateSphere` and `Stiefel` use a
real/imaginary split of the ambient array instead and need no frame — which is
also far cheaper, $2Nm$ coefficients against a matrix frame's $d^2$ (2 048
against ~10⁶ at ten qubits). A manifold that needs no frame builds no projector.

**The per-step context.** `m.context(phi)` returns a `GeometricContext`: every
geometric quantity a step needs, each a lazily-computed cached property, grouped
by what it costs.

| tier | quantities | cost |
| --- | --- | --- |
| 0 — base point | `point`, `jacobian`, `A`, `F0`, `gammas`, `omegas`, `fidelity`, `infidelity` | one propagator, one Jacobian, **one** logarithm |
| 1 — direction | `V`, `Omega`, `omega_norm2`, `velocity`, `xi_rel` | free: a contraction of tier 0's Jacobian |
| 2 — curvature | `W`, `acceleration`, `chi`, `q`, `q_exact`, `rho` | one directional HVP plus the manifold's Riemannian Hessian |
| 3 — ray | `point_at(t)`, `infidelity_at(t)`, `distance_at(t)` | one propagator per trial point |

Laziness is load-bearing rather than an optimisation: `Gecko` reads only
`omegas`, and so never traces the matrix logarithm at all. Only tier 0 is
direction-free — the search direction arrives via `ctx.set_direction(coeffs)`
after the least-squares solve (which needs `gammas`/`omegas`) and may be set
once; every direction-dependent property raises before that.

The context is **trace-time only**: build one per step inside the jitted update,
and never return it from a jitted function or put it in a `scan`/`while_loop`
carry.

**Migrating from `params.<name>_fn`.** The eleven cached function properties are
gone; each has a home on the manifold:

| was | is |
| --- | --- |
| `params.compute_point_fn` | `params.manifold.compute_point` |
| `params.fid_U_fn(U)` / `infid_U_fn(U)` | `params.manifold.fidelity(U, target)` / `.infidelity(...)` |
| `params.fid_U_fn(params.compute_point_fn(phi))` | `params.manifold.fidelity_at(phi)` |
| `params.infid_fn(phi)` | `params.manifold.infidelity_at(phi)` |
| `params.grad_fn` / `params.hess_fn` | `params.manifold.value_and_grad` / `.hessian` |
| `params.jac_fn` | `params.manifold.tangent.jacobian` |
| `params.geo_fn(U)` | `-params.manifold.log(U, target)` (at the base point; no `U·` to undo) |
| `params.project_omegas_fn` | `params.manifold.coefficients(point, tangent)` — the frame lives on `manifold.tangent.frame` |
| `params.gammas_and_omegas(phi, key)` | `params.manifold.context(phi).gammas` / `.omegas` |
| `params.free(...)` | *(new)* the free parameter columns, in the pipeline's dtype |

`params.projective` still reads as before; it now delegates to the manifold.

### `Geope` (`geope.py`)

```python
Geope(params, verbose=False, history=None)
```

`Geope` requires a `Parameters` object as its single positional argument. The optimisation functions, initial parameters, drift, constraints, pulse constraints, seed, initialisation spread, projective flag and `param_transform` are all read from `params`. Passing anything other than a `Parameters` raises `TypeError`.

| Parameter | Description |
|-----------|-------------|
| `params` | a `Parameters` instance bundling all inputs |
| `verbose` | print per-step progress |
| `history` | optional `History` logger (`None` = no logging) |

The iteration cap, the line search, and the three run-control knobs are arguments of `optimize`, not constructor fields:

```python
from geope import ApproximateQuadraticArmijo, Armijo, GoldenSection, QuadraticArmijo

optimize(max_steps=1000,
         line_search=GoldenSection(),        # default; or Armijo(), QuadraticArmijo()
         precision=0.9999999,
         max_step_size=0.9, gram_schmidt_step_size=1.3)
```

| `optimize` argument | Description |
|---------------------|-------------|
| `max_steps` | maximum number of optimisation steps |
| `line_search` | a `LineSearch` object tuning the geodesic step size; defaults to `GoldenSection()` |
| `precision` | target fidelity threshold (host-side; no recompile) |
| `max_step_size` | maximum line-search step (baked into the JIT; changing it recompiles) |
| `gram_schmidt_step_size` | step size for the Gram–Schmidt fallback (host-side; a falsy value disables it) |

The line searches are immutable config objects (frozen dataclasses):

- `GoldenSection(tol=1e-5)` — golden-section search (the default). Like every line search it reports a per-step evaluation count in its state (`{"n_eval"}`).
- `Armijo(c1=1e-4, beta=0.5, t_min=1e-8)` — first-order backtracking Armijo on the squared geodesic distance. It seeds at the full bracket step $-t_{\max}$ and backtracks, taking the slope from the objective value alone ($s = 2F_0$, exact under the tangent matching $\Omega = -A$), so it forms no derivative of the product unitary. Works in every mode, `param_transform` included.
- `QuadraticArmijo(c1=1e-4, beta=0.5, t_min=1e-8)` — geometry-aware second-order line search: seeds the step from the SU(N) curvature (clipped to the bracket, falling back to the full step when the curvature is non-positive) and enforces sufficient decrease with Armijo backtracking (standard/projective mode only).
- `ApproximateQuadraticArmijo(c1=1e-4, beta=0.5, t_min=1e-8)` — the same algorithm, but with the *exact* curvature. `QuadraticArmijo` builds $\psi''(0)$ using $\lVert\Omega\rVert_F^2$ for the intrinsic term $\langle\Omega,\mathcal{K}_A\Omega\rangle_F$, which is only valid when the achieved tangent $\Omega$ is parallel to the geodesic tangent $A$ — i.e. only when the least-squares solve for the search direction leaves no residual. This variant evaluates the form properly, so the residual couples into the curvature through the Riemannian Hessian as it should. Since $\mathcal{K}_A\preceq I$ it always seeds a **longer** step. Costs one extra `eigh` on a group or the state sphere, and one small operator exponential on `Stiefel` (standard mode only; on `Stiefel` it also needs `projective=False`, see below).

    Whether it changes anything is structural: the solve has `piecewise_steps × K_proj` unknowns against `K_basis` equations, so once there are enough pulse segments it is underdetermined, fits the geodesic tangent exactly, and the two curvatures coincide — the correction only bites for short pulses or thin control sets. `GeometricContext.xi_rel` reports the residual as the (scale-invariant) sine of the angle between $\Omega$ and $A$, and tracks `ls_diagnostics["residual_rel"]` closely; it is `0` exactly when the two curvatures agree.

The three differ in what they evaluate per step, not just in flops: `GoldenSection` evaluates the cheap infidelity many times; `Armijo` evaluates the `logm`-bearing geodesic distance a few times; `QuadraticArmijo` adds one `logm` plus one directional HVP to seed its step. Note that a wider `max_step_size` is what makes the quadratic seed worth its cost — at the default the model minimiser usually falls outside $[-t_{\max}, 0]$ and is clipped to $-t_{\max}$, which is exactly where `Armijo` starts anyway.

The line-search object and `max_step_size` bake into JIT-compiled functions that `optimize` builds on first use and reuses across calls; the frozen-dataclass value equality means two equal line searches (e.g. the per-call default `GoldenSection()`) reuse the compiled functions, while changing the object or `max_step_size` triggers a one-off recompile. `precision` and `gram_schmidt_step_size` are host-side only — changing them never recompiles.

Live state and logging:

- The current parameters and fidelity live on `params` (`params.parameters`, `params.fidelity`); `Geope` updates them in place each step, and `optimize(max_steps=...)` returns the `Parameters` instance itself — so the user has a single handle for both inputs and the final result.
- `step_size` — the transient last line-search step size.
- `ls_diagnostics` — diagnostics of the **most recent** geodesic least-squares solve (step 4 of the algorithm below), as a dict of host-side scalars:

    | Key | Meaning |
    |-----|---------|
    | `residual` | $\lVert A\,\mathrm{sol}-b\rVert_2$, the absolute misfit |
    | `residual_rel` | $\lVert A\,\mathrm{sol}-b\rVert_2/\lVert b\rVert_2 \in [0,1]$ — the fraction of the geodesic direction the controls cannot reproduce. Dimensionless, so comparable across steps and problems; **this is the headline number**. `0.0` when $\lVert b\rVert_2$ vanishes |
    | `rank` | numerical rank of the least-squares system |
    | `cond` | condition number over the singular values retained above the `rcond` cutoff |

    Before the first solve of a run these hold `nan` / `-1` sentinels. A high `residual_rel` with full `rank` means the geodesic direction genuinely leaves the controllable subspace; a rank drop or a large `cond` instead means the solve itself is degenerate. Because `coeffs` is renormalised to fixed norm after the solve, these rate the *direction* fit, not the error of the step taken.

    These are **not** logged by default. To record them, pass a `logging_fn` (see `History` below) or read them from a callback.
- `history` — an optional `History` logger (`None` unless one was passed). When supplied, the full run trajectory and `best_*` helpers are available on it (see below).

### `History` (`history.py`)

```python
History(logging_fn=None)
```

An opt-in, configurable run log. Pass one to `Geope` (`history=History()`) and the full trajectory is recorded into `geope.history`; leave it `None` and no history is kept (the final answer still lives on `params`).

By default each step records five columns — `parameters` (a full-basis snapshot), `fidelities`, `infidelities`, `step_sizes`, and an integer `steps` counter derived from the log length. Pass `logging_fn` to record arbitrary per-step values instead: it receives the running `Geope` and returns a `dict` of `column -> value` (e.g. `History(logging_fn=lambda g: {"fid": float(g.params.fidelity)})`).

This is how you log anything the optimiser exposes but does not record by default — for instance the least-squares diagnostics:

```python
h = geope.History(logging_fn=lambda g: {
    "fidelities": g.params.fidelity,
    "ls_residual_rel": g.ls_diagnostics["residual_rel"],
    "ls_rank": g.ls_diagnostics["rank"],
    "ls_cond": g.ls_diagnostics["cond"],
})
g = geope.Geope(params, history=h)
g.optimize()
h.to_dataframe()   # ls_residual_rel alongside fidelities
```

A `logging_fn` must return the **same keys on every call**. `record` is schema-free (it appends per key), and `len(history)` reports the first column's length, so a key that only appears from some step onwards leaves the columns ragged and makes `to_dataframe()` raise. Guard values that are not yet available at step 0 — recorded during `init()`, before the first step — rather than omitting the key; the `ls_*` attributes already do this themselves via their `nan` / `-1` sentinels.

| Member | Description |
|--------|-------------|
| `record(geope)` | append one row via `logging_fn`; called by `Geope` each step |
| `reset()` | drop all rows |
| `len(history)` | number of recorded rows |
| `history.<col>` / `history["<col>"]` | a logged column (the same list) |
| `keys()` | the logged column names |
| `to_dataframe()` | the logs as a `pandas.DataFrame` |
| `best_fidelity` | `max(fidelities)` (or `None`) |
| `best_parameters` | parameters at the highest-fidelity step (or `None`) |
| `best_basis_coefficients` | best parameters mapped through `param_transform` if set |
| `to_dict()` | best solution as a control-style dict (`{}` if unavailable) |

The best-over-trajectory helpers need the default `fidelities`/`parameters` columns; under a custom `logging_fn` that omits them they degrade to `None`/`{}` rather than raising. Note `params.parameters` is the single current array while `history.parameters` is the list of per-step snapshots. A `History` is meant for a single run.

## Core algorithm: `optimize()`

```
for each step:
    1. Extract free_params = parameters[:, proj_drift_indices]

    2. Compute the geodesic direction:
       U  = manifold.compute_point(free_params)        # ctx.point
       g  = -i · logm(U† U_T)                       # generator in u(d)
       g  = g - Tr(g)/d · I        if projective    # drop global-phase generator
       Γ  = U · g                                   # geodesic tangent
       γ  = project(Γ) / d                          # coefficients in basis

       `logm` here is `geope.jax.logm_unitary`, the unitary specialisation:
       `U† U_T` is a product of unitaries, hence normal, and a matrix is normal
       exactly when its complex Schur form is *diagonal*. So the principal log
       is just the scalar log of that diagonal, and the inverse
       scaling-and-squaring machinery in the general `geope.jax.logm` — which
       exists to handle a non-zero super-diagonal — is skipped entirely. This
       is exact, not an approximation. The same substitution applies to
       `traceless_log` in the geometry-aware line searches.

       It is also *more* accurate than the general path on an important class
       of targets: Hermitian unitaries (Hadamard, the Paulis, CNOT, Toffoli)
       have an eigenvalue at exactly −1, sitting on the principal branch cut,
       which `logm_unitary` resolves to ~1e-16 against the general path's
       ~1e-7.

    3. Compute the Jacobian projections:
       ω[g, k] = project(i · ∂U/∂φ_{g,k})

    4. Solve the constrained least-squares problem:
       sol = argmin ||ω^T · sol - γ||
       (optionally through a constraint+pulse expander E)
       → residual / rank / condition number recorded in geope.ls_diagnostics

    5. Normalise and line-search:
       coeffs = sol · sqrt(N_g) / ||sol||
       dt     = argmin infid(φ + t · coeffs)        # over t ∈ [-t_max, 0]
       φ_new  = φ + dt · coeffs

    6. If fidelity decreased, Gram–Schmidt fallback:
       proj_c = random_direction ⊥ coeffs
       try ±proj_c, keep the side with higher fidelity
```

The line search interval $[-t_{\max}, 0]$ is the toward-target half-line under the algorithm's sign convention: solving $\omega^\top \cdot \mathrm{sol} = \gamma$ matches the achieved velocity $\Omega$ to $A$, the geodesic tangent *pointing away from* the target, so negative `dt` is what approaches it. Zeroth-order searches minimise `ctx.infidelity_at`, which is non-negative in both `projective` modes; the Armijo family minimises `ctx.distance_at`, the squared geodesic distance. A step is kept when it reduced **its own** objective by more than `PROGRESS_RTOL` relatively; otherwise the Gram-Schmidt fallback replaces it. Convergence is always tested on the fidelity.

### Key functions

- **`ctx.gammas` / `ctx.omegas`** — the least-squares operands, and the per-iteration core: one propagator, one Jacobian and one logarithm produce both. **Tangent vectors are mapped into the manifold's own representation before being resolved into coefficients** (on a group, left-trivialised by $U^\dagger$), and this is load-bearing: the Pauli basis is Hermitian, so the projection keeps only the traceless-Hermitian part of what it is handed, while the raw geodesic tangent and Jacobian columns are $U\cdot(\text{skew-Hermitian})$ and mostly fall outside it. Left-trivialising first makes the projection lossless and the least squares an honest $\langle\cdot,\cdot\rangle_F$-orthogonal projection of the geodesic tangent onto $\mathrm{Im}(\mathrm{D}\Phi)$ — which is what the second-order line searches assume. Left translation is itself an isometry; it is the projection *after* it that would be lossy, so the two cannot be commuted. Both operands go through the same coefficient map, so the residual compares like with like.
- **`linear_comb_projected_coeffs_multigate(ω, γ, E, *, return_diagnostics=False)`** — least-squares solve, optionally through a constraint expander $E$. With `return_diagnostics=True` it returns `(sol, diagnostics)`, the second a dict of `residual` / `residual_rel` / `rank` / `cond` scalars; these come free from the same `lstsq` call and are what `Geope` surfaces as `ls_diagnostics`.
- **`Geope.update_step(free_params, ls_state)`** — the whole step, in one jitted function: open the context, solve, renormalise the direction, attach it, line-search. It returns the accepted step alongside the line search's objective at $t=0$ and at the accepted step, which is what the loop tests for progress (see below).

## Callbacks

`Geope.optimize`, `Grape.optimize` and every `Gecko` post-processing method (`smooth`, `smooth_frequency`, `filter_frequency`, `speed`, `length`, `robust`, `bound`) accept a `callbacks` argument — a single callable, or a list/tuple of callables — invoked at the **end of every step** (after the state update and, for `Geope`, after pulse-template enforcement). Use them to log, plot, checkpoint, implement custom early-stopping, or drive dynamic behaviour.

Each callback has the signature:

```python
def callback(step, history, optimizer) -> bool:
    ...
```

- `step` — the 1-based index of the step just completed.
- `history` — the optimiser's `History` object, or `None` if none was attached.
- `optimizer` — the live `Geope` / `Grape` / `Gecko` instance (read `optimizer.params.parameters`, `optimizer.params.fidelity`, etc.).

**Stopping semantics.** All callbacks run every step (so their side-effects always execute), then the loop **stops if any callback returns a falsy value** (`False`, `None`, `0`, …). The loop continues only while *every* callback returns truthy — so a pure logging callback **must** `return True`.

> For `Gecko`, callbacks fire once per null-space iteration. During the loop the running geometry is visible via `optimizer.params.parameters`; `optimizer.params.fidelity` and `optimizer.step_size` are finalised only after the loop returns. `Gecko`'s `History` still records only at the start and end of a pass.

Example — custom early stopping plus logging:

```python
losses = []

def stop_on_plateau(step, history, geope):
    losses.append(float(geope.params.fidelity))
    # stop if the last 5 fidelities barely moved
    if len(losses) >= 5 and (max(losses[-5:]) - min(losses[-5:])) < 1e-6:
        return False
    return True

geope.optimize(max_steps=1000, callbacks=stop_on_plateau)
```

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

`Parameters.n_experimental_params` sets the input dimension. When `param_transform` is set, the manifold's chart is wrapped to apply `vmap(τ)` over the gate axis, embed the result into the proj+drift slots, broadcast drift coefficients, and delegate to the unitary-product code. The Jacobian and its pullback are replaced by split-real-imaginary versions (real intermediates in `τ` would otherwise drop the imaginary part under holomorphic autodiff), and the chart loses its exponential-product structure — so `tangent.generators` is `None`, which is the single signal that drops both second differentials: `tangent.hvp` (and with it the curvature tier and the second-order line searches) and `tangent.hessian` (so `Manifold.hessian` falls back to autodiff). The gradient stays exact, because the pullback is still available — just autodiff-flavoured.

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
g = geope.Geope(params, history=geope.History())
g.optimize(max_steps=300, precision=1 - 1e-7)
print(float(g.params.fidelity))        # final fidelity (lives on Parameters)
print(g.params.basis_coefficients)     # current params mapped through param_transform
print(g.history.best_fidelity)         # best fidelity over the trajectory
```

### Practical implications

- `Gecko`'s null-space methods (`speed`, `length`, `robust`) must use `parameter_indices`, not `parameter_labels`, when `param_transform` is set — labels no longer correspond to optimised parameters. `Gecko` raises `ValueError` otherwise. (`Gecko` supports experimental parameters in every construction mode: when reusing a `Geope` the engine is already wrapped; when built from `params` it re-wraps a fresh engine.)
- Internally `param_transform` mode uses `float64`; basis-coefficient mode uses `complex128`. Tolerances and bounds you supply should match.

## Phase-sensitive vs projective

The two fidelities differ in how the trace is taken:

$$
F_{\text{proj}}(U, U_T) = \frac{|\mathrm{Tr}(U_T^\dagger U)|}{d}, \qquad
F_{\text{full}}(U, U_T) = \frac{\mathrm{Re}\,\mathrm{Tr}(U_T^\dagger U)}{d}.
$$

You choose by passing the **manifold**: `SpecialUnitaryGroup(d)` — the default
— gives the projective fidelity (traceless $\mathfrak{su}(d)$ tangents, phase
quotiented out), and `UnitaryGroup(d)` gives the phase-sensitive one (all of
$\mathfrak u(d)$, the phase controllable). That single choice carries the
fidelity, the tangent projection and the geodesic with it — nothing else in the
pipeline branches on it, and `params.projective` simply reports what the manifold
says.

$F_{\text{proj}} \in [0,1]$ is invariant under $U \mapsto e^{i\theta}U$ (the global phase is unobservable). $F_{\text{full}} \in [-1,1]$ is not. Use `manifold=UnitaryGroup(d)` only when the absolute phase matters — for example, when the gate is a sub-block of a larger coherent unitary, or when stitching multiple gates whose relative phase enters the composite fidelity.

Two pathologies to keep in mind for phase-sensitive mode:

- **Traceless targets** (Hadamard, single-qubit $X/Y/Z$, etc.) make the gradient of $F_{\text{full}}$ vanish at $U = I$ in every direction; a random init near identity has no descent direction. Use larger `init_spread` or non-zero `init_values`.
- **Stopping criterion**. `precision = 0.9999999` is meaningful for $F_{\text{proj}}$. For $F_{\text{full}}$ the same threshold is valid near the optimum (both fidelities agree as $U \to U_T$), but the optimiser may transit negative-fidelity regions on its way — that's geometry, not a bug.

## Subspace synthesis on Stiefel manifolds

When only an $m$-dimensional subspace of the target matters — a gate mediated by
a bosonic mode whose final state is irrelevant, a subspace encoding, a state
preparation — the rest of the unitary is **redundancy**. Optimising over it is
wasted work. Pass a `Stiefel` manifold instead and it is quotiented out:

```python
from geope import Parameters, Geope, Stiefel

# Two spins coupled through one boson (max 2 bosons -> Fock dim 3), so N = 12.
# Score only the four spin states with the boson in vacuum: m = 4.
E = np.zeros((12, 4), complex)
for spin in range(4):
    E[3 * spin, spin] = 1.0                      # |spin> (x) |0>

params = Parameters(
    basis=construct_full_spin_boson_basis(2, 1, 2),
    projected_basis=construct_restricted_spin_boson_basis(
        2, 1, {1: ["x", "y"], 2: ["x", "y"]}, 2),
    target=E @ np.diag([1, 1, 1, -1]),           # CZ on the vacuum subspace
    piecewise_steps=10,
    manifold=Stiefel(dim=12, frame=4, base_point=E),
)
Geope(params).optimize(max_steps=300)
```

A point is an orthonormal $m$-frame $Q \in \mathbb C^{N\times m}$, the chart is
$\Phi(\phi) = U(\phi)E$, and the fidelity
$\lvert\mathrm{Tr}(Q_\star^\dagger Q)\rvert/m$ scores only the frame. In the
example above the boson starts and ends in vacuum without appearing anywhere in
the objective — leaving it simply is not free.

| argument | meaning |
|---|---|
| `dim` | the ambient dimension $N$ |
| `frame` | the number of scored columns $m$; a point is `(N, m)` |
| `base_point` | the frame $E$ the pulse acts on; defaults to the first $m$ basis states. Inherited from `Manifold`, where it is $\Phi(0)$ for every space |
| `projective` | keyword-only, default `True`; as for SU/U, whether a global phase is physical |

Three things to know:

- **The metric is the canonical one**, $\langle\Delta,\Upsilon\rangle_Q =
  \mathrm{Tr}(\Delta^\dagger(\mathbb 1 - \tfrac12 QQ^\dagger)\Upsilon)$, not the
  embedded Frobenius metric: rotations *within* the frame carry half the weight
  of leakage out of it. These are different Riemannian manifolds with different
  geodesics.
- **The logarithm is iterative** (Zimmermann–Hüper), unlike every other manifold
  here. It costs a $2m\times2m$ Schur decomposition per iteration and typically
  converges in 5–10, so keep $m$ modest. At $m = N$ there is no redundancy left
  and `SpecialUnitaryGroup` is the better choice; at $m = 1$ prefer
  `StateSphere`, whose logarithm is closed-form.
- **`ApproximateQuadraticArmijo` needs `projective=False`.** The Riemannian
  Hessian of $\tfrac12 d^2$ exists here, but it is not a scalar function of one
  adjoint the way the group's $\frac{\mathrm{ad}}2\coth\frac{\mathrm{ad}}2$ is —
  a general Stiefel manifold is normal homogeneous but not *symmetric*. It is
  instead read off the blocks of one operator exponential (the Jacobi equation
  has constant coefficients in a homogeneous moving frame); see
  `geope.jax.stiefel_hessian_quadratic_form`. With `projective=True` the phase
  alignment makes the objective the squared distance on the $\mathrm U(1)$
  *quotient*, whose Hessian carries an extra O'Neill term this does not have, so
  `ctx.q_exact` and `ctx.rho` raise `NotImplementedError` there rather than
  silently returning a form that is a few percent wrong. `GoldenSection`,
  `Armijo` and `QuadraticArmijo` all work in either mode.

    Cost is $O(m^6)$ and **independent of $N$** — the Jacobi operator
    block-diagonalises, and the sector that scales with the ambient dimension
    reduces to right multiplications on an $m\times m$ Gram. Measured at 0.08 ms
    for $m = 2$ and 6 ms for $m = 8$; past that prefer `QuadraticArmijo`.

## Null-space optimisation (`Gecko`)

After the main GEOPE loop has converged, the null space of the Jacobian $\omega$ represents directions in parameter space that don't change the unitary to first order. Stepping along these lets you optimise secondary objectives while preserving fidelity.

These passes live on a separate optimiser, **`Gecko`**, which post-processes a solution. A `Gecko` is constructed from a `Parameters` object — the same object a `Geope` uses:

- `Gecko(p)` — operate on whatever solution `p.parameters` holds.
- `Gecko(g.params)` — post-process a `Geope` result straight after `g.optimize(...)`. Because the optimisation functions are cached on `params`, this reuses `Geope`'s already-compiled functions rather than recompiling.

**The solution does not have to come from `Geope`.** `Gecko` operates on the current `params.parameters` — that array can be a `Geope` result, but it can equally be a solution found by any other method (a different optimiser, an analytic/hand-crafted pulse, an imported result, …). Just put the parameters into a `Parameters` object describing the same system (`basis`, `projected_basis`/`drift_basis`, `target`, `piecewise_steps`, and any `param_transform`) and call `Gecko(p)`; it refines the imported solution while preserving its fidelity. (When `params` has never been evaluated, `Gecko` computes the baseline fidelity itself on construction.)

When you pass a `Geope`'s `params`, the `Parameters` object is shared with that `Geope`, so a pass with `piecewise_steps_multiplier > 1` advances the shared state forward (`params.parameters` and `params.piecewise_steps` move to the new count together).

### Available objectives (methods on `Gecko`)

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

### Null-space algorithm: `Gecko._null_space_optimisation()`

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

# Run — updates params (parameters/fidelity) in place; returns the same Parameters
g = geope.Geope(params, history=geope.History())
result = g.optimize(max_steps=1000, precision=0.9999)
print(float(result.fidelity))        # final fidelity (lives on Parameters)
print(result.to_dict())              # current solution as a control dict
print(g.history.best_fidelity)       # best over the trajectory

# Null-space passes — fidelity preserved — live on Gecko, which
# shares the converged optimiser's Parameters (and its cached functions).
gk = geope.Gecko(g.params)
gk.smooth(piecewise_steps_multiplier=2, smoothing_rate=0.05, diff_tol=1e-3)
gk.smooth_frequency(smoothing_rate=0.05, diff_tol=1e-3)
gk.bound({"x": (-1, 1), "z": (-1, 1)}, method='projected_gradient')
gk.robust(parameter_labels=["XII", "IXI", "IIX"], delta=0.01)
gk.speed(parameter_labels=["XII", "IXI", "IIX"])
gk.length()
```

### Refining a solution from another method

`Gecko` does not require the solution to have been produced by `Geope`. Drop any
fidelity-achieving solution — from a different optimiser, an analytic construction,
or an imported result — into a `Parameters` describing the same system, then build a
`Gecko` directly from it:

```python
# `phi` is a (piecewise_steps, K_full) parameter array obtained elsewhere.
params = geope.Parameters(
    basis=full,
    control=control,
    drift=drift,
    target=target,
    piecewise_steps=phi.shape[0],
    init_values=phi,          # the externally-found solution
)

gk = geope.Gecko(params)          # no Geope needed
gk.smooth(piecewise_steps_multiplier=2, smoothing_rate=0.05, diff_tol=1e-3)
print(float(gk.params.fidelity))  # baseline computed on construction, preserved by the pass
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
