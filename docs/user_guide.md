# Using GEOPE with `Parameters`

A guide to the geodesic projective parameter-estimation optimiser for synthesising target unitaries on a Lie-algebraic control model, with practical notes on `Parameters`, experimental parameters via `param_transform`, and pulse-shape constraints.

---

## 1. What GEOPE does

### 1.1 The control problem

GEOPE solves the following gate-synthesis problem. You have:

- a target unitary $U_T \in U(d)$;
- a **full** Lie algebra basis $\{B_k\}_{k=1}^{K}$ of Hermitian generators spanning some subspace of $\mathfrak{u}(d)$ (in `geope`, normally Pauli strings);
- a **projected** sub-basis $\{B_k\}_{k \in \mathcal{C}}$ — the *controllable* generators, the ones the experimenter can drive;
- optionally, a **drift** sub-basis $\{B_k\}_{k \in \mathcal{D}}$ — generators with fixed, uncontrollable coefficients;
- a number of piecewise-constant gate segments $N_g \ge 1$.

The piecewise gate produced from a parameter array $\phi \in \mathbb{R}^{N_g \times K_{\text{pd}}}$ (proj+drift) is

$$
U(\phi) \;=\; \prod_{g=1}^{N_g} \exp\!\Bigl(i \sum_{k \in \mathcal{C} \cup \mathcal{D}} \phi_{g,k}\, B_k\Bigr).
$$

The goal is to find $\phi$ such that $U(\phi)$ matches $U_T$. "Matches" is measured by a fidelity:

$$
F_{\text{proj}}(U, U_T) \;=\; \frac{|\mathrm{Tr}(U_T^\dagger U)|}{d}, \qquad
F_{\text{full}}(U, U_T) \;=\; \frac{\mathrm{Re}\,\mathrm{Tr}(U_T^\dagger U)}{d}.
$$

$F_{\text{proj}} \in [0,1]$ is **projective** — invariant under global phase $U \mapsto e^{i\theta}U$; the default. $F_{\text{full}} \in [-1,1]$ is **phase-sensitive**; selected with `projective=False`.

### 1.2 The geodesic step

GEOPE is a first-order method that, at each iterate $U_k$, walks along the **geodesic** on $U(d)$ pointing toward $U_T$, projected onto the controllable directions in parameter space.

The geodesic generator at $U_k$ is

$$
g_k \;=\; -i\,\log\!\bigl(U_k^\dagger U_T\bigr) \;\in\; \mathfrak{u}(d).
$$

In projective mode the trace is removed, giving an $\mathfrak{su}(d)$ generator:

$$
g_k \;\leftarrow\; g_k - \tfrac{\mathrm{Tr}(g_k)}{d}\,\mathbb{1}.
$$

The geodesic *tangent* at $U_k$ is $\Gamma_k = U_k g_k$. Its projection onto the basis is

$$
\boldsymbol{\gamma}_k \;=\; \mathrm{proj}(\Gamma_k)/d \;\in\; \mathbb{R}^{K}.
$$

For each parameter $\phi_{g,k}$, GEOPE forms the projected partial derivative

$$
\boldsymbol{\omega}_{g,k} \;=\; \mathrm{proj}\!\bigl(i\,\partial U/\partial\phi_{g,k}\bigr) \;\in\; \mathbb{R}^{K}.
$$

The parameter step $\delta\phi$ is the least-squares solution to

$$
\Omega^\top \,\delta\phi \;\approx\; \boldsymbol{\gamma}_k, \qquad \Omega_{:,(g,k)} = \boldsymbol{\omega}_{g,k}.
$$

`coeffs = sol` is normalised to $\|\boldsymbol{\mathrm{coeffs}}\| = \sqrt{N_g}$. A one-dimensional line search along the ray $\phi + t \cdot \boldsymbol{\mathrm{coeffs}}$ for $t \in [-t_{\max}, 0]$ (the toward-target half-line; see the sign-convention note below) chooses the actual step. The new parameters are appended to the history.

When the geodesic step fails to improve fidelity, the optimiser falls back to a Gram–Schmidt-orthogonalised random direction.

### 1.3 Null-space refinement

After the main optimisation has converged, you can move within the *null space* of the parameter-to-unitary Jacobian to optimise auxiliary objectives without losing fidelity. The geometric picture: at the converged $\phi^\star$,

$$
\dim \ker J(\phi^\star) \;=\; \dim(\text{parameter space}) - \mathrm{rank}\bigl(J(\phi^\star)\bigr).
$$

If this kernel is non-trivial (a generically over-parameterised problem), each method projects a gradient of its cost onto $\ker J$ and takes a normalised step. Available costs:

| Method | Cost minimised | Notes |
| ------ | -------------- | ----- |
| `smooth(...)` | $\sum_g \|\phi_{g+1} - \phi_g\|^2$ | difference between adjacent segments |
| `smooth_frequency(...)` | $\sum_{m\ge 1}\|\widehat\phi(m)\|^2$ | high-frequency spectral power |
| `filter_frequency(filter_fn, ...)` | $\|\widehat\phi - \mathcal{F}(\widehat\phi)\|^2$ | distance to a user filter |
| `speed(params, ...)` | $\max_{g,k\in P}\|\phi_{g,k}\|$ | peak amplitude (raises gate-speed limit) |
| `length(params, ...)` | $\sum_g \|\phi_g\|_2$ | pulse length |
| `robust(params, delta, ...)` | $1 - \min_{|\delta_k|\le \Delta} F$ | worst-case fidelity under δ perturbations |
| `bound(parameter_bounds, ...)` | $\max(\phi - u_b, l_b - \phi)$ | enforce a box constraint |

These all preserve fidelity to within the rate × number-of-iterations tolerance.

### 1.4 Sign-convention note

The line-search interval is $[-t_{\max}, 0]$, not $[0, t_{\max}]$, because of how `coeffs` is oriented: solving $\Omega^\top \cdot \delta\phi = \boldsymbol\gamma$ yields a $\delta\phi$ such that stepping in **+coeffs** moves *away* from $U_T$, so the toward-target ray is **dt < 0**. The line search minimises the engine's `infid_U_fn` over that interval and reports `fidelity = 1 - infid`. You don't normally need to think about this — the convention is internal — but it's the answer to "why negative dt".

---

## 2. Building a `Parameters` object

`Parameters` is the high-level entry point that bundles every input the optimiser needs.

### 2.1 The basis

Pick a Pauli basis with one of the `utils` constructors:

```python
import geope

basis = geope.construct_full_pauli_basis(n=2)              # all 4^n - 1 Pauli strings
basis = geope.construct_two_body_pauli_basis(n=4)          # ≤ 2-body terms
basis = geope.construct_Heisenberg_pauli_basis(n=4)        # single-body + {XX, YY, ZZ}
basis = geope.construct_restricted_pauli_basis(n=4, restriction=['xx', 'yy'])
```

Each `Basis` carries:

- `basis.basis` — the $(K, d, d)$ tensor of Hermitian matrices.
- `basis.labels` — Pauli-string labels like `'XI'`, `'ZZ'`.
- `basis.plot_labels` — LaTeX-formatted labels for plotting.
- `basis.interaction_qubits` — qubit-index tuples per element.
- `basis.n`, `basis.dim`, `basis.lie_algebra_dim` — sizes.

### 2.2 Control and drift dictionaries

The **control** dict picks the controllable subset of `basis`:

```python
control = {
    1: ['x', 'y', 'z'],         # single-qubit controls on qubit 1
    2: ['x', 'y'],              # qubit 2: only X, Y
    (1, 2): ['xx', 'yy'],       # two-qubit coupling: XX, YY on the (1,2) pair
}
```

Keys are 1-indexed qubit identifiers (or tuples for multi-qubit interactions); values are lower-case interaction labels. Anything matching the dict is added to the projected basis.

The **drift** dict has the same format but selects generators with fixed coefficients:

```python
drift = {(1, 2): ['zz']}   # an always-on ZZ coupling between qubits 1 and 2
```

If you don't pass `drift_values`, drift coefficients default to ones.

### 2.3 The target

`target` is just an `np.ndarray` of shape $(d,d)$:

```python
import numpy as np
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
```

### 2.4 Piecewise gates and initial values

```python
params = geope.Parameters(
    basis=basis,
    control=control,
    drift=drift,
    target=H,
    piecewise_steps=8,
    init_values=None,         # → random in (-init_spread*π, init_spread*π)
    drift_values=None,        # → ones
    init_spread=0.1,
    seed=0,
)
```

`init_values` can be:

- `None` — random uniform.
- A dict in the same format as `control`, listing concrete starting amplitudes (broadcast across all $N_g$ segments).
- An `np.ndarray` of shape $(N_g, K_{\text{full}})$.

### 2.5 Linear equality constraints

`constraints` accepts either a list of `np.ndarray` vectors $c$ (length $K_{\text{proj}}$) representing $c \cdot \phi^{\text{proj}} = 0$, or a list of control-style dicts that are converted into such vectors. Internally `merge_constraints` resolves overlapping constraints before they are turned into an expander matrix that re-parameterises the projected space.

Use this to enforce, e.g., that two parameters share the same value across all gate segments.

### 2.6 Bounds

```python
params = geope.Parameters(
    ...,
    bounds={'x': (-0.5, 0.5), 'y': (-0.5, 0.5)},
)
```

This is only consumed by `Geope.bound(...)`, not the main GEOPE loop. See Section 4.

### 2.7 Phase-sensitive vs projective

Set `projective=False` to use $F_{\text{full}}$ rather than $F_{\text{proj}}$. Use this only when the global phase of $U_T$ is physically meaningful (e.g. matching a specific representative for downstream composition).

> **Caveat.** Phase-sensitive optimisation has a known pathology when $\mathrm{Tr}(U_T) = 0$: the gradient of $\mathrm{Re}\,\mathrm{Tr}(U_T^\dagger U)$ vanishes at the identity in every basis direction, so a random init close to $I$ may have no descent direction. Use a larger `init_spread` (≥ 0.5) or a non-zero `init_values` dict to break the symmetry.

---

## 3. Running the optimisation

The minimal three-line pattern:

```python
params = geope.Parameters(basis=basis, control=control, target=U_T,
                          piecewise_steps=8, seed=0)
result = geope.Geope(params, max_steps=500, precision=1 - 1e-7).optimize()
print(result.best_fidelity)
```

After `optimize()`, the `result` (which is the same `Parameters` instance) has its mutable history populated:

- `result.parameters` — list of $(N_g, K_{\text{full}})$ arrays, one per recorded step.
- `result.fidelities`, `result.infidelities` — per-step scalars.
- `result.step_sizes`, `result.steps` — step magnitudes and counters.
- `result.best_fidelity`, `result.best_parameters` — convenience accessors.
- `result.best_basis_coefficients` — best params mapped through `param_transform` if set.
- `result.to_dict()` — best solution as a human-readable control-style dict.

### 3.1 Optimiser knobs

These are on `Geope.__init__`, not on `Parameters`:

```python
geope.Geope(
    params,
    max_steps=500,                       # cap on iterations
    precision=1 - 1e-7,                  # stopping fidelity
    max_step_size=0.9,                   # line-search clip
    gram_schmidt_step_size=1.3,          # fallback step magnitude
    line_search_method='golden_section', # or 'difference_step'
    verbose=False,
)
```

### 3.2 Auxiliary passes

Post-convergence refinement does not require re-building the optimiser:

```python
g = geope.Geope(params, max_steps=500, precision=1 - 1e-7)
g.optimize()
g.smooth(piecewise_steps_multiplier=2, smoothing_rate=0.05, diff_tol=1e-3)
g.smooth_frequency(smoothing_rate=0.05, diff_tol=1e-3)
g.speed(parameter_labels=['X'], optimization_rate=0.05)
g.length(optimization_rate=0.01)
g.robust(parameter_labels=['X'], delta=0.02, num_samples=5)
g.bound({'X': (-0.5, 0.5)}, method='projected_gradient')
```

Each returns `(success, iters)`. `piecewise_steps_multiplier > 1` subdivides each existing gate segment into that many smaller ones (interpolating linearly), giving more degrees of freedom for the null-space pass to use.

---

## 4. Pulse-shape constraints (`pulse_constraints`)

### 4.1 What they enforce

A pulse constraint on parameter $k$ forces the time profile $\phi_k(g)$, $g=0,\dots,N_g-1$, to lie on a one-dimensional subspace spanned by a fixed unit template $t_k$:

$$
\phi_k(g) \;=\; \alpha_k\, t_k(g), \qquad \|t_k\| = 1, \quad \alpha_k \in \mathbb{R}.
$$

After every GEOPE iteration (and every null-space update), the time profile is re-projected:

$$
\phi_k \;\leftarrow\; \bigl(\phi_k \cdot t_k\bigr)\, t_k.
$$

This is what makes the constraint *strict* in practice rather than soft.

### 4.2 The expander

Concretely, the flat parameter vector $\Phi \in \mathbb{R}^{N_g K_{\text{proj}}}$ is replaced by free parameters $\psi \in \mathbb{R}^{n_{\text{free}}}$ via

$$
\Phi \;=\; E \psi, \qquad
n_{\text{free}} = N_g(K_{\text{proj}} - |P|) \;+\; |P|,
$$

where $P$ is the set of constrained indices and:

- each unconstrained $k$ contributes $N_g$ columns to $E$ (the identity on its time profile);
- each constrained $k \in P$ contributes a single column equal to $t_k$ down its $N_g$ rows.

When you also have linear-equality `constraints` with expander $C$, the combined expander used by the geodesic step is

$$
E_{\text{comb}} \;=\; (I_{N_g}\!\otimes\!C)\,\bigl(I_{N_g}\!\otimes\!C\bigr)^{+}\,E,
$$

i.e. $E$ projected onto the column space of the Kronecker-lifted $C$.

### 4.3 Template choice

The template $t_k$ is derived from the *current* solution at the moment `optimize()` is called:

- if the column $\phi_k$ from `self.parameters[-1]` has $\|\phi_k\| > 10^{-12}$, use $t_k = \phi_k / \|\phi_k\|$;
- otherwise fall back to the flat template $t_k = \mathbf{1}/\sqrt{N_g}$ (a square pulse).

So the typical pattern is to first run an unconstrained optimisation, inspect the pulse shape, decide which parameters should respect that shape, and *then* re-run with `pulse_constraints` to lock the shape in while allowing further refinement. Or to specify a desired template via `init_values` directly.

### 4.4 Specifying constraints

```python
params = geope.Parameters(
    ...,
    pulse_constraints={'X': True, 'Y': True},   # dict: keys are projected-basis labels
)
# or equivalently:
params = geope.Parameters(..., pulse_constraints=['X', 'Y'])
```

The values in the dict are ignored — only the keys matter. The label format follows the `Basis.labels` convention (e.g. `'X'`, `'YI'`, `'XX'`).

### 4.5 Worked example: flat X, free Y

```python
import numpy as np
import geope

# Single-qubit, target = Hadamard, piecewise gate over 8 segments
basis = geope.construct_full_pauli_basis(1)
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

params = geope.Parameters(
    basis=basis,
    control={1: ['x', 'y']},        # only X and Y controllable
    target=H,
    piecewise_steps=8,
    pulse_constraints={'X': True},  # X must be a single scalar × time shape
    seed=0,
)
result = geope.Geope(params, max_steps=400, precision=1 - 1e-6).optimize()

X_profile = result.best_parameters[:, basis.labels.index('X')]
print("X profile:", X_profile.real)
print("X profile / norm:", X_profile.real / np.linalg.norm(X_profile.real))
print("best fidelity:", float(result.best_fidelity))
```

After convergence the X column will be a single template repeated across segments; only the Y column has independent values per segment.

---

## 5. Experimental parameters (`param_transform`)

### 5.1 When to use

GEOPE's "native" parameters are basis coefficients $\phi^{\text{proj}}_{g,k}$ — pure real numbers indexing $\sum_k \phi_{g,k} B_k$. But in practice, the **experimentally controllable** quantities are often different:

- a single drive amplitude that couples to multiple basis elements through a $\sin$/$\cos$ phase;
- a pulse parametrised by a small number of shape coefficients (Gaussian, Slepian, etc.);
- a calibration map $\phi^{\text{proj}} = f(\text{voltage}, \text{frequency})$ with $f$ non-linear.

You want to optimise directly over those experimental knobs $\phi^{\text{exp}}$, with the algorithm internally seeing the corresponding basis coefficients through a user-supplied callable $\tau$:

$$
\phi^{\text{proj}}_{g,\cdot} \;=\; \tau\bigl(\phi^{\text{exp}}_{g,\cdot}\bigr) \;\;\;\text{or}\;\;\; \tau\bigl(\phi^{\text{exp}}_{g,\cdot},\, g\bigr).
$$

### 5.2 The callable contract

`param_transform` must be a JAX-traceable callable. Two signatures are accepted:

- **Step-independent**: `tau(phi)` where `phi.shape == (n_experimental_params,)`.
- **Step-dependent**: `tau(phi, step_index)` where `step_index` is a scalar `int32`.

The output must be a 1-D array. Its length is either:

- equal to `projected_basis.lie_algebra_dim` — the coefficients are taken as the projected-basis coefficients directly; or
- equal to `basis.lie_algebra_dim` — the relevant projected entries are extracted automatically via `projected_basis.overlap(basis)`.

`Parameters.n_experimental_params` controls the input dimension; if you don't pass it, it defaults to `projected_basis.lie_algebra_dim`.

### 5.3 Helper: `make_per_element_transform`

For the common case where each experimental parameter maps to one basis coefficient through an independent scalar function:

```python
import jax.numpy as jnp
import geope

tau = geope.make_per_element_transform([
    jnp.cos,                       # phi[0] → cos(phi[0])    (e.g. X coefficient)
    jnp.sin,                       # phi[1] → sin(phi[1])    (e.g. Y coefficient)
    lambda x: 0.5 * x,             # phi[2] → 0.5 * phi[2]  (e.g. Z coefficient)
    None,                          # phi[3] passes through unchanged
])
```

### 5.4 Worked example: Rabi rotation in $(A, \varphi)$

A common Rabi-drive parametrisation is one amplitude $A$ and one phase $\varphi$, coupling to the $(X, Y)$ generators as

$$
H_{\text{drive}}(t) = A \,\cos(\varphi)\, X \;+\; A \,\sin(\varphi)\, Y.
$$

```python
import numpy as np
import jax.numpy as jnp
import geope

basis = geope.construct_full_pauli_basis(1)   # K_proj = 3 (X, Y, Z labelled XI/YI/ZI? no, just X,Y,Z)

def rabi_transform(phi):                        # phi = (A, varphi)
    A, varphi = phi[0], phi[1]
    return jnp.array([A * jnp.cos(varphi),      # X coefficient
                      A * jnp.sin(varphi),      # Y coefficient
                      0.0])                     # Z coefficient

theta = np.pi / 3
RX = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
               [-1j*np.sin(theta/2),  np.cos(theta/2)]], dtype=complex)

params = geope.Parameters(
    basis=basis,
    control={1: ['x', 'y', 'z']},
    target=RX,
    piecewise_steps=4,
    param_transform=rabi_transform,
    n_experimental_params=2,                    # phi has 2 entries per segment
    init_spread=0.3,
    seed=0,
)

result = geope.Geope(params, max_steps=300, precision=1 - 1e-7).optimize()
print("final fidelity:", float(result.best_fidelity))
print("best (A, varphi) per segment:\n", result.best_parameters)
print("induced basis coefficients:\n", result.best_basis_coefficients)
```

### 5.5 Internal consequences

When `param_transform` is set, the engine's `compute_U_fn` is wrapped to apply $\tau$ via `jax.vmap` over the gate axis, then embed into the proj+drift parameter slots before delegating to the original unitary-product code. The Jacobian is replaced by a split-real-imaginary version because user-supplied $\tau$ functions often carry real-valued intermediates that lose imaginary parts under the usual holomorphic Jacobian. Engine index masks are overridden so the rest of `Geope` (line search, Gram–Schmidt fallback, null-space passes) operate uniformly over $\phi^{\text{exp}}$.

You don't need to think about any of this; it's all transparent. But it has two practical implications:

- **No `parameter_labels` argument in null-space methods.** When `param_transform` is set, the projected basis labels no longer correspond to the indices of the optimised parameters. Pass `parameter_indices` (integers into $\phi^{\text{exp}}$) instead. `Geope` raises a clear `ValueError` otherwise.
- **dtype.** Internally, experimental parameters are `float64`; basis-coefficient mode uses `complex128`. Tolerances and bounds you supply should match.

---

## 6. Auxiliary null-space methods in detail

All five methods follow the same template:

1. Compute the omega tensor and its null space `ker(Ω) ≈ I - Ω^+ Ω`.
2. Form the cost gradient $\nabla C(\phi)$ via `jax.value_and_grad`.
3. Project $-\nabla C$ onto $\ker(\Omega)$ via a least-squares solve.
4. Take a normalised step `rate * x / ||x||` along the projection.

Because the step stays in the null space, fidelity is preserved to first order at each iteration. Over many iterations small drift accumulates, so the auxiliary passes run with `diff_tol` rather than a precise fidelity guard — they stop when the auxiliary cost stops improving, not when fidelity drops below threshold.

### 6.1 `smooth_frequency`

```python
g.smooth_frequency(smoothing_rate=0.05, max_smoothing_steps=200, diff_tol=1e-3)
```

Cost is the mean spectral power above DC,

$$
C(\phi) = \frac{1}{(N_g/2)\,K_{\text{proj}}}\sum_{m \ge 1, k} \bigl|\widehat{\phi_k}(m)\bigr|^2,
$$

with $\widehat{\phi_k} = \mathrm{rfft}_g \phi_k$ along the gate axis. DC ($m=0$) is excluded so the average amplitude is not penalised — only the *variation* is.

### 6.2 `filter_frequency`

```python
def low_pass(rfft_array):           # rfft_array.shape = (N_g//2 + 1, K_proj)
    cutoff = 2
    mask = jnp.arange(rfft_array.shape[0]) < cutoff
    return rfft_array * mask[:, None]

g.filter_frequency(low_pass, smoothing_rate=0.05, diff_tol=1e-3)
```

Cost is $\|\widehat\phi - \mathcal{F}(\widehat\phi)\|^2$. By Parseval this equals the time-domain $L^2$ distance to the filtered pulse, so minimising it drives $\phi$ toward its filtered version while preserving fidelity.

### 6.3 `speed`, `length`

`speed` minimises $\max_{g, k \in P}|\phi_{g,k}|$ — the peak control amplitude. `length` minimises $\sum_g \sqrt{\sum_{k\in P}\phi_{g,k}^2 + \|d_g\|^2}$ — the total pulse length, including a constant drift contribution. Both reduce the energy you have to put into the gate.

```python
g.speed(parameter_labels=['X', 'Y'], optimization_rate=0.05, diff_tol=1e-3)
g.length(optimization_rate=0.01, diff_tol=1e-3)
```

### 6.4 `robust`

```python
g.robust(parameter_labels=['X'], delta=0.02, num_samples=5,
         optimization_rate=0.05, diff_tol=1e-3)
```

Cost is $1 - \min_{\boldsymbol{\delta} \in \mathcal{S}} F\bigl(U(\phi + \sum_k \delta_k e_k)\bigr)$, where $\mathcal{S}$ is the Cartesian grid of `num_samples` evenly-spaced values in $[-\Delta, +\Delta]$ for each parameter in $P$. Minimising this maximises the worst-case fidelity over that perturbation box. Each $\delta_k$ is applied **uniformly** to all gate segments (so it models a systematic calibration error, not per-segment noise). The total grid is $\text{num\_samples}^{|P|}$ points — keep $|P|$ small.

### 6.5 `bound`

```python
g.bound({'X': (-0.5, 0.5)}, method='projected_gradient', bounding_rate=0.05)
```

Two methods:

- `'mid_point'` / `'mp'` — pulls parameters toward the centre of the feasible box via a least-squares projection.
- `'projected_gradient'` / `'pg'` — gradient of the maximum-violation $\max(\phi - u_b, l_b - \phi)$, projected onto the null space.

`pg` typically converges faster when most parameters are already feasible; `mp` is gentler and more useful when starting from a heavily-violating point.

---

## 7. Phase-sensitive vs projective

The default `projective=True` should be your first choice. Use `projective=False` only when:

- you need the absolute phase of the gate (e.g. matching a stored reference rather than an equivalence class);
- the gate is a sub-block of a larger, coherent unitary so global phase becomes a relative phase against spectators;
- you're stitching several gates and the inter-gate phases enter the composite fidelity.

The two are coupled internally: `projective=False` activates the U-geodesic (no trace subtraction) **and** rebinds `infid_U_fn` to $1 - F_{\text{full}}$. The line search is correct for both modes (it minimises infidelity, which is always non-negative). The two pathologies to watch out for:

- **Traceless targets.** When $\mathrm{Tr}(U_T) = 0$ (e.g. Hadamard, single-qubit Pauli $X$, $Y$, $Z$, etc.), $F_{\text{full}}$ has a vanishing gradient at $U = I$ in every controllable direction. A random initialisation near the identity will have no descent direction. Use larger `init_spread` (≥ 0.5) or seed `init_values` away from zero.
- **Stopping criterion.** `precision = 0.9999999` makes sense for $F_{\text{proj}} \in [0,1]$. For $F_{\text{full}}$ the same number still works *near the optimum* (both fidelities agree as $U \to U_T$), but the optimiser may pass through negative-fidelity regions on the way. That's fine — it isn't a stopping criterion failure, just the geometry of the cost surface.

---

## 8. Full worked example

A two-qubit XX-coupled chain, controllable only on $\{X_i, Y_i, ZZ\}$, with `param_transform` to drive XY rotations by a single amplitude and phase per qubit, pulse-shape constraint on the coupling, post-optimisation smoothing and robustness.

```python
import numpy as np
import jax.numpy as jnp
import geope

# --- 1. Target: SWAP gate
SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=complex)

# --- 2. Basis: full 2-qubit Pauli
basis = geope.construct_full_pauli_basis(2)

# --- 3. Controllable subset: single-qubit XY drives + a tunable ZZ coupling
control = {
    1:      ['x', 'y'],
    2:      ['x', 'y'],
    (1, 2): ['zz'],
}

# --- 4. (No drift in this example)
# --- 5. Experimental parametrisation:
#     phi^exp per segment = (A1, varphi1, A2, varphi2, J_zz)  →  basis coefficients

def tau(phi):
    A1, p1, A2, p2, Jzz = phi
    out = jnp.zeros(5)                  # 5 controllable basis elements
    # The projected basis ordering follows basis.labels, filtered by `control`.
    # In this example the order happens to be (X1, Y1, X2, Y2, ZZ).
    out = out.at[0].set(A1 * jnp.cos(p1))   # X on qubit 1
    out = out.at[1].set(A1 * jnp.sin(p1))   # Y on qubit 1
    out = out.at[2].set(A2 * jnp.cos(p2))   # X on qubit 2
    out = out.at[3].set(A2 * jnp.sin(p2))   # Y on qubit 2
    out = out.at[4].set(Jzz)                # ZZ
    return out

# --- 6. Build Parameters with the experimental parametrisation
params = geope.Parameters(
    basis=basis,
    control=control,
    target=SWAP,
    piecewise_steps=12,
    param_transform=tau,
    n_experimental_params=5,
    pulse_constraints=[4],          # by INDEX (not label) — required with param_transform
    init_spread=0.3,
    seed=0,
)

# --- 7. Optimise
g = geope.Geope(params, max_steps=2000, precision=1 - 1e-7, verbose=False)
g.optimize()
print(f"main optimisation: F = {float(params.best_fidelity):.7f}")

# --- 8. Post-process: smooth in time, then enforce robustness on the
#       Rabi amplitudes (parameter indices 0 and 2).
g.smooth(piecewise_steps_multiplier=2, smoothing_rate=0.05, diff_tol=1e-3)
print(f"after smoothing:  F = {float(params.best_fidelity):.7f}")

g.robust(parameter_indices=(0, 2), delta=0.02, num_samples=5,
         optimization_rate=0.05, diff_tol=1e-3)
print(f"after robustness: F = {float(params.best_fidelity):.7f}")

# --- 9. Inspect the final solution
print("Best (A1, varphi1, A2, varphi2, Jzz) per segment:")
print(np.round(params.best_parameters, 4))
```

The patterns to take from this example:

- The experimental parametrisation lives entirely in `tau`. Optimisation never sees the raw basis coefficients on the input side.
- Pulse constraints reference parameter **indices** (not labels) because labels don't make sense in experimental space.
- Robustness, smoothing, and other auxiliaries also reference indices and otherwise look identical to the non-`param_transform` case.
- `params.best_basis_coefficients` gives you the induced basis coefficients $\tau(\phi^{\text{exp}})$ across all gate segments, in case you want them for downstream analysis (e.g. computing the realised Hamiltonian).

---

## 9. Cheat sheet

| Task | Code |
| ---- | ---- |
| All Pauli strings on n qubits | `geope.construct_full_pauli_basis(n)` |
| Only ≤2-body terms | `geope.construct_two_body_pauli_basis(n)` |
| Heisenberg (single + XX/YY/ZZ) | `geope.construct_Heisenberg_pauli_basis(n)` |
| Build Parameters | `geope.Parameters(basis=..., control=..., target=..., piecewise_steps=...)` |
| Run | `geope.Geope(params, max_steps=..., precision=...).optimize()` |
| Best fidelity | `params.best_fidelity` |
| Best parameters | `params.best_parameters` |
| Solution as a dict | `params.to_dict()` |
| Phase-sensitive | `Parameters(..., projective=False)` |
| Pulse-shape constraint | `Parameters(..., pulse_constraints=['X','Y'])` |
| Experimental parameters | `Parameters(..., param_transform=tau, n_experimental_params=K)` |
| Smooth in time | `Geope(...).smooth(...)` |
| Suppress high frequencies | `Geope(...).smooth_frequency(...)` |
| Apply a frequency filter | `Geope(...).filter_frequency(filter_fn, ...)` |
| Limit peak amplitude | `Geope(...).speed(parameter_labels=[...], ...)` |
| Limit total pulse length | `Geope(...).length(parameter_labels=[...], ...)` |
| Robustness to δ noise | `Geope(...).robust(parameter_labels=[...], delta=..., ...)` |
| Enforce box bounds | `Geope(...).bound({label: (lo, hi)}, ...)` |
