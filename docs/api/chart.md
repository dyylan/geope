# geope.geometry.chart

The **ambient layer**. Every manifold in the library is a submanifold of one
ambient space $\mathcal A = \mathbb C^{N\times m}$, and the pulse acts on all of
them the same way — by left multiplication with a product of piecewise-constant
exponentials:

$$U(\phi) = \prod_g \exp\Bigl(i\sum_k \phi_{g,k} G_k\Bigr),
\qquad \Phi(\phi) = U(\phi)\,x_0 .$$

The chart is therefore the **orbit map** of that one action through a base point
$x_0 = \Phi(0)$: the identity on a matrix group (where the propagator *is* the
point), a state on `StateSphere`, a frame on `Stiefel`. Only $x_0$ varies, which
is why no manifold writes chart code of its own — `Manifold.bind` composes what
is here, once.

This module owns the whole **jet** $(\Phi,\ \mathrm D\Phi,\ \mathrm D^2\Phi)$,
i.e. everything valued in $\mathcal A$. What happens in the tangent space — the
metric, the coefficient frame, the logarithm, the fidelity — is the
[manifold's](manifolds.md), so the two never mix. It imports nothing but JAX and
`geope.jax`, sitting *below* the manifolds rather than beside them.

Each factory returns an un-jitted callable, so JIT compilation happens once,
lazily, when the optimiser's top-level `update_step` is first traced.

## The propagator

::: geope.geometry.chart.compute_matrices_params_list_fn

::: geope.geometry.chart.get_compute_matrices_params_list_fn

## The chart, and its second differential

::: geope.geometry.chart.get_chart_fn

::: geope.geometry.chart.get_chart_hvp_fn

## Its Jacobians

::: geope.geometry.chart.get_jacobian_fn

::: geope.geometry.chart.get_split_jacobian_fn

## Related

The rest of what used to live in `engine.py` moved to the owner of the
mathematics in question:

| what | where |
| --- | --- |
| the fidelity formulas, and the manual propagator Hessian | [manifolds](manifolds.md) — they are the *group's* |
| the autodiff Hessian and its HVP | [hessian](hessian.md) — `geope.jax.hessian` |
| the `param_transform` chart wrapper | `geope.parameters` — the one place that reads a `Parameters` |
