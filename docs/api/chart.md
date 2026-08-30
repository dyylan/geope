# geope.geometry.chart

The **pulse model**: the product of piecewise-constant exponentials every
manifold in the library is coordinatised by,

$$\Phi(\phi) = \prod_g \exp\Bigl(i\sum_k \phi_{g,k} G_k\Bigr),$$

together with the two ways of differentiating it. It belongs to no single
manifold — `MatrixLieGroup` takes it as its chart directly and `StateSphere`
composes it with a base state — so it sits below them, importing nothing but
JAX.

Each factory returns an un-jitted callable, so JIT compilation happens once,
lazily, when the optimiser's top-level `update_step` is first traced.

## The chart

::: geope.geometry.chart.compute_matrices_params_list_fn

::: geope.geometry.chart.get_compute_matrices_params_list_fn

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
| the `param_transform` chart wrapper | `geope.geometry.binding` — the one place that reads a `Parameters` |
