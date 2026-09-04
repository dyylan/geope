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

## The chart

::: geope.geometry.chart.get_chart_fn

## The landed jet

The whole of $(\mathrm D\Phi,\ \mathrm D\Phi^\intercal,\ \mathrm D^2\Phi)$, built
from the propagator recursions in `geope.jax` and landed on the base point. Right
multiplication by $x_0$ is linear, so it commutes with differentiation and each
term lands termwise — which is why a homogeneous space reuses the group's
machinery wholesale and no manifold writes chart code of its own.

::: geope.geometry.chart.get_chart_jacobian_fn

::: geope.geometry.chart.get_chart_vjp_fn

::: geope.geometry.chart.get_chart_hvp_fn

::: geope.geometry.chart.get_chart_hessian_fn

## The autodiff Jacobians

`get_jacobian_fn` is the reference the manual path is tested against.
`get_split_jacobian_fn` and `get_split_vjp_fn` are the `param_transform` path's
only option: a user-supplied transform has no exponential-product structure to
pull back through, and its real intermediates would lose their imaginary part
under a holomorphic derivative.

::: geope.geometry.chart.get_jacobian_fn

::: geope.geometry.chart.get_split_jacobian_fn

::: geope.geometry.chart.get_split_vjp_fn

## Related

The rest of what used to live in `engine.py` moved to the owner of the
mathematics in question:

| what | where |
| --- | --- |
| the fidelity formulas | [manifolds](manifolds.md) — they are the *group's* |
| the objective's gradient and Hessian, assembled from this jet | [manifolds](manifolds.md) — `Manifold.value_and_grad` / `.hessian` |
| the autodiff Hessian and its HVP | [hessian](hessian.md) — `geope.jax.hessian` |
| the `param_transform` chart wrapper | `geope.parameters` — the one place that reads a `Parameters` |
