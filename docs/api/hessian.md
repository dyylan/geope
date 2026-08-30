# geope.jax.hessian

Second derivatives of the pulse model, by two independent routes.

The **manual propagator** functions exploit the product-of-exponentials
structure of `geope.geometry.chart`: `hvp_propagator` gets a single direction in
$O(G)$ with one `jax.lax.scan`, while `hessian_propagator` materialises the dense
$(G, G, d, d, K, K)$ tensor. The **autodiff** pair works on any chart, including
the `param_transform` path where no exponential-product structure exists — which
is why `Manifold.hessian` falls back to it there.

`su_hessian_quadratic_form` is a different object: the Riemannian Hessian of the
squared geodesic distance, which supplies
`SpecialUnitaryGroup.hessian_quadratic_form` and the curvature the second-order
line searches seed their step from.

## Manual propagator derivatives

::: geope.jax.hessian.hvp_propagator

::: geope.jax.hessian.get_hvp_propagator

::: geope.jax.hessian.hessian_propagator

::: geope.jax.hessian.get_hessian_propagator

## Autodiff

::: geope.jax.hessian.hvp_forward_over_reverse

::: geope.jax.hessian.get_hessian_fn

## Riemannian curvature

::: geope.jax.hessian.su_hessian_quadratic_form
