# geope.jax.hessian

Second derivatives of the pulse model, by two independent routes.

The **manual propagator** functions exploit the product-of-exponentials
structure of `geope.geometry.chart`: `hvp_propagator` gets a single direction in
$O(G)$ with one `jax.lax.scan`, while `hessian_propagator` materialises the dense
$(G, G, d, d, K, K)$ tensor. The **autodiff** pair works on any chart, including
the `param_transform` path where no exponential-product structure exists — which
is why `Manifold.hessian` falls back to it there.

`su_hessian_quadratic_form` and `stiefel_hessian_quadratic_form` are a different
object: the Riemannian Hessian of the squared geodesic distance, which supplies
`SpecialUnitaryGroup.hessian_quadratic_form` and
`Stiefel.hessian_quadratic_form`, and the curvature the second-order line
searches seed their step from. The two are shaped very differently — a
bi-invariant group's is a scalar function $\frac{\mathrm{ad}}2\coth
\frac{\mathrm{ad}}2$ of one adjoint operator, evaluated without ever forming it,
while a general Stiefel manifold is *not* symmetric and needs the blocks of an
operator exponential solving the Jacobi equation.

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

::: geope.jax.hessian.stiefel_hessian_quadratic_form
