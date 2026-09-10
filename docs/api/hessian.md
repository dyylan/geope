# geope.jax.hessian

Second derivatives of the pulse model, by two independent routes.

The **manual propagator** functions exploit the product-of-exponentials
structure of `geope.geometry.chart`, in three shapes. `hvp_propagator` gets a
single direction in $O(G)$ with one `jax.lax.scan`. `hessian_vjp_propagator` gets
*every pair*, contracted against one ambient covector, by splitting the middle
propagator so that each half depends on one gate index only — two
$O(G)$-propagated derivative trajectories and one Gram matrix, which is the
acceleration of Goodwin & Vinding, *Phys. Rev. Research* **5**, L012042 (2023),
§III. `hessian_propagator` materialises the dense $(G, G, d, d, K, K)$ tensor at
$O(G^2K^2d^3)$ and $O(G^2K^2d^2)$, and is the **reference** the pullback is
checked against — the same standing `geope.geometry.chart.get_jacobian_fn` has
one order down.

The first two are live, landed on the base point by
`geope.geometry.chart.get_chart_hvp_fn` and
`geope.geometry.chart.get_chart_hessian_vjp_fn` and read off
`geope.geometry.tangent.TangentBundle`. Note these are derivatives of the
*propagator*, not of an objective: `Manifold.hessian` assembles the infidelity
Hessian from the pullback plus the manifold's own cost derivatives.

The **autodiff** pair works on any chart, including the `param_transform` path
where no exponential-product structure exists — which is why `Manifold.hessian`
falls back to it there. It is also the reference the manual path is tested
against.

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

::: geope.jax.hessian.hessian_vjp_propagator

::: geope.jax.hessian.get_hessian_vjp_propagator

::: geope.jax.hessian.hessian_propagator

::: geope.jax.hessian.get_hessian_propagator

## Autodiff

::: geope.jax.hessian.hvp_forward_over_reverse

::: geope.jax.hessian.get_hessian_fn

## Riemannian curvature

::: geope.jax.hessian.su_hessian_quadratic_form

::: geope.jax.hessian.stiefel_hessian_quadratic_form
