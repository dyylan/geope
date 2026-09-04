# geope.jax.jacobian

First derivatives of the pulse model, in both directions.

`jacobian_propagator` builds the whole $(G, d, d, K)$ tensor
$\partial U/\partial\phi_{g,k}$; `vjp_propagator` contracts an ambient covector
against every one of those columns *without* building it. Both share the two
`jax.lax.scan` partial products (the gates before and after the one being
differentiated), and both are exact — they differ only in what they are for. A
Jacobian is what the geodesic least-squares problem needs, one column per
parameter; a gradient is not, and paying $O(G\,d^3K)$ to build a tensor you
immediately contract away is the difference between the two paths.

`jvp_propagator` is the third direction: one parameter-space direction pushed
forward, in $O(G)$, without either.

## The Jacobian

::: geope.jax.jacobian.jacobian_propagator

::: geope.jax.jacobian.get_jacobian_propagator

## The pullback

::: geope.jax.jacobian.vjp_propagator

::: geope.jax.jacobian.get_vjp_propagator

## The pushforward

::: geope.jax.jacobian.jvp_propagator

::: geope.jax.jacobian.get_jvp_propagator
