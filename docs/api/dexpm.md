# geope.jax.dexpm

::: geope.jax.dexpm.Ui

::: geope.jax.dexpm.get_Ui_fn

::: geope.jax.dexpm.dexpm_block

::: geope.jax.dexpm.dexpm

::: geope.jax.dexpm.dexpm_batched

::: geope.jax.dexpm.get_dexpm

## The adjoint

The per-gate transpose of the above: all `K` overlaps of one covector with
`dexpm`'s directions, from a single eigendecomposition and without ever forming
the `(d, d, K)` tensor. This is the step `geope.jax.get_vjp_propagator` is built
from.

::: geope.jax.dexpm.adj_expm_eig

::: geope.jax.dexpm.get_adj_expm_eig

::: geope.jax.dexpm.adj_expm

::: geope.jax.dexpm.get_adj_expm
