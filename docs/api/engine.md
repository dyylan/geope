# geope.engine

`engine.py` is a collection of **pure function factories**. Each returns an
un-jitted callable; JIT compilation happens once, lazily, when the optimiser's
top-level `update_step` is first traced. The optimisers read these (lazily
built and cached) off the `Parameters` object — there is no engine class.

## Fidelity / infidelity

::: geope.engine.fidelity

<!-- ::: geope.engine.get_fidelity_fn
::: geope.engine.infidelity
::: geope.engine.get_infidelity_fn
::: geope.engine.fidelity_full
::: geope.engine.infidelity_full -->

## Unitary computation

::: geope.engine.get_compute_matrices_params_list_fn

## Geodesic, Jacobian, Hessian, gammas & omegas

::: geope.engine.get_geodesic_hamiltonian_fn

::: geope.engine.get_jacobian_fn

::: geope.engine.get_gammas_fn

::: geope.engine.get_omegas_fn

::: geope.engine.get_gammas_and_omegas_fn

::: geope.engine.get_hessian_fn

## param_transform helpers

::: geope.engine.wrap_compute_U_param_transform

::: geope.engine.get_split_jacobian_fn
