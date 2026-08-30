"""Binding a `Parameters` object's configuration to its manifold."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax import Array

from .chart import get_jacobian_fn, get_split_jacobian_fn
from .lie.pauli_projector import get_project_omegas_fn, get_project_omegas_fn_otf
from .manifold import Manifold
from .tangent import TangentBundle

if TYPE_CHECKING:
    from ..parameters import Parameters


def bind_manifold(params: "Parameters") -> Manifold:
    """Bind ``params``'s manifold to the pulse chart ``params`` describes.

    Assembles the chart $\\Phi$ (built by the manifold from the proj+drift
    generators, and wrapped when ``params.param_transform`` is set), its
    Jacobian, the coefficient-space projector and the mask of solvable columns —
    then hands them to `Manifold.bind`.

    This is the one place that knows both `Parameters` and the geometry layer;
    it is reached through the memoised `Parameters.manifold`, so a `Geope` and a
    `Gecko` sharing one `Parameters` share the callables and JAX reuses their
    compiled traces.

    Args:
        params: The `Parameters` describing the system.

    Returns:
        The bound `Manifold`.
    """
    compute_U = params.manifold_spec.chart(params.proj_drift_basis)
    if params.param_transform is None:
        jacobian = get_jacobian_fn(compute_U)
        # The chart is a plain product of exponentials in these generators, so
        # the manual propagator HVP and Hessian apply.
        generators = params.proj_drift_basis
        # Only the projected columns are solvable; the drift is held fixed.
        columns = params.proj_indices_projdrift_basis
    else:
        compute_U = wrap_compute_U_param_transform(params, compute_U)
        # Holomorphic autodiff through the real-valued user transform would drop
        # the imaginary part of the intermediates.
        jacobian = get_split_jacobian_fn(compute_U)
        # Experimental space: no exponential-product structure to exploit, and
        # every column is free.
        generators = None
        columns = None

    if params.basis.n > 5:
        # Materialising the full projector is too memory-heavy up here.
        project = get_project_omegas_fn_otf(params.basis, batch_size=None)
    else:
        project = get_project_omegas_fn(params.basis)

    tangent = TangentBundle(
        basis=params.basis,
        project=project,
        jacobian=jacobian,
        hvp=None if generators is None else params.manifold_spec.chart_hvp(generators),
        generators=generators,
        columns=columns,
    )
    return params.manifold_spec.bind(
        target=params.target, compute_U=compute_U, tangent=tangent
    )


def wrap_compute_U_param_transform(
    params: "Parameters", raw_compute_U: Callable[[Array], Array]
) -> Callable[[Array], Array]:
    r"""Wrap the chart to honour ``params.param_transform``.

    The user-facing experimental parameters $\phi^{\mathrm{exp}}$ are mapped to
    projected-basis coefficients via ``params.param_transform`` (possibly
    step-dependent), embedded into the proj+drift basis, and combined with the
    drift before the original ``raw_compute_U`` is called — so the whole pipeline
    then runs uniformly on $\phi^{\mathrm{exp}}$.

    It lives here rather than with the chart because it is the one piece of chart
    plumbing that reads a `Parameters`: the index masks, the drift values and the
    transform itself all come off it.

    Returned un-jitted so it fuses into the enclosing ``@jax.jit`` update step
    on first ``optimize()``.

    Args:
        params: The `Parameters` object carrying ``param_transform``.
        raw_compute_U: The projected-basis chart.

    Returns:
        The wrapped experimental-space chart.
    """
    n_exp = params.n_experimental_params
    n_proj_drift = params.proj_drift_basis.lie_algebra_dim
    proj_idx_pd = params.proj_indices_projdrift_basis
    drift_idx_pd = params.drift_indices_projdrift_basis

    # Detect step-dependence: tau(phi) vs tau(phi, step_index)
    _step_dependent = len(inspect.signature(params.param_transform).parameters) >= 2

    # Detect whether transform outputs full-basis or projected-basis coefficients
    _test_out = (
        params.param_transform(jnp.zeros(n_exp), 0)
        if _step_dependent
        else params.param_transform(jnp.zeros(n_exp))
    )
    tf_out_dim = _test_out.shape[0]
    n_proj = params.projected_basis.lie_algebra_dim
    if tf_out_dim != n_proj:
        _extract = jnp.array(
            np.where(np.array(params.projected_basis.overlap(params.basis)))[0]
        )
    else:
        _extract = None

    if params.drift_parameters is not None:
        _drift = jnp.array(params.drift_parameters, dtype=jnp.float64)
    else:
        _drift = None

    def _wrapped_compute_U(
        exp_params,
        _raw=raw_compute_U,
        _tf=params.param_transform,
        _pi=proj_idx_pd,
        _di=drift_idx_pd,
        _npd=n_proj_drift,
        _dr=_drift,
        _ext=_extract,
        _step_dep=_step_dependent,
    ):
        if _step_dep:
            ctrl = jax.vmap(_tf)(exp_params, jnp.arange(exp_params.shape[0]))
        else:
            ctrl = jax.vmap(_tf)(exp_params)
        if _ext is not None:
            ctrl = ctrl[:, _ext]
        # Promote dtype so complex tracing through real intermediates works
        _dtype = jnp.result_type(ctrl.dtype, exp_params.dtype)
        ctrl = ctrl.astype(_dtype)
        full = jnp.zeros((exp_params.shape[0], _npd), dtype=_dtype)
        full = full.at[:, _pi].set(ctrl)
        if _dr is not None:
            full = full.at[:, _di].set(
                jnp.broadcast_to(
                    _dr.astype(_dtype), (exp_params.shape[0], _dr.shape[0])
                )
            )
        return _raw(full)

    return _wrapped_compute_U
