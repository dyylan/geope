from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_enable_x64", True)

from .jax.logm import logm
from .jax.dexpm import (
    get_Ui_fn,
    get_dexpm,
    get_dexpm_eig,
    get_d2expm,
    get_d2expm_eig,
)
from .jax.jacobian import manual_jacobian
from .jax.hessian import manual_hessian

import inspect
from functools import partial
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .parameters import Parameters


def fidelity(unitary: Array, target_unitary: Array) -> Array:
    """Compute the fidelity between a unitary and a target unitary.

    The fidelity is defined as the normalised absolute value of the
    Hilbert-Schmidt inner product between the two matrices.

    Args:
        unitary: The unitary ``Array`` to evaluate.
        target_unitary: The target unitary ``Array``.

    Returns:
        A scalar fidelity ``Array`` in the range $[0, 1]$.
    """
    return jnp.abs(jnp.einsum("ji,ji->", target_unitary.conj(), unitary)) / len(
        target_unitary[0]
    )


def get_fidelity_fn(target_unitary: Array) -> Callable[[Array], Array]:
    """Create a partial fidelity function with a fixed target unitary.

    Args:
        target_unitary: The target unitary ``Array`` to bind.

    Returns:
        A ``Callable[[Array], Array]`` that accepts a single unitary
        and returns the fidelity against ``target_unitary``.
    """
    return partial(fidelity, target_unitary=target_unitary)


def infidelity(unitary: Array, target_unitary: Array) -> Array:
    """Projective infidelity $1 - F_{\\mathrm{proj}}(U, U_T)$.

    Args:
        unitary: The unitary ``Array`` to evaluate.
        target_unitary: The target unitary ``Array``.

    Returns:
        A scalar infidelity ``Array`` in $[0, 1]$.
    """
    return 1 - jnp.abs(jnp.einsum("ji,ji->", target_unitary.conj(), unitary)) / len(
        target_unitary[0]
    )


def get_infidelity_fn(target_unitary: Array) -> Callable[[Array], Array]:
    """Create a partial projective-infidelity function with a fixed target.

    Args:
        target_unitary: The target unitary ``Array`` to bind.

    Returns:
        A ``Callable[[Array], Array]`` returning $1 - F_{\\mathrm{proj}}$.
    """
    return partial(infidelity, target_unitary=target_unitary)


def fidelity_full(unitary: Array, target_unitary: Array) -> Array:
    """Phase-sensitive (non-projective) fidelity.

    $F_{\\mathrm{full}}(U, U_T) = \\mathrm{Re}\\,\\mathrm{Tr}(U_T^\\dagger U) / d$.
    Unlike the projective fidelity, this is sensitive to a global phase
    on $U$ and lies in $[-1, 1]$.

    Args:
        unitary: The unitary ``Array`` to evaluate.
        target_unitary: The target unitary ``Array``.

    Returns:
        A scalar fidelity ``Array`` in $[-1, 1]$.
    """
    return jnp.real(jnp.einsum("ji,ji->", target_unitary.conj(), unitary)) / len(
        target_unitary[0]
    )


def get_fidelity_full_fn(target_unitary: Array) -> Callable[[Array], Array]:
    """Create a partial phase-sensitive fidelity function with a fixed target.

    Args:
        target_unitary: The target unitary ``Array`` to bind.

    Returns:
        A ``Callable[[Array], Array]`` returning $F_{\\mathrm{full}}$.
    """
    return partial(fidelity_full, target_unitary=target_unitary)


def infidelity_full(unitary: Array, target_unitary: Array) -> Array:
    """Phase-sensitive infidelity $1 - F_{\\mathrm{full}}(U, U_T)$.

    Args:
        unitary: The unitary ``Array`` to evaluate.
        target_unitary: The target unitary ``Array``.

    Returns:
        A scalar infidelity ``Array`` in $[0, 2]$.
    """
    return 1 - jnp.real(jnp.einsum("ji,ji->", target_unitary.conj(), unitary)) / len(
        target_unitary[0]
    )


def get_infidelity_full_fn(target_unitary: Array) -> Callable[[Array], Array]:
    """Create a partial phase-sensitive infidelity function with a fixed target.

    Args:
        target_unitary: The target unitary ``Array`` to bind.

    Returns:
        A ``Callable[[Array], Array]`` returning $1 - F_{\\mathrm{full}}$.
    """
    return partial(infidelity_full, target_unitary=target_unitary)


def compute_matrices_params_list_fn(params_list: Array, basis: Array) -> Array:
    """Compute the product unitary from a list of parameter vectors.

    For each parameter vector in `params_list`, constructs a Hamiltonian
    as a linear combination of the `basis` elements, exponentiates it,
    and accumulates the product unitary via `jax.lax.scan`.

    Args:
        params_list: ``Array`` of shape ``(piecewise_steps, K)`` where each row
            contains the Lie-algebra coefficients for one gate segment.
        basis: ``Array`` of shape ``(K, d, d)`` of Hermitian basis matrices.

    Returns:
        The product unitary ``Array`` of shape ``(d, d)``.
    """

    def step(U, params):
        A = jnp.tensordot(params, basis, axes=[[-1], [0]])
        Ui = jax.scipy.linalg.expm(1j * A)
        U_new = jnp.matmul(Ui, U)
        return U_new, None

    U0 = jnp.eye(basis.shape[1], dtype=basis.dtype)
    U_final, _ = jax.lax.scan(step, U0, jnp.stack(params_list))
    return U_final


def get_compute_matrices_params_list_fn(basis: np.ndarray) -> Callable[[Array], Array]:
    """Create a partial unitary-computation function with a fixed basis.

    Args:
        basis: Array of shape ``(K, d, d)`` of Hermitian basis matrices.

    Returns:
        A ``Callable[[Array], Array]`` that accepts a parameter list
        and returns the product unitary.
    """
    return partial(compute_matrices_params_list_fn, basis=basis)


def geodesic_hamiltonian(
    unitary: Array,
    target_unitary: Array,
    projective: bool = True,
    key: Array = jax.random.key(0),
) -> Array:
    """Compute the geodesic Hamiltonian between a unitary and a target.

    Computes the generator $g = -i\\log(U^\\dagger U_T) \\in \\mathfrak{u}(d)$
    and returns $U g'$ where $g' = g - \\frac{\\mathrm{Tr}(g)}{d}\\mathbb{1}$
    (the SU part) when ``projective=True``, or $g' = g$ (full U) when
    ``projective=False``.

    Args:
        unitary: The current unitary ``Array``.
        target_unitary: The target unitary ``Array``.
        projective: If ``True``, subtract the global-phase generator
            (SU geodesic). If ``False``, keep it (U geodesic).
            Defaults to ``True``.
        key: JAX random key forwarded to ``logm``. Defaults to
            ``jax.random.key(0)``.

    Returns:
        The geodesic tangent ``Array`` $U g'$ at the current unitary.
    """
    g = -1.0j * logm(jnp.einsum("ji,jk->ik", unitary.conj(), target_unitary), key=key)
    if projective:
        Id = jnp.eye(g.shape[0])
        global_phase = jnp.real(jnp.einsum("ij,ji->", Id, g)) / g.shape[0]
        g = g - global_phase * Id
    return unitary @ g


def get_geodesic_hamiltonian_fn(
    target_unitary: Array, projective: bool = True
) -> Callable[[Array, Array], Array]:
    """Create a partial geodesic Hamiltonian function with a fixed target.

    Args:
        target_unitary: The target unitary ``Array`` to bind.
        projective: If ``True``, return the projective (SU) geodesic.
            Defaults to ``True``.

    Returns:
        A ``Callable[[Array, Array], Array]`` that accepts a unitary and a
        JAX random key and returns the geodesic Hamiltonian.
    """
    return partial(
        geodesic_hamiltonian, target_unitary=target_unitary, projective=projective
    )


def hvp_forward_over_reverse(
    f: Callable[[Array], Array], params: Array, v: Array
) -> Array:
    """Compute a Hessian-vector product via forward-over-reverse mode.

    Args:
        f: Scalar-valued callable of ``params``.
        params: Parameter ``Array`` at which to evaluate.
        v: Tangent ``Array`` for the Hessian-vector product.

    Returns:
        The Hessian-vector product $\\nabla^2 f \\cdot v$.
    """
    v = v.reshape(params.shape)
    return jax.jvp(jax.grad(f), (params,), (v,))[1]


def get_jacobian_fn(compute_U_fn: Callable[[Array], Array]) -> Callable[[Array], Array]:
    """Build the autodiff Jacobian of the unitary w.r.t. parameters.

    Returns the holomorphic ``jax.jacobian`` of ``compute_U_fn``. This is the
    live Jacobian path for *all* system sizes: the manual Jacobian
    (``geope.jax.jacobian.get_jacobian_manual``) exists and is independently
    tested, but is not currently wired into the optimisation pipeline (the
    autodiff path historically overwrote it for the >5-qubit branch — see
    issue #4). The returned function is left un-jitted so it fuses into the
    enclosing ``@jax.jit`` update step on first ``optimize()``.

    Args:
        compute_U_fn: Callable mapping a parameter list to the product unitary.

    Returns:
        A ``Callable[[Array], Array]`` returning the Jacobian of the unitary.
    """
    return jax.jacobian(compute_U_fn, argnums=0, holomorphic=True)


def get_gammas_fn(
    compute_U_fn: Callable[[Array], Array],
    geo_fn: Callable[..., Array],
    project_omegas_fn: Callable[[Array], Array],
) -> Callable[[Array, Array], Array]:
    """Build the projected geodesic-Hamiltonian (``gammas``) function.

    Computes the unitary, its geodesic Hamiltonian towards the target, and
    projects that onto the Pauli basis (normalised by the dimension). Returned
    un-jitted so it composes inside an enclosing ``@jax.jit``.

    Args:
        compute_U_fn: Parameter-list -> unitary.
        geo_fn: ``(unitary, key) -> geodesic Hamiltonian``.
        project_omegas_fn: Projection of matrices onto the Lie-algebra basis.

    Returns:
        A ``Callable[[Array, Array], Array]`` ``gammas(free_params, key)``.
    """

    def gammas(free_params: Array, key: Array) -> Array:
        unitary = compute_U_fn(free_params)
        gammaU = geo_fn(unitary, key=key)  # seed for logm
        return project_omegas_fn(jnp.expand_dims(gammaU, axis=0)).squeeze(axis=0) / (
            gammaU.shape[0]
        )

    return gammas


def get_omegas_fn(
    jac_fn: Callable[[Array], Array],
    project_omegas_fn: Callable[[Array], Array],
    proj_indices: np.ndarray,
    has_proj_drift: bool,
) -> Callable[[Array], Array]:
    """Build the projected per-gate Jacobian (``omegas``) function.

    Projects the Jacobian of each gate (w.r.t. each parameter) onto the Pauli
    basis, optionally restricting to the projected indices within the combined
    proj+drift basis. Returned un-jitted so it composes inside an enclosing
    ``@jax.jit``.

    Args:
        jac_fn: Jacobian of the unitary w.r.t. the free parameters.
        project_omegas_fn: Projection of matrices onto the Lie-algebra basis.
        proj_indices: Projected indices within the proj+drift basis.
        has_proj_drift: Whether the proj+drift basis is non-empty (gates the
            projected-index restriction; mirrors the legacy
            ``np.any(proj_drift_basis)`` check).

    Returns:
        A ``Callable[[Array], Array]`` ``omegas(free_params)``.
    """

    def omegas(free_params: Array) -> Array:
        dUs = jnp.array(jac_fn(free_params))
        dUs_t = jnp.transpose(dUs, [2, 3, 0, 1])
        omegas_steps_phis = jnp.array(
            [project_omegas_fn(1.0j * omegaUs) for omegaUs in dUs_t]
        )
        if has_proj_drift:
            omegas_steps_phis = omegas_steps_phis.at[:, proj_indices, :].get()
        return omegas_steps_phis

    return omegas


def get_gammas_and_omegas_fn(
    compute_U_fn: Callable[[Array], Array],
    jac_fn: Callable[[Array], Array],
    geo_fn: Callable[..., Array],
    project_omegas_fn: Callable[[Array], Array],
    proj_indices: np.ndarray,
    has_proj_drift: bool,
) -> Callable[[Array, Array], tuple[Array, Array]]:
    """Build the combined gammas-and-omegas function used by the GEOPE step.

    Gammas are the projected geodesic Hamiltonian coefficients; omegas encode
    the Jacobian of each gate w.r.t. each parameter, projected onto the Pauli
    basis. This is the single combined body the GEOPE update step calls (one
    ``compute_U_fn`` and one ``jac_fn`` evaluation), matching the legacy
    numerics; :func:`get_gammas_fn` / :func:`get_omegas_fn` are the separately
    testable halves. Returned un-jitted so it fuses into the enclosing
    ``@jax.jit`` update step on first ``optimize()``.

    Args:
        compute_U_fn: Parameter-list -> unitary.
        jac_fn: Jacobian of the unitary w.r.t. the free parameters.
        geo_fn: ``(unitary, key) -> geodesic Hamiltonian``.
        project_omegas_fn: Projection of matrices onto the Lie-algebra basis.
        proj_indices: Projected indices within the proj+drift basis.
        has_proj_drift: Whether the proj+drift basis is non-empty.

    Returns:
        A ``Callable[[Array, Array], tuple[Array, Array]]``
        ``gammas_and_omegas(free_params, key) -> (gammaU_params, omegas)``.
    """

    def gammas_and_omegas(free_params: Array, key: Array) -> tuple[Array, Array]:
        unitary = compute_U_fn(free_params)
        gammaU = geo_fn(unitary, key=key)  # seed for logm
        gammaU_params = project_omegas_fn(jnp.expand_dims(gammaU, axis=0)).squeeze(
            axis=0
        ) / (gammaU.shape[0])

        dUs = jnp.array(jac_fn(free_params))
        dUs_t = jnp.transpose(dUs, [2, 3, 0, 1])
        omegas_steps_phis = jnp.array(
            [project_omegas_fn(1.0j * omegaUs) for omegaUs in dUs_t]
        )

        if has_proj_drift:
            omegas_steps_phis = omegas_steps_phis.at[:, proj_indices, :].get()

        return gammaU_params, omegas_steps_phis

    return gammas_and_omegas


def get_hessian_fn(infid_fn: Callable[[Array], Array]) -> Callable[[Array], Array]:
    """Build the full Hessian function via forward-over-reverse HVPs.

    Materialises the Hessian of ``infid_fn`` by mapping a Hessian-vector
    product over the identity matrix's columns. Returned un-jitted so it fuses
    into the enclosing ``@jax.jit`` update step.

    Args:
        infid_fn: Scalar-valued infidelity callable of the free parameters.

    Returns:
        A ``Callable[[Array], Array]`` ``hess(y)`` returning the Hessian.
    """

    def hess(y: Array) -> Array:
        return jax.vmap(lambda x: hvp_forward_over_reverse(infid_fn, y, x))(
            jnp.eye(y.size, dtype=y.dtype)
        )

    return hess


def get_hessian_manual_fn(
    basis: np.ndarray,
    target: Array,
    projective: bool = True,
    method: str = "eig",
    hermitian: bool = True,
) -> Callable[[Array], Array]:
    r"""Build the infidelity Hessian manually (Goodwin–Kuprov NR-GRAPE).

    Analytic drop-in for `get_hessian_fn`: returns ``hess(y) -> (P, P)`` with
    ``P = y.size``, the Hessian of the same infidelity that `get_hessian_fn`
    differentiates by autodiff, but built from the manual propagator
    derivatives (`manual_jacobian`, `manual_hessian`) rather than from
    forward-over-reverse HVPs.

    Let $z = \mathrm{Tr}(U_T^\dagger U)$, $\partial_a z$, $\partial_a\partial_b z$
    be obtained by contracting $U$, $\partial U$, $\partial^2 U$ against
    $U_T^\dagger$. For the phase-sensitive cost $C = 1 - \mathrm{Re}(z)/d$ the
    Hessian is the linear contraction $-\mathrm{Re}(\partial_a\partial_b z)/d$;
    for the projective cost $C = 1 - |z|/d$,

    $$\partial_a\partial_b|z| =
        \frac{\mathrm{Re}(\overline{\partial_a z}\,\partial_b z)
              + \mathrm{Re}(\bar z\,\partial_a\partial_b z)}{|z|}
        - \frac{\mathrm{Re}(\bar z\,\partial_a z)\,
                \mathrm{Re}(\bar z\,\partial_b z)}{|z|^3}.$$

    Like the projective fidelity itself, this is singular as $|z| \to 0$ (the
    near-identity / traceless-target gotcha) — the autodiff Hessian shares that.

    Memory note: this materialises the dense propagator Hessian
    (`manual_hessian`, $O(G^2 d^2 K^2)$); intended for the small systems where
    NR-GRAPE is used.

    Args:
        basis: Proj+drift basis ``(K, d, d)`` — the same basis the bound
            ``compute_U_fn`` uses.
        target: Target unitary ``(d, d)``.
        projective: Match the projective (``True``) or phase-sensitive
            (``False``) infidelity.
        method: ``"eig"`` (spectral, default) or ``"block"`` (auxiliary-matrix)
            per-step derivatives.
        hermitian: For ``method="eig"``, assume real parameters (skew-Hermitian
            generators) and use the faster ``eigh``-based spectral derivatives.
            Set ``False`` for complex-valued parameters.

    Returns:
        A ``Callable[[Array], Array]`` ``hess(y)`` returning the ``(P, P)``
        infidelity Hessian. Left un-jitted so it fuses into the enclosing
        ``@jax.jit`` update step.
    """
    Ui_fn = get_Ui_fn(basis)
    if method == "eig":
        jac_step = get_dexpm_eig(basis, hermitian=hermitian)
        hess_step = get_d2expm_eig(basis, hermitian=hermitian)
    elif method == "block":
        jac_step = get_dexpm(basis)
        hess_step = get_d2expm(basis)
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'eig' or 'block'.")

    compute_U = get_compute_matrices_params_list_fn(basis)
    t_conj = jnp.asarray(target).conj()
    d = jnp.asarray(target).shape[0]

    def hess(y: Array) -> Array:
        U = compute_U(y)
        dU = manual_jacobian(y, Ui_fn, jac_step)  # (G, d, d, K)
        H = manual_hessian(y, Ui_fn, jac_step, hess_step)  # (G, G, d, d, K, K)

        # Contract the propagator and its derivatives with U_T^dagger.
        z = jnp.einsum("ab,ab->", t_conj, U)
        dz = jnp.einsum("ab,iabk->ik", t_conj, dU)  # (G, K)
        d2z = jnp.einsum("ab,ijabkl->ijkl", t_conj, H)  # (G, G, K, K)

        n_g, n_k = y.shape
        P = n_g * n_k
        dz_f = dz.reshape(P)
        d2z_f = jnp.transpose(d2z, (0, 2, 1, 3)).reshape(P, P)

        if not projective:
            return -jnp.real(d2z_f) / d

        r = jnp.abs(z)
        z_bar = jnp.conj(z)
        re_zdz = jnp.real(z_bar * dz_f)  # (P,)
        term1 = (
            jnp.real(jnp.outer(jnp.conj(dz_f), dz_f)) + jnp.real(z_bar * d2z_f)
        ) / r
        term2 = jnp.outer(re_zdz, re_zdz) / r**3
        return -(term1 - term2) / d

    return hess


def wrap_compute_U_param_transform(
    params: "Parameters", raw_compute_U: Callable[[Array], Array]
) -> Callable[[Array], Array]:
    """Wrap ``compute_U`` to honour ``params.param_transform``.

    The user-facing experimental parameters $\\phi^{\\mathrm{exp}}$ are mapped to
    projected-basis coefficients via ``params.param_transform`` (possibly
    step-dependent), embedded into the proj+drift basis, and combined with the
    drift before the original ``raw_compute_U`` is called.

    Returned un-jitted so it fuses into the enclosing ``@jax.jit`` update step
    on first ``optimize()``.

    Args:
        params: The ``Parameters`` object carrying ``param_transform``.
        raw_compute_U: The projected-basis unitary-computation function.

    Returns:
        The wrapped experimental-space ``compute_U`` callable.
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


def get_split_jacobian_fn(
    compute_U_fn: Callable[[Array], Array]
) -> Callable[[Array], Array]:
    """Build a real/imag-split Jacobian of ``compute_U_fn``.

    Used on the ``param_transform`` path: differentiating through the
    real-valued user transform with a holomorphic Jacobian would discard the
    imaginary part of intermediates, so the unitary is split into real and
    imaginary parts, each differentiated, then recombined.

    Returned un-jitted so it fuses into the enclosing ``@jax.jit`` update step.

    Args:
        compute_U_fn: The (wrapped) experimental-space unitary function.

    Returns:
        A ``Callable[[Array], Array]`` returning the complex Jacobian.
    """

    def _split_U(x):
        U = compute_U_fn(x)
        return jnp.stack([jnp.real(U), jnp.imag(U)])

    _raw_jac_split = jax.jacobian(_split_U, argnums=0)

    def _jac_fn(x):
        jac_split = _raw_jac_split(x)
        return jac_split[0] + 1j * jac_split[1]

    return _jac_fn
