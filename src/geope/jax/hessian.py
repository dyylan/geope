from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from functools import partial
from typing import Callable

from .dexpm import (
    get_Ui_fn,
    get_dexpm,
    get_dexpm_eig,
    get_d2expm,
    get_d2expm_eig,
    expm_hvp,
    expm_hvp_eig,
)


def hessian_propagator(
    params: Array,
    Ui_fn: Callable[[Array], Array],
    jac_fn: Callable[[Array], Array],
    hess_step_fn: Callable[[Array], Array],
) -> Array:
    r"""Compute the full Hessian propagator of the product unitary.

    Second-derivative analogue of `geope.jax.jacobian_propagator`. With the product
    convention $U = U_{G-1} \cdots U_1 U_0$, $U_i = \exp(i\sum_k x_{i,k} B_k)$
    (each gate left-multiplied), the mixed derivative with respect to gates
    $i$ and $j$ leaves all other gates untouched:

    - same gate ($i = j$):
      $\partial^2 U / \partial x_{i,k}\partial x_{i,l}
        = L_i\,(\partial^2 U_i / \partial x_{i,k}\partial x_{i,l})\,R_i$;
    - distinct gates ($i > j$):
      $\partial^2 U / \partial x_{i,k}\partial x_{j,l}
        = L_i\,(\partial U_i/\partial x_{i,k})\,M_{ij}\,
          (\partial U_j/\partial x_{j,l})\,R_j$,

    where $R_i = U_{i-1}\cdots U_0$ (exclusive prefix), $L_i = U_{G-1}\cdots
    U_{i+1}$ (exclusive suffix), and $M_{ij} = U_{i-1}\cdots U_{j+1}
    = R_i (U_j R_j)^\dagger$ (the middle product, using unitarity). The $i < j$
    blocks follow from symmetry,
    $H_{ij,kl} = H_{ji,lk}$. Prefix/suffix/middle products are built with two
    ``jax.lax.scan`` passes and a single batched matmul; the assembly is a set
    of vectorised einsums (no Python loop over gates).

    Memory note: the returned tensor is dense with shape ``(G, G, d, d, K, K)``,
    i.e. $O(G^2 d^2 K^2)$. For the infidelity-cost Hessian, prefer
    `geope.engine.get_hessian_propagator_fn`, which contracts on the fly and never
    materialises this object.

    Args:
        params: Parameter ``Array`` of shape ``(G, K)``.
        Ui_fn: Callable mapping a coefficient ``Array`` to a unitary ``Array``.
        jac_fn: Per-gate first derivative, ``(K,) -> (d, d, K)`` (e.g. `dexpm`).
        hess_step_fn: Per-gate second derivative, ``(K,) -> (d, d, K, K)``
            (e.g. `d2expm`).

    Returns:
        An ``Array`` of shape ``(G, G, d, d, K, K)`` with
        ``H[i, j, :, :, k, l] = d^2 U / d x_{i,k} d x_{j,l}``.
    """
    gates = jax.vmap(Ui_fn)(params)  # (G, d, d)
    dU = jax.vmap(jac_fn)(params)  # (G, d, d, K)
    d2 = jax.vmap(hess_step_fn)(params)  # (G, d, d, K, K)

    eye = jnp.eye(gates.shape[1], dtype=gates.dtype)

    def step_right(R, g):
        return g @ R, R

    Rs = jax.lax.scan(step_right, eye, gates)[1]  # exclusive prefix R_i

    def step_left(L, g):
        return L @ g, L

    Ls = jax.lax.scan(step_left, eye, gates, reverse=True)[1]  # exclusive suffix L_i

    # Inclusive prefix P_incl[j] = U_j R_j; middle product M[i, j] = R_i P_incl[j]^†.
    Pincl = jnp.einsum("iab,ibc->iac", gates, Rs)
    M = jnp.einsum("iab,jcb->ijac", Rs, jnp.conj(Pincl))  # (G, G, d, d)

    # Off-diagonal (ordered i > j): L_i dU_i M_ij dU_j R_j.
    P = jnp.einsum("iab,ibek->iaek", Ls, dU)  # (L_i dU_i,k)   [i, a, e, k]
    Q = jnp.einsum("jfgl,jgc->jfcl", dU, Rs)  # (dU_j,l R_j)   [j, f, c, l]
    lower = jnp.einsum("iaek,ijef,jfcl->ijackl", P, M, Q)  # valid where i > j

    # Diagonal blocks: L_i d2_i R_i.
    diag = jnp.einsum("iab,ibekl,iec->iackl", Ls, d2, Rs)  # (G, d, d, K, K)

    # i < j blocks by symmetry: H[i,j,a,c,k,l] = lower[j,i,a,c,l,k].
    upper = jnp.swapaxes(jnp.transpose(lower, (1, 0, 2, 3, 4, 5)), -1, -2)

    G = gates.shape[0]
    i_idx = jnp.arange(G)[:, None]
    j_idx = jnp.arange(G)[None, :]
    m = lambda mask: mask[:, :, None, None, None, None]

    H = jnp.where(m(i_idx > j_idx), lower, 0.0)
    H = H + jnp.where(m(i_idx < j_idx), upper, 0.0)
    H = H + jnp.where(m(i_idx == j_idx), diag[:, None], 0.0)
    return H


def get_hessian_propagator(
    gate_basis: Array, method: str = "eig", hermitian: bool = True
) -> Callable[[Array], Array]:
    """Create a JIT-compiled manual propagator-Hessian function.

    Wrapped in ``jax.jit`` so it compiles once and is reused across calls.

    Args:
        gate_basis: ``Array`` of Hermitian basis matrices of shape ``(K, d, d)``.
        method: Per-step derivative method. ``"eig"`` (default) uses the
            spectral per-step derivatives (`dexpm_eig` / `d2expm_eig`);
            ``"block"`` uses the block-exponential derivatives (`dexpm` /
            `d2expm`), which handle non-Hermitian generators and ignore
            ``hermitian``.
        hermitian: Assume real parameters (skew-Hermitian generators) and use
            the faster ``eigh``-based per-gate derivatives. Set ``False`` for
            complex-valued parameters. Only affects ``method="eig"``.

    Returns:
        A ``Callable[[Array], Array]`` accepting a parameter array of shape
        ``(G, K)`` and returning the Hessian of shape ``(G, G, d, d, K, K)``.
    """
    Ui_fn = get_Ui_fn(gate_basis)
    if method == "eig":
        jac_fn = get_dexpm_eig(gate_basis, hermitian=hermitian)
        hess_step_fn = get_d2expm_eig(gate_basis, hermitian=hermitian)
    elif method == "block":
        jac_fn = get_dexpm(gate_basis)
        hess_step_fn = get_d2expm(gate_basis)
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'eig' or 'block'.")
    return jax.jit(
        partial(
            hessian_propagator, Ui_fn=Ui_fn, jac_fn=jac_fn, hess_step_fn=hess_step_fn
        )
    )


def hvp_propagator(
    params: Array,
    direction: Array,
    step_fn: Callable[[Array, Array], tuple[Array, Array, Array]],
) -> tuple[Array, Array, Array]:
    r"""Directional second derivative (HVP) of the product unitary in $O(G)$.

    Computes the value and the first and second single-direction derivatives of
    the product unitary along a parameter-space direction $p$, without forming
    the full Jacobian or the dense $(G, G, d, d, K, K)$ Hessian of
    `hessian_propagator`. With the product convention of
    :func:`geope.engine.compute_matrices_params_list_fn`,
    $\phi(\theta) = U_{G-1} \cdots U_1 U_0$ with each gate left-multiplied,
    define the partial product $X_g = U_g \cdots U_0$ with derivatives
    $V_g = \dot X_g(0)$, $W_g = \ddot X_g(0)$ along $\theta(t) = \theta + t p$.
    Since $X_g(t) = U_g(t) X_{g-1}(t)$, the product rule gives the linear-time
    recursion (the left-multiply mirror of the note's convention)

    $$X_g = U_g X_{g-1}, \qquad
      V_g = U_g V_{g-1} + E_g X_{g-1},$$
    $$W_g = U_g W_{g-1} + 2\,E_g V_{g-1} + G_g X_{g-1},$$

    with $X_{-1} = I$, $V_{-1} = W_{-1} = 0$, where $U_g = \exp(iA_g)$,
    $E_g = D\exp(iA_g)[iB_g]$, and $G_g = D^2\exp(iA_g)[iB_g, iB_g]$ are the
    per-gate value and directional derivatives (`step_fn`). After all $G$ gates,
    $X_{G-1} = \phi(\theta)$, $V_{G-1} = D\phi_\theta[p]$, and
    $W_{G-1} = D^2\phi_\theta[p, p]$. The cross term $2\,E_g V_{g-1}$ collects
    every cross-pulse interaction, so no explicit $O(G^2)$ double sum is needed;
    the recursion is a single ``jax.lax.scan``. This is the second-order sibling
    of `geope.jax.jacobian.jvp_propagator`.

    Args:
        params: Parameter ``Array`` of shape ``(G, K)``.
        direction: Direction ``Array`` of shape ``(G, K)`` (the $p$ above).
        step_fn: Per-gate step ``(x, p) -> (U, E, G)`` mapping a gate's
            coefficients and direction to its value and first/second directional
            derivatives (e.g. `geope.jax.expm_hvp_eig`).

    Returns:
        Tuple ``(X, V, W)`` of matrices of shape ``(d, d)``: the product unitary
        $\phi(\theta)$, its directional derivative $D\phi_\theta[p]$, and its
        directional second derivative $D^2\phi_\theta[p, p]$.
    """
    Us, Es, Gs = jax.vmap(step_fn)(params, direction)  # each (G, d, d)

    eye = jnp.eye(Us.shape[1], dtype=Us.dtype)

    def step(carry, gate):
        X, V, W = carry
        U, E, G = gate
        X_new = U @ X
        V_new = U @ V + E @ X
        W_new = U @ W + 2.0 * (E @ V) + G @ X
        return (X_new, V_new, W_new), None

    zero = jnp.zeros_like(eye)
    (X, V, W), _ = jax.lax.scan(step, (eye, zero, zero), (Us, Es, Gs))
    return X, V, W


def get_hvp_propagator(
    gate_basis: Array, method: str = "eig", hermitian: bool = True
) -> Callable[[Array, Array], tuple[Array, Array, Array]]:
    """Create a JIT-compiled directional-HVP propagator for a given gate basis.

    Wraps `hvp_propagator` with a per-gate step built from ``gate_basis`` and is
    wrapped in ``jax.jit`` so it compiles once and is reused across calls.

    Args:
        gate_basis: ``Array`` of Hermitian basis matrices of shape ``(K, d, d)``.
        method: Per-gate method. ``"eig"`` (default) uses the spectral
            `geope.jax.expm_hvp_eig`; ``"block"`` uses the block-exponential
            `geope.jax.expm_hvp` (ignores ``hermitian``).
        hermitian: Assume real parameters and use the ``eigh``-based per-gate
            step. Set ``False`` for complex-valued parameters. Only affects
            ``method="eig"``.

    Returns:
        A ``Callable[[Array, Array], tuple[Array, Array, Array]]`` accepting
        parameters and a direction, both of shape ``(G, K)``, and returning the
        triple ``(phi, Dphi[p], D2phi[p, p])`` of shape ``(d, d)``.
    """
    if method == "eig":
        step_fn = partial(expm_hvp_eig, basis=gate_basis, hermitian=hermitian)
    elif method == "block":
        step_fn = partial(expm_hvp, basis=gate_basis)
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'eig' or 'block'.")
    return jax.jit(partial(hvp_propagator, step_fn=step_fn))
