from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from functools import partial
from typing import Callable

from .dexpm import get_Ui_fn, get_dexpm_eig, get_d2expm_eig


def manual_hessian(
    params: Array,
    Ui_fn: Callable[[Array], Array],
    jac_fn: Callable[[Array], Array],
    hess_step_fn: Callable[[Array], Array],
) -> Array:
    r"""Compute the full Hessian of the product unitary manually.

    Second-derivative analogue of `geope.jax.manual_jacobian`. With the product
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
    `geope.engine.get_hessian_manual_fn`, which contracts on the fly and never
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


def get_hessian_manual(
    gate_basis: Array, hermitian: bool = True
) -> Callable[[Array], Array]:
    """Create a JIT-compiled manual propagator-Hessian function.

    Uses the spectral per-step derivatives (`dexpm_eig` / `d2expm_eig`) and is
    wrapped in ``jax.jit`` so it compiles once and is reused across calls.

    Args:
        gate_basis: ``Array`` of Hermitian basis matrices of shape ``(K, d, d)``.
        hermitian: Assume real parameters (skew-Hermitian generators) and use
            the faster ``eigh``-based per-gate derivatives. Set ``False`` for
            complex-valued parameters.

    Returns:
        A ``Callable[[Array], Array]`` accepting a parameter array of shape
        ``(G, K)`` and returning the Hessian of shape ``(G, G, d, d, K, K)``.
    """
    Ui_fn = get_Ui_fn(gate_basis)
    jac_fn = get_dexpm_eig(gate_basis, hermitian=hermitian)
    hess_step_fn = get_d2expm_eig(gate_basis, hermitian=hermitian)
    return jax.jit(
        partial(manual_hessian, Ui_fn=Ui_fn, jac_fn=jac_fn, hess_step_fn=hess_step_fn)
    )
