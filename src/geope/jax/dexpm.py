from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from functools import partial
from typing import Callable


def Ui(x: Array, basis: Array) -> Array:
    """Compute a unitary from a linear combination of Hermitian basis matrices.

    Constructs $U = \\exp(i \\sum_k x_k B_k)$.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        basis: Array of Hermitian matrices of shape ``(K, d, d)``.

    Returns:
        A unitary matrix of shape ``(d, d)``.
    """
    A = jnp.tensordot(x, basis, axes=[[-1], [0]])
    return jax.scipy.linalg.expm(1j * A)


def get_Ui_fn(basis: Array) -> Callable[[Array], Array]:
    """Create a partial unitary function with a fixed basis.

    Args:
        basis: Array of Hermitian matrices of shape ``(K, d, d)``.

    Returns:
        A callable that accepts a coefficient vector and returns
        the corresponding unitary matrix.
    """
    return partial(Ui, basis=basis)


@jax.jit
def dexpm_block(A: Array, x: Array) -> Array:
    """Compute the derivative of the matrix exponential via the block method.

    Implements the block-matrix approach of
    `Al-Mohy & Higham (2009) <https://arxiv.org/pdf/1506.00628>`_,
    Eq. (31), extracting $d\\exp(iA)/dA \\cdot x$ from the upper-right
    block of $\\exp(i[[A, x], [0, A]])$.

    Args:
        A: The Hamiltonian matrix of shape ``(d, d)``.
        x: The direction matrix of shape ``(d, d)``.

    Returns:
        The directional derivative matrix of shape ``(d, d)``.
    """
    dim = A.shape[0]
    # Create block matrix
    block_mat = jnp.block([[A, x], [jnp.zeros_like(A), A]])
    # Take matrix exponential
    dblock_mat = jax.scipy.linalg.expm(1j * block_mat)
    # Upper right block contains derivative
    return dblock_mat[:dim, dim:]


def dexpm(x: Array, basis: Array) -> Array:
    """Compute the derivative of the exponential map for all basis directions.

    For each basis element $B_k$, computes
    $\\partial \\exp(i \\sum_j x_j B_j) / \\partial x_k$.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        basis: Array of Hermitian matrices of shape ``(K, d, d)``.

    Returns:
        An array of shape ``(d, d, K)`` whose last axis indexes the
        partial derivatives with respect to each coefficient.
    """
    # Construct argument of exponential
    A = jnp.tensordot(x, basis, axes=[[-1], [0]])
    # For each element in the basis, get the derivative. Stack in last axis.
    return jax.vmap(lambda b: dexpm_block(A, b), out_axes=2)(basis)


def dexpm_batched(x: Array, basis: Array, batch_size: int) -> Array:
    """Batched derivative of the exponential map.

    Same as `dexpm` but uses ``jax.lax.map`` with a configurable
    `batch_size` to limit peak memory usage.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        basis: Array of Hermitian matrices of shape ``(K, d, d)``.
        batch_size: Number of basis elements to process per batch.

    Returns:
        An array of shape ``(d, d, K)``.
    """
    # Construct argument of exponential
    A = jnp.tensordot(x, basis, axes=[[-1], [0]])
    # For each element in the basis, get the derivative. Stack in last axis.
    return jnp.transpose(
        jax.lax.map(lambda b: dexpm_block(A, b), basis, batch_size=batch_size),
        axes=(1, 2, 0),
    )


def _eig(
    x: Array, basis: Array, hermitian: bool = True
) -> tuple[Array, Array, Array]:
    r"""Diagonalise $M = i \sum_j x_j B_j = V \mathrm{diag}(\mu) V^{-1}$.

    For real coefficients ``x`` the generator $A = \sum_j x_j B_j$ is Hermitian
    and $M$ is skew-Hermitian, so the default ``hermitian=True`` path uses
    ``jnp.linalg.eigh`` on $A$: this is faster, yields a *unitary* eigenvector
    matrix (so $V^{-1} = V^\dagger$, avoiding an explicit inverse), and is
    supported on GPU/TPU (unlike the general ``jnp.linalg.eig``). Set
    ``hermitian=False`` to diagonalise the general (possibly non-normal) $M$ via
    ``jnp.linalg.eig`` — required only when ``x`` has a non-zero imaginary part.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        basis: Array of Hermitian matrices of shape ``(K, d, d)``.
        hermitian: Assume real coefficients (Hermitian ``A``) and use ``eigh``.

    Returns:
        Tuple ``(mu, V, Vinv)`` of shapes ``(d,)``, ``(d, d)``, ``(d, d)``.
    """
    A = jnp.tensordot(x, basis, axes=[[-1], [0]])
    if hermitian:
        w, V = jnp.linalg.eigh(A)
        return 1j * w, V, jnp.conj(V).T
    mu, V = jnp.linalg.eig(1j * A)
    return mu, V, jnp.linalg.inv(V)


def _spectral_factors(
    x: Array, basis: Array, hermitian: bool = True
) -> tuple[Array, Array, Array]:
    r"""Eigendecomposition factors shared by the spectral derivative variants.

    Diagonalises $M = i \sum_j x_j B_j = V \mathrm{diag}(\mu) V^{-1}$ and builds
    the divided-difference matrix $\Delta$ of $\exp$, using the diagonal limit
    $\Delta_{pp} = e^{\mu_p}$ where eigenvalues (nearly) coincide.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        basis: Array of Hermitian matrices of shape ``(K, d, d)``.
        hermitian: Assume real coefficients and diagonalise via ``eigh`` (see
            `_eig`).

    Returns:
        Tuple ``(V, Vinv, delta)`` of shapes ``(d, d)``, ``(d, d)``, ``(d, d)``.
    """
    mu, V, Vinv = _eig(x, basis, hermitian=hermitian)
    exp_mu = jnp.exp(mu)

    dmu = mu[:, None] - mu[None, :]
    degenerate = jnp.abs(dmu) < 1e-12
    safe_dmu = jnp.where(degenerate, 1.0, dmu)
    delta = jnp.where(
        degenerate,
        exp_mu[:, None] * jnp.ones_like(dmu),
        (exp_mu[:, None] - exp_mu[None, :]) / safe_dmu,
    )
    return V, Vinv, delta


def _second_divided_differences(mu: Array, tol: float = 1e-7) -> Array:
    r"""Second divided differences of ``exp`` over an eigenvalue spectrum.

    Returns the symmetric tensor ``T[p, r, q] = exp[mu_p, mu_r, mu_q]`` where

    $$f[a,b,c] = \frac{f[b,c] - f[a,b]}{c - a}, \quad
      f[a,b] = \frac{e^a - e^b}{a - b},$$

    with the coincidence limits handled by ``where``-guards: $f[a,a] = e^a$,
    $f[a,a,c] = (f[a,c] - e^a)/(c-a)$, and $f[a,a,a] = \tfrac12 e^a$.

    Args:
        mu: Eigenvalues of shape ``(d,)``.
        tol: Threshold below which two eigenvalues are treated as coincident.

    Returns:
        Tensor of shape ``(d, d, d)``.
    """
    a = mu[:, None, None]
    b = mu[None, :, None]
    c = mu[None, None, :]
    exp_a = jnp.exp(a)

    def f1(x: Array, y: Array) -> Array:
        dxy = x - y
        near = jnp.abs(dxy) < tol
        return jnp.where(
            near,
            jnp.exp(0.5 * (x + y)),
            (jnp.exp(x) - jnp.exp(y)) / jnp.where(near, 1.0, dxy),
        )

    fab = f1(a, b)
    fbc = f1(b, c)

    dca = c - a
    near_ca = jnp.abs(dca) < tol
    t_main = (fbc - fab) / jnp.where(near_ca, 1.0, dca)

    # Limit as c -> a: f[a, a, b] = (f[a, b] - e^a)/(b - a), itself -> e^a/2 at b -> a.
    dba = b - a
    near_ba = jnp.abs(dba) < tol
    t_limit = jnp.where(
        near_ba,
        0.5 * exp_a * jnp.ones_like(b),
        (fab - exp_a) / jnp.where(near_ba, 1.0, dba),
    )

    return jnp.where(near_ca, t_limit, t_main)


def dexpm_eig(x: Array, basis: Array, hermitian: bool = True) -> Array:
    r"""Derivative of the exponential map via the spectral (Fréchet) method.

    Computes the same quantity as `dexpm` — for each basis element $B_k$,
    $\partial \exp(i \sum_j x_j B_j) / \partial x_k$ — but from a single
    eigendecomposition rather than ``K`` block-matrix exponentials, which is
    substantially faster for large ``K``.

    Writing $M = i \sum_j x_j B_j = V \mathrm{diag}(\mu) V^{-1}$, the
    directional derivative of $\exp(M)$ along $E$ is
    $V\,(\Delta \circ (V^{-1} E V))\,V^{-1}$, where $\Delta$ is the matrix of
    divided differences of $\exp$,
    $\Delta_{pq} = (e^{\mu_p} - e^{\mu_q}) / (\mu_p - \mu_q)$ with the limit
    $\Delta_{pp} = e^{\mu_p}$ on (near-)degenerate eigenvalues. The relevant
    directions are $E_k = i B_k$.

    For real coefficients ``x`` the generator is Hermitian and $M$ is
    anti-Hermitian, so ``V`` is well-conditioned; the method also handles
    general (diagonalisable) complex ``x``.

    See `dexpm_eig_batched` for a variant that chunks the ``K`` directions to
    bound peak memory.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        basis: Array of Hermitian matrices of shape ``(K, d, d)``.
        hermitian: Assume real coefficients (skew-Hermitian ``M``) and
            diagonalise via ``eigh`` — see `_eig`. Set ``False`` for complex
            coefficients.

    Returns:
        An array of shape ``(d, d, K)`` whose last axis indexes the
        partial derivatives with respect to each coefficient.
    """
    V, Vinv, delta = _spectral_factors(x, basis, hermitian=hermitian)

    E = 1j * basis  # directions dM/dx_k, shape (K, d, d)
    C = jnp.einsum("pi,kij,jq->kpq", Vinv, E, V)  # V^{-1} E_k V
    D = delta[None] * C  # Delta o (V^{-1} E_k V)
    dexp_k = jnp.einsum("ip,kpq,qj->kij", V, D, Vinv)  # V (...) V^{-1}
    return jnp.moveaxis(dexp_k, 0, -1)


def dexpm_eig_batched(
    x: Array, basis: Array, batch_size: int, hermitian: bool = True
) -> Array:
    """Batched spectral derivative of the exponential map.

    Same result as `dexpm_eig`, but the shared eigendecomposition is computed
    once and the per-direction transform is applied with ``jax.lax.map`` in
    chunks of `batch_size`, bounding the peak memory of the otherwise
    ``(K, d, d)`` intermediates.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        basis: Array of Hermitian matrices of shape ``(K, d, d)``.
        batch_size: Number of basis directions to process per chunk.

    Returns:
        An array of shape ``(d, d, K)``.
    """
    V, Vinv, delta = _spectral_factors(x, basis, hermitian=hermitian)

    def per_direction(b):
        # b is a single basis matrix (d, d); E = i b is its direction.
        C = Vinv @ (1j * b) @ V
        return V @ (delta * C) @ Vinv

    return jnp.transpose(
        jax.lax.map(per_direction, basis, batch_size=batch_size),
        axes=(1, 2, 0),
    )


def _expm_block13(A: Array, x_a: Array, x_b: Array) -> Array:
    r"""Top-right ``(1, 3)`` block of the ``3d x 3d`` auxiliary exponential.

    Returns the ``(1, 3)`` block of
    $\exp\!\big(i [[A, x_a, 0], [0, A, x_b], [0, 0, A]]\big)$, i.e. the
    *ordered* second-derivative integral with ``x_a`` applied to the left of
    ``x_b`` (Van Loan / Goodwin & Kuprov).
    """
    dim = A.shape[0]
    Z = jnp.zeros_like(A)
    block_mat = jnp.block([[A, x_a, Z], [Z, A, x_b], [Z, Z, A]])
    eblock = jax.scipy.linalg.expm(1j * block_mat)
    return eblock[:dim, 2 * dim : 3 * dim]


def d2expm_block(A: Array, x_a: Array, x_b: Array) -> Array:
    r"""Mixed second derivative of $\exp(iA)$ via the auxiliary-matrix method.

    Goodwin & Kuprov's (and Van Loan's) extension of the 2x2 block trick to
    second order. The top-right ``(1, 3)`` block of the ``3d x 3d`` exponential
    gives only the *ordered* term (``x_a`` left of ``x_b``); the symmetric mixed
    derivative is the sum of both orderings,

    $$\partial^2_{ab}\exp(iA)
        = \mathrm{block}_{13}(A, x_a, x_b) + \mathrm{block}_{13}(A, x_b, x_a).$$

    (For ``x_a = x_b`` this reduces to twice the single block.)

    Args:
        A: The Hamiltonian matrix of shape ``(d, d)``.
        x_a: First direction matrix of shape ``(d, d)``.
        x_b: Second direction matrix of shape ``(d, d)``.

    Returns:
        The mixed second-derivative matrix of shape ``(d, d)``.
    """
    return _expm_block13(A, x_a, x_b) + _expm_block13(A, x_b, x_a)


def d2expm(x: Array, basis: Array) -> Array:
    r"""Second derivative of the exponential map for all basis-direction pairs.

    For each pair $(B_k, B_l)$, computes
    $\partial^2 \exp(i \sum_j x_j B_j) / \partial x_k \partial x_l$ via the
    auxiliary-matrix method. Only the ``K^2`` ordered blocks are exponentiated;
    the symmetric result is their transpose-sum.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        basis: Array of Hermitian matrices of shape ``(K, d, d)``.

    Returns:
        An array of shape ``(d, d, K, K)`` whose last two axes index the pair
        of coefficients; symmetric under their exchange.
    """
    A = jnp.tensordot(x, basis, axes=[[-1], [0]])
    ordered = jax.vmap(
        lambda Bk: jax.vmap(lambda Bl: _expm_block13(A, Bk, Bl))(basis)
    )(basis)  # (K, K, d, d), ordered (k left of l)
    pairs = ordered + jnp.swapaxes(ordered, 0, 1)  # symmetrise both orderings
    return jnp.transpose(pairs, (2, 3, 0, 1))


def d2expm_eig(x: Array, basis: Array, hermitian: bool = True) -> Array:
    r"""Second derivative of the exponential map via the spectral method.

    Computes the same ``(d, d, K, K)`` tensor as `d2expm` from a single
    eigendecomposition using the second-order Daleckii-Krein formula. Writing
    $M = V \mathrm{diag}(\mu) V^{-1}$ and $\tilde{G}_k = V^{-1}(iB_k)V$,

    $$(\partial^2\exp)_{pq}
        = \sum_r T_{prq}\,
          \big(\tilde{G}_{k,pr}\tilde{G}_{l,rq} + \tilde{G}_{l,pr}\tilde{G}_{k,rq}\big),$$

    where $T$ is the second divided difference of $\exp$
    (`_second_divided_differences`), then mapped back with $V(\cdot)V^{-1}$. This
    is substantially faster than `d2expm` for large ``K`` (one eigendecomposition
    instead of ``K^2`` block exponentials).

    Args:
        x: Coefficient vector of shape ``(K,)``.
        basis: Array of Hermitian matrices of shape ``(K, d, d)``.
        hermitian: Assume real coefficients (skew-Hermitian ``M``) and
            diagonalise via ``eigh`` — see `_eig`. Set ``False`` for complex
            coefficients.

    Returns:
        An array of shape ``(d, d, K, K)``; symmetric under exchange of the
        last two axes.
    """
    mu, V, Vinv = _eig(x, basis, hermitian=hermitian)
    T = _second_divided_differences(mu)  # (d, d, d) indexed [p, r, q]

    E = 1j * basis  # directions, (K, d, d)
    Gt = jnp.einsum("pi,kij,jq->kpq", Vinv, E, V)  # V^{-1} E_k V, [k, p, q]

    # term[k,l,p,q] = sum_r T[p,r,q] Gt[k,p,r] Gt[l,r,q]; symmetrise over (k,l).
    term = jnp.einsum("prq,kpr,lrq->klpq", T, Gt, Gt)
    W = term + jnp.swapaxes(term, 0, 1)
    d2 = jnp.einsum("ip,klpq,qj->klij", V, W, Vinv)  # (K, K, d, d)
    return jnp.transpose(d2, (2, 3, 0, 1))


def d2expm_eig_batched(
    x: Array, basis: Array, batch_size: int, hermitian: bool = True
) -> Array:
    """Batched spectral second derivative of the exponential map.

    Same result as `d2expm_eig`, but the per-``k`` slabs of the ``(K, K, d, d)``
    intermediate are produced with ``jax.lax.map`` over the first direction in
    chunks of `batch_size`, bounding peak memory.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        basis: Array of Hermitian matrices of shape ``(K, d, d)``.
        batch_size: Number of first-directions to process per chunk.

    Returns:
        An array of shape ``(d, d, K, K)``.
    """
    mu, V, Vinv = _eig(x, basis, hermitian=hermitian)
    T = _second_divided_differences(mu)
    E = 1j * basis
    Gt = jnp.einsum("pi,kij,jq->kpq", Vinv, E, V)  # [k, p, q]

    def per_first_direction(Gk):
        # Gk = V^{-1} E_k V, shape (d, d). Returns the (K, d, d) slab for this k.
        term = jnp.einsum("prq,pr,lrq->lpq", T, Gk, Gt)
        term_sym = term + jnp.einsum("prq,lpr,rq->lpq", T, Gt, Gk)
        return jnp.einsum("ip,lpq,qj->lij", V, term_sym, Vinv)

    slabs = jax.lax.map(per_first_direction, Gt, batch_size=batch_size)  # (K, K, d, d)
    return jnp.transpose(slabs, (2, 3, 0, 1))


def get_dexpm(basis: Array, batch_size: int | None = None) -> Callable[[Array], Array]:
    """Create a JIT-compiled exponential-map derivative function.

    Args:
        basis: Array of Hermitian matrices of shape ``(K, d, d)``.
        batch_size: Optional batch size. If ``None``, the full vmap
            variant is used; otherwise the batched variant.

    Returns:
        A callable that accepts a coefficient vector and returns
        the derivative array of shape ``(d, d, K)``.
    """
    if batch_size is None:
        return jax.jit(partial(dexpm, basis=basis))
    else:
        return partial(dexpm_batched, basis=basis, batch_size=batch_size)


def get_dexpm_eig(
    basis: Array, batch_size: int | None = None, hermitian: bool = True
) -> Callable[[Array], Array]:
    """Create a JIT-compiled spectral exponential-map derivative function.

    Wraps `dexpm_eig` (or `dexpm_eig_batched`) with a fixed basis. The
    full-``vmap`` variant is the fast default used by
    `geope.jax.get_jacobian_manual`.

    Args:
        basis: Array of Hermitian matrices of shape ``(K, d, d)``.
        batch_size: Optional batch size. If ``None``, the full variant is used;
            otherwise the directions are chunked to bound peak memory.
        hermitian: Assume real coefficients (skew-Hermitian ``M``) and use
            ``eigh`` — see `_eig`. Set ``False`` for complex coefficients.

    Returns:
        A callable that accepts a coefficient vector and returns
        the derivative array of shape ``(d, d, K)``.
    """
    if batch_size is None:
        return jax.jit(partial(dexpm_eig, basis=basis, hermitian=hermitian))
    else:
        return jax.jit(
            partial(
                dexpm_eig_batched,
                basis=basis,
                batch_size=batch_size,
                hermitian=hermitian,
            )
        )


def get_d2expm(basis: Array, batch_size: int | None = None) -> Callable[[Array], Array]:
    """Create a JIT-compiled block second-derivative function.

    Wraps `d2expm` with a fixed basis (the auxiliary-matrix method).

    Args:
        basis: Array of Hermitian matrices of shape ``(K, d, d)``.
        batch_size: Currently only the full variant is provided; kept for
            signature parity with `get_d2expm_eig`.

    Returns:
        A callable that accepts a coefficient vector and returns the
        second-derivative array of shape ``(d, d, K, K)``.
    """
    return jax.jit(partial(d2expm, basis=basis))


def get_d2expm_eig(
    basis: Array, batch_size: int | None = None, hermitian: bool = True
) -> Callable[[Array], Array]:
    """Create a JIT-compiled spectral second-derivative function.

    Wraps `d2expm_eig` (or `d2expm_eig_batched`) with a fixed basis. The full
    variant is the fast default used by `geope.jax.get_hessian_manual`.

    Args:
        basis: Array of Hermitian matrices of shape ``(K, d, d)``.
        batch_size: Optional batch size. If ``None``, the full variant is used;
            otherwise the first direction is chunked to bound peak memory.
        hermitian: Assume real coefficients (skew-Hermitian ``M``) and use
            ``eigh`` — see `_eig`. Set ``False`` for complex coefficients.

    Returns:
        A callable that accepts a coefficient vector and returns the
        second-derivative array of shape ``(d, d, K, K)``.
    """
    if batch_size is None:
        return jax.jit(partial(d2expm_eig, basis=basis, hermitian=hermitian))
    else:
        return jax.jit(
            partial(
                d2expm_eig_batched,
                basis=basis,
                batch_size=batch_size,
                hermitian=hermitian,
            )
        )
