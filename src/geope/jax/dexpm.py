r"""Per-gate exponentials of the chart generators, and their derivatives.

**The contract of this module is sign-free.** Every function here works on a
generator array ``basis`` of shape ``(K, d, d)`` and computes (derivatives of)

$$U(x) = \exp\Bigl(\sum_k x_k E_k\Bigr),$$

with derivative directions the generators $E_k$ themselves. No factor of $i$ is
inserted or assumed anywhere: the physics convention — that the pipeline's
generators are $E_k = -i\,G_k$ for a Hermitian basis $G_k$, i.e.
$U = e^{-iH}$ — lives entirely in the one seam where
`geope.geometry.manifold.Manifold.bind` builds the generator array, and these
kernels inherit it through their input.

In the pipeline the $E_k$ are **skew-Hermitian** and the coefficients real, so
the combination $M = \sum_k x_k E_k$ is skew-Hermitian; the spectral variants'
``hermitian=True`` fast path exploits that structure (see `_eig`). The block
variants assume nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array


def Ui(x: Array, basis: Array) -> Array:
    r"""Compute a matrix from a linear combination of generators.

    Constructs $U = \exp(\sum_k x_k E_k)$.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        basis: Generator array of shape ``(K, d, d)`` (skew-Hermitian in the
            pipeline; see the module docstring).

    Returns:
        A matrix of shape ``(d, d)``.
    """
    M = jnp.tensordot(x, basis, axes=[[-1], [0]])
    return jax.scipy.linalg.expm(M)


def get_Ui_fn(basis: Array) -> Callable[[Array], Array]:
    """Create a partial gate function with a fixed generator array.

    Args:
        basis: Generator array of shape ``(K, d, d)``.

    Returns:
        A callable that accepts a coefficient vector and returns the
        corresponding gate matrix.
    """
    return partial(Ui, basis=basis)


@jax.jit
def dexpm_block(A: Array, x: Array) -> Array:
    r"""Compute the derivative of the matrix exponential via the block method.

    Implements the block-matrix approach of
    `Al-Mohy & Higham (2009) <https://arxiv.org/pdf/1506.00628>`_,
    Eq. (31), extracting $d\exp(A)/dA \cdot x$ from the upper-right
    block of $\exp([[A, x], [0, A]])$.

    Args:
        A: The generator combination $\sum_k x_k E_k$, of shape ``(d, d)``.
        x: The direction matrix of shape ``(d, d)``.

    Returns:
        The directional derivative matrix of shape ``(d, d)``.
    """
    dim = A.shape[0]
    # Create block matrix
    block_mat = jnp.block([[A, x], [jnp.zeros_like(A), A]])
    # Take matrix exponential
    dblock_mat = jax.scipy.linalg.expm(block_mat)
    # Upper right block contains derivative
    return dblock_mat[:dim, dim:]


def dexpm(x: Array, basis: Array) -> Array:
    r"""Compute the derivative of the exponential map for all generator directions.

    For each generator $E_k$, computes
    $\partial \exp(\sum_j x_j E_j) / \partial x_k$.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        basis: Generator array of shape ``(K, d, d)``.

    Returns:
        An array of shape ``(d, d, K)`` whose last axis indexes the
        partial derivatives with respect to each coefficient.
    """
    # Construct argument of exponential
    M = jnp.tensordot(x, basis, axes=[[-1], [0]])
    # For each generator, get the derivative. Stack in last axis.
    return jax.vmap(lambda e: dexpm_block(M, e), out_axes=2)(basis)


def dexpm_batched(x: Array, basis: Array, batch_size: int) -> Array:
    """Batched derivative of the exponential map.

    Same as `dexpm` but uses ``jax.lax.map`` with a configurable
    `batch_size` to limit peak memory usage.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        basis: Generator array of shape ``(K, d, d)``.
        batch_size: Number of generators to process per batch.

    Returns:
        An array of shape ``(d, d, K)``.
    """
    # Construct argument of exponential
    M = jnp.tensordot(x, basis, axes=[[-1], [0]])
    # For each generator, get the derivative. Stack in last axis.
    return jnp.transpose(
        jax.lax.map(lambda e: dexpm_block(M, e), basis, batch_size=batch_size),
        axes=(1, 2, 0),
    )


def _eig(x: Array, basis: Array, hermitian: bool = True) -> tuple[Array, Array, Array]:
    r"""Diagonalise $M = \sum_j x_j E_j = V \mathrm{diag}(\mu) V^{-1}$.

    For real coefficients ``x`` and skew-Hermitian generators $M$ is
    skew-Hermitian, so $-iM$ is Hermitian and the default ``hermitian=True``
    path uses ``jnp.linalg.eigh`` on it: this is faster, yields a *unitary*
    eigenvector matrix (so $V^{-1} = V^\dagger$, avoiding an explicit inverse),
    and is supported on GPU/TPU (unlike the general ``jnp.linalg.eig``). The
    internal rotation by $\mp i$ is a self-inverse **numerical device with no
    convention content** — like `geope.jax.logm`'s branch-cut arithmetic — and
    the eigenvalues returned are $M$'s own, $\mu = i\,\mathrm{eigh}(-iM)$. Set
    ``hermitian=False`` to diagonalise a general (possibly non-normal) $M$ via
    ``jnp.linalg.eig`` — required when ``x`` has a non-zero imaginary part or
    the generators are not skew-Hermitian.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        basis: Generator array of shape ``(K, d, d)``.
        hermitian: Assume a skew-Hermitian combination and use ``eigh``.

    Returns:
        Tuple ``(mu, V, Vinv)`` of shapes ``(d,)``, ``(d, d)``, ``(d, d)``.
    """
    M = jnp.tensordot(x, basis, axes=[[-1], [0]])
    if hermitian:
        w, V = jnp.linalg.eigh(-1j * M)
        return 1j * w, V, jnp.conj(V).T
    mu, V = jnp.linalg.eig(M)
    return mu, V, jnp.linalg.inv(V)


def _first_divided_differences(mu: Array) -> Array:
    r"""First divided differences of ``exp`` over an eigenvalue spectrum.

    Returns the matrix
    $\Delta_{pq} = (e^{\mu_p} - e^{\mu_q}) / (\mu_p - \mu_q)$ with the diagonal
    limit $\Delta_{pp} = e^{\mu_p}$ on (near-)degenerate eigenvalues.

    Args:
        mu: Eigenvalues of shape ``(d,)``.

    Returns:
        Tensor of shape ``(d, d)``.
    """
    exp_mu = jnp.exp(mu)
    dmu = mu[:, None] - mu[None, :]
    degenerate = jnp.abs(dmu) < 1e-12
    safe_dmu = jnp.where(degenerate, 1.0, dmu)
    return jnp.where(
        degenerate,
        exp_mu[:, None] * jnp.ones_like(dmu),
        (exp_mu[:, None] - exp_mu[None, :]) / safe_dmu,
    )


def _spectral_factors(
    x: Array, basis: Array, hermitian: bool = True
) -> tuple[Array, Array, Array]:
    r"""Eigendecomposition factors shared by the spectral derivative variants.

    Diagonalises $M = \sum_j x_j E_j = V \mathrm{diag}(\mu) V^{-1}$ and builds
    the divided-difference matrix $\Delta$ of $\exp$, using the diagonal limit
    $\Delta_{pp} = e^{\mu_p}$ where eigenvalues (nearly) coincide.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        basis: Generator array of shape ``(K, d, d)``.
        hermitian: Assume a skew-Hermitian combination and diagonalise via
            ``eigh`` (see `_eig`).

    Returns:
        Tuple ``(V, Vinv, delta)`` of shapes ``(d, d)``, ``(d, d)``, ``(d, d)``.
    """
    mu, V, Vinv = _eig(x, basis, hermitian=hermitian)
    delta = _first_divided_differences(mu)
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

    Computes the same quantity as `dexpm` — for each generator $E_k$,
    $\partial \exp(\sum_j x_j E_j) / \partial x_k$ — but from a single
    eigendecomposition rather than ``K`` block-matrix exponentials, which is
    substantially faster for large ``K``.

    Writing $M = \sum_j x_j E_j = V \mathrm{diag}(\mu) V^{-1}$, the
    directional derivative of $\exp(M)$ along $E$ is
    $V\,(\Delta \circ (V^{-1} E V))\,V^{-1}$, where $\Delta$ is the matrix of
    divided differences of $\exp$,
    $\Delta_{pq} = (e^{\mu_p} - e^{\mu_q}) / (\mu_p - \mu_q)$ with the limit
    $\Delta_{pp} = e^{\mu_p}$ on (near-)degenerate eigenvalues. The relevant
    directions are the generators $E_k$ themselves.

    For real coefficients ``x`` and skew-Hermitian generators $M$ is
    skew-Hermitian, so ``V`` is well-conditioned; the method also handles
    general (diagonalisable) $M$.

    See `dexpm_eig_batched` for a variant that chunks the ``K`` directions to
    bound peak memory.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        basis: Generator array of shape ``(K, d, d)``.
        hermitian: Assume a skew-Hermitian combination and diagonalise via
            ``eigh`` — see `_eig`. Set ``False`` for complex coefficients or
            non-skew generators.

    Returns:
        An array of shape ``(d, d, K)`` whose last axis indexes the
        partial derivatives with respect to each coefficient.
    """
    V, Vinv, delta = _spectral_factors(x, basis, hermitian=hermitian)

    C = jnp.einsum("pi,kij,jq->kpq", Vinv, basis, V)  # V^{-1} E_k V
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
        basis: Generator array of shape ``(K, d, d)``.
        batch_size: Number of generator directions to process per chunk.

    Returns:
        An array of shape ``(d, d, K)``.
    """
    V, Vinv, delta = _spectral_factors(x, basis, hermitian=hermitian)

    def per_direction(e):
        # e is a single generator (d, d), and its own direction.
        C = Vinv @ e @ V
        return V @ (delta * C) @ Vinv

    return jnp.transpose(
        jax.lax.map(per_direction, basis, batch_size=batch_size),
        axes=(1, 2, 0),
    )


def adj_expm_eig(x: Array, b: Array, basis: Array, hermitian: bool = True) -> Array:
    r"""Adjoint of the exponential-map derivative: all ``K`` overlaps, no ``(d, d, K)``.

    Returns the vector of Frobenius overlaps of a single covector ``b`` with every
    partial derivative of `Ui`,

    $$t_k = \mathrm{Tr}\bigl(b^\dagger\,\partial_k \exp(\textstyle\sum_j x_j E_j)\bigr),$$

    which is what a chain rule through the exponential actually needs — the
    ``(d, d, K)`` tensor `dexpm_eig` builds is contracted away immediately, and
    building it first costs $O(d^3 K)$ where this costs $O(d^3 + d^2 K)$.

    Writing $M = \sum_j x_j E_j = V\,\mathrm{diag}(\mu)\,V^{-1}$ and $\Delta$ for
    the divided differences of $\exp$, `dexpm_eig` gives
    $\partial_k e^M = V(\Delta \circ V^{-1}E_kV)V^{-1}$, so

    $$t_k = \mathrm{Tr}\bigl(\bar A\,E_k\bigr),\qquad
      \bar A = V S^\intercal V^{-1},\quad
      S_{pq} = \Delta_{pq}\,\bigl(V^{-1} b^\dagger V\bigr)_{qp}.$$

    The ``K`` directions therefore cost one $(K, d, d)$ tensordot against a single
    $d \times d$ matrix, and the two rotations that build $\bar A$ are paid once.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        b: The covector to contract against, of shape ``(d, d)``. Paired with
            $\partial_k e^M$ through $\mathrm{Tr}(b^\dagger \cdot)$, i.e. ``b`` is
            *conjugated*.
        basis: Generator array of shape ``(K, d, d)``.
        hermitian: Assume a skew-Hermitian combination and diagonalise via
            ``eigh`` — see `_eig`. Set ``False`` for complex coefficients or
            non-skew generators.

    Returns:
        A complex ``Array`` of shape ``(K,)``. Real-valued cost gradients take
        ``2 * jnp.real(...)`` of it; the raw complex value is what a projective
        (absolute-value) cost needs.
    """
    V, Vinv, delta = _spectral_factors(x, basis, hermitian=hermitian)

    # S[p, q] = delta[p, q] * (Vinv b^dagger V)[q, p]
    S = delta * (Vinv @ jnp.conj(b).T @ V).T
    A_bar = V @ S.T @ Vinv
    # t_k = Tr(A_bar E_k) = sum_ij A_bar[i, j] E_k[j, i]
    return jnp.einsum("ij,kji->k", A_bar, basis)


def adj_expm(x: Array, b: Array, basis: Array) -> Array:
    r"""`adj_expm_eig` by the block-exponential route — the slow, general path.

    The auxiliary-matrix method has no spectral structure to exploit, so this
    genuinely does build `dexpm`'s ``(d, d, K)`` tensor and contract it. It exists
    for parity with ``method="block"`` elsewhere in this module, and because the
    block method tolerates arbitrary generators; prefer `adj_expm_eig`.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        b: The covector to contract against, of shape ``(d, d)``.
        basis: Generator array of shape ``(K, d, d)``.

    Returns:
        A complex ``Array`` of shape ``(K,)``.
    """
    return jnp.einsum("ij,ijk->k", jnp.conj(b), dexpm(x, basis))


def _expm_block13(A: Array, x_a: Array, x_b: Array) -> Array:
    r"""Top-right ``(1, 3)`` block of the ``3d x 3d`` auxiliary exponential.

    Returns the ``(1, 3)`` block of
    $\exp\!\big([[A, x_a, 0], [0, A, x_b], [0, 0, A]]\big)$, i.e. the
    *ordered* second-derivative integral with ``x_a`` applied to the left of
    ``x_b`` (Van Loan / Goodwin & Kuprov).
    """
    dim = A.shape[0]
    Z = jnp.zeros_like(A)
    block_mat = jnp.block([[A, x_a, Z], [Z, A, x_b], [Z, Z, A]])
    eblock = jax.scipy.linalg.expm(block_mat)
    return eblock[:dim, 2 * dim : 3 * dim]


def d2expm_block(A: Array, x_a: Array, x_b: Array) -> Array:
    r"""Mixed second derivative of $\exp(A)$ via the auxiliary-matrix method.

    Goodwin & Kuprov's (and Van Loan's) extension of the 2x2 block trick to
    second order. The top-right ``(1, 3)`` block of the ``3d x 3d`` exponential
    gives only the *ordered* term (``x_a`` left of ``x_b``); the symmetric mixed
    derivative is the sum of both orderings,

    $$\partial^2_{ab}\exp(A)
        = \mathrm{block}_{13}(A, x_a, x_b) + \mathrm{block}_{13}(A, x_b, x_a).$$

    (For ``x_a = x_b`` this reduces to twice the single block.)

    Args:
        A: The generator combination $\sum_k x_k E_k$, of shape ``(d, d)``.
        x_a: First direction matrix of shape ``(d, d)``.
        x_b: Second direction matrix of shape ``(d, d)``.

    Returns:
        The mixed second-derivative matrix of shape ``(d, d)``.
    """
    return _expm_block13(A, x_a, x_b) + _expm_block13(A, x_b, x_a)


def d2expm(x: Array, basis: Array) -> Array:
    r"""Second derivative of the exponential map for all generator pairs.

    For each pair $(E_k, E_l)$, computes
    $\partial^2 \exp(\sum_j x_j E_j) / \partial x_k \partial x_l$ via the
    auxiliary-matrix method. Only the ``K^2`` ordered blocks are exponentiated;
    the symmetric result is their transpose-sum.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        basis: Generator array of shape ``(K, d, d)``.

    Returns:
        An array of shape ``(d, d, K, K)`` whose last two axes index the pair
        of coefficients; symmetric under their exchange.
    """
    M = jnp.tensordot(x, basis, axes=[[-1], [0]])
    ordered = jax.vmap(lambda Ek: jax.vmap(lambda El: _expm_block13(M, Ek, El))(basis))(
        basis
    )  # (K, K, d, d), ordered (k left of l)
    pairs = ordered + jnp.swapaxes(ordered, 0, 1)  # symmetrise both orderings
    return jnp.transpose(pairs, (2, 3, 0, 1))


def d2expm_eig(x: Array, basis: Array, hermitian: bool = True) -> Array:
    r"""Second derivative of the exponential map via the spectral method.

    Computes the same ``(d, d, K, K)`` tensor as `d2expm` from a single
    eigendecomposition using the second-order Daleckii-Krein formula. Writing
    $M = V \mathrm{diag}(\mu) V^{-1}$ and $\tilde{G}_k = V^{-1}E_kV$,

    $$(\partial^2\exp)_{pq}
        = \sum_r T_{prq}\,
          \big(\tilde{G}_{k,pr}\tilde{G}_{l,rq} + \tilde{G}_{l,pr}\tilde{G}_{k,rq}\big),$$

    where $T$ is the second divided difference of $\exp$
    (`_second_divided_differences`), then mapped back with $V(\cdot)V^{-1}$. This
    is substantially faster than `d2expm` for large ``K`` (one eigendecomposition
    instead of ``K^2`` block exponentials).

    Args:
        x: Coefficient vector of shape ``(K,)``.
        basis: Generator array of shape ``(K, d, d)``.
        hermitian: Assume a skew-Hermitian combination and diagonalise via
            ``eigh`` — see `_eig`. Set ``False`` for complex coefficients or
            non-skew generators.

    Returns:
        An array of shape ``(d, d, K, K)``; symmetric under exchange of the
        last two axes.
    """
    mu, V, Vinv = _eig(x, basis, hermitian=hermitian)
    T = _second_divided_differences(mu)  # (d, d, d) indexed [p, r, q]

    Gt = jnp.einsum("pi,kij,jq->kpq", Vinv, basis, V)  # V^{-1} E_k V, [k, p, q]

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
        basis: Generator array of shape ``(K, d, d)``.
        batch_size: Number of first-directions to process per chunk.

    Returns:
        An array of shape ``(d, d, K, K)``.
    """
    mu, V, Vinv = _eig(x, basis, hermitian=hermitian)
    T = _second_divided_differences(mu)
    Gt = jnp.einsum("pi,kij,jq->kpq", Vinv, basis, V)  # [k, p, q]

    def per_first_direction(Gk):
        # Gk = V^{-1} E_k V, shape (d, d). Returns the (K, d, d) slab for this k.
        term = jnp.einsum("prq,pr,lrq->lpq", T, Gk, Gt)
        term_sym = term + jnp.einsum("prq,lpr,rq->lpq", T, Gt, Gk)
        return jnp.einsum("ip,lpq,qj->lij", V, term_sym, Vinv)

    slabs = jax.lax.map(per_first_direction, Gt, batch_size=batch_size)  # (K, K, d, d)
    return jnp.transpose(slabs, (2, 3, 0, 1))


def expm_jvp(x: Array, p: Array, basis: Array) -> tuple[Array, Array]:
    r"""Directional first derivative of the exponential map (block method).

    For $M = \sum_k x_k E_k$ and direction $B = \sum_k p_k E_k$, returns the
    pair

    $$U = \exp(M), \qquad E = D\exp(M)[B],$$

    i.e. the value and the single-direction (JVP) derivative, rather than the
    full per-parameter stack of `dexpm`. Both are read off the upper blocks of a
    single ``2d x 2d`` block exponential (Al-Mohy & Higham):
    $\exp([[M, B], [0, M]]) = [[U, E], [0, U]]$.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        p: Direction-coefficient vector of shape ``(K,)``.
        basis: Generator array of shape ``(K, d, d)``.

    Returns:
        Tuple ``(U, E)`` of matrices of shape ``(d, d)``.
    """
    M = jnp.tensordot(x, basis, axes=[[-1], [0]])
    B = jnp.tensordot(p, basis, axes=[[-1], [0]])
    dim = M.shape[0]
    block_mat = jnp.block([[M, B], [jnp.zeros_like(M), M]])
    e = jax.scipy.linalg.expm(block_mat)
    return e[:dim, :dim], e[:dim, dim:]


def expm_jvp_eig(
    x: Array, p: Array, basis: Array, hermitian: bool = True
) -> tuple[Array, Array]:
    r"""Directional first derivative of the exponential map (spectral method).

    Same result as `expm_jvp` but from a single eigendecomposition, the
    single-direction specialisation of `dexpm_eig`. Writing
    $M = \sum_k x_k E_k = V \mathrm{diag}(\mu) V^{-1}$ and
    $\tilde{B} = V^{-1}BV$ for $B = \sum_k p_k E_k$,

    $$U = V \mathrm{diag}(e^\mu) V^{-1}, \qquad
      E = V\,(\Delta \circ \tilde{B})\,V^{-1},$$

    with $\Delta$ the divided-difference matrix of $\exp$
    (`_first_divided_differences`).

    Args:
        x: Coefficient vector of shape ``(K,)``.
        p: Direction-coefficient vector of shape ``(K,)``.
        basis: Generator array of shape ``(K, d, d)``.
        hermitian: Assume a skew-Hermitian combination and use ``eigh`` — see
            `_eig`. Set ``False`` for complex coefficients or non-skew
            generators.

    Returns:
        Tuple ``(U, E)`` of matrices of shape ``(d, d)``.
    """
    mu, V, Vinv = _eig(x, basis, hermitian=hermitian)
    delta = _first_divided_differences(mu)

    B = jnp.tensordot(p, basis, axes=[[-1], [0]])
    Bt = Vinv @ B @ V  # V^{-1} B V

    U = (V * jnp.exp(mu)[None, :]) @ Vinv
    E = V @ (delta * Bt) @ Vinv
    return U, E


def expm_hvp(x: Array, p: Array, basis: Array) -> tuple[Array, Array, Array]:
    r"""Directional first and second derivatives of the exponential map (block).

    For $M = \sum_k x_k E_k$ and direction $B = \sum_k p_k E_k$, returns

    $$U = \exp(M), \quad E = D\exp(M)[B], \quad G = D^2\exp(M)[B, B],$$

    all read off a single ``3d x 3d`` block exponential (Van Loan / Goodwin &
    Kuprov). With the upper-triangular block $[[M, B, 0], [0, M, B], [0, 0, M]]$,
    the top row of its exponential gives $U$, $E$, and the *ordered*
    second-derivative integral; the symmetric $G$ is twice that block. This is
    the per-gate step used by `geope.jax.hessian.hvp_propagator`.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        p: Direction-coefficient vector of shape ``(K,)``.
        basis: Generator array of shape ``(K, d, d)``.

    Returns:
        Tuple ``(U, E, G)`` of matrices of shape ``(d, d)``.
    """
    M = jnp.tensordot(x, basis, axes=[[-1], [0]])
    B = jnp.tensordot(p, basis, axes=[[-1], [0]])
    dim = M.shape[0]
    Z = jnp.zeros_like(M)
    block_mat = jnp.block([[M, B, Z], [Z, M, B], [Z, Z, M]])
    e = jax.scipy.linalg.expm(block_mat)
    U = e[:dim, :dim]
    E = e[:dim, dim : 2 * dim]
    G = 2.0 * e[:dim, 2 * dim : 3 * dim]
    return U, E, G


def expm_hvp_eig(
    x: Array, p: Array, basis: Array, hermitian: bool = True
) -> tuple[Array, Array, Array]:
    r"""Directional first and second derivatives of the exponential map (spectral).

    Same result as `expm_hvp` but from a single eigendecomposition, the
    single-direction specialisation of `dexpm_eig` / `d2expm_eig`. With
    $M = \sum_k x_k E_k = V \mathrm{diag}(\mu) V^{-1}$ and
    $\tilde{B} = V^{-1}BV$ for $B = \sum_k p_k E_k$,

    $$U = V \mathrm{diag}(e^\mu) V^{-1}, \qquad
      E = V\,(\Delta \circ \tilde{B})\,V^{-1},$$
    $$G_{ij} = V\,\Big(2\sum_r T_{prq}\,\tilde{B}_{pr}\tilde{B}_{rq}\Big)\,V^{-1},$$

    where $\Delta$ is the first (`_first_divided_differences`) and $T$ the second
    (`_second_divided_differences`) divided difference of $\exp$.

    Args:
        x: Coefficient vector of shape ``(K,)``.
        p: Direction-coefficient vector of shape ``(K,)``.
        basis: Generator array of shape ``(K, d, d)``.
        hermitian: Assume a skew-Hermitian combination and use ``eigh`` — see
            `_eig`. Set ``False`` for complex coefficients or non-skew
            generators.

    Returns:
        Tuple ``(U, E, G)`` of matrices of shape ``(d, d)``.
    """
    mu, V, Vinv = _eig(x, basis, hermitian=hermitian)
    delta = _first_divided_differences(mu)
    T = _second_divided_differences(mu)  # (d, d, d) indexed [p, r, q]

    B = jnp.tensordot(p, basis, axes=[[-1], [0]])
    Bt = Vinv @ B @ V  # V^{-1} B V

    U = (V * jnp.exp(mu)[None, :]) @ Vinv
    E = V @ (delta * Bt) @ Vinv
    # term[p, q] = sum_r T[p, r, q] Bt[p, r] Bt[r, q]; G symmetrises to 2*term.
    term = jnp.einsum("prq,pr,rq->pq", T, Bt, Bt)
    G = V @ (2.0 * term) @ Vinv
    return U, E, G


def get_dexpm(basis: Array, batch_size: int | None = None) -> Callable[[Array], Array]:
    """Create a JIT-compiled exponential-map derivative function.

    Args:
        basis: Generator array of shape ``(K, d, d)``.
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

    Wraps `dexpm_eig` (or `dexpm_eig_batched`) with a fixed generator array. The
    full-``vmap`` variant is the fast default used by
    `geope.jax.get_jacobian_propagator`.

    Args:
        basis: Generator array of shape ``(K, d, d)``.
        batch_size: Optional batch size. If ``None``, the full variant is used;
            otherwise the directions are chunked to bound peak memory.
        hermitian: Assume a skew-Hermitian combination and use ``eigh`` — see
            `_eig`. Set ``False`` for complex coefficients or non-skew
            generators.

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


def get_adj_expm_eig(
    basis: Array, hermitian: bool = True
) -> Callable[[Array, Array], Array]:
    """Create a JIT-compiled spectral exponential-map adjoint.

    Wraps `adj_expm_eig` with a fixed generator array. This is the per-gate step
    `geope.jax.get_vjp_propagator` is built from.

    Args:
        basis: Generator array of shape ``(K, d, d)``.
        hermitian: Assume a skew-Hermitian combination and use ``eigh`` — see
            `_eig`. Set ``False`` for complex coefficients or non-skew
            generators.

    Returns:
        A callable accepting a coefficient vector ``(K,)`` and a covector
        ``(d, d)``, returning the complex overlaps of shape ``(K,)``.
    """
    return jax.jit(partial(adj_expm_eig, basis=basis, hermitian=hermitian))


def get_adj_expm(basis: Array) -> Callable[[Array, Array], Array]:
    """Create a JIT-compiled block-method exponential-map adjoint.

    Wraps `adj_expm` with a fixed generator array — the slow path, kept for
    parity with ``method="block"``.

    Args:
        basis: Generator array of shape ``(K, d, d)``.

    Returns:
        A callable accepting a coefficient vector ``(K,)`` and a covector
        ``(d, d)``, returning the complex overlaps of shape ``(K,)``.
    """
    return jax.jit(partial(adj_expm, basis=basis))


def get_d2expm(basis: Array, batch_size: int | None = None) -> Callable[[Array], Array]:
    """Create a JIT-compiled block second-derivative function.

    Wraps `d2expm` with a fixed generator array (the auxiliary-matrix method).

    Args:
        basis: Generator array of shape ``(K, d, d)``.
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

    Wraps `d2expm_eig` (or `d2expm_eig_batched`) with a fixed generator array.
    The full variant is the fast default used by
    `geope.jax.get_hessian_propagator`.

    Args:
        basis: Generator array of shape ``(K, d, d)``.
        batch_size: Optional batch size. If ``None``, the full variant is used;
            otherwise the first direction is chunked to bound peak memory.
        hermitian: Assume a skew-Hermitian combination and use ``eigh`` — see
            `_eig`. Set ``False`` for complex coefficients or non-skew
            generators.

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
