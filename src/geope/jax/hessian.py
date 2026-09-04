from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from functools import lru_cache, partial
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
    `geope.geometry.manifold.Manifold.hessian`, which contracts on the fly and never
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
    :func:`geope.geometry.chart.compute_matrices_params_list_fn`,
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


# --- the autodiff alternative to the manual propagator above -----------------


def hvp_forward_over_reverse(
    f: Callable[[Array], Array], params: Array, v: Array
) -> Array:
    r"""Compute a Hessian-vector product via forward-over-reverse mode.

    The chart-agnostic counterpart of `hvp_propagator`: it differentiates any
    scalar callable rather than exploiting the product-of-exponentials structure,
    so it works on the ``param_transform`` path where the manual propagator
    cannot.

    Args:
        f: Scalar-valued callable of ``params``.
        params: Parameter ``Array`` at which to evaluate.
        v: Tangent ``Array`` for the Hessian-vector product.

    Returns:
        The Hessian-vector product $\nabla^2 f \cdot v$.
    """
    v = v.reshape(params.shape)
    return jax.jvp(jax.grad(f), (params,), (v,))[1]


def get_hessian_fn(infid_fn: Callable[[Array], Array]) -> Callable[[Array], Array]:
    """Build the full Hessian function via forward-over-reverse HVPs.

    Materialises the Hessian of ``infid_fn`` by mapping `hvp_forward_over_reverse`
    over the identity matrix's columns — the autodiff drop-in for
    `get_hessian_propagator`, and what `geope.geometry.Manifold.hessian` falls
    back to when no analytic form is available. Returned un-jitted so it fuses
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


def su_hessian_quadratic_form(A: Array, Omega: Array) -> tuple[Array, Array]:
    r"""The Riemannian-Hessian quadratic form of the squared geodesic distance.

    Evaluates $\langle\Omega,\mathcal K_A\Omega\rangle_F$, the intrinsic term of the
    second derivative of $F=\tfrac12 d_g(\cdot,y)^2$ along a curve with
    left-trivialised velocity $\Omega$, where

    $$\mathcal K_A=\frac{\operatorname{ad}_A}{2}
      \coth\!\left(\frac{\operatorname{ad}_A}{2}\right)$$

    is the left-trivialised Riemannian Hessian at $x$ with
    $A=\log_{\min}(y^\dagger x)$.

    No operator is ever formed. $\mathcal K_A$ is diagonal in the eigenbasis of
    $A$: since $A\in\mathfrak{su}(N)$ is skew-Hermitian, $-iA$ is Hermitian and a
    single ``eigh`` gives $A=Q\,i\operatorname{diag}(\theta)\,Q^\dagger$. On the
    root plane of the pair $(j,k)$ the operator $\operatorname{ad}_A$ has
    eigenvalue $i\delta_{jk}$ with $\delta_{jk}=\theta_j-\theta_k$, so
    $\mathcal K_A$ has the real eigenvalue $h(\delta_{jk})$ and, with
    $\tilde\Omega=Q^\dagger\Omega Q$,

    $$\langle\Omega,\mathcal K_A\Omega\rangle_F
      =\sum_{j,k}h(\delta_{jk})\,\bigl|\tilde\Omega_{jk}\bigr|^2,
      \qquad
      h(\delta)=\frac{\delta}{2}\cot\!\left(\frac{\delta}{2}\right),
      \quad h(0)=1,$$

    which is manifestly real.

    Unlike the surrogate $\|\Omega\|_F^2$, this is exact for an **arbitrary**
    $\Omega$ — it does not assume the tangent matching $\Omega\parallel A$. The
    two agree exactly when $\Omega\parallel A$, because $[A,A]=0$ gives
    $\mathcal K_AA=A$ (the radial eigenvalue is $1$).

    Since $h$ decreases from $1$ to $0$ on $[0,\pi)$, the eigenphase spread
    $\rho=\Delta(A)=\max_j\theta_j-\min_j\theta_j$ bounds the whole spectrum:
    for $\rho<\pi$,

    $$\mu(\rho)I\preceq\mathcal K_A\preceq I,
      \qquad \mu(\rho)=\frac{\rho}{2}\cot\!\left(\frac{\rho}{2}\right)>0,$$

    so $\|\Omega\|_F^2$ is always an *upper* bound on the returned value there.
    Beyond $\rho=\pi$ the form is indefinite, and $h$ diverges as
    $|\delta|\to2\pi$ — the cut locus, where $\log_{\min}$ stops being unique.
    ``rho`` is returned alongside so callers can test for that.

    Args:
        A: Skew-Hermitian ``Array`` of shape ``(d, d)``; the minimum-norm
            logarithm $A=\log_{\min}(y^\dagger x)$, traceless for
            $\mathfrak{su}(N)$.
        Omega: ``Array`` of shape ``(d, d)``; the left-trivialised velocity
            $\Omega=x^\dagger\,d\phi_\theta[p]$.

    Returns:
        A tuple ``(value, rho)`` of real scalars: the quadratic form
        $\langle\Omega,\mathcal K_A\Omega\rangle_F$ and the eigenphase spread
        $\rho=\Delta(A)$, both from the same eigendecomposition.

    Example:
        ```python
        # Radial direction: the form reduces to ||A||_F^2 exactly.
        value, rho = su_hessian_quadratic_form(A, A)
        ```
    """
    # -1j * A is Hermitian for skew-Hermitian A, so eigh gives the eigenphases
    # theta_j of A directly (A = Q i diag(theta) Q^dagger).
    theta, Q = jnp.linalg.eigh(-1j * A)
    delta = theta[:, None] - theta[None, :]

    # h(delta) = (delta/2) cot(delta/2), continuously extended to h(0) = 1. The
    # diagonal always hits delta = 0, so the small branch is never unused. Both
    # branches of a `where` are evaluated, so feed tan() a *safe* argument -
    # otherwise tan(0) in the discarded branch poisons the result (and any
    # derivative) with nan.
    small = jnp.abs(delta) < 1e-8
    half = 0.5 * jnp.where(small, 1.0, delta)
    h = jnp.where(small, 1.0 - delta**2 / 12.0, half / jnp.tan(half))

    Omega_tilde = Q.conj().T @ Omega @ Q
    value = jnp.sum(h * jnp.abs(Omega_tilde) ** 2)
    rho = theta[-1] - theta[0]  # eigh returns ascending eigenvalues
    return value, rho


@lru_cache(maxsize=None)
def _skew_hermitian_basis(m: int) -> np.ndarray:
    r"""An orthonormal real basis of the skew-Hermitian $m\times m$ matrices.

    Orthonormal under $\mathrm{Re}\,\mathrm{Tr}(X^\dagger Y)$, with $m^2$
    elements: the diagonal $ie_{kk}$, and for each $j<k$ the real and imaginary
    rotations $(E_{jk}-E_{kj})/\sqrt2$ and $i(E_{jk}+E_{kj})/\sqrt2$. It depends
    on nothing but ``m``, so it is built once per frame size and cached.

    **Returns numpy, and the cache holds numpy.** Memoising a `jax.Array` here
    would be a tracer leak: the first call happens inside whichever trace gets
    there first, and every later trace would then reuse a value belonging to a
    dead one — `jax.errors.UnexpectedTracerError` the moment a second pulse
    length is compiled. The caller converts, inside its own trace.
    """
    basis = np.zeros((m * m, m, m), dtype=complex)
    root_half = 1.0 / np.sqrt(2.0)
    index = 0
    for k in range(m):
        basis[index, k, k] = 1j
        index += 1
    for j in range(m):
        for k in range(j + 1, m):
            basis[index, j, k], basis[index, k, j] = root_half, -root_half
            index += 1
            basis[index, j, k] = basis[index, k, j] = 1j * root_half
            index += 1
    basis.flags.writeable = False
    return basis


def _jacobi_coupled_sector(a: Array, b: Array, c: Array, d1: Array) -> Array:
    r"""The $(C, D_1)$ half of $\langle Z, E_{12}^{-1}E_{11}Z\rangle$.

    The genuinely coupled sector: real dimension $m^2 + 2pm$, and the only place
    a dense operator exponential is needed. Its size depends on $m$ and $p$
    alone, never on $N$.

    The two operators are built by handing `jax.jacfwd` the real-linear packed
    maps — for a linear function the forward Jacobian *is* the matrix, so no
    vec/kron convention can be got wrong (the same reasoning as the reference
    implementation in ``tests/test_factories.py``).
    """
    m = a.shape[0]
    p = b.shape[0]
    dtype = a.dtype
    basis = jnp.asarray(_skew_hermitian_basis(m), dtype=dtype)
    n_skew = m * m
    n = n_skew + 2 * p * m

    def pack(skew: Array, block: Array) -> Array:
        # Elementwise, not an einsum: Re Tr(E_k^dag C) contracts *matching*
        # indices of conj(E_k) and C, and transposing one of them here would
        # silently break the radial identity K_S S = S.
        coeffs = jnp.real(jnp.sum(jnp.conj(basis) * skew, axis=(1, 2)))
        return jnp.concatenate(
            [coeffs, jnp.real(block).ravel(), jnp.imag(block).ravel()]
        )

    def unpack(vector: Array) -> tuple[Array, Array]:
        skew = jnp.tensordot(vector[:n_skew].astype(dtype), basis, axes=(0, 0))
        real, imag = vector[n_skew : n_skew + p * m], vector[n_skew + p * m :]
        return skew, real.reshape(p, m) + 1j * imag.reshape(p, m)

    def apply_l(vector: Array) -> Array:
        skew, block = unpack(vector)
        return pack(
            a @ skew - skew @ a - jnp.conj(b).T @ block + jnp.conj(block).T @ b,
            b @ skew - block @ a,
        )

    def apply_m(vector: Array) -> Array:
        skew, block = unpack(vector)
        commutator = block @ jnp.conj(b).T - b @ jnp.conj(block).T
        return pack(jnp.zeros((m, m), dtype), -commutator @ b)

    origin = jnp.zeros(n)
    l_matrix = jax.jacfwd(apply_l)(origin)
    m_matrix = jax.jacfwd(apply_m)(origin)

    # The Jacobi equation z'' + L z' - M z = 0 as a first-order system.
    generator = jnp.block(
        [[jnp.zeros((n, n)), jnp.eye(n)], [m_matrix, -l_matrix]],
    )
    propagated = jax.scipy.linalg.expm(generator)
    e_11, e_12 = propagated[:n, :n], propagated[:n, n:]

    # z(0) = Z, z(1) = 0  =>  z'(0) = -E_12^{-1} E_11 Z.
    start = pack(c, d1)
    slope_skew, slope_block = unpack(-jnp.linalg.solve(e_12, e_11 @ start))
    # -<Z, z'(0)>, in the canonical inner product 1/2 Re Tr(C^dag C') + Re Tr(D^dag D').
    return -(
        0.5 * jnp.real(jnp.trace(jnp.conj(c).T @ slope_skew))
        + jnp.real(jnp.trace(jnp.conj(d1).T @ slope_block))
    )


def _jacobi_right_sector(a: Array, b: Array, gram: Array) -> Array:
    r"""The $D_2$ half, where both operators are *right* multiplications.

    $\mathsf L(D_2) = -D_2A$ and $\mathsf M(D_2) = -D_2B^\dagger B$ act
    identically on every row, so the whole sector — of real dimension
    $2m(N-2m)$ — collapses to one $2m\times2m$ complex exponential, and enters
    the quadratic form only through the $m\times m$ Gram
    $\Xi^\dagger(\mathbb 1 - QQ^\dagger - Q_1Q_1^\dagger)\Xi$.
    """
    m = a.shape[0]
    dtype = a.dtype
    # [d, e]' = [d, e] G with d' = e and e' = -d B^dag B + e A.
    generator = jnp.block(
        [
            [jnp.zeros((m, m), dtype), -jnp.conj(b).T @ b],
            [jnp.eye(m, dtype=dtype), a],
        ]
    )
    propagated = jax.scipy.linalg.expm(generator)
    f_11, f_21 = propagated[:m, :m], propagated[m:, :m]
    # Y = F_11 F_21^{-1}, i.e. Y F_21 = F_11, solved as F_21^T Y^T = F_11^T.
    y = jnp.linalg.solve(f_21.T, f_11.T).T
    return jnp.real(jnp.trace(gram @ y))


def stiefel_hessian_quadratic_form(
    point: Array, delta: Array, omega: Array
) -> tuple[Array, Array]:
    r"""The canonical-Stiefel Riemannian-Hessian quadratic form.

    Evaluates $g_Q(\Omega, \mathcal K^{\mathrm{St}}_\Delta\Omega)$, the intrinsic
    term of $\psi''(0)$ for $\psi = \tfrac12 d_g(\cdot, Q_\star)^2$ under the
    canonical metric $g_Q(\Xi,\Upsilon) = \mathrm{Re}\,\mathrm{Tr}
    (\Xi^\dagger(\mathbb 1 - \tfrac12 QQ^\dagger)\Upsilon)$.

    **There is no eigenangle formula here.** A general Stiefel manifold is
    normal homogeneous but *not* symmetric, so neither the group's
    $\frac{\operatorname{ad}}2\coth\frac{\operatorname{ad}}2$ nor the sphere's
    $\theta\cot\theta$ applies. What survives is that, once the logarithm is
    known, the Jacobi equation has **constant coefficients** in a homogeneous
    moving frame. With $A = Q^\dagger\Delta$, $Q_\perp B = (\mathbb 1 -
    QQ^\dagger)\Delta$ and a tangent written as $\Xi = QC + Q_\perp D$,

    $$\mathsf L_S(C,D) = \bigl([A,C] - B^\dagger D + D^\dagger B,\; BC - DA\bigr),
      \qquad
      \mathsf M_S(C,D) = \bigl(0,\; -(DB^\dagger - BD^\dagger)B\bigr),$$

    and $z'' + \mathsf L_Sz' - \mathsf M_Sz = 0$. Blocking the unit-time solution
    operator of that system as $e^{\mathbb A_S} = \bigl(\begin{smallmatrix}E_{11}
    & E_{12}\\ E_{21} & E_{22}\end{smallmatrix}\bigr)$ with $\mathbb A_S =
    \bigl(\begin{smallmatrix}0 & \mathbb 1\\ \mathsf M_S &
    -\mathsf L_S\end{smallmatrix}\bigr)$ gives $\mathcal K^{\mathrm{St}}_S =
    E_{12}^{-1}E_{11} - \tfrac12\mathsf L_S$, and since $\mathsf L_S$ is
    skew-adjoint the *diagonal* form drops its second term entirely.

    **The operator block-diagonalises, which is what makes this affordable.**
    Choosing the complement as $Q_\perp = (Q_1\mid Q_2)$ with
    $\mathrm{range}((\mathbb 1 - QQ^\dagger)\Delta) \subseteq
    \mathrm{span}(Q_1)$ confines $B$ to $Q_1$, and both operators then split
    along $(C, D_1)\oplus D_2$. On $D_2$ they reduce to *right* multiplications
    identical for every row (`_jacobi_right_sector`), so only $(C, D_1)$ needs a
    dense exponential (`_jacobi_coupled_sector`). The naive route over the whole
    $2Nm - m^2$-dimensional tangent space costs $O((Nm)^3)$; this costs
    $O(m^6)$, **independent of $N$**, and never forms $Q_2$. Prefer
    `geope.line_searches.QuadraticArmijo` once $m$ grows past roughly 8.

    **Not even in $\Delta$.** Unlike `su_hessian_quadratic_form`, whose kernel
    $h$ is even, this form genuinely depends on the sign: $-\Delta$ points at the
    geodesic reflection of the target, which is a different Hessian. Only at
    $m = N$, where the manifold *is* the group, does evenness return. Callers
    holding the context's ``A = -log(point, target)`` must negate it.

    In the two limits it reduces correctly: at $m = N$ the vertical space
    vanishes, $\mathsf M_S = 0$, and this reproduces
    `su_hessian_quadratic_form` — value and ``rho`` alike — to double precision;
    at $\Delta = 0$ both operators vanish, $\mathcal K = \mathbb 1$, and the form
    is exactly $\lVert\Omega\rVert_Q^2$.

    Args:
        point: The base frame $Q$, ``(N, m)`` with $Q^\dagger Q = \mathbb 1_m$.
        delta: The tangent $\Delta = \mathrm{Log}_Q(Q_\star)$ at ``point``,
            pointing **toward** the target.
        omega: The tangent $\Omega$ the form is evaluated on, ``(N, m)``.

    Returns:
        A tuple ``(value, rho)`` of real scalars: the quadratic form
        $g_Q(\Omega, \mathcal K^{\mathrm{St}}_\Delta\Omega)$, and the eigenphase
        spread $\rho$ of the horizontal lift $S$ as the cut-locus diagnostic.

    References:
        R. Zimmermann and K. Hüper, *Computing the Riemannian logarithm on the
        Stiefel manifold*, SIAM J. Matrix Anal. Appl. **43**, 953 (2022).

        W. Ziller, *The Jacobi equation on naturally reductive compact
        Riemannian homogeneous spaces*, Comment. Math. Helv. **52**, 573 (1977).
    """
    n_dim, m = point.shape
    p = min(m, n_dim - m)
    dtype = jnp.result_type(point, delta, omega, jnp.complex128)
    point, delta, omega = point.astype(dtype), delta.astype(dtype), omega.astype(dtype)

    # Including `point` in the QR is what keeps Q_1 orthogonal to it even when
    # (1 - QQ^dag) Delta is rank deficient: a QR of the residual alone completes
    # with arbitrary columns that need not miss Q's span, and the D_2 reduction
    # below would then be projecting against a non-orthogonal split. It also
    # degrades correctly when 2m > N, where the reduced QR returns only N
    # columns and the slice picks up exactly p = N - m of them.
    q1 = jnp.linalg.qr(jnp.concatenate([point, delta], axis=1))[0][:, m : m + p]

    a = jnp.conj(point).T @ delta  # (m, m), skew-Hermitian
    b = jnp.conj(q1).T @ delta  # (p, m)
    c = jnp.conj(point).T @ omega  # (m, m), skew-Hermitian
    d1 = jnp.conj(q1).T @ omega  # (p, m)
    # The Q_2 sector enters only through this m x m Gram, which is why the
    # ambient complement is never materialised.
    gram = jnp.conj(omega).T @ (omega - point @ c - q1 @ d1)

    value = _jacobi_coupled_sector(a, b, c, d1) + _jacobi_right_sector(a, b, gram)

    # rho: the eigenphase spread of the horizontal lift S. At m = N (p = 0) it is
    # S = A and this is the group's spread verbatim.
    lift = jnp.block([[a, -jnp.conj(b).T], [b, jnp.zeros((p, p), dtype)]])
    phases = jnp.linalg.eigvalsh(-1j * lift)  # ascending
    low, high = phases[0], phases[-1]
    if n_dim > m + p:
        # S annihilates the rest of the ambient space, so 0 is an eigenvalue too.
        low, high = jnp.minimum(low, 0.0), jnp.maximum(high, 0.0)
    return value, high - low
