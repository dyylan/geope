from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from functools import partial
from typing import Callable

from .dexpm import get_dexpm, get_dexpm_eig, expm_jvp, expm_jvp_eig


def Ui(x: Array, basis: Array) -> Array:
    """Compute a unitary from a linear combination of Hermitian basis matrices.

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
        A callable that takes a coefficient vector and returns
        the corresponding unitary.
    """
    return partial(Ui, basis=basis)


def jacobian_propagator(
    params: Array, Ui_fn: Callable[[Array], Array], jac_fn: Callable[[Array], Array]
) -> Array:
    r"""Compute the full Jacobian propagator of the product unitary.

    The product unitary follows the convention of
    :func:`geope.geometry.chart.compute_matrices_params_list_fn`, where each gate is
    left-multiplied onto the accumulator,

    $$U = U_{G-1} \cdots U_1 U_0, \qquad U_i = \exp\!\Big(i \sum_k x_{i,k} G_k\Big).$$

    The derivative with respect to a parameter of gate $i$ leaves every other
    gate untouched, so it is a product with a single factor replaced by the
    per-gate derivative:

    $$\frac{\partial U}{\partial x_{i,k}}
        = \underbrace{U_{G-1} \cdots U_{i+1}}_{L_i}\,
          \frac{\partial U_i}{\partial x_{i,k}}\,
          \underbrace{U_{i-1} \cdots U_0}_{R_i}.$$

    Both the left ($L_i$, exclusive suffix product) and right ($R_i$, exclusive
    prefix product) partial products are obtained in $O(G)$ matrix
    multiplications with two ``jax.lax.scan`` passes, after which the per-gate
    derivative blocks are combined with a single vectorised ``einsum``. This is
    the equivalent of differentiating the whole sequence with autodiff, but
    built explicitly from the per-gate derivative ``jac_fn``.

    Args:
        params: Parameter ``Array`` of shape ``(G, K)``.
        Ui_fn: Callable mapping a coefficient ``Array`` to a unitary ``Array``.
        jac_fn: Callable computing the per-gate Jacobian ``Array`` of shape
            ``(d, d, K)`` (e.g. :func:`geope.jax.dexpm`).

    Returns:
        An ``Array`` of shape ``(G, d, d, K)`` containing the full Jacobian.
    """
    # Per-gate unitaries (G, d, d) and per-gate derivatives (G, d, d, K).
    gates = jax.vmap(Ui_fn)(params)
    jacs = jax.vmap(jac_fn)(params)

    eye = jnp.eye(gates.shape[1], dtype=gates.dtype)

    # Exclusive prefix products: R[i] = gates[i-1] @ ... @ gates[0], R[0] = I.
    # Emit the running product *before* folding in the current gate.
    def step_right(R, g):
        return g @ R, R

    Rs = jax.lax.scan(step_right, eye, gates)[1]

    # Exclusive suffix products: L[i] = gates[G-1] @ ... @ gates[i+1], L[G-1] = I.
    # Scan in reverse so the running product holds the gates processed so far.
    def step_left(L, g):
        return L @ g, L

    Ls = jax.lax.scan(step_left, eye, gates, reverse=True)[1]

    # Block_i[a, c, k] = L_i[a, b] jac_i[b, e, k] R_i[e, c].
    return jax.vmap(lambda L, J, R: jnp.einsum("ab,bek,ec->ack", L, J, R))(Ls, jacs, Rs)


def get_jacobian_propagator(
    gate_basis: Array, method: str = "eig", hermitian: bool = True
) -> Callable[[Array], Array]:
    """Create a JIT-compiled manual Jacobian function for a given gate basis.

    The returned function is wrapped in ``jax.jit`` so it is compiled once and
    reused across calls.

    Args:
        gate_basis: ``Array`` of Hermitian basis matrices of shape ``(K, d, d)``.
        method: Per-gate derivative method. ``"eig"`` (default) uses the
            spectral method (`geope.jax.dexpm_eig`); ``"block"`` uses the
            block-exponential method (`geope.jax.dexpm`), which handles
            non-Hermitian generators and ignores ``hermitian``.
        hermitian: Assume real parameters (skew-Hermitian generators) and use
            the faster ``eigh``-based per-gate derivative. Set ``False`` for
            complex-valued parameters. Only affects ``method="eig"``.

    Returns:
        A ``Callable[[Array], Array]`` that accepts a parameter array
        of shape ``(G, K)`` and returns the Jacobian of shape
        ``(G, d, d, K)``.
    """
    Ui_fn = get_Ui_fn(gate_basis)
    if method == "eig":
        jac_fn = get_dexpm_eig(gate_basis, hermitian=hermitian)
    elif method == "block":
        jac_fn = get_dexpm(gate_basis)
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'eig' or 'block'.")
    return jax.jit(partial(jacobian_propagator, Ui_fn=Ui_fn, jac_fn=jac_fn))


def jvp_propagator(
    params: Array,
    direction: Array,
    step_fn: Callable[[Array, Array], tuple[Array, Array]],
) -> tuple[Array, Array]:
    r"""Directional first derivative (JVP) of the product unitary in $O(G)$.

    Computes the value and the single-direction derivative of the product
    unitary along a parameter-space direction $p$, without forming the full
    Jacobian. With the product convention of
    :func:`geope.geometry.chart.compute_matrices_params_list_fn`,
    $\phi(\theta) = U_{G-1} \cdots U_1 U_0$ with each gate left-multiplied,
    define the partial product $X_g = U_g \cdots U_0$ and its derivative
    $V_g = \dot X_g(0)$ along $\theta(t) = \theta + t p$. Since
    $X_g(t) = U_g(t) X_{g-1}(t)$, the product rule gives the linear-time
    recursion

    $$X_g = U_g X_{g-1}, \qquad V_g = U_g V_{g-1} + E_g X_{g-1},$$

    with $X_{-1} = I$, $V_{-1} = 0$, where $U_g = \exp(iA_g)$ and
    $E_g = D\exp(iA_g)[iB_g]$ are the per-gate value and directional derivative
    (`step_fn`). After all $G$ gates, $X_{G-1} = \phi(\theta)$ and
    $V_{G-1} = D\phi_\theta[p]$. The recursion is a single ``jax.lax.scan``; this
    is the forward-mode (JVP) analogue of `jacobian_propagator`, and the
    first-order sibling of `geope.jax.hessian.hvp_propagator`.

    Args:
        params: Parameter ``Array`` of shape ``(G, K)``.
        direction: Direction ``Array`` of shape ``(G, K)`` (the $p$ above).
        step_fn: Per-gate step ``(x, p) -> (U, E)`` mapping a gate's
            coefficients and direction to its value and directional derivative
            (e.g. `geope.jax.expm_jvp_eig`).

    Returns:
        Tuple ``(X, V)`` of matrices of shape ``(d, d)``: the product unitary
        $\phi(\theta)$ and its directional derivative $D\phi_\theta[p]$.
    """
    Us, Es = jax.vmap(step_fn)(params, direction)  # each (G, d, d)

    eye = jnp.eye(Us.shape[1], dtype=Us.dtype)

    def step(carry, gate):
        X, V = carry
        U, E = gate
        X_new = U @ X
        V_new = U @ V + E @ X
        return (X_new, V_new), None

    (X, V), _ = jax.lax.scan(step, (eye, jnp.zeros_like(eye)), (Us, Es))
    return X, V


def get_jvp_propagator(
    gate_basis: Array, method: str = "eig", hermitian: bool = True
) -> Callable[[Array, Array], tuple[Array, Array]]:
    """Create a JIT-compiled directional-JVP propagator for a given gate basis.

    Wraps `jvp_propagator` with a per-gate step built from ``gate_basis`` and is
    wrapped in ``jax.jit`` so it compiles once and is reused across calls.

    Args:
        gate_basis: ``Array`` of Hermitian basis matrices of shape ``(K, d, d)``.
        method: Per-gate method. ``"eig"`` (default) uses the spectral
            `geope.jax.expm_jvp_eig`; ``"block"`` uses the block-exponential
            `geope.jax.expm_jvp` (ignores ``hermitian``).
        hermitian: Assume real parameters and use the ``eigh``-based per-gate
            step. Set ``False`` for complex-valued parameters. Only affects
            ``method="eig"``.

    Returns:
        A ``Callable[[Array, Array], tuple[Array, Array]]`` accepting parameters
        and a direction, both of shape ``(G, K)``, and returning the pair
        ``(phi, Dphi[p])`` of shape ``(d, d)``.
    """
    if method == "eig":
        step_fn = partial(expm_jvp_eig, basis=gate_basis, hermitian=hermitian)
    elif method == "block":
        step_fn = partial(expm_jvp, basis=gate_basis)
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'eig' or 'block'.")
    return jax.jit(partial(jvp_propagator, step_fn=step_fn))
