"""Forward-map and directional-derivative propagator benchmarks.

Times the three un-benchmarked hot kernels of the GEOPE inner loop, as a
function of system size ``n`` (Hilbert dimension ``d = 2**n``) and pulse length
``G`` (``piecewise_steps``):

* ``compute_matrices`` — the forward product unitary
  :func:`geope.geometry.chart.compute_matrices_params_list_fn` (a ``jax.lax.scan`` of
  ``expm(i * sum_k x_k B_k)``).
* ``jvp`` — the directional first derivative
  :func:`geope.jax.get_jvp_propagator` (single-``scan`` ``(X, V)`` recursion).
* ``hvp`` — the directional second derivative
  :func:`geope.jax.get_hvp_propagator` (single-``scan`` ``(X, V, W)``).

The *full* ``jacobian_propagator`` / ``hessian_propagator`` are already covered
by ``test_bench_jacobian.py`` / ``test_bench_hessian.py`` and are out of scope.

**Realistic poly(n) basis.** Unlike the other benchmarks (which use the full
``4**n - 1`` Pauli basis), this one uses :func:`construct_restricted_pauli_basis`
with single-qubit X/Y/Z on every qubit plus nearest-neighbour ZZ couplings, so
the generator count is ``K = 4n - 1`` (linear in ``n``), as in a real control
problem. With ``K = O(n)``, the per-gate Hamiltonian assembly ``O(K * d**2)`` is
subdominant to the ``O(d**3)`` exponential/eigendecomposition, so the expected
scaling is ``O(G * d**3)``: **linear in G** and **~8x per added qubit**.

Timing follows the repo convention — warm up once (compile outside the timed
region), then time steady-state calls that end in ``block_until_ready`` so device
work (not async dispatch) is measured.

Run with, e.g.::

    pytest benchmarks/test_bench_propagators.py \
        --benchmark-group-by=param:size \
        --benchmark-columns=mean,median,rounds

which places the ``compute_matrices``/``jvp``/``hvp`` bars for each ``(n, G)``
side by side.
"""

import jax
import jax.numpy as jnp
import pytest

from geope.geometry.chart import get_compute_matrices_params_list_fn
from geope.jax import get_jvp_propagator, get_hvp_propagator
from geope.utils import construct_restricted_pauli_basis

from conftest import make_params, warm

# System-size sweep at fixed G (isolates the d axis; expect ~8x per qubit) and
# pulse-length sweep at fixed n (isolates the G axis; expect linear). The shared
# (2, 10) point lives in SIZES_N, so SIZES_G omits it to avoid a duplicate id.
SIZES_N = [(1, 10), (2, 10), (3, 10), (4, 10)]
SIZES_G = [(2, 1), (2, 5), (2, 20), (2, 40)]
SIZES = SIZES_N + SIZES_G
SIZE_IDS = [f"n{n}-G{g}" for n, g in SIZES]


def make_restricted_basis(n: int) -> jnp.ndarray:
    """Poly(n) control basis as a ``(K, d, d)`` complex128 array.

    Single-qubit X/Y/Z on each of ``n`` qubits (``3n`` generators) plus
    nearest-neighbour ZZ couplings on the ``n - 1`` bonds, giving
    ``K = 4n - 1`` and ``d = 2**n``.
    """
    restriction: dict = {}
    for i in range(1, n + 1):
        restriction[i] = ["x", "y", "z"]
    for i in range(1, n):
        restriction[(i, i + 1)] = ["zz"]
    return jnp.asarray(construct_restricted_pauli_basis(n, restriction).basis)


def _real_params_and_dir(G: int, K: int):
    """Deterministic real ``(G, K)`` params and direction for the eig path.

    ``method="eig", hermitian=True`` assumes real coefficients (so
    ``A = sum_k x_k B_k`` is Hermitian), hence float64 rather than the complex
    inputs used for the holomorphic forward map.
    """
    k1, k2 = jax.random.split(jax.random.key(0))
    params = jax.random.normal(k1, (G, K))
    direction = jax.random.normal(k2, (G, K))
    return params, direction


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_compute_matrices_exec(benchmark, size):
    n, G = size
    basis = make_restricted_basis(n)
    params = make_params(G, basis.shape[0], jax.random.key(0))  # (G, K) complex128
    # get_compute_matrices_params_list_fn returns a bare partial (unlike the JVP/
    # HVP factories, which jit internally). In production it is always consumed
    # inside a jitted function, so wrap it here to time device work, not the
    # per-call re-trace/compile of the lax.scan.
    fn = jax.jit(get_compute_matrices_params_list_fn(basis))
    warm(fn, params)
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(params)), rounds=20, warmup_rounds=1
    )


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_jvp_propagator_exec(benchmark, size):
    n, G = size
    basis = make_restricted_basis(n)
    p, v = _real_params_and_dir(G, basis.shape[0])
    fn = get_jvp_propagator(basis, method="eig")
    warm(fn, p, v)
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(p, v)), rounds=20, warmup_rounds=1
    )


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_hvp_propagator_exec(benchmark, size):
    n, G = size
    basis = make_restricted_basis(n)
    p, v = _real_params_and_dir(G, basis.shape[0])
    fn = get_hvp_propagator(basis, method="eig")
    warm(fn, p, v)
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(p, v)), rounds=20, warmup_rounds=1
    )


def test_propagators_match_autodiff():
    """Guard: the directional propagators must agree with autodiff of the
    forward product unitary (else the timings compare different work)."""
    n, G = 2, 4
    basis = make_restricted_basis(n)
    p, v = _real_params_and_dir(G, basis.shape[0])
    compute_U = get_compute_matrices_params_list_fn(basis)

    # Real eig-path inputs, cast to complex for the holomorphic forward map so
    # the two paths build the same Hermitian A = sum_k x_k B_k.
    p_c = p.astype(jnp.complex128)
    v_c = v.astype(jnp.complex128)

    # First order: value and directional derivative.
    X_jvp, V_jvp = get_jvp_propagator(basis, method="eig")(p, v)
    X_ref, V_ref = jax.jvp(compute_U, (p_c,), (v_c,))
    assert jnp.allclose(X_jvp, X_ref, atol=1e-8)
    assert jnp.allclose(V_jvp, V_ref, atol=1e-8)

    # Second order: nest jax.jvp to get D^2 phi[v, v].
    X_hvp, V_hvp, W_hvp = get_hvp_propagator(basis, method="eig")(p, v)
    _, W_ref = jax.jvp(lambda q: jax.jvp(compute_U, (q,), (v_c,))[1], (p_c,), (v_c,))
    assert jnp.allclose(X_hvp, X_ref, atol=1e-8)
    assert jnp.allclose(V_hvp, V_ref, atol=1e-8)
    assert jnp.allclose(W_hvp, W_ref, atol=1e-8)
