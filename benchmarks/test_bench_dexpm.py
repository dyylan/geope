"""Per-step Jacobian benchmarks: block vs. spectral vs. naive autodiff.

Compares three ways to compute a single gate's derivative
``(K,) -> (d, d, K)``:

* :func:`geope.jax.dexpm` — the Al-Mohy & Higham block-exponential method
  (``K`` block matrix exponentials).
* :func:`geope.jax.dexpm_eig` — the spectral / Fréchet method (one
  eigendecomposition + BLAS matmuls); this is what `get_jacobian_propagator` uses.
* autodiff of a single exponential gate ``jax.jacobian(Ui_fn, holomorphic=True)``.

All produce the same ``(d, d, K)`` layout, so no transpose is needed.
Also includes the optionally-batched ``dexpm`` (``get_dexpm(basis,
batch_size=...)``) which trades execution speed for lower peak memory.

Run with, e.g.::

    pytest benchmarks/test_bench_dexpm.py \\
        --benchmark-group-by=param:n --benchmark-columns=mean,median,rounds
"""

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from geope.jax import dexpm, dexpm_eig, get_dexpm, get_dexpm_eig, get_Ui_fn

from conftest import make_basis, warm

N_QUBITS = [
    1,
    2,
    3,
]  # d = 2**n, K = 4**n - 1
COMPILE_ROUNDS = 3


def _setup(n):
    basis = make_basis(n)
    K = basis.shape[0]
    x = jax.random.normal(jax.random.key(0), (K,)).astype(jnp.complex128)
    return basis, x


@pytest.mark.parametrize("n", N_QUBITS)
def test_dexpm_exec(benchmark, n):
    basis, x = _setup(n)
    fn = get_dexpm(basis)  # already @jax.jit
    warm(fn, x)
    benchmark.pedantic(lambda: jax.block_until_ready(fn(x)), rounds=20, warmup_rounds=1)


@pytest.mark.parametrize("n", N_QUBITS)
def test_dexpm_eig_exec(benchmark, n):
    basis, x = _setup(n)
    fn = get_dexpm_eig(basis)  # spectral method, one eigendecomposition
    warm(fn, x)
    benchmark.pedantic(lambda: jax.block_until_ready(fn(x)), rounds=20, warmup_rounds=1)


@pytest.mark.parametrize("n", N_QUBITS)
def test_dexpm_batched_exec(benchmark, n):
    basis, x = _setup(n)
    fn = get_dexpm(basis, batch_size=4)
    warm(fn, x)
    benchmark.pedantic(lambda: jax.block_until_ready(fn(x)), rounds=20, warmup_rounds=1)


@pytest.mark.parametrize("n", N_QUBITS)
def test_dexpm_autodiff_exec(benchmark, n):
    basis, x = _setup(n)
    Ui_fn = get_Ui_fn(basis)
    fn = jax.jit(jax.jacobian(Ui_fn, holomorphic=True))
    warm(fn, x)
    benchmark.pedantic(lambda: jax.block_until_ready(fn(x)), rounds=20, warmup_rounds=1)


@pytest.mark.parametrize("n", N_QUBITS)
def test_dexpm_compile(benchmark, n):
    basis, x = _setup(n)

    def compile_once():
        return jax.jit(partial(dexpm, basis=basis)).lower(x).compile()

    benchmark.pedantic(compile_once, rounds=COMPILE_ROUNDS, warmup_rounds=0)


@pytest.mark.parametrize("n", N_QUBITS)
def test_dexpm_autodiff_compile(benchmark, n):
    basis, x = _setup(n)
    Ui_fn = get_Ui_fn(basis)

    def compile_once():
        return jax.jit(jax.jacobian(Ui_fn, holomorphic=True)).lower(x).compile()

    benchmark.pedantic(compile_once, rounds=COMPILE_ROUNDS, warmup_rounds=0)


def test_dexpm_matches_autodiff():
    """Guard: block, spectral, and autodiff must agree on the per-step Jacobian."""
    basis, x = _setup(2)
    jac_auto = jax.jacobian(get_Ui_fn(basis), holomorphic=True)(x)  # (d, d, K)
    assert jnp.allclose(dexpm(x, basis), jac_auto, atol=1e-8)
    assert jnp.allclose(dexpm_eig(x, basis), jac_auto, atol=1e-8)
