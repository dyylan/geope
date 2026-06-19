"""Per-step second-derivative benchmarks: block vs. spectral vs. autodiff.

Compares three ways to compute a single gate's second derivative
``(K,) -> (d, d, K, K)``:

* :func:`geope.jax.d2expm` — Goodwin & Kuprov's auxiliary-matrix method
  (``K^2`` block exponentials of ``3d x 3d`` matrices).
* :func:`geope.jax.d2expm_eig` — the spectral second-divided-difference method
  (one eigendecomposition); this is what `get_hessian_manual` uses.
* autodiff ``jax.jacfwd(jax.jacrev(Ui_fn))``.

Run with, e.g.::

    pytest benchmarks/test_bench_d2expm.py \\
        --benchmark-group-by=param --benchmark-columns=mean,median,rounds
"""

import jax
import jax.numpy as jnp
import pytest

from geope.jax import d2expm, d2expm_eig, get_d2expm, get_d2expm_eig, get_Ui_fn

from conftest import make_basis, warm

N_QUBITS = [1, 2, 3]
COMPILE_ROUNDS = 3


def _setup(n):
    basis = make_basis(n)
    K = basis.shape[0]
    x = jax.random.normal(jax.random.key(0), (K,)).astype(jnp.complex128)
    return basis, x


@pytest.mark.parametrize("n", N_QUBITS)
def test_d2expm_eig_exec(benchmark, n):
    basis, x = _setup(n)
    fn = get_d2expm_eig(basis)
    warm(fn, x)
    benchmark.pedantic(lambda: jax.block_until_ready(fn(x)), rounds=10, warmup_rounds=1)


@pytest.mark.parametrize("n", N_QUBITS)
def test_d2expm_block_exec(benchmark, n):
    basis, x = _setup(n)
    fn = get_d2expm(basis)
    warm(fn, x)
    benchmark.pedantic(lambda: jax.block_until_ready(fn(x)), rounds=10, warmup_rounds=1)


@pytest.mark.parametrize("n", N_QUBITS)
def test_d2expm_autodiff_exec(benchmark, n):
    basis, x = _setup(n)
    Ui_fn = get_Ui_fn(basis)
    fn = jax.jit(jax.jacfwd(jax.jacrev(Ui_fn, holomorphic=True), holomorphic=True))
    warm(fn, x)
    benchmark.pedantic(lambda: jax.block_until_ready(fn(x)), rounds=10, warmup_rounds=1)


@pytest.mark.parametrize("n", N_QUBITS)
def test_d2expm_eig_compile(benchmark, n):
    basis, x = _setup(n)
    benchmark.pedantic(
        lambda: get_d2expm_eig(basis).lower(x).compile(),
        rounds=COMPILE_ROUNDS,
        warmup_rounds=0,
    )


def test_d2expm_matches_autodiff():
    """Guard: block, spectral, and autodiff agree on the per-step Hessian."""
    basis, x = _setup(2)
    auto = jax.jacfwd(jax.jacrev(get_Ui_fn(basis), holomorphic=True), holomorphic=True)(
        x
    )
    assert jnp.allclose(d2expm(x, basis), auto, atol=1e-8)
    assert jnp.allclose(d2expm_eig(x, basis), auto, atol=1e-8)
