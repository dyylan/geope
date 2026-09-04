"""Full-sequence **propagator** Hessian benchmarks: manual vs. autodiff.

Benchmarks ``d^2U/dphi^2`` itself — `get_hessian_propagator` (spectral,
prefix/suffix products) against autodiff
``jax.jacfwd(jax.jacrev(compute_point))``.

The *cost* Hessian that used to sit here moved to
``benchmarks/test_bench_objectives.py``: it is now assembled at the manifold
level from this tensor plus the manifold's own cost derivatives, rather than by a
group-specific function, so it belongs with the gradient it shares that assembly
with.

``*_exec`` benchmarks are warmed up and timed with ``block_until_ready``.

Run with, e.g.::

    pytest benchmarks/test_bench_hessian.py \
        --benchmark-group-by=param:size --benchmark-columns=mean,median,rounds
"""

import jax
import jax.numpy as jnp
import pytest

from geope.geometry.chart import get_compute_matrices_params_list_fn
from geope.jax import get_hessian_propagator

from conftest import make_basis

# (n_qubits, n_steps). Kept small: the propagator Hessian is O(G^2 d^2 K^2).
SIZES = [(1, 2), (2, 2), (2, 3)]
SIZE_IDS = [f"n{n}-G{g}" for n, g in SIZES]


def _setup(size, real=False):
    n, n_steps = size
    basis = make_basis(n)
    K = basis.shape[0]
    params = jax.random.normal(jax.random.key(0), (n_steps, K)) * 0.3
    if not real:
        params = params.astype(jnp.complex128)
    return n, basis, params


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_propagator_hessian_propagator_exec(benchmark, size):
    _, basis, params = _setup(size)
    fn = get_hessian_propagator(basis)
    jax.block_until_ready(fn(params))
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(params)), rounds=10, warmup_rounds=1
    )


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_propagator_hessian_autodiff_exec(benchmark, size):
    _, basis, params = _setup(size)
    compute_point = get_compute_matrices_params_list_fn(basis)
    fn = jax.jit(
        jax.jacfwd(jax.jacrev(compute_point, holomorphic=True), holomorphic=True)
    )
    jax.block_until_ready(fn(params))
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(params)), rounds=10, warmup_rounds=1
    )


def test_manual_matches_autodiff():
    """Guard: the two paths must compute the same Hessian (else the execution
    benchmarks are not comparing equivalent work)."""
    _, basis, params = _setup((2, 2))
    compute_point = get_compute_matrices_params_list_fn(basis)

    manual = get_hessian_propagator(basis)(params)  # (G, G, d, d, K, K)
    auto = jax.jacfwd(jax.jacrev(compute_point, holomorphic=True), holomorphic=True)(
        params
    )  # (d, d, G, K, G, K)
    auto = jnp.einsum("...ikjl->ij...kl", auto)  # -> (G, G, d, d, K, K)

    assert jnp.allclose(manual, auto, atol=1e-8)
