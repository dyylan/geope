"""Full-sequence Hessian benchmarks: manual vs. autodiff.

Two objects are benchmarked:

* the **propagator** Hessian ``d^2U/dphi^2`` — `get_hessian_manual` (spectral,
  prefix/suffix products) vs autodiff ``jax.jacfwd(jax.jacrev(compute_U))``;
* the **infidelity-cost** Hessian ``(P, P)`` — `get_hessian_manual_fn`
  (Goodwin-Kuprov) vs the autodiff `get_hessian_fn` over the same infidelity.

``*_exec`` benchmarks are warmed up and timed with ``block_until_ready``.

Run with, e.g.::

    pytest benchmarks/test_bench_hessian.py \\
        --benchmark-group-by=param:size --benchmark-columns=mean,median,rounds
"""

import jax
import jax.numpy as jnp
import pytest

from geope.engine import (
    get_compute_matrices_params_list_fn,
    get_hessian_fn,
    get_hessian_manual_fn,
    get_infidelity_fn,
)
from geope.jax import get_hessian_manual
from geope.utils import qft_unitary

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
def test_propagator_hessian_manual_exec(benchmark, size):
    _, basis, params = _setup(size)
    fn = get_hessian_manual(basis)
    jax.block_until_ready(fn(params))
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(params)), rounds=10, warmup_rounds=1
    )


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_propagator_hessian_autodiff_exec(benchmark, size):
    _, basis, params = _setup(size)
    compute_U = get_compute_matrices_params_list_fn(basis)
    fn = jax.jit(jax.jacfwd(jax.jacrev(compute_U, holomorphic=True), holomorphic=True))
    jax.block_until_ready(fn(params))
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(params)), rounds=10, warmup_rounds=1
    )


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_cost_hessian_manual_exec(benchmark, size):
    n, basis, params = _setup(size, real=True)
    target = jnp.asarray(qft_unitary(n))
    fn = jax.jit(get_hessian_manual_fn(basis, target, projective=True))
    jax.block_until_ready(fn(params))
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(params)), rounds=10, warmup_rounds=1
    )


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_cost_hessian_autodiff_exec(benchmark, size):
    n, basis, params = _setup(size, real=True)
    target = jnp.asarray(qft_unitary(n))
    compute_U = get_compute_matrices_params_list_fn(basis)
    infid_U = get_infidelity_fn(target)
    fn = jax.jit(get_hessian_fn(lambda x: infid_U(compute_U(x))))
    jax.block_until_ready(fn(params))
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(params)), rounds=10, warmup_rounds=1
    )
