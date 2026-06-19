"""Full-sequence Jacobian benchmarks: manual stitch vs. naive autodiff.

Compares :func:`geope.jax.get_jacobian_propagator` (the hand-written
``(G, K) -> (G, d, d, K)`` stitch) against the production autodiff path
``jax.jacobian(compute_U_fn, holomorphic=True)`` built by
:func:`geope.engine.get_jacobian_fn` over the ``jax.lax.scan`` product
unitary.

Two benchmark families per size:

* ``*_exec``    — steady-state execution (warmed up; timed call ends in
  ``block_until_ready`` so device work, not async dispatch, is measured).
* ``*_compile`` — one-time XLA compilation cost, isolated via AOT
  ``lower().compile()``. Both paths are single jitted functions.

Run with, e.g.::

    pytest benchmarks/test_bench_jacobian.py \\
        --benchmark-group-by=param:size \\
        --benchmark-columns=mean,median,rounds

which places the manual and autodiff bars for each ``(n, G)`` side by side.
"""

import jax
import jax.numpy as jnp
import pytest

from geope.engine import get_compute_matrices_params_list_fn, get_jacobian_fn
from geope.jax import get_jacobian_propagator

from conftest import make_basis, make_params, warm

# (n_qubits, n_steps): d = 2**n, K = 4**n - 1.
SIZES = [(1, 1), (1, 10), (2, 1), (2, 10), (3, 1), (3, 10),]
SIZE_IDS = [f"n{n}-G{g}" for n, g in SIZES]

# Few rounds for compilation benchmarks — each round recompiles and is slow.
COMPILE_ROUNDS = 3


def _setup(size):
    n, n_steps = size
    basis = make_basis(n)
    K = basis.shape[0]
    params = make_params(n_steps, K, jax.random.key(0))
    return basis, params


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_jacobian_propagator_exec(benchmark, size):
    basis, params = _setup(size)
    fn = get_jacobian_propagator(basis)
    warm(fn, params)
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(params)), rounds=10, warmup_rounds=1
    )


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_jacobian_autodiff_exec(benchmark, size):
    basis, params = _setup(size)
    compute_U_fn = get_compute_matrices_params_list_fn(basis)
    fn = jax.jit(get_jacobian_fn(compute_U_fn))
    warm(fn, params)
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(params)), rounds=10, warmup_rounds=1
    )


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_jacobian_autodiff_compile(benchmark, size):
    basis, params = _setup(size)
    compute_U_fn = get_compute_matrices_params_list_fn(basis)

    # AOT lower().compile() recompiles on every call, isolating XLA compile
    # cost from execution and from the jit dispatch cache.
    def compile_once():
        return jax.jit(get_jacobian_fn(compute_U_fn)).lower(params).compile()

    benchmark.pedantic(compile_once, rounds=COMPILE_ROUNDS, warmup_rounds=0)


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_jacobian_propagator_compile(benchmark, size):
    basis, params = _setup(size)

    # get_jacobian_propagator returns a single jitted function, so XLA compilation
    # can be isolated via AOT lower().compile(), the same as the autodiff path.
    def compile_once():
        return get_jacobian_propagator(basis).lower(params).compile()

    benchmark.pedantic(compile_once, rounds=COMPILE_ROUNDS, warmup_rounds=0)


def test_manual_matches_autodiff():
    """Guard: the two paths must compute the same Jacobian (else the
    execution benchmarks are not comparing equivalent work)."""
    basis, params = _setup((2, 3))
    compute_U_fn = get_compute_matrices_params_list_fn(basis)

    jac_manual = get_jacobian_propagator(basis)(params)  # (G, d, d, K)
    jac_auto = get_jacobian_fn(compute_U_fn)(params)  # (d, d, G, K)
    jac_auto = jnp.transpose(jac_auto, (2, 0, 1, 3))  # -> (G, d, d, K)

    assert jnp.allclose(jac_manual, jac_auto, atol=1e-8)
