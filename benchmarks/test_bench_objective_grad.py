r"""Objective-**gradient** benchmarks: the propagator pullback vs. autodiff.

`Manifold.value_and_grad` is what GRAPE actually calls, rather than the bare
propagator derivatives the sibling modules time, and it is benchmarked against
the autodiff reference it replaced, `Manifold.value_and_grad_autodiff`.

This is the interesting comparison. Reverse-mode autodiff already gets the whole
gradient in $O(G)$ propagator-sized adjoints, so a manual path only wins if it
also avoids the $(G, d, d, K)$ Jacobian — which `geope.jax.vjp_propagator` does,
landing the partial products on the covector and contracting each gate's
derivative spectrally. Both routes are then $O(G(d^3 + d^2K))$, and this measures
the constant.

The Hessian assembled from the same jet is in
``benchmarks/test_bench_objective_hessian.py``; it has three routes and its own
size ranges, which is why the two do not share a module.

``*_exec`` benchmarks are warmed up and timed with ``block_until_ready``.

Run with, e.g.::

    pytest benchmarks/test_bench_objective_grad.py \
        --benchmark-group-by=param:size --benchmark-columns=mean,median,rounds
"""

import jax
import jax.numpy as jnp
import pytest

from conftest import make_su_manifold, warm

# (n_qubits, n_steps): d = 2**n, K = 4**n - 1. The gradient is linear in K, so it
# reaches sizes the dense Hessian cannot.
GRAD_SIZES = [(1, 10), (2, 5), (2, 20), (3, 5), (3, 20)]
GRAD_IDS = [f"n{n}-G{g}" for n, g in GRAD_SIZES]


@pytest.mark.parametrize("size", GRAD_SIZES, ids=GRAD_IDS)
def test_grad_propagator_exec(benchmark, size):
    manifold, free = make_su_manifold(size)
    fn = manifold.value_and_grad
    warm(fn, free)
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(free)), rounds=10, warmup_rounds=1
    )


@pytest.mark.parametrize("size", GRAD_SIZES, ids=GRAD_IDS)
def test_grad_autodiff_exec(benchmark, size):
    manifold, free = make_su_manifold(size)
    fn = jax.jit(manifold.value_and_grad_autodiff)
    warm(fn, free)
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(free)), rounds=10, warmup_rounds=1
    )


# ---------------------------------------------------------------------------
# Guard — the benchmarks must be comparing equivalent work
# ---------------------------------------------------------------------------


def test_grad_matches_autodiff():
    manifold, free = make_su_manifold((2, 5))
    value, grad = manifold.value_and_grad(free)
    ref_value, ref_grad = manifold.value_and_grad_autodiff(free)
    assert jnp.allclose(value, ref_value, atol=1e-12)
    assert jnp.allclose(grad, ref_grad, atol=1e-9)
