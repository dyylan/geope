"""Objective-derivative benchmarks: the propagator assembly vs. autodiff.

These are the quantities GRAPE actually calls — `Manifold.value_and_grad` and
`Manifold.hessian` — rather than the bare propagator derivatives the sibling
modules time. Each is benchmarked against the autodiff reference it replaced,
`Manifold.value_and_grad_autodiff` and `Manifold.hessian_autodiff`.

The gradient is the interesting one. Reverse-mode autodiff already gets the whole
gradient in $O(G)$ propagator-sized adjoints, so a manual path only wins if it
also avoids the $(G, d, d, K)$ Jacobian — which
`geope.jax.vjp_propagator` does, landing the partial products on the covector and
contracting each gate's derivative spectrally. Both routes are then
$O(G(d^3 + d^2K))$, and this measures the constant.

The Hessian is dense, $O(G^2 d^2 K^2)$, so its sizes are kept small.

``*_exec`` benchmarks are warmed up and timed with ``block_until_ready``.

Run with, e.g.::

    pytest benchmarks/test_bench_objectives.py \
        --benchmark-group-by=param:size --benchmark-columns=mean,median,rounds
"""

import jax
import jax.numpy as jnp
import pytest

from geope.geometry import SpecialUnitaryGroup
from geope.parameters import Parameters
from geope.utils import construct_full_pauli_basis, qft_unitary

from conftest import warm

# (n_qubits, n_steps): d = 2**n, K = 4**n - 1. The gradient is linear in K, so it
# reaches sizes the dense Hessian cannot.
GRAD_SIZES = [(1, 10), (2, 5), (2, 20), (3, 5), (3, 20)]
GRAD_IDS = [f"n{n}-G{g}" for n, g in GRAD_SIZES]

# The Hessian is O(G^2 d^2 K^2) in both flops and memory.
HESS_SIZES = [(1, 2), (2, 2), (2, 3)]
HESS_IDS = [f"n{n}-G{g}" for n, g in HESS_SIZES]


def _manifold(size):
    """A bound `SpecialUnitaryGroup` and a real pulse of the right shape.

    The pulse is real ``float64``: that is the regime GRAPE runs in, and it is
    the one where the autodiff reference is exact (differentiating holomorphically
    w.r.t. a ``complex128`` array with a zero imaginary part gives a gradient with
    a spurious imaginary part — see `Manifold.value_and_grad_autodiff`).
    """
    n, n_steps = size
    basis = construct_full_pauli_basis(n)
    params = Parameters(
        basis=basis,
        projected_basis=basis,
        target=qft_unitary(n),
        piecewise_steps=n_steps,
        seed=0,
        manifold=SpecialUnitaryGroup(2**n),
    )
    free = jnp.real(params.free()).astype(jnp.float64)
    return params.manifold, free


# ---------------------------------------------------------------------------
# The gradient
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", GRAD_SIZES, ids=GRAD_IDS)
def test_grad_propagator_exec(benchmark, size):
    manifold, free = _manifold(size)
    fn = manifold.value_and_grad
    warm(fn, free)
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(free)), rounds=10, warmup_rounds=1
    )


@pytest.mark.parametrize("size", GRAD_SIZES, ids=GRAD_IDS)
def test_grad_autodiff_exec(benchmark, size):
    manifold, free = _manifold(size)
    fn = jax.jit(manifold.value_and_grad_autodiff)
    warm(fn, free)
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(free)), rounds=10, warmup_rounds=1
    )


# ---------------------------------------------------------------------------
# The Hessian
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", HESS_SIZES, ids=HESS_IDS)
def test_hessian_propagator_exec(benchmark, size):
    manifold, free = _manifold(size)
    fn = manifold.hessian
    warm(fn, free)
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(free)), rounds=10, warmup_rounds=1
    )


@pytest.mark.parametrize("size", HESS_SIZES, ids=HESS_IDS)
def test_hessian_autodiff_exec(benchmark, size):
    manifold, free = _manifold(size)
    fn = jax.jit(manifold.hessian_autodiff)
    warm(fn, free)
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(free)), rounds=10, warmup_rounds=1
    )


# ---------------------------------------------------------------------------
# Guards — the benchmarks must be comparing equivalent work
# ---------------------------------------------------------------------------


def test_grad_matches_autodiff():
    manifold, free = _manifold((2, 5))
    value, grad = manifold.value_and_grad(free)
    ref_value, ref_grad = manifold.value_and_grad_autodiff(free)
    assert jnp.allclose(value, ref_value, atol=1e-12)
    assert jnp.allclose(grad, ref_grad, atol=1e-9)


def test_hessian_matches_autodiff():
    manifold, free = _manifold((2, 3))
    assert jnp.allclose(
        manifold.hessian(free), manifold.hessian_autodiff(free), atol=1e-8
    )
