r"""Objective-**Hessian** benchmarks: the pair pullback vs. the dense tensor vs. autodiff.

`Manifold.hessian` is what `geope.Grape`'s Newton rules call. Three routes
compute the same $(P, P)$ matrix, and the $G$ sweep at the bottom is the point of
the comparison:

- `Manifold.hessian`, the live one, which pulls one ambient covector back through
  every pair (`geope.jax.hessian_vjp_propagator`) — two $O(G)$-propagated
  derivative trajectories and one Gram matrix, $O(GKd^3 + G^2K^2md)$ flops and
  $O(GKmd + (GK)^2)$ memory;
- ``_dense_hessian``, what it used to do: build the dense
  $(G, G, d, d, K, K)$ $\mathrm D^2\Phi$ and *then* contract —
  $O(G^2K^2d^3)$ flops and $O(G^2K^2d^2)$ memory, which is what limited the
  Newton step's reachable sizes;
- `Manifold.hessian_autodiff`, $P$ forward-over-reverse HVPs.

The gradient, assembled from the same jet, is in
``benchmarks/test_bench_objective_grad.py``. The *propagator* Hessian
$\mathrm D^2\Phi$ on its own is in ``benchmarks/test_bench_hessian.py``.

``*_exec`` benchmarks are warmed up and timed with ``block_until_ready``.

Run with, e.g.::

    pytest benchmarks/test_bench_objective_hessian.py \
        --benchmark-group-by=param:size --benchmark-columns=mean,median,rounds
"""

import jax
import jax.numpy as jnp
import pytest

from geope.geometry.chart import get_chart_hessian_fn

from conftest import make_su_manifold, warm

# Three system sizes all three Hessian routes can still reach. What bounds the
# range is the *dense* route's O(G^2 d^2 K^2) memory, not the live one's.
HESS_SIZES = [(1, 10), (2, 5), (3, 5)]
HESS_IDS = [f"n{n}-G{g}" for n, g in HESS_SIZES]

HESS_SCALING_SIZES = [(2, 5), (2, 10), (2, 20), (2, 40), (2, 100), (3, 40)]
HESS_SCALING_IDS = [f"n{n}-G{g}" for n, g in HESS_SCALING_SIZES]


def _dense_hessian(manifold):
    """`Manifold.hessian` as it was before the pullback: materialise, then contract.

    The route this module measures the acceleration against. Kept here rather
    than in the library because nothing calls it any more — the dense
    $\\mathrm D^2\\Phi$ survives only as `geope.geometry.chart.get_chart_hessian_fn`,
    the reference the tests check the pullback against.
    """
    hessian_fn = get_chart_hessian_fn(
        manifold.tangent.generators.basis, manifold.base_point
    )
    target = manifold.target
    grad_axes = tuple(range(2, 2 + manifold.ambient_ndim))

    @jax.jit
    def dense(phi):
        p = phi.size
        point = manifold.compute_point(phi)
        columns = jnp.moveaxis(manifold.tangent.jacobian(phi), (-2, -1), (0, 1))
        columns = columns.reshape((p, *manifold.ambient_shape))
        grad = manifold.cost_gradient(point, target)
        chart = 2.0 * jnp.real(
            jnp.sum(
                jnp.conj(jnp.expand_dims(grad, (0, 1, -2, -1))) * hessian_fn(phi),
                axis=grad_axes,
            )
        )
        chart = jnp.transpose(chart, (0, 2, 1, 3)).reshape(p, p)
        return manifold.cost_hessian_form(point, target, columns) + chart

    return dense


@pytest.mark.parametrize("size", HESS_SIZES, ids=HESS_IDS)
def test_hessian_propagator_exec(benchmark, size):
    manifold, free = make_su_manifold(size)
    fn = manifold.hessian
    warm(fn, free)
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(free)), rounds=10, warmup_rounds=1
    )


@pytest.mark.parametrize("size", HESS_SIZES, ids=HESS_IDS)
def test_hessian_dense_exec(benchmark, size):
    manifold, free = make_su_manifold(size)
    fn = _dense_hessian(manifold)
    warm(fn, free)
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(free)), rounds=10, warmup_rounds=1
    )


@pytest.mark.parametrize("size", HESS_SIZES, ids=HESS_IDS)
def test_hessian_autodiff_exec(benchmark, size):
    manifold, free = make_su_manifold(size)
    fn = jax.jit(manifold.hessian_autodiff)
    warm(fn, free)
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(free)), rounds=10, warmup_rounds=1
    )


# --- the G sweep: linear vs quadratic propagation ---------------------------


@pytest.mark.parametrize("size", HESS_SCALING_SIZES, ids=HESS_SCALING_IDS)
def test_hessian_propagator_scaling_exec(benchmark, size):
    manifold, free = make_su_manifold(size)
    fn = manifold.hessian
    warm(fn, free)
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(free)), rounds=5, warmup_rounds=1
    )


@pytest.mark.parametrize("size", HESS_SCALING_SIZES, ids=HESS_SCALING_IDS)
def test_hessian_dense_scaling_exec(benchmark, size):
    manifold, free = make_su_manifold(size)
    fn = _dense_hessian(manifold)
    warm(fn, free)
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(free)), rounds=5, warmup_rounds=1
    )


# ---------------------------------------------------------------------------
# Guards — the benchmarks must be comparing equivalent work
# ---------------------------------------------------------------------------


def test_hessian_matches_autodiff():
    manifold, free = make_su_manifold((2, 3))
    assert jnp.allclose(
        manifold.hessian(free), manifold.hessian_autodiff(free), atol=1e-8
    )


def test_hessian_matches_the_dense_route():
    """The two timed manual routes must be computing the same matrix."""
    manifold, free = make_su_manifold((2, 3))
    assert jnp.allclose(
        manifold.hessian(free), _dense_hessian(manifold)(free), atol=1e-10
    )
