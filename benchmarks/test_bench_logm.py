"""Matrix-logarithm benchmarks: general algorithm vs. unitary specialisation.

Compares the two ways of computing the principal log of the argument the
geodesic step actually sees, ``U^dagger U_T``:

* :func:`geope.jax.logm` — the general Al-Mohy & Higham inverse
  scaling-and-squaring method (complex Schur, a ``while_loop`` of triangular
  square roots, randomised 1-norm estimates, Pade quadrature).
* :func:`geope.jax.logm_unitary` — the specialisation used by the optimiser:
  the argument is unitary, hence normal, so its complex Schur form is
  diagonal and the log is the scalar log of that diagonal.

Both the per-call cost and the trace/compile cost are timed. The latter is
kept separate because it is the larger absolute difference and is paid again
whenever ``Geope`` re-traces (e.g. on a line-search config change).

Run with, e.g.::

    pytest benchmarks/test_bench_logm.py \
        --benchmark-group-by=param:n --benchmark-columns=mean,median,rounds
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geope.jax import logm, logm_unitary
from geope.utils import qft_unitary

from conftest import warm

N_QUBITS = [
    1,
    2,
    3,
]  # d = 2**n, K = 4**n - 1
COMPILE_ROUNDS = 3
KEY = jax.random.key(0)


def _setup(n):
    """The geodesic step's argument ``U^dagger U_T`` for an ``n``-qubit problem.

    ``U`` is a random-Hermitian-generated unitary standing in for the current
    iterate and ``U_T`` is the QFT target, so the spectrum matches what the
    optimiser really encounters rather than that of a bare random unitary.
    """
    d = 2**n
    rng = np.random.default_rng(n)
    H = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    H = (H + H.conj().T) / 2
    U = jax.scipy.linalg.expm(1j * 0.3 * jnp.asarray(H))
    target = jnp.asarray(qft_unitary(n))
    return U.conj().T @ target


def _setup_branch_cut(n):
    """An argument with eigenvalues driven up against the ``theta -> pi`` cut.

    Timed only, no accuracy assertion: the accuracy story here is covered by
    ``tests/test_logm.py``, and the point of the benchmark is that neither
    implementation changes cost near the cut.
    """
    d = 2**n
    rng = np.random.default_rng(n + 50)
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    V, _ = np.linalg.qr(A)
    theta = np.linspace(-np.pi + 1e-9, np.pi - 1e-9, d)
    return jnp.asarray((V * np.exp(1j * theta)) @ V.conj().T)


@pytest.mark.parametrize("n", N_QUBITS)
def test_logm_exec(benchmark, n):
    U = _setup(n)
    fn = jax.jit(lambda x: logm(x, KEY))
    warm(fn, U)
    benchmark.pedantic(lambda: jax.block_until_ready(fn(U)), rounds=20, warmup_rounds=1)


@pytest.mark.parametrize("n", N_QUBITS)
def test_logm_unitary_exec(benchmark, n):
    U = _setup(n)
    fn = jax.jit(lambda x: logm_unitary(x, KEY))
    warm(fn, U)
    benchmark.pedantic(lambda: jax.block_until_ready(fn(U)), rounds=20, warmup_rounds=1)


@pytest.mark.parametrize("n", N_QUBITS)
def test_logm_branch_cut_exec(benchmark, n):
    U = _setup_branch_cut(n)
    fn = jax.jit(lambda x: logm(x, KEY))
    warm(fn, U)
    benchmark.pedantic(lambda: jax.block_until_ready(fn(U)), rounds=20, warmup_rounds=1)


@pytest.mark.parametrize("n", N_QUBITS)
def test_logm_unitary_branch_cut_exec(benchmark, n):
    U = _setup_branch_cut(n)
    fn = jax.jit(lambda x: logm_unitary(x, KEY))
    warm(fn, U)
    benchmark.pedantic(lambda: jax.block_until_ready(fn(U)), rounds=20, warmup_rounds=1)


@pytest.mark.parametrize("n", N_QUBITS)
def test_logm_compile(benchmark, n):
    U = _setup(n)

    def compile_once():
        return jax.jit(lambda x: logm(x, KEY)).lower(U).compile()

    benchmark.pedantic(compile_once, rounds=COMPILE_ROUNDS, warmup_rounds=0)


@pytest.mark.parametrize("n", N_QUBITS)
def test_logm_unitary_compile(benchmark, n):
    U = _setup(n)

    def compile_once():
        return jax.jit(lambda x: logm_unitary(x, KEY)).lower(U).compile()

    benchmark.pedantic(compile_once, rounds=COMPILE_ROUNDS, warmup_rounds=0)


def test_logm_unitary_matches_logm():
    """Guard: the two paths must agree, and the specialised one stays in su(d)."""
    U = _setup(2)
    L = logm_unitary(U, KEY)
    assert jnp.allclose(L, logm(U, KEY), atol=1e-8)
    # -i log(U) is Hermitian for unitary U, so log(U) is anti-Hermitian.
    assert float(jnp.max(jnp.abs(L + L.conj().T))) < 1e-14
