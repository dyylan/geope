import sys
from pathlib import Path

# Ensure the src/ layout is importable without pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import jax

# float64/complex128 throughout, matching every test/source module.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from geope.utils import construct_full_pauli_basis


def make_basis(n: int) -> jnp.ndarray:
    """Full ``n``-qubit Pauli basis as a ``(K, d, d)`` array.

    ``K = 4**n - 1`` (identity excluded) and ``d = 2**n``.
    """
    return jnp.asarray(construct_full_pauli_basis(n).basis)


def make_params(n_steps: int, K: int, key: jax.Array) -> jnp.ndarray:
    """Random ``(n_steps, K)`` parameter array in complex128.

    Inputs are complex because the autodiff Jacobians use
    ``holomorphic=True`` and the manual path also expects complex
    coefficients.
    """
    return jax.random.normal(key, (n_steps, K)).astype(jnp.complex128)


def warm(fn, *args):
    """Force compilation and return a device-ready result.

    Used both to warm up a function before timing and to materialise a
    result for the correctness guards.
    """
    return jax.block_until_ready(fn(*args))


# ---------------------------------------------------------------------------
# Non-timing metric registry for the line-search benchmark.
#
# ``pytest-benchmark``'s terminal table shows only walltime (per-round timing);
# steps / fevals / final fidelity are constant across rounds (deterministic runs)
# and so belong in a side table. ``test_bench_line_searches.py`` appends one row
# per (problem, method) here, and the ``pytest_terminal_summary`` hook below
# prints them grouped by problem after the run. A ``pytest_terminal_summary``
# defined in a test module is *not* collected — it must live in a conftest.
# ---------------------------------------------------------------------------
LINESEARCH_BENCH_ROWS = []


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print the steps / fevals / fidelity comparison, grouped by problem."""
    if not LINESEARCH_BENCH_ROWS:
        return

    tr = terminalreporter
    tr.write_sep("=", "line-search comparison (steps / fevals / final fidelity)")

    header = (
        f"{'problem':<12} {'method':<12} {'steps':>7} {'fevals':>8} {'fidelity':>12}"
    )
    tr.write_line(header)
    tr.write_line("-" * len(header))

    # Preserve first-seen problem order, methods within a problem.
    seen = []
    for row in LINESEARCH_BENCH_ROWS:
        if row["problem"] not in seen:
            seen.append(row["problem"])
    for problem in seen:
        for row in LINESEARCH_BENCH_ROWS:
            if row["problem"] != problem:
                continue
            tr.write_line(
                f"{row['problem']:<12} {row['method']:<12} "
                f"{row['steps']:>7d} {row['fevals']:>8d} {row['fidelity']:>12.9f}"
            )
