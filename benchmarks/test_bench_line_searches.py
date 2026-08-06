"""Line-search comparison: fevals / steps / walltime across a difficulty ladder.

Runs the full GEOPE geodesic optimiser end-to-end with each pluggable line
search (:class:`geope.GoldenSection`, :class:`geope.Adam`,
:class:`geope.QuadraticArmijo`) on a ladder of gate-synthesis problems from easy
(single-qubit rotation) to hard (3-qubit Toffoli), and reports for each:

* **walltime** — total ``optimize()`` wall-clock (warmed; the timed call ends in
  ``block_until_ready`` so device work, not async dispatch, is measured). This is
  the column ``pytest-benchmark`` prints.
* **steps** — optimiser iterations to reach ``precision`` (or the cap).
* **fevals** — line-search 1-D-objective evaluations summed over the run (the
  uniform ``n_eval`` every line search now threads in its state). This counts
  *only* the line search, not the geodesic Jacobian or the Gram-Schmidt fallback.

``steps``/``fevals``/``final fidelity`` are deterministic (fixed seed), so they
are read once after timing, stashed in ``benchmark.extra_info`` (→
``--benchmark-json``) and appended to the ``conftest.LINESEARCH_BENCH_ROWS``
registry, which the ``pytest_terminal_summary`` hook in ``conftest.py`` prints as
a side table grouped by problem.

Run with, e.g.::

    pytest benchmarks/test_bench_line_searches.py \
        --benchmark-group-by=param:problem \
        --benchmark-columns=mean,median,rounds

which places the three line searches side by side for each problem.
"""

import numpy as np
import jax
import pytest

from geope import (
    Adam,
    Armijo,
    Geope,
    GoldenSection,
    History,
    Parameters,
    QuadraticArmijo,
)
from geope.utils import (
    construct_full_pauli_basis,
    construct_Heisenberg_pauli_basis,
    multicontrol_unitary,
    qft_unitary,
)

from conftest import LINESEARCH_BENCH_ROWS

SEED = 0
PRECISION = 0.9999999

_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)


def _rx(theta):
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


# --- Problem ladder (easy -> hard), all standard projective mode -------------
# QuadraticArmijo needs the SU(N) geometry, so no problem uses param_transform.
# Each builder returns a *fresh* Parameters so runs never share live state.


def _rot_1q():
    return Parameters(
        basis=construct_full_pauli_basis(1),
        control={1: ["x", "y", "z"]},
        target=_rx(np.pi / 3),
        piecewise_steps=3,
        seed=SEED,
        projective=True,
    )


def _cnot_2q():
    return Parameters(
        basis=construct_full_pauli_basis(2),
        projected_basis=construct_Heisenberg_pauli_basis(2),
        target=_CNOT,
        piecewise_steps=6,
        seed=SEED,
        projective=True,
    )


def _qft_2q():
    return Parameters(
        basis=construct_full_pauli_basis(2),
        control={1: ["x", "z"], 2: ["x", "z"], (1, 2): ["zz"]},
        target=qft_unitary(2),
        piecewise_steps=10,
        seed=SEED,
        projective=True,
    )


_DRIFT_3Q = {(1, 2): ["zz"], (2, 3): ["zz"], (1, 3): ["zz"]}
_DRIFT_VALUES_3Q = {
    (1, 2): {"zz": 1.0},
    (2, 3): {"zz": 1.0},
    (1, 3): {"zz": 1.0},
}
_CONTROL_3Q = {1: ["x", "z"], 2: ["x", "z"], 3: ["x", "z"]}


def _qft_3q():
    return Parameters(
        basis=construct_full_pauli_basis(3),
        control=_CONTROL_3Q,
        drift=_DRIFT_3Q,
        drift_values=_DRIFT_VALUES_3Q,
        target=qft_unitary(3),
        piecewise_steps=20,
        seed=SEED,
        projective=True,
    )


def _toffoli_3q():
    return Parameters(
        basis=construct_full_pauli_basis(3),
        control=_CONTROL_3Q,
        drift=_DRIFT_3Q,
        drift_values=_DRIFT_VALUES_3Q,
        target=multicontrol_unitary(_X, 2),
        piecewise_steps=20,
        seed=SEED,
        projective=True,
    )


# (builder, max_steps, rounds) per problem id, easy -> hard.
PROBLEMS = {
    "rot-1q": (_rot_1q, 200, 5),
    "cnot-2q": (_cnot_2q, 200, 5),
    "qft-2q": (_qft_2q, 300, 5),
    "qft-3q": (_qft_3q, 500, 3),
    "toffoli-3q": (_toffoli_3q, 500, 3),
}
PROBLEM_IDS = list(PROBLEMS)

METHODS = {
    "golden": GoldenSection(),
    "adam": Adam(),
    "armijo": Armijo(),
    "quad_armijo": QuadraticArmijo(),
}
METHOD_IDS = list(METHODS)


def _feval_logger(g):
    """History column set that also captures the line search's per-step n_eval.

    ``line_search_state`` is ``None`` at the step-0 row recorded during
    ``init()`` (before ``optimize`` sets it), so guard for it.
    """
    st = g.line_search_state or {}
    return {
        "steps": len(g.history),
        "fidelities": g.params.fidelity,
        "n_eval": int(st.get("n_eval", 0)),
    }


@pytest.mark.parametrize("method_name", METHOD_IDS, ids=METHOD_IDS)
@pytest.mark.parametrize("problem", PROBLEM_IDS, ids=PROBLEM_IDS)
def test_line_search_optimize(benchmark, problem, method_name):
    builder, max_steps, rounds = PROBLEMS[problem]
    line_search = METHODS[method_name]
    g = Geope(builder(), history=History(logging_fn=_feval_logger))

    # Warm up: the first optimize() triggers the one-off XLA compile of
    # update_step for this (line_search, max_step_size) config.
    g.init(seed=SEED)
    g.optimize(max_steps=max_steps, line_search=line_search, precision=PRECISION)

    def setup():
        # Fresh, identical initial parameters/state each round (init() keeps the
        # compile memo; optimize() re-init()s the line-search state itself).
        g.init(seed=SEED)
        return (), {}

    def run():
        g.optimize(max_steps=max_steps, line_search=line_search, precision=PRECISION)
        jax.block_until_ready(g.params.parameters)

    benchmark.pedantic(run, setup=setup, rounds=rounds, warmup_rounds=0)

    # Deterministic run -> read the constant metrics from the last timed round.
    n_steps = len(g.history) - 1  # step-0 row + one per optimise iteration
    fevals = int(sum(int(x) for x in g.history["n_eval"]))
    fidelity = float(g.params.fidelity)

    benchmark.extra_info.update(steps=n_steps, fevals=fevals, final_fidelity=fidelity)
    LINESEARCH_BENCH_ROWS.append(
        {
            "problem": problem,
            "method": method_name,
            "steps": n_steps,
            "fevals": fevals,
            "fidelity": fidelity,
        }
    )


def test_all_methods_make_progress():
    """Guard: every line search drives CNOT to high fidelity, so the timings
    compare methods that actually work (not one that stalls)."""
    for method_name, line_search in METHODS.items():
        g = Geope(_cnot_2q(), history=History())
        g.optimize(max_steps=100, line_search=line_search, precision=PRECISION)
        assert (
            g.history.best_fidelity > 0.99
        ), f"{method_name} stalled at {float(g.history.best_fidelity)}"
