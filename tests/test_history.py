"""
Tests for geope/utils/history.py.

Covers the opt-in ``History`` run log used by the optimisers:
  - default logged columns and item/attribute access,
  - ``to_dataframe`` / ``to_dict`` / ``reset`` / ``len``,
  - ``best_fidelity`` and ``best_basis_coefficients`` helpers,
  - a custom ``logging_fn`` overriding the default columns,
  - the back-ref to the source ``Parameters``.
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geope.geope import Geope
from geope.parameters import Parameters
from geope.utils.history import History
from geope.utils import (
    construct_full_pauli_basis,
    construct_Heisenberg_pauli_basis,
)


def _params_2q(
    cnot,
    full_basis_2q,
    projected_basis_2q,
    *,
    drift_basis=None,
    drift_values=None,
    init_values=None,
    constraints=None,
    piecewise_steps=1,
    seed=42,
    init_spread=0.1,
    pulse_constraints=None,
    projective=True,
    param_transform=None,
    n_experimental_params=None,
):
    """Build a Parameters bundle from the raw test fixtures.

    Helper for tests that need to construct a ``Geope`` from the
    Heisenberg / full Pauli basis fixtures rather than from a
    control dict.
    """
    return Parameters(
        basis=full_basis_2q,
        projected_basis=projected_basis_2q,
        drift_basis=drift_basis,
        drift_values=drift_values,
        init_values=init_values,
        target=cnot,
        piecewise_steps=piecewise_steps,
        constraints=constraints,
        pulse_constraints=pulse_constraints,
        init_spread=init_spread,
        seed=seed,
        projective=projective,
        param_transform=param_transform,
        n_experimental_params=n_experimental_params,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cnot():
    return jnp.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )


@pytest.fixture
def full_basis_2q():
    """Full 2-qubit Pauli basis (15 elements)."""
    return construct_full_pauli_basis(2)


@pytest.fixture
def projected_basis_2q():
    """Heisenberg 2-qubit basis (9 elements ⊂ 15) — a proper subset of the full basis."""
    return construct_Heisenberg_pauli_basis(2)


@pytest.fixture
def params_2q(cnot, full_basis_2q, projected_basis_2q):
    return _params_2q(cnot, full_basis_2q, projected_basis_2q)


# ---------------------------------------------------------------------------
# Tests — History (opt-in run log)
# ---------------------------------------------------------------------------


class TestHistory:
    def test_no_history_is_none(self, params_2q):
        g = Geope(params_2q)
        assert g.history is None
        # the final result is still available on Parameters
        g.optimize(max_steps=3)
        assert g.params.fidelity is not None

    def test_default_columns(self, params_2q):
        g = Geope(params_2q, history=History())
        g.optimize(max_steps=3)
        assert set(g.history.keys()) == {
            "parameters",
            "fidelities",
            "infidelities",
            "step_sizes",
            "steps",
        }

    def test_attribute_is_item(self, params_2q):
        g = Geope(params_2q, history=History())
        g.optimize(max_steps=3)
        assert g.history.fidelities is g.history["fidelities"]

    def test_unknown_column_raises(self, params_2q):
        g = Geope(params_2q, history=History())
        with pytest.raises(AttributeError):
            _ = g.history.not_a_column

    def test_to_dataframe_length(self, params_2q):
        g = Geope(params_2q, history=History())
        g.optimize(max_steps=4)
        assert len(g.history.to_dataframe()) == len(g.history)

    def test_reset_empties(self, params_2q):
        g = Geope(params_2q, history=History())
        g.optimize(max_steps=3)
        g.history.reset()
        assert len(g.history) == 0
        assert g.history.best_fidelity is None

    def test_backref_to_params(self, params_2q):
        g = Geope(params_2q, history=History())
        assert g.history.params is params_2q

    def test_best_fidelity_is_max(self, params_2q):
        g = Geope(params_2q, history=History())
        g.optimize(max_steps=5)
        assert g.history.best_fidelity == max(g.history.fidelities)

    def test_custom_logging_fn(self, params_2q):
        g = Geope(
            params_2q,
            history=History(logging_fn=lambda gg: {"fid": float(gg.params.fidelity)}),
        )
        g.optimize(max_steps=5, precision=0.0)
        # only the custom column is logged
        assert list(g.history.keys()) == ["fid"]
        # the loop still converges (reads params.fidelity, not a column)
        assert g.params.fidelity is not None
        # best-helpers degrade gracefully when the default columns are absent
        assert g.history.best_fidelity is None
        assert g.history.to_dict() == {}

    def test_params_to_dict_reflects_current(self, params_2q):
        g = Geope(params_2q)
        g.optimize(max_steps=5)
        # to_dict over the current params is a non-empty control dict
        assert params_2q.to_dict() != {}

    def test_best_basis_coefficients_requires_backref(self, full_basis_2q):
        # A bare History with logged columns but no back-ref must raise.
        n = full_basis_2q.lie_algebra_dim
        h = History()
        h.logs = {"fidelities": [0.5], "parameters": [np.zeros((1, n))]}
        with pytest.raises(ValueError):
            _ = h.best_basis_coefficients
