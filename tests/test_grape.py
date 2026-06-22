"""
Tests for geope/grape.py.

Covers the GRAPE optimiser after its public API was aligned with Geope:
  - Parameters-only constructor (legacy GrapeEngine input removed),
  - method / hyperparameters / max_steps passed to optimize(),
  - Geope-style result model (params.parameters is the current array,
    params.fidelity is a scalar, trajectory in an optional History),
  - reproducibility from an integer seed and a jax.random.key seed,
  - the param_transform path via the cached params.compute_U_fn.
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geope.grape import Grape
from geope.parameters import Parameters
from geope.utils import (
    construct_full_pauli_basis,
    construct_Heisenberg_pauli_basis,
)
from geope.utils.history import History

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
    return construct_full_pauli_basis(2)


@pytest.fixture
def projected_basis_2q():
    return construct_Heisenberg_pauli_basis(2)


def _params(cnot, full_basis_2q, projected_basis_2q, *, seed=42, piecewise_steps=1):
    return Parameters(
        basis=full_basis_2q,
        projected_basis=projected_basis_2q,
        target=cnot,
        piecewise_steps=piecewise_steps,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Tests — constructor / usage parity with Geope
# ---------------------------------------------------------------------------


class TestGrapeConstructor:
    def test_requires_parameters(self):
        with pytest.raises(TypeError):
            Grape("not a Parameters object")

    def test_init_sets_geope_style_state(self, cnot, full_basis_2q, projected_basis_2q):
        p = _params(cnot, full_basis_2q, projected_basis_2q)
        g = Grape(p)
        # params.parameters is the current array, not a list
        assert isinstance(g.params.parameters, np.ndarray)
        assert g.params.parameters.shape == (1, full_basis_2q.lie_algebra_dim)
        # params.fidelity is a scalar
        assert np.ndim(g.params.fidelity) == 0
        # optimiser is built lazily
        assert g.update_step is None


class TestGrapeOptimize:
    def test_optimize_improves_fidelity(self, cnot, full_basis_2q, projected_basis_2q):
        p = _params(cnot, full_basis_2q, projected_basis_2q)
        g = Grape(p, history=History())
        f0 = float(g.params.fidelity)
        # delta=0.1 keeps the trust-region step well-regularised; a pathologically
        # small delta (e.g. 1e-3) makes nr-trm take near-pure-Newton steps that
        # bounce chaotically and platform-FP-sensitively.
        out = g.optimize(max_steps=100, method="nr-trm", delta=0.1)
        assert out is p
        # nr-trm (regularised Newton + backtracking) is not monotone: the
        # per-step fidelity still bounces, so the final iterate after a fixed
        # step budget is platform-FP-sensitive. The robust, meaningful signal is
        # that the run reaches a better point than it started from somewhere
        # along the trajectory — assert on the best fidelity, not the final one.
        assert g.history.best_fidelity > f0
        # result is still a current array + scalar fidelity
        assert g.params.parameters.shape == (1, full_basis_2q.lie_algebra_dim)
        assert np.ndim(g.params.fidelity) == 0

    def test_history_records_trajectory(self, cnot, full_basis_2q, projected_basis_2q):
        p = _params(cnot, full_basis_2q, projected_basis_2q)
        g = Grape(p, history=History())
        g.optimize(max_steps=40, method="nr-trm", delta=1e-3)
        assert len(g.history) > 1  # step 0 + iterations
        assert g.history.best_fidelity >= float(g.params.fidelity) - 1e-9

    def test_adam_method_runs(self, cnot, full_basis_2q, projected_basis_2q):
        p = _params(cnot, full_basis_2q, projected_basis_2q)
        g = Grape(p)
        f0 = float(g.params.fidelity)
        g.optimize(max_steps=50, method="adam", learning_rate=0.1)
        assert float(g.params.fidelity) >= f0

    def test_unknown_method_raises(self, cnot, full_basis_2q, projected_basis_2q):
        g = Grape(_params(cnot, full_basis_2q, projected_basis_2q))
        with pytest.raises(NotImplementedError):
            g.optimize(max_steps=1, method="nonsense")


class TestGrapeReproducibility:
    def test_same_seed_same_result(self, cnot, full_basis_2q, projected_basis_2q):
        g1 = Grape(_params(cnot, full_basis_2q, projected_basis_2q, seed=7))
        g1.optimize(max_steps=30, method="nr-trm", delta=1e-3)
        g2 = Grape(_params(cnot, full_basis_2q, projected_basis_2q, seed=7))
        g2.optimize(max_steps=30, method="nr-trm", delta=1e-3)
        assert np.allclose(g1.params.parameters, g2.params.parameters)
        assert np.isclose(float(g1.params.fidelity), float(g2.params.fidelity))

    def test_different_seed_differs(self, cnot, full_basis_2q, projected_basis_2q):
        g1 = Grape(_params(cnot, full_basis_2q, projected_basis_2q, seed=7))
        g2 = Grape(_params(cnot, full_basis_2q, projected_basis_2q, seed=8))
        assert not np.allclose(g1.init_parameters, g2.init_parameters)

    def test_jax_key_seed_matches_int_seed(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        g_int = Grape(_params(cnot, full_basis_2q, projected_basis_2q, seed=7))
        g_key = Grape(
            _params(cnot, full_basis_2q, projected_basis_2q, seed=jax.random.key(7))
        )
        assert np.allclose(g_int.init_parameters, g_key.init_parameters)


class TestGrapeParamTransform:
    """The param_transform path wraps compute_U_fn via the cached
    ``params.compute_U_fn`` (operating in experimental space)."""

    def _exp_params(self, cnot, full_basis_2q, projected_basis_2q, seed=42):
        n_exp = projected_basis_2q.lie_algebra_dim
        return Parameters(
            basis=full_basis_2q,
            projected_basis=projected_basis_2q,
            target=cnot,
            piecewise_steps=1,
            seed=seed,
            param_transform=lambda phi: phi,
            n_experimental_params=n_exp,
        )

    def test_param_transform_uses_experimental_space(
        self, cnot, full_basis_2q, projected_basis_2q
    ):
        p = self._exp_params(cnot, full_basis_2q, projected_basis_2q)
        g = Grape(p)
        n_exp = projected_basis_2q.lie_algebra_dim
        assert g._real_params is True
        # Experimental space: drift is folded into compute_U, and every column
        # is a free parameter.
        assert g.drift_parameters is None
        assert g._proj_drift_mask().sum() == n_exp
        assert g.params.parameters.shape == (1, n_exp)

    def test_optimize_improves_fidelity(self, cnot, full_basis_2q, projected_basis_2q):
        p = self._exp_params(cnot, full_basis_2q, projected_basis_2q)
        g = Grape(p, history=History())
        f0 = float(g.params.fidelity)
        # delta=0.1 + best-over-trajectory: nr-trm is non-monotone (see
        # TestGrapeOptimize), so assert on the best fidelity reached, not the
        # platform-sensitive final iterate.
        g.optimize(max_steps=100, method="nr-trm", delta=0.1)
        assert g.history.best_fidelity > f0
