"""
Tests for geope/grape.py.

Covers the GRAPE optimiser after its alignment with the geope.py
`jax.random.key` + key-splitting RNG:
  - end-to-end optimisation improves fidelity (Parameters API),
  - reproducibility from an integer seed,
  - a jax.random.key passed as the seed matches the equivalent int seed,
  - the legacy GrapeEngine constructor path (which exercises Grape's own
    _split_key / prepare_random_parameters draw).
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geope.grape import Grape, GrapeEngine
from geope.parameters import Parameters
from geope.utils import (
    construct_full_pauli_basis,
    construct_Heisenberg_pauli_basis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cnot():
    return jnp.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]],
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
# Tests
# ---------------------------------------------------------------------------

class TestGrapeOptimize:
    def test_optimize_improves_fidelity(self, cnot, full_basis_2q, projected_basis_2q):
        p = _params(cnot, full_basis_2q, projected_basis_2q)
        g = Grape(p, method="nr-trm", delta=1e-3, max_steps=100)
        f0 = g.fidelities[0]
        g.optimize()
        assert g.fidelities[-1] > f0
        assert g.fidelities[-1] > 0.99

    def test_returns_parameters_object(self, cnot, full_basis_2q, projected_basis_2q):
        p = _params(cnot, full_basis_2q, projected_basis_2q)
        g = Grape(p, method="nr-trm", delta=1e-3, max_steps=10)
        out = g.optimize()
        assert out is p


class TestGrapeReproducibility:
    def test_same_seed_same_init(self, cnot, full_basis_2q, projected_basis_2q):
        g1 = Grape(_params(cnot, full_basis_2q, projected_basis_2q, seed=7),
                   method="nr-trm", delta=1e-3, max_steps=0)
        g2 = Grape(_params(cnot, full_basis_2q, projected_basis_2q, seed=7),
                   method="nr-trm", delta=1e-3, max_steps=0)
        assert np.allclose(g1.init_parameters, g2.init_parameters)

    def test_different_seed_differs(self, cnot, full_basis_2q, projected_basis_2q):
        g1 = Grape(_params(cnot, full_basis_2q, projected_basis_2q, seed=7),
                   method="nr-trm", delta=1e-3, max_steps=0)
        g2 = Grape(_params(cnot, full_basis_2q, projected_basis_2q, seed=8),
                   method="nr-trm", delta=1e-3, max_steps=0)
        assert not np.allclose(g1.init_parameters, g2.init_parameters)

    def test_trajectory_reproducible(self, cnot, full_basis_2q, projected_basis_2q):
        g1 = Grape(_params(cnot, full_basis_2q, projected_basis_2q, seed=7),
                   method="nr-trm", delta=1e-3, max_steps=30)
        g1.optimize()
        g2 = Grape(_params(cnot, full_basis_2q, projected_basis_2q, seed=7),
                   method="nr-trm", delta=1e-3, max_steps=30)
        g2.optimize()
        assert np.allclose(np.array(g1.fidelities), np.array(g2.fidelities))

    def test_jax_key_seed_matches_int_seed(self, cnot, full_basis_2q, projected_basis_2q):
        g_int = Grape(_params(cnot, full_basis_2q, projected_basis_2q, seed=7),
                      method="nr-trm", delta=1e-3, max_steps=0)
        g_key = Grape(_params(cnot, full_basis_2q, projected_basis_2q, seed=jax.random.key(7)),
                      method="nr-trm", delta=1e-3, max_steps=0)
        assert np.allclose(g_int.init_parameters, g_key.init_parameters)


class TestGrapeLegacyEngineAPI:
    """The legacy `Grape(GrapeEngine, seed=...)` path draws its own initial
    parameters via Grape._split_key / prepare_random_parameters(key=...)."""

    def _engine(self, cnot, full_basis_2q, projected_basis_2q):
        return GrapeEngine(
            target_unitary=cnot,
            full_basis=full_basis_2q,
            projected_basis=projected_basis_2q,
            piecewise_steps=1,
        )

    def test_legacy_init_reproducible(self, cnot, full_basis_2q, projected_basis_2q):
        g1 = Grape(self._engine(cnot, full_basis_2q, projected_basis_2q),
                   seed=11, method="nr-trm", delta=1e-3, max_steps=0)
        g2 = Grape(self._engine(cnot, full_basis_2q, projected_basis_2q),
                   seed=11, method="nr-trm", delta=1e-3, max_steps=0)
        assert np.allclose(g1.init_parameters, g2.init_parameters)

    def test_legacy_optimize_runs(self, cnot, full_basis_2q, projected_basis_2q):
        g = Grape(self._engine(cnot, full_basis_2q, projected_basis_2q),
                  seed=11, method="nr-trm", delta=1e-3, max_steps=20)
        f0 = g.fidelities[0]
        result = g.optimize()
        assert isinstance(result, bool)
        assert g.fidelities[-1] >= f0


class TestGrapeParamTransform:
    """The param_transform path now wraps compute_U_fn via
    GrapeEngine.wrap_param_transform (mirroring GeopeEngine)."""

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

    def test_wrap_param_transform_overrides_engine(self, cnot, full_basis_2q, projected_basis_2q):
        p = self._exp_params(cnot, full_basis_2q, projected_basis_2q)
        g = Grape(p, method="nr-trm", delta=1e-3, max_steps=0)
        n_exp = projected_basis_2q.lie_algebra_dim
        assert g._real_params is True
        assert g.engine.drift_basis is None
        # experimental space: indices are an all-true mask of length n_exp
        assert g.engine.proj_drift_indices.sum() == n_exp
        assert g.init_parameters.shape == (1, n_exp)

    def test_optimize_improves_fidelity(self, cnot, full_basis_2q, projected_basis_2q):
        p = self._exp_params(cnot, full_basis_2q, projected_basis_2q)
        g = Grape(p, method="nr-trm", delta=1e-3, max_steps=100)
        f0 = g.fidelities[0]
        g.optimize()
        assert g.fidelities[-1] > f0
        assert g.fidelities[-1] > 0.99

    def test_reproducible(self, cnot, full_basis_2q, projected_basis_2q):
        g1 = Grape(self._exp_params(cnot, full_basis_2q, projected_basis_2q, seed=5),
                   method="nr-trm", delta=1e-3, max_steps=0)
        g2 = Grape(self._exp_params(cnot, full_basis_2q, projected_basis_2q, seed=5),
                   method="nr-trm", delta=1e-3, max_steps=0)
        assert np.allclose(g1.init_parameters, g2.init_parameters)
