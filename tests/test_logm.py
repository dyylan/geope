"""Tests for geope.jax.logm (matrix logarithm via inverse scaling-and-squaring).

Regression coverage for the near-identity branch: the inverse
scaling-and-squaring must reproduce ``scipy.linalg.logm`` even when the input is
close to the identity (small eigenvalue angles), where an earlier bug returned a
power-of-2 multiple of the true logarithm — see the diagonal-replacement fix in
``_logm_triu``.
"""

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import scipy.linalg as sla

from geope.jax import logm

KEY = jax.random.key(0)

_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)


def _maxdiff(M):
    L = np.array(logm(jnp.asarray(M), KEY))
    return float(np.max(np.abs(L - sla.logm(M))))


@pytest.mark.parametrize("theta", [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0, 2.0, 3.0])
def test_diagonal_rotation_matches_scipy(theta):
    # exp(i*theta*Z) is diagonal with eigenvalues e^{+-i*theta}; the log must be
    # exactly +-i*theta for every angle, including the small (near-I) ones that
    # previously came back scaled by 2^-m.
    assert _maxdiff(sla.expm(1j * theta * _Z)) < 1e-12


@pytest.mark.parametrize("scale", [0.02, 0.1, 0.3, 0.7, 1.5])
def test_generic_2x2_matches_scipy(scale):
    # Non-normal-diagonal su(2) direction (nonzero superdiagonal after Schur).
    M = sla.expm(1j * scale * (0.3 * _X + 0.2 * _Z))
    assert _maxdiff(M) < 1e-12


@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("scale", [0.03, 0.3, 1.0])
def test_random_unitary_matches_scipy(n, scale):
    # Random Hermitian generator -> unitary near I (small scale) and generic.
    rng = np.random.default_rng(n)
    H = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    H = (H + H.conj().T) / 2
    assert _maxdiff(sla.expm(1j * scale * H)) < 1e-10


def test_expm_logm_roundtrip_near_identity():
    # exp(logm(M)) == M for a near-identity unitary (the failing regime).
    rng = np.random.default_rng(1)
    H = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    H = (H + H.conj().T) / 2
    M = sla.expm(1j * 0.05 * H)
    L = np.array(logm(jnp.asarray(M), KEY))
    assert np.max(np.abs(sla.expm(L) - M)) < 1e-10
