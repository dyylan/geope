"""Tests for geope.jax.logm: both matrix-logarithm implementations.

Covers the general ``logm`` (inverse scaling-and-squaring) and the
unitary-specialised ``logm_unitary`` (Schur), which is
what the geodesic step and the geometry-aware line searches actually call.
Since every argument the optimiser passes is unitary, the shared correctness
tests are parametrised over both implementations at identical tolerances.

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

from geope.jax import logm, logm_unitary

KEY = jax.random.key(0)

_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)

# Every input below is a unitary ``expm(1j * H)``, so both the general
# inverse scaling-and-squaring ``logm`` and the unitary-specialised
# ``logm_unitary`` must handle it. Parametrising the shared helpers over both
# keeps the two implementations held to identical tolerances.
IMPLS = [logm, logm_unitary]
IMPL_IDS = ["logm", "logm_unitary"]


def _maxdiff(M, impl=logm):
    L = np.array(impl(jnp.asarray(M), KEY))
    return float(np.max(np.abs(L - sla.logm(M))))


@pytest.mark.parametrize("impl", IMPLS, ids=IMPL_IDS)
@pytest.mark.parametrize("theta", [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0, 2.0, 3.0])
def test_diagonal_rotation_matches_scipy(theta, impl):
    # exp(i*theta*Z) is diagonal with eigenvalues e^{+-i*theta}; the log must be
    # exactly +-i*theta for every angle, including the small (near-I) ones that
    # previously came back scaled by 2^-m.
    assert _maxdiff(sla.expm(1j * theta * _Z), impl) < 1e-12


@pytest.mark.parametrize("impl", IMPLS, ids=IMPL_IDS)
@pytest.mark.parametrize("scale", [0.02, 0.1, 0.3, 0.7, 1.5])
def test_generic_2x2_matches_scipy(scale, impl):
    # Non-normal-diagonal su(2) direction (nonzero superdiagonal after Schur).
    M = sla.expm(1j * scale * (0.3 * _X + 0.2 * _Z))
    assert _maxdiff(M, impl) < 1e-12


@pytest.mark.parametrize("impl", IMPLS, ids=IMPL_IDS)
@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("scale", [0.03, 0.3, 1.0])
def test_random_unitary_matches_scipy(n, scale, impl):
    # Random Hermitian generator -> unitary near I (small scale) and generic.
    rng = np.random.default_rng(n)
    H = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    H = (H + H.conj().T) / 2
    assert _maxdiff(sla.expm(1j * scale * H), impl) < 1e-10


@pytest.mark.parametrize("impl", IMPLS, ids=IMPL_IDS)
def test_expm_logm_roundtrip_near_identity(impl):
    # exp(logm(M)) == M for a near-identity unitary (the failing regime).
    rng = np.random.default_rng(1)
    H = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    H = (H + H.conj().T) / 2
    M = sla.expm(1j * 0.05 * H)
    L = np.array(impl(jnp.asarray(M), KEY))
    assert np.max(np.abs(sla.expm(L) - M)) < 1e-10


# ---------------------------------------------------------------------------
# Properties specific to the unitary-specialised path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("scale", [0.03, 0.3, 1.0])
def test_logm_unitary_matches_logm(n, scale):
    # The two implementations must agree, not merely each match scipy.
    rng = np.random.default_rng(n + 100)
    H = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    H = (H + H.conj().T) / 2
    M = jnp.asarray(sla.expm(1j * scale * H))
    assert np.max(np.abs(np.array(logm_unitary(M, KEY) - logm(M, KEY)))) < 1e-10


@pytest.mark.parametrize("n", [2, 4, 8])
def test_logm_unitary_is_anti_hermitian(n):
    # log(U) = i*H with H Hermitian. Assembling from a unitary Schur basis and
    # a purely imaginary diagonal holds this to round-off, a little tighter
    # than the general Pade path manages.
    rng = np.random.default_rng(n + 200)
    H = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    H = (H + H.conj().T) / 2
    L = logm_unitary(jnp.asarray(sla.expm(1j * 0.7 * H)), KEY)
    assert float(jnp.max(jnp.abs(L + L.conj().T))) < 1e-14


_HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)


@pytest.mark.parametrize(
    "M",
    [_HADAMARD, _X, _Z, _CNOT],
    ids=["hadamard", "x", "z", "cnot"],
)
def test_logm_unitary_on_hermitian_unitaries(M):
    # Regression: Hadamard, the Paulis and CNOT are Hermitian unitaries, so an
    # eigenvalue sits at exactly -1 -- precisely on the principal branch cut.
    # These are ordinary synthesis targets, not edge cases, and an earlier
    # Cayley-transform implementation returned zeros/NaN for all of them.
    # Taking the scalar log of the (diagonal, because normal) Schur factor is
    # exact here, and beats the inverse scaling-and-squaring path by ~9 orders
    # of magnitude -- hence the much tighter tolerance than ``logm`` can meet.
    L = np.array(logm_unitary(jnp.asarray(M), KEY))
    assert np.max(np.abs(L - sla.logm(M))) < 1e-13
    assert np.max(np.abs(sla.expm(L) - M)) < 1e-13


def test_logm_unitary_beats_logm_on_the_branch_cut():
    # Guards the accuracy claim above rather than just the absolute tolerance:
    # if the general path ever improves, this should be revisited, and if the
    # specialised one regresses to Pade-level accuracy we want to hear about it.
    M = jnp.asarray(_HADAMARD)
    ref = sla.logm(_HADAMARD)
    err_unitary = np.max(np.abs(np.array(logm_unitary(M, KEY)) - ref))
    err_general = np.max(np.abs(np.array(logm(M, KEY)) - ref))
    assert err_unitary < err_general
