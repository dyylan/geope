"""
Tests for `geope.geometry.euclidean.SpecialEuclideanGroup`.

The hook contracts it shares with every manifold run in ``test_manifolds.py``; this
file covers what is particular to rigid motions: the product-metric geometry in
closed form, and serial-arm inverse kinematics — a UR5's product-of-exponentials
forward kinematics as a GEOPE chart — checked against an independent
Denavit–Hartenberg reference in numpy.

Tested items:
  Functions:
    - hat / vee / so3_log_vee
  Classes:
    - SpecialEuclideanGroup (membership, log, distance, Hessian form, the chart's
                             omegas and gammas, and an exact-lift Geope run)
  Methods:
    - Basis.overlap (non-Hermitian, non-orthogonal frames)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as spla

jax.config.update("jax_enable_x64", True)

from geope.geometry import Basis, SpecialEuclideanGroup
from geope.geometry.basis import traces
from geope.geometry.euclidean import hat, so3_log_vee, vee
from geope.geope import Geope, least_squares
from geope.line_searches import Armijo
from geope.parameters import Parameters

# ---------------------------------------------------------------------------
# A UR5, from its standard DH table — the reference the chart is checked against
# ---------------------------------------------------------------------------

UR5_D = [0.089459, 0.0, 0.0, 0.10915, 0.09465, 0.0823]
UR5_A = [0.0, -0.425, -0.39225, 0.0, 0.0, 0.0]
UR5_ALPHA = [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]
#: A regular configuration: elbow (q3) and wrist (q5) away from 0 and pi.
Q_REGULAR = np.array([0.4, -1.2, 1.5, -0.8, 1.1, 0.3])


def _dh_frames(q):
    """The DH frames ``T_0 .. T_6`` of the UR5 at joint angles ``q``."""
    frames = [np.eye(4)]
    for theta, d, a, alpha in zip(q, UR5_D, UR5_A, UR5_ALPHA, strict=True):
        ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
        link = np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        frames.append(frames[-1] @ link)
    return frames


def _fk(q):
    return _dh_frames(q)[-1]


def _screws():
    """Space-frame screw axes ``(omega, -omega x q)`` from the zero configuration."""
    frames = _dh_frames(np.zeros(6))
    return np.array(
        [
            np.r_[f[:3, 2], -np.cross(f[:3, 2], f[:3, 3])]
            for f in frames[:-1]  # joint i turns about z of frame i - 1
        ]
    )


def _screw_matrices(screws):
    out = np.zeros((len(screws), 4, 4))
    out[:, :3, :3] = np.asarray(hat(screws[:, :3]))
    out[:, :3, 3] = screws[:, 3:]
    return out


def _to_params(q):
    """Joint angles to the chart's ``(n, 1)`` pulse: later segments act on the left."""
    return np.asarray(q, dtype=float)[::-1].reshape(-1, 1)


def _from_params(params):
    return np.real(np.asarray(params))[::-1, 0]


def _ur5_problem(q0, target, length_scale=0.5):
    """UR5 inverse kinematics: one joint per segment, landed on the home pose."""
    basis = Basis(1j * _screw_matrices(_screws()), labels=list("ABCDEF"))

    def one_joint_per_segment(x, g):
        return x[0] * jax.nn.one_hot(5 - g, 6, dtype=jnp.float64)

    return Parameters(
        basis=basis,
        projected_basis=basis,
        target=target,
        piecewise_steps=6,
        manifold=SpecialEuclideanGroup(
            length_scale=length_scale, base_point=_fk(np.zeros(6))
        ),
        param_transform=one_joint_per_segment,
        n_experimental_params=1,
        init_values=_to_params(q0),
    )


def _pose(rotation_vector, translation):
    pose = np.eye(4)
    pose[:3, :3] = spla.expm(np.asarray(hat(np.asarray(rotation_vector, float))))
    pose[:3, 3] = translation
    return pose


@pytest.fixture
def rng():
    return np.random.default_rng(7)


# ===================================================================
# Tests — the SO(3) primitives
# ===================================================================


class TestRotations:
    def test_vee_inverts_hat(self, rng):
        w = rng.normal(size=(5, 3))
        np.testing.assert_allclose(np.asarray(vee(hat(w))), w, atol=1e-15)

    @pytest.mark.parametrize("angle", [1e-9, 1e-4, 0.7, 2.0, 3.0])
    def test_so3_log_inverts_expm(self, rng, angle):
        axis = rng.normal(size=3)
        w = angle * axis / np.linalg.norm(axis)
        rotation = spla.expm(np.asarray(hat(w)))
        np.testing.assert_allclose(
            np.asarray(so3_log_vee(jnp.asarray(rotation))), w, atol=1e-12
        )

    def test_so3_log_gradient_is_finite_at_the_identity(self):
        def f(w):
            return jnp.sum(so3_log_vee(jax.scipy.linalg.expm(hat(w))) ** 2)

        grad = jax.grad(f)(jnp.zeros(3))
        assert np.all(np.isfinite(np.asarray(grad)))


# ===================================================================
# Tests — the manifold's own geometry
# ===================================================================


class TestGeometry:
    def test_rejects_a_bad_length_scale(self):
        with pytest.raises(ValueError, match="length_scale"):
            SpecialEuclideanGroup(length_scale=0.0)

    def test_rejects_an_off_manifold_base_point(self):
        with pytest.raises(ValueError, match="rigid transform"):
            SpecialEuclideanGroup(base_point=2.0 * np.eye(4))

    @pytest.mark.parametrize(
        "corrupt",
        [
            lambda x: x @ np.diag([1.0, 1.0, 1.001, 1.0]),  # R not orthogonal
            lambda x: x @ np.diag([1.0, 1.0, -1.0, 1.0]),  # a reflection
            lambda x: x + np.eye(4)[[3]] * 0.1,  # last row
            lambda x: x + 1e-3j,  # complex
        ],
    )
    def test_validate_point_rejects_non_rigid_transforms(self, corrupt):
        x = _pose([0.2, -0.4, 0.9], [1.0, 2.0, 3.0])
        SpecialEuclideanGroup().validate_point(x)
        with pytest.raises(ValueError):
            SpecialEuclideanGroup().validate_point(corrupt(x))

    def test_distance_is_the_product_metric(self, rng):
        ell = 0.3
        m = SpecialEuclideanGroup(length_scale=ell)
        w, dp = np.array([0.4, -1.1, 0.3]), rng.normal(size=3)
        x = _pose(rng.normal(size=3), rng.normal(size=3))
        y = x.copy()
        y[:3, :3] = x[:3, :3] @ spla.expm(np.asarray(hat(w)))
        y[:3, 3] += dp
        d2 = ell**2 * w @ w + dp @ dp
        assert float(m.distance2(x, y)) == pytest.approx(0.5 * d2, rel=1e-12)

    def test_log_generates_the_product_geodesic(self, rng):
        """Rotate along the SO(3) geodesic, translate in a straight line: lands on y."""
        m = SpecialEuclideanGroup(length_scale=0.8)
        x = _pose(rng.normal(size=3), rng.normal(size=3))
        y = _pose(rng.normal(size=3), rng.normal(size=3))
        a = np.asarray(m.log(x, y))
        end = np.eye(4)
        end[:3, :3] = x[:3, :3] @ spla.expm(a[:3, :3])
        end[:3, 3] = x[:3, 3] + a[:3, 3]
        np.testing.assert_allclose(end, y, atol=1e-12)

    def test_distance_is_left_invariant(self, rng):
        m = SpecialEuclideanGroup(length_scale=0.5)
        x, y, g = (_pose(rng.normal(size=3), rng.normal(size=3)) for _ in range(3))
        assert float(m.distance2(g @ x, g @ y)) == pytest.approx(
            float(m.distance2(x, y)), rel=1e-10
        )

    @pytest.mark.parametrize("ell", [1.0, 0.3])
    def test_hessian_form_is_the_second_derivative_along_a_geodesic(self, rng, ell):
        """Along a geodesic the chart bends nothing, so the form *is* the curvature."""
        m = SpecialEuclideanGroup(length_scale=ell)
        x = _pose(0.8 * rng.normal(size=3), rng.normal(size=3))
        y = _pose(0.8 * rng.normal(size=3), rng.normal(size=3))
        w, u = rng.normal(size=3), rng.normal(size=3)

        def f(t):
            point = np.eye(4)
            point[:3, :3] = x[:3, :3] @ spla.expm(t * np.asarray(hat(w)))
            point[:3, 3] = x[:3, 3] + t * u
            return float(m.distance2(point, y))

        h = 1e-3
        curvature = (-f(-2 * h) + 16 * f(-h) - 30 * f(0.0) + 16 * f(h) - f(2 * h)) / (
            12 * h * h
        )
        omega = np.zeros((4, 4))
        omega[:3, :3] = np.asarray(hat(w))
        omega[:3, 3] = u
        value, _ = m.hessian_quadratic_form(x, -m.log(x, y), omega)
        assert float(value) == pytest.approx(curvature, rel=1e-6)


# ===================================================================
# Tests — the screw basis survives `Parameters`' masks
# ===================================================================


class TestScrewBasis:
    def test_overlap_keeps_every_screw(self):
        """The trace pairing's column sums cancel on these screws; overlap must not."""
        b = 1j * _screw_matrices(_screws())
        naive = np.sum(np.asarray(traces(b, b)), axis=0)
        assert np.isclose(naive, 0).any()  # the trap: a summed plain trace drops joints
        basis = Basis(b, labels=list("ABCDEF"))
        assert basis.overlap(basis).all()

    def test_overlap_keeps_pure_translations(self):
        """A translation is nilpotent, so Tr(B B') vanishes against every element."""
        t = np.zeros((3, 4, 4))
        t[:, :3, 3] = np.eye(3)
        basis = Basis(1j * t, labels=list("ABC"))
        assert basis.overlap(basis).all()

    def test_parameters_keep_every_joint(self):
        params = _ur5_problem(Q_REGULAR, _fk(Q_REGULAR))
        assert params.projected_indices.all()


# ===================================================================
# Tests — UR5 inverse kinematics through the chart
# ===================================================================


class TestSerialArm:
    def test_the_chart_is_the_forward_kinematics(self, rng):
        m = _ur5_problem(Q_REGULAR, _fk(Q_REGULAR)).manifold
        for q in rng.uniform(-np.pi, np.pi, (5, 6)):
            point = np.asarray(m.compute_point(jnp.asarray(_to_params(q))))
            np.testing.assert_allclose(np.real(point), _fk(q), atol=1e-13)
            assert np.allclose(np.imag(point), 0.0)

    def test_omegas_are_the_metric_weighted_jacobian(self, rng):
        r"""Column $j$ is $(\ell\,\omega_b,\ \dot p)$ for a unit rate of joint $j$."""
        ell = 0.5
        q = Q_REGULAR + 0.3 * rng.normal(size=6)
        m = _ur5_problem(q, _fk(Q_REGULAR), length_scale=ell).manifold
        omegas = np.asarray(m.context(jnp.asarray(_to_params(q))).omegas)[::-1]
        rotation = _fk(q)[:3, :3]
        h = 1e-6
        for j in range(6):
            step = np.eye(6)[j] * h
            dx = (_fk(q + step) - _fk(q - step)) / (2 * h)
            expected = np.r_[ell * vee(rotation.T @ dx[:3, :3]), dx[:3, 3]]
            np.testing.assert_allclose(omegas[j].ravel(), expected, atol=1e-8)

    def test_gammas_are_the_weighted_pose_error(self, rng):
        ell = 0.5
        q = Q_REGULAR + 0.3 * rng.normal(size=6)
        target = _fk(Q_REGULAR)
        ctx = _ur5_problem(q, target, ell).manifold.context(jnp.asarray(_to_params(q)))
        x = _fk(q)
        w = np.asarray(so3_log_vee(x[:3, :3].T @ target[:3, :3]))
        e = np.r_[ell * w, target[:3, 3] - x[:3, 3]]
        np.testing.assert_allclose(np.asarray(ctx.gammas), -e, atol=1e-12)
        assert float(ctx.F0) == pytest.approx(0.5 * e @ e, rel=1e-12)

    def test_the_exact_lift_has_slope_minus_two_f(self, rng):
        """A square, full-rank Jacobian matches the log exactly: F'(0) = -2F."""
        q = Q_REGULAR + 0.3 * rng.normal(size=6)
        m = _ur5_problem(q, _fk(Q_REGULAR)).manifold
        ctx = m.context(jnp.asarray(_to_params(q)))
        sol, diagnostics, _ = least_squares(ctx, None)
        assert float(diagnostics["residual_rel"]) < 1e-10
        ctx.set_direction(-m.tangent.embed(sol))
        h = 1e-6
        slope = (float(ctx.distance_at(h)) - float(ctx.distance_at(-h))) / (2 * h)
        assert float(ctx.velocity) == pytest.approx(-2 * float(ctx.F0), rel=1e-10)
        assert slope == pytest.approx(float(ctx.velocity), rel=1e-6)

    def test_the_exact_lift_converges_quadratically_on_full_steps(self):
        """Geope's raw least-squares step with Armijo from t = 1 is Newton-like IK."""
        q_star = Q_REGULAR
        q0 = q_star + np.array([0.3, -0.25, 0.2, 0.3, -0.2, 0.25])
        target = _fk(q_star)
        params = _ur5_problem(q0, target)
        log = []

        def record(step, history, optimizer):
            value = optimizer.params.manifold.infidelity_at(optimizer.params.free())
            log.append((float(value), int(optimizer.line_search_state["n_eval"])))
            return log[-1][0] > 1e-26

        Geope(params).optimize(
            max_steps=30,
            line_search=Armijo(radius_factor=None),
            max_step_size=1.0,
            least_squares_args={"method": "svd", "normalized": False},
            gram_schmidt_step_size=0,
            precision=np.inf,  # stop on the callback: fidelity saturates at 1.0
            callbacks=record,
        )
        r = [np.sqrt(2 * value) for value, _ in log]
        assert r[-1] < 1e-12
        np.testing.assert_allclose(
            _fk(_from_params(params.parameters)), target, atol=1e-10
        )
        # Full steps once in the quadratic regime, and r_{k+1} <= C r_k^2 there.
        local = [k for k in range(len(r) - 1) if 1e-7 < r[k] < 0.1]
        assert local
        assert all(log[k + 1][1] == 1 for k in local)
        assert max(r[k + 1] / r[k] ** 2 for k in local) < 50.0
