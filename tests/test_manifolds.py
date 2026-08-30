"""
Tests for the `geope.geometry.manifold.Manifold` hook contracts.

Every test in ``TestHookContracts`` is written **once** and run against every
manifold in the library — the two matrix Lie groups and the state sphere, which
is deliberately *not* a group. That is the point: if a contract test needs to
know which space it is looking at, the abstraction has leaked.

Tested items:
  Classes:
    - Manifold             (the eleven hooks, as contracts)
    - MatrixLieGroup       (trivialisation, via the contracts)
    - UnitaryGroup / SpecialUnitaryGroup
    - StateSphere          (the non-group case, plus its own primitives and an
                            end-to-end state-preparation run)
"""

from types import SimpleNamespace

import pytest
import numpy as np
import scipy.linalg as spla

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geope.geometry import (
    MatrixLieGroup,
    SpecialUnitaryGroup,
    StateSphere,
    Stiefel,
    UnitaryGroup,
)
from geope.geometry.lie.groups import (
    fidelity,
    fidelity_full,
    infidelity,
    infidelity_full,
)
from geope.geope import Geope, linear_comb_projected_coeffs_multigate
from geope.jax import su_hessian_quadratic_form
from geope.gecko import Gecko
from geope.parameters import Parameters
from geope.line_searches import (
    ApproximateQuadraticArmijo,
    Armijo,
    GoldenSection,
    QuadraticArmijo,
)
from geope.utils import (
    construct_full_pauli_basis,
    construct_full_spin_boson_basis,
    construct_Heisenberg_pauli_basis,
    construct_restricted_spin_boson_basis,
)
from geope.utils.history import History

CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
KET_00 = np.array([1, 0, 0, 0], dtype=complex)
BELL = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
# An orthonormal 2-frame in C^4: the first two columns of CNOT. Only this
# subspace of the gate is scored, which is the redundancy Stiefel exploits.
CNOT_FRAME = CNOT[:, :2]

PIECES = 2


# ---------------------------------------------------------------------------
# One problem per manifold, built the way a user would
# ---------------------------------------------------------------------------


def _params(**kwargs):
    return Parameters(
        basis=construct_full_pauli_basis(2),
        projected_basis=construct_Heisenberg_pauli_basis(2),
        piecewise_steps=PIECES,
        seed=0,
        **kwargs,
    )


def _su_problem(target_scale=1.0):
    """SU(4): the default gate-synthesis geometry."""
    return _params(target=target_scale * CNOT)


def _u_problem(target_scale=1.0):
    """U(4): the phase-sensitive geometry."""
    return _params(target=target_scale * CNOT, projective=False)


def _sphere_problem(target_scale=1.0):
    """CP^3: state preparation — a homogeneous space, not a group."""
    return _params(
        target=target_scale * BELL,
        manifold=StateSphere(dim=4, base_point=KET_00),
    )


def _stiefel_problem(target_scale=1.0):
    """St_2(C^4): an orthonormal 2-frame — the 1 < m < N redundancy case."""
    return _params(
        target=target_scale * CNOT_FRAME,
        manifold=Stiefel(dim=4, frame=2),
    )


def _stiefel_phase_problem(target_scale=1.0):
    """The same frame, phase-sensitive — the mode whose Hessian is exact."""
    return _params(
        target=target_scale * CNOT_FRAME,
        manifold=Stiefel(dim=4, frame=2, projective=False),
    )


PROBLEMS = {
    "SU(d)": _su_problem,
    "U(d)": _u_problem,
    "CP(n-1)": _sphere_problem,
    "St(4,2)": _stiefel_problem,
    "St(4,2) phase": _stiefel_phase_problem,
}


@pytest.fixture(params=list(PROBLEMS))
def space(request):
    """A bound manifold with a base point, a direction, and its derivatives.

    Everything a hook contract needs, and nothing manifold-specific.
    """
    params = PROBLEMS[request.param]()
    manifold = params.manifold
    k = params.proj_drift_basis.lie_algebra_dim
    free = (
        jax.random.normal(jax.random.key(3), (PIECES, k)).astype(jnp.complex128) * 0.4
    )
    coeffs = np.zeros((PIECES, k))
    coeffs[:, params.proj_indices_projdrift_basis] = 0.35
    coeffs = jnp.asarray(coeffs)

    point = manifold.compute_U(free)
    other = manifold.compute_U(free + 0.7 * coeffs)
    velocity = _velocity(manifold, free, coeffs)
    # Whether the manifold supplies a closed-form Riemannian Hessian is a
    # *capability*, not an identity: `param_transform` already withdraws tier 2
    # from the groups, and projective Stiefel measures distance on a quotient
    # whose Hessian is not implemented. Probe for it so the contracts below stay
    # ignorant of which space they are on.
    try:
        manifold.hessian_quadratic_form(point, velocity, velocity)
        has_curvature = True
    except NotImplementedError:
        has_curvature = False
    return SimpleNamespace(
        has_curvature=has_curvature,
        name=request.param,
        params=params,
        manifold=manifold,
        free=free,
        coeffs=coeffs,
        point=point,
        other=other,
        # Two independent tangent vectors at ``point``, drawn from the chart —
        # which is where the ones the algorithm actually uses come from.
        u=manifold.to_tangent(point, velocity),
        v=manifold.to_tangent(point, _velocity(manifold, free, 0.6 * coeffs[::-1])),
    )


def _skip_without_curvature(space):
    """Skip a contract that only applies where tier 2 exists."""
    if not space.has_curvature:
        pytest.skip(f"{space.name} does not supply a Riemannian Hessian")


def _velocity(manifold, free, coeffs):
    """The chart's ambient velocity ``DPhi_free[coeffs]``."""
    jac = jnp.asarray(manifold.tangent.jacobian(free))
    return jnp.tensordot(jac, coeffs, axes=[[-2, -1], [0, 1]])


# ===================================================================
# Tests — the hook contracts, for every manifold
# ===================================================================


class TestHookContracts:
    def test_dims_are_consistent(self, space):
        m = space.manifold
        assert space.point.shape == tuple(m.ambient_shape)
        assert m.ambient_ndim == len(m.ambient_shape)
        assert 0 < m.manifold_dim

    # --- membership --------------------------------------------------------

    def test_validate_point_accepts_the_chart_s_points(self, space):
        """Whatever the chart produces is on the manifold, by construction."""
        space.manifold.validate_point(space.point)
        space.manifold.validate_point(space.other)

    def test_validate_point_rejects_a_scaled_point(self, space):
        """Scaling leaves every manifold here: it breaks unitarity and unit norm alike."""
        with pytest.raises(ValueError):
            space.manifold.validate_point(np.asarray(1.5 * space.point))

    def test_parameters_rejects_an_off_manifold_target(self, space):
        """The check runs at configuration time, not on the traced `bind` path."""
        with pytest.raises(ValueError):
            PROBLEMS[space.name](target_scale=1.5)

    # --- the metric --------------------------------------------------------

    def test_inner_is_symmetric(self, space):
        m = space.manifold
        assert np.isclose(
            float(m.inner(space.point, space.u, space.v)),
            float(m.inner(space.point, space.v, space.u)),
            atol=1e-12,
        )

    def test_inner_is_positive_definite_on_a_nonzero_tangent(self, space):
        m = space.manifold
        assert float(m.norm2(space.point, space.u)) > 0
        zero = jnp.zeros_like(space.u)
        assert np.isclose(float(m.norm2(space.point, zero)), 0.0)

    def test_inner_is_bilinear(self, space):
        m = space.manifold
        lhs = m.inner(space.point, 2.0 * space.u + 3.0 * space.v, space.v)
        rhs = 2.0 * m.inner(space.point, space.u, space.v) + 3.0 * m.norm2(
            space.point, space.v
        )
        assert np.isclose(float(lhs), float(rhs), rtol=1e-10)

    # --- to_tangent --------------------------------------------------------

    def test_to_tangent_is_linear(self, space):
        m = space.manifold
        a, b = jnp.asarray(space.u), jnp.asarray(space.v)
        np.testing.assert_allclose(
            np.asarray(m.to_tangent(space.point, 2.0 * a - 0.5 * b)),
            np.asarray(
                2.0 * m.to_tangent(space.point, a) - 0.5 * m.to_tangent(space.point, b)
            ),
            atol=1e-12,
        )

    def test_to_tangent_batches_over_leading_axes(self, space):
        m = space.manifold
        batch = jnp.stack([space.u, space.v])
        out = m.to_tangent(space.point, batch)
        assert out.shape == batch.shape
        np.testing.assert_allclose(
            np.asarray(out[0]),
            np.asarray(m.to_tangent(space.point, space.u)),
            atol=1e-12,
        )

    # --- coefficients ------------------------------------------------------

    def test_coefficients_are_real(self, space):
        c = space.manifold.coefficients(space.point, space.u)
        assert not jnp.iscomplexobj(c)

    def test_coefficients_are_linear(self, space):
        m = space.manifold
        lhs = m.coefficients(space.point, 2.0 * space.u - 0.5 * space.v)
        rhs = 2.0 * m.coefficients(space.point, space.u) - 0.5 * m.coefficients(
            space.point, space.v
        )
        np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), atol=1e-12)

    def test_coefficients_are_metric_consistent(self, space):
        # sum_k c_k(u) c_k(v) == const * inner(u, v), with one const for the
        # manifold. Checked on tangents drawn from the chart, which is where the
        # vectors the least-squares problem sees come from: a coefficient map is
        # allowed to resolve onto a subspace (an incomplete basis does exactly
        # that), so the invariant is stated where it has to hold.
        m = space.manifold
        cu = m.coefficients(space.point, space.u)
        cv = m.coefficients(space.point, space.v)
        const = float(jnp.sum(cu * cu)) / float(m.norm2(space.point, space.u))
        assert const > 0
        assert np.isclose(
            float(jnp.sum(cu * cv)),
            const * float(m.inner(space.point, space.u, space.v)),
            rtol=1e-9,
        )

    def test_coefficients_batch_to_a_trailing_axis(self, space):
        m = space.manifold
        batch = jnp.stack([space.u, space.v])
        out = m.coefficients(space.point, batch)
        assert out.shape[0] == 2
        np.testing.assert_allclose(
            np.asarray(out[1]),
            np.asarray(m.coefficients(space.point, space.v)),
            atol=1e-12,
        )

    # --- log and distance --------------------------------------------------

    def test_log_of_a_point_with_itself_vanishes(self, space):
        a = space.manifold.log(space.point, space.point)
        np.testing.assert_allclose(np.asarray(a), 0.0, atol=1e-10)

    def test_log_lands_in_the_tangent_space(self, space):
        # to_tangent is the identity on tangent vectors' representation, so a
        # logarithm must already be expressed in it.
        m = space.manifold
        a = m.log(space.point, space.other)
        assert float(m.norm2(space.point, a)) > 0

    def test_log_norm_is_the_geodesic_distance(self, space):
        m = space.manifold
        a = m.log(space.point, space.other)
        assert np.isclose(
            0.5 * float(m.norm2(space.point, a)),
            float(m.distance2(space.point, space.other)),
            rtol=1e-10,
        )

    def test_distance_is_symmetric_and_vanishes_on_the_diagonal(self, space):
        m = space.manifold
        assert float(m.distance2(space.point, space.point)) < 1e-18
        assert np.isclose(
            float(m.distance2(space.point, space.other)),
            float(m.distance2(space.other, space.point)),
            rtol=1e-9,
        )

    def test_log_is_jittable(self, space):
        m = space.manifold
        a = jax.jit(m.log)(space.point, space.other)
        assert a.shape == tuple(m.ambient_shape)

    # --- the chart's second differential -----------------------------------

    def test_tangent_acceleration_matches_finite_differences(self, space):
        # The definitive check on the hook: Omega-dot is *defined* as the
        # derivative of to_tangent along the chart's curve, so differentiate that
        # numerically and compare. A wrong Christoffel/product-rule term shows up
        # here and nowhere else.
        m = space.manifold

        def omega(t):
            phi = space.free + t * space.coeffs
            return m.to_tangent(m.compute_U(phi), _velocity(m, phi, space.coeffs))

        h = 1e-5
        fd = (omega(h) - omega(-h)) / (2.0 * h)

        _, v, w = m.tangent.hvp(jnp.real(space.free), space.coeffs)
        exact = m.tangent_acceleration(space.point, v, w)

        scale = float(jnp.linalg.norm(jnp.ravel(jnp.asarray(fd))))
        np.testing.assert_allclose(
            np.asarray(exact), np.asarray(fd), atol=1e-6 * max(scale, 1.0)
        )

    def test_hvp_velocity_matches_the_jacobian_contraction(self, space):
        m = space.manifold
        _, v, _ = m.tangent.hvp(jnp.real(space.free), space.coeffs)
        np.testing.assert_allclose(
            np.asarray(v),
            np.asarray(_velocity(m, space.free, space.coeffs)),
            atol=1e-10,
        )

    def test_hvp_point_matches_the_chart(self, space):
        m = space.manifold
        x, _, _ = m.tangent.hvp(jnp.real(space.free), space.coeffs)
        np.testing.assert_allclose(np.asarray(x), np.asarray(space.point), atol=1e-10)

    # --- the intrinsic curvature -------------------------------------------

    def test_hessian_form_is_radially_exact(self, space):
        _skip_without_curvature(space)
        # K_A A = A on any manifold, so the form must return ||A||^2 on the
        # radial direction — the identity QuadraticArmijo's surrogate leans on.
        m = space.manifold
        a = m.log(space.point, space.other)
        value, spread = m.hessian_quadratic_form(space.point, a, a)
        assert np.isclose(float(value), float(m.norm2(space.point, a)), rtol=1e-9)
        assert float(spread) > 0

    def test_hessian_form_never_exceeds_the_surrogate(self, space):
        _skip_without_curvature(space)
        # K_A <= I inside the convex region, so ||Omega||^2 bounds it above.
        m = space.manifold
        a = 0.25 * m.log(space.point, space.other)
        value, _ = m.hessian_quadratic_form(space.point, a, space.u)
        assert float(value) <= float(m.norm2(space.point, space.u)) + 1e-9

    def test_hessian_form_is_finite_at_a_vanishing_tangent(self, space):
        _skip_without_curvature(space)
        m = space.manifold
        zero = jnp.zeros_like(space.u)
        value, spread = m.hessian_quadratic_form(space.point, zero, space.u)
        # K_0 = I, so the form degenerates to the plain squared norm.
        assert np.isclose(float(value), float(m.norm2(space.point, space.u)), rtol=1e-9)
        assert float(spread) < 1e-9

    # --- the cost ----------------------------------------------------------

    def test_fidelity_is_one_at_the_target(self, space):
        m = space.manifold
        assert np.isclose(float(m.fidelity(space.point, space.point)), 1.0, atol=1e-12)

    def test_infidelity_vanishes_at_the_target(self, space):
        m = space.manifold
        assert abs(float(m.infidelity(space.point, space.point))) < 1e-12

    def test_parameter_space_objectives_agree_with_the_ambient_ones(self, space):
        m = space.manifold
        assert np.isclose(
            float(m.fidelity_at(space.free)),
            float(m.fidelity(space.point, m.target)),
            atol=1e-12,
        )
        assert np.isclose(
            float(m.infidelity_at(space.free)),
            float(m.infidelity(space.point, m.target)),
            atol=1e-12,
        )

    # --- the chart ---------------------------------------------------------

    def test_chart_maps_into_the_ambient_space(self, space):
        m = space.manifold
        assert m.compute_U(space.free).shape == tuple(m.ambient_shape)

    def test_hessian_is_available(self, space):
        m = space.manifold
        h = m.hessian(space.free)
        n = space.free.size
        assert jnp.asarray(h).reshape(n, n).shape == (n, n)


# ===================================================================
# Tests — a full GeometricContext step, for every manifold
# ===================================================================


class TestContextOnEveryManifold:
    def test_a_full_geodesic_step_runs(self, space):
        m = space.manifold
        ctx = m.context(space.free)
        sol = linear_comb_projected_coeffs_multigate(ctx.omegas, ctx.gammas, None)
        ctx.set_direction(m.tangent.embed(sol))
        exact = (ctx.q_exact, ctx.rho) if space.has_curvature else ()
        for value in (ctx.F0, ctx.s, ctx.q, ctx.xi_rel, *exact):
            assert np.isfinite(float(value)), value
        for t in (0.0, -0.1):
            assert np.isfinite(float(ctx.distance_at(t)))
            assert np.isfinite(float(ctx.infidelity_at(t)))

    def test_s_and_q_exact_are_the_derivatives_they_claim_to_be(self, space):
        """``s`` and ``q_exact`` against a finite difference of ``distance_at``.

        The one test that ties the whole of tier 2 to the objective it describes
        — the manifold's intrinsic Hessian *and* the chart's `accel` together,
        along the real parameter ray rather than along a geodesic. Nothing else
        in the suite compares a curvature to an actual second derivative, which
        is exactly how a wrong closed form survives every algebraic contract.
        """
        _skip_without_curvature(space)
        m = space.manifold
        ctx = m.context(space.free)
        sol = linear_comb_projected_coeffs_multigate(ctx.omegas, ctx.gammas, None)
        ctx.set_direction(m.tangent.embed(sol))

        h = 1e-4
        f = [float(ctx.distance_at(t)) for t in (-2 * h, -h, 0.0, h, 2 * h)]
        slope = (f[0] - 8 * f[1] + 8 * f[3] - f[4]) / (12 * h)
        curvature = (-f[0] + 16 * f[1] - 30 * f[2] + 16 * f[3] - f[4]) / (12 * h * h)
        # `s` *is* psi'(0), positive on a descent direction — which is why the
        # bracket is [-t_max, 0] and the accepted step comes out negative.
        assert float(ctx.s) == pytest.approx(slope, rel=1e-6, abs=1e-9)
        assert float(ctx.q_exact) == pytest.approx(curvature, rel=1e-5, abs=1e-8)

    def test_the_solved_direction_descends(self, space):
        # The whole algorithm in one assertion: the least-squares direction has a
        # positive slope on the distance objective (GEOPE's sign convention), so
        # stepping *negatively* along it reduces the geodesic distance.
        m = space.manifold
        ctx = m.context(space.free)
        sol = linear_comb_projected_coeffs_multigate(ctx.omegas, ctx.gammas, None)
        ctx.set_direction(m.tangent.embed(sol))
        assert float(ctx.s) > 0
        assert float(ctx.distance_at(-1e-3)) < float(ctx.F0)

    def test_omegas_and_gammas_share_their_units(self, space):
        # Both operands go through the same `coefficients`, so the residual is a
        # comparison of like with like.
        m = space.manifold
        ctx = m.context(space.free)
        assert ctx.gammas.shape[-1] == ctx.omegas.shape[-1]

    def test_shapes_follow_the_ambient_rank(self, space):
        m = space.manifold
        ctx = m.context(space.free)
        # (*ambient, G, K_free) for the Jacobian, whatever the ambient rank.
        assert ctx.jacobian.shape[: m.ambient_ndim] == tuple(m.ambient_shape)
        assert ctx.omegas.shape[0] == PIECES


# ===================================================================
# Tests — StateSphere's own primitives
# ===================================================================


class TestStateSphere:
    @pytest.fixture
    def sphere(self):
        return StateSphere(dim=4, base_point=KET_00)

    def test_is_not_a_lie_group(self, sphere):
        # The premise of the whole exercise.
        assert not isinstance(sphere, MatrixLieGroup)
        assert sphere.ambient_shape == (4,)
        assert sphere.manifold_dim == 6  # dim_R CP^3 = 2n - 2

    def test_rejects_a_mis_shaped_base_point(self):
        with pytest.raises(ValueError, match=r"\(4,\) base_point"):
            StateSphere(dim=4, base_point=np.zeros(3, dtype=complex))

    def test_rejects_a_non_unit_base_point(self):
        with pytest.raises(ValueError, match="unit vector"):
            StateSphere(dim=4, base_point=2.0 * KET_00)

    def test_tangents_are_orthogonal_to_the_point(self, sphere):
        x = jnp.asarray(BELL)
        z = jnp.asarray([0.3 + 0.1j, -0.2, 0.5j, 0.4])
        u = sphere.to_tangent(x, z)
        # Horizontal: no component along x at all, phase direction included.
        assert abs(complex(jnp.vdot(x, u))) < 1e-12

    def test_log_generates_the_great_circle(self, sphere):
        # exp_x(t * Log_x(y)) = cos(t|A|) x + sin(t|A|) A/|A| must reach y at
        # t = 1, up to the phase the projective geometry quotients out.
        x, y = jnp.asarray(KET_00), jnp.asarray(BELL)
        a = sphere.log(x, y)
        theta = jnp.sqrt(sphere.norm2(x, a))
        reached = jnp.cos(theta) * x + jnp.sin(theta) * a / theta
        assert np.isclose(float(sphere.fidelity(reached, y)), 1.0, atol=1e-10)

    def test_distance_is_the_fubini_study_angle(self, sphere):
        x, y = jnp.asarray(KET_00), jnp.asarray(BELL)
        theta = np.arccos(abs(complex(np.vdot(np.asarray(x), np.asarray(y)))))
        assert np.isclose(float(sphere.distance2(x, y)), 0.5 * theta**2, rtol=1e-10)

    def test_geometry_is_phase_invariant(self, sphere):
        x, y = jnp.asarray(KET_00), jnp.asarray(BELL)
        phased = jnp.exp(1j * 0.9) * y
        assert np.isclose(
            float(sphere.fidelity(x, phased)), float(sphere.fidelity(x, y)), atol=1e-12
        )
        assert np.isclose(
            float(sphere.distance2(x, phased)),
            float(sphere.distance2(x, y)),
            rtol=1e-10,
        )

    def test_orthogonal_states_are_a_quarter_turn_apart(self, sphere):
        x = jnp.asarray(KET_00)
        y = jnp.asarray([0, 1, 0, 0], dtype=complex)
        assert np.isclose(float(sphere.fidelity(x, y)), 0.0, atol=1e-12)
        assert np.isclose(float(sphere.distance2(x, y)), 0.5 * (np.pi / 2) ** 2)


# ===================================================================
# Tests — state preparation end to end on a non-group manifold
# ===================================================================


class TestStateSphereEndToEnd:
    @staticmethod
    def _problem(pieces=4):
        return Parameters(
            basis=construct_full_pauli_basis(2),
            projected_basis=construct_Heisenberg_pauli_basis(2),
            target=BELL,
            piecewise_steps=pieces,
            seed=7,
            manifold=StateSphere(dim=4, base_point=KET_00),
        )

    @pytest.mark.parametrize(
        "line_search", [GoldenSection(), ApproximateQuadraticArmijo()]
    )
    def test_geope_prepares_a_bell_state(self, line_search):
        # The payoff: the optimiser, the line searches and the context run
        # unchanged on a manifold with no group structure.
        g = Geope(self._problem(), history=History())
        g.optimize(max_steps=80, line_search=line_search)
        assert g.history.best_fidelity > 0.999

    def test_projective_flag_comes_from_the_manifold(self):
        p = self._problem()
        assert p.projective is True
        assert p.manifold.projective is True

    def test_gecko_runs_on_the_sphere(self):
        # Gecko reads only `omegas`, so it needs no logarithm and no group.
        p = self._problem()
        Geope(p).optimize(max_steps=60)
        fidelity_before = float(p.fidelity)
        Gecko(p).smooth(smoothing_rate=0.01, max_smoothing_steps=5, diff_tol=1e-6)
        assert float(p.fidelity) > fidelity_before - 1e-2


# ===================================================================
# Tests — Stiefel: the canonical metric and the iterative logarithm
# ===================================================================


def _rand_frame(n, m, rng):
    """A uniformly random orthonormal m-frame in C^n."""
    z = rng.normal(size=(n, m)) + 1j * rng.normal(size=(n, m))
    return jnp.asarray(np.linalg.qr(z)[0], dtype=jnp.complex128)


def _rand_tangent(manifold, point, rng, scale=0.4):
    """A random tangent at ``point``, scaled inside the injectivity radius."""
    n, m = point.shape
    z = jnp.asarray(rng.normal(size=(n, m)) + 1j * rng.normal(size=(n, m)))
    d = manifold.to_tangent(point, z)
    return scale * d / jnp.linalg.norm(d)


class TestStiefel:
    """The primitives Edelman and Zimmermann–Hüper supply, and their invariants.

    An iterative logarithm that is subtly wrong converges GEOPE slowly rather
    than failing, so these check it against an independent construction (`exp`)
    and against the two manifolds already trusted at the endpoints m = 1 and
    m = N.
    """

    def test_dims_and_flag(self):
        assert Stiefel(6, 3).ambient_shape == (6, 3)
        # dim_R St_m(C^N) = 2Nm - m^2, less one for the quotiented global phase
        assert Stiefel(6, 3, projective=False).manifold_dim == 2 * 6 * 3 - 9
        assert Stiefel(6, 3).manifold_dim == 2 * 6 * 3 - 9 - 1
        assert Stiefel(6, 3).projective is True

    def test_rejects_a_bad_shape_or_frame(self):
        with pytest.raises(ValueError, match="1 <= frame <= dim"):
            Stiefel(3, 5)
        with pytest.raises(ValueError, match="base_frame"):
            Stiefel(4, 2, jnp.eye(4, 3, dtype=complex))
        with pytest.raises(ValueError, match="orthonormal"):
            Stiefel(4, 2, 2.0 * jnp.eye(4, 2, dtype=complex))

    def test_exp_lands_on_the_manifold(self):
        rng = np.random.default_rng(0)
        for n, m in [(4, 2), (6, 3), (8, 4)]:
            man = Stiefel(n, m)
            q = _rand_frame(n, m, rng)
            out = man.exp(q, _rand_tangent(man, q, rng))
            assert np.allclose(np.conj(out).T @ out, np.eye(m), atol=1e-10)

    def test_log_inverts_exp(self):
        """The non-projective round trip is exact: no gauge in the way."""
        rng = np.random.default_rng(1)
        for n, m in [(4, 2), (6, 3), (8, 4)]:
            man = Stiefel(n, m, projective=False)
            q = _rand_frame(n, m, rng)
            d = _rand_tangent(man, q, rng)
            assert np.allclose(
                np.asarray(man.log(q, man.exp(q, d))), np.asarray(d), atol=1e-10
            )

    def test_projective_log_inverts_exp_up_to_gauge(self):
        """Projectively, `exp(log(Q*))` recovers Q* only up to a global phase.

        The alignment picks the overlap-real representative of the U(1) orbit,
        which is the submersion's horizontal lift exactly at m=1 and to first
        order beyond — so the *point* comes back exactly, the tangent does not.
        """
        rng = np.random.default_rng(2)
        for n, m in [(4, 2), (6, 3), (8, 4)]:
            man = Stiefel(n, m)
            q = _rand_frame(n, m, rng)
            q_star = man.exp(q, _rand_tangent(man, q, rng, 0.5))
            back = man.exp(q, man.log(q, q_star))
            assert float(man.fidelity(back, q_star)) == pytest.approx(1.0, abs=1e-10)

    def test_log_is_phase_blind_only_when_projective(self):
        rng = np.random.default_rng(3)
        q = _rand_frame(6, 3, rng)
        for projective, blind in [(True, True), (False, False)]:
            man = Stiefel(6, 3, projective=projective)
            q_star = man.exp(q, _rand_tangent(man, q, rng, 0.5))
            base = man.log(q, q_star)
            moved = man.log(q, jnp.exp(1j * 1.0) * q_star)
            same = np.allclose(np.asarray(base), np.asarray(moved), atol=1e-9)
            assert same is blind
            # The fidelity is phase-blind either way only in projective mode.
            assert (
                float(man.fidelity(q, q_star))
                == pytest.approx(
                    float(man.fidelity(q, jnp.exp(1j) * q_star)), abs=1e-12
                )
            ) is blind

    def test_degenerate_targets_stay_finite(self):
        """Q* -> Q is where a converging run spends its time, and the QR there
        is of a vanishing matrix. B -> 0 annihilates whatever basis it returns."""
        rng = np.random.default_rng(4)
        # Non-projective, so the norm of the log is the norm of the tangent:
        # the projective variant strips the phase component and shrinks it.
        man = Stiefel(6, 3, projective=False)
        q = _rand_frame(6, 3, rng)
        assert np.allclose(np.asarray(man.log(q, q)), 0.0, atol=1e-12)
        for scale in (1e-7, 1e-12, 1e-14):
            d = _rand_tangent(man, q, rng, scale)
            out = np.asarray(man.log(q, man.exp(q, d)))
            assert np.all(np.isfinite(out))
            assert float(np.linalg.norm(out)) == pytest.approx(scale, rel=1e-2)

    def test_distance_matches_the_closed_form(self):
        r"""$L[\gamma] = \sqrt{\tfrac12\mathrm{Tr}(A^\dagger A) + \mathrm{Tr}(B^\dagger B)}$."""
        rng = np.random.default_rng(5)
        for n, m in [(4, 2), (8, 4)]:
            man = Stiefel(n, m, projective=False)
            q = _rand_frame(n, m, rng)
            d = _rand_tangent(man, q, rng, 0.5)
            a = np.conj(q).T @ d
            b = np.linalg.qr(np.asarray(d - q @ a))[1]
            closed = 0.5 * np.trace(np.conj(a).T @ a) + np.trace(np.conj(b).T @ b)
            assert float(man.norm2(q, d)) == pytest.approx(
                float(np.real(closed)), rel=1e-10
            )

    def test_coefficients_carry_the_canonical_metric_exactly(self):
        """The constant of the `coefficients` contract is 1, not merely fixed.

        This is what lets the geodesic least squares be posed in the canonical
        norm without the optimiser knowing anything about it.
        """
        rng = np.random.default_rng(6)
        man = Stiefel(8, 4)
        q = _rand_frame(8, 4, rng)
        u = _rand_tangent(man, q, rng, 0.7)
        v = _rand_tangent(man, q, rng, 0.9)
        dot = float(man.coefficients(q, u) @ man.coefficients(q, v))
        assert dot == pytest.approx(float(man.inner(q, u, v)), rel=1e-10)

    def test_m_equals_1_reproduces_the_state_sphere(self):
        """The projective m=1 case *is* CP^{n-1}, checked against StateSphere."""
        rng = np.random.default_rng(7)
        stiefel = Stiefel(5, 1)
        sphere = StateSphere(dim=5, base_point=jnp.eye(5, dtype=complex)[:, 0])
        for _ in range(3):
            q = _rand_frame(5, 1, rng)
            q_star = stiefel.exp(q, _rand_tangent(stiefel, q, rng, 0.6))
            assert np.allclose(
                np.asarray(stiefel.log(q, q_star))[:, 0],
                np.asarray(sphere.log(q[:, 0], q_star[:, 0])),
                atol=1e-10,
            )
            assert float(stiefel.fidelity(q, q_star)) == pytest.approx(
                float(sphere.fidelity(q[:, 0], q_star[:, 0])), abs=1e-12
            )

    def test_m_equals_N_reduces_to_the_group(self):
        """At m=N there is no complement: the log collapses to Q logm(Q'V), and
        the canonical metric is exactly half the Frobenius one."""
        rng = np.random.default_rng(8)
        man = Stiefel(4, 4, projective=False)
        q = _rand_frame(4, 4, rng)
        d = _rand_tangent(man, q, rng, 0.5)
        q_star = man.exp(q, d)
        group = np.asarray(q) @ spla.logm(np.asarray(np.conj(q).T @ q_star))
        assert np.allclose(np.asarray(man.log(q, q_star)), group, atol=1e-10)
        assert float(man.norm2(q, d)) == pytest.approx(
            0.5 * float(jnp.linalg.norm(d)) ** 2, rel=1e-10
        )

    def test_curvature_is_gated_not_approximated_under_projective(self):
        """Phase alignment makes the objective the U(1) *quotient's* squared
        distance, whose Hessian carries an O'Neill term this does not have. It
        fails loudly rather than silently degrading `ApproximateQuadraticArmijo`
        to a form that is a few percent wrong."""
        rng = np.random.default_rng(9)
        man = Stiefel(6, 3)
        q = _rand_frame(6, 3, rng)
        d = _rand_tangent(man, q, rng)
        with pytest.raises(NotImplementedError, match="QuadraticArmijo"):
            man.hessian_quadratic_form(q, d, d)

    @pytest.mark.parametrize(
        "n, m",
        [(6, 2), (4, 2), (6, 3), (8, 2), (9, 3), (5, 1), (5, 3), (4, 4)],
        ids="6x2 4x2 6x3 8x2 9x3 5x1 5x3 4x4".split(),
    )
    def test_hessian_form_matches_a_finite_difference(self, n, m):
        """The definitive check, across every regime of ``p = min(m, N-m)``.

        ``2m < N`` leaves a live D_2 sector, ``2m == N`` empties it, ``2m > N``
        truncates it, and ``m == N`` removes the complement altogether. Moving
        along a *geodesic* zeroes the chart's acceleration, so the second
        derivative of the squared distance is the intrinsic term alone.
        """
        rng = np.random.default_rng(100 + 10 * n + m)
        man = Stiefel(n, m, projective=False)
        q = _rand_frame(n, m, rng)
        target = man.exp(q, _rand_tangent(man, q, rng, 0.6))
        xi = _rand_tangent(man, q, rng, 1.0)

        h = 1e-4
        vals = [
            float(man.distance2(man.exp(q, t * xi), target))
            for t in (-2 * h, -h, 0.0, h, 2 * h)
        ]
        fd = (-vals[0] + 16 * vals[1] - 30 * vals[2] + 16 * vals[3] - vals[4]) / (
            12 * h * h
        )
        # The context's convention: A points *away* from the target.
        a = -man.log(q, target)
        value, _ = man.hessian_quadratic_form(q, a, xi)
        assert float(value) == pytest.approx(fd, rel=1e-6)

    def test_hessian_form_is_not_even_in_a(self):
        """Unlike su(d), -A addresses the geodesic *reflection* of the target,
        which is a different Hessian. This is why the hook negates the context's
        ``A = -log`` instead of passing it through as the group does."""
        rng = np.random.default_rng(31)
        man = Stiefel(6, 2, projective=False)
        q = _rand_frame(6, 2, rng)
        a = _rand_tangent(man, q, rng, 0.6)
        xi = _rand_tangent(man, q, rng, 1.0)
        forward, _ = man.hessian_quadratic_form(q, a, xi)
        backward, _ = man.hessian_quadratic_form(q, -a, xi)
        assert not np.isclose(float(forward), float(backward), rtol=1e-6)

    def test_m_equals_N_reproduces_the_group_hessian(self):
        """At m=N the vertical space vanishes, M_S = 0, and the block-Jacobi
        operator collapses to the group's ad/2 coth(ad/2) — value and spread
        alike, up to the canonical metric's factor of one half."""
        rng = np.random.default_rng(32)
        man = Stiefel(4, 4, projective=False)
        q = _rand_frame(4, 4, rng)
        a = _rand_tangent(man, q, rng, 0.7)
        xi = _rand_tangent(man, q, rng, 1.0)
        value, spread = man.hessian_quadratic_form(q, -a, xi)
        # Left-trivialise: on a group the canonical metric is half the Frobenius.
        group, rho = su_hessian_quadratic_form(jnp.conj(q).T @ a, jnp.conj(q).T @ xi)
        assert float(value) == pytest.approx(0.5 * float(group), rel=1e-10)
        assert float(spread) == pytest.approx(float(rho), rel=1e-10)

    def test_log_is_jittable_and_nests_in_a_while_loop(self):
        """`Armijo` calls `distance_at` inside its own `lax.while_loop`, so the
        iterative log has to survive being nested in one."""
        rng = np.random.default_rng(10)
        man = Stiefel(6, 3)
        q = _rand_frame(6, 3, rng)
        q_star = man.exp(q, _rand_tangent(man, q, rng, 0.5))
        eager = man.log(q, q_star)
        assert np.allclose(
            np.asarray(jax.jit(man.log)(q, q_star)), np.asarray(eager), atol=1e-12
        )

        def nested(a, b):
            step = lambda s: (s[0] + man.distance2(a, b), s[1] + 1)
            return jax.lax.while_loop(lambda s: s[1] < 3, step, (0.0, 0))[0]

        assert float(jax.jit(nested)(q, q_star)) == pytest.approx(
            3.0 * float(man.distance2(q, q_star)), rel=1e-10
        )


class TestStiefelEndToEnd:
    """GEOPE on a genuine 1 < m < N quotient, with no optimiser changes."""

    @staticmethod
    def _problem(pieces=4):
        return Parameters(
            basis=construct_full_pauli_basis(2),
            projected_basis=construct_Heisenberg_pauli_basis(2),
            target=CNOT_FRAME,
            piecewise_steps=pieces,
            seed=11,
            manifold=Stiefel(dim=4, frame=2),
        )

    @pytest.mark.parametrize(
        "line_search",
        [GoldenSection(), Armijo(), QuadraticArmijo()],
        ids=["golden", "armijo", "quad_armijo"],
    )
    def test_geope_synthesises_the_frame(self, line_search):
        g = Geope(self._problem(), history=History())
        g.optimize(max_steps=120, line_search=line_search)
        assert g.history.best_fidelity > 0.999

    def test_approximate_quadratic_armijo_is_refused_when_projective(self):
        g = Geope(self._problem())
        with pytest.raises(NotImplementedError, match="QuadraticArmijo"):
            g.optimize(max_steps=2, line_search=ApproximateQuadraticArmijo())

    @staticmethod
    def _phase_problem(pieces):
        return Parameters(
            basis=construct_full_pauli_basis(2),
            projected_basis=construct_Heisenberg_pauli_basis(2),
            target=CNOT_FRAME,
            piecewise_steps=pieces,
            seed=11,
            manifold=Stiefel(dim=4, frame=2, projective=False),
        )

    @pytest.mark.parametrize("pieces", [1, 4])
    def test_approximate_quadratic_armijo_runs_phase_sensitively(self, pieces):
        """Where the Hessian *is* exact, the residual-aware search runs and
        converges — the whole point of implementing the block-Jacobi form.

        Two pulse lengths, so two separate compilations of `update_step` in one
        process: the Hessian caches its skew basis on the frame size, and a
        cache holding anything trace-local would leak from the first into the
        second.
        """
        g = Geope(self._phase_problem(pieces), history=History())
        g.optimize(max_steps=150, line_search=ApproximateQuadraticArmijo())
        assert g.history.best_fidelity > 0.999

    def test_only_the_frame_is_scored(self):
        """The fidelity ignores the complement — that *is* the redundancy."""
        p = self._problem()
        Geope(p).optimize(max_steps=120)
        q = p.manifold.compute_U(p.free())
        assert float(p.manifold.fidelity(q, jnp.asarray(CNOT_FRAME))) > 0.999


class TestStiefelSpinBoson:
    """The note's 1 < m < N case: a gate mediated by a bosonic mode.

    Two spins coupled through one boson, truncated at $d = 2$ bosons, so
    $N = 4(d+1) = 12$ and $m = 4$. Only how the pulse acts on the four spin
    states *with the boson in vacuum* is scored; everything it does in the
    boson-occupied sector is redundancy the quotient discards.
    """

    TRUNC = 2  # max boson number; Fock dimension is TRUNC + 1
    FOCK = TRUNC + 1
    N = 4 * FOCK
    M = 4

    @classmethod
    def _vacuum_frame(cls):
        """The m columns |spin s> (x) |0>, i.e. column FOCK*s of the identity."""
        e = np.zeros((cls.N, cls.M), dtype=complex)
        for s in range(cls.M):
            e[cls.FOCK * s, s] = 1.0
        return e

    @classmethod
    def _problem(cls, pieces=10):
        e = cls._vacuum_frame()
        cz = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)
        return Parameters(
            basis=construct_full_spin_boson_basis(2, 1, cls.TRUNC),
            projected_basis=construct_restricted_spin_boson_basis(
                2, 1, {1: ["x", "y"], 2: ["x", "y"]}, cls.TRUNC
            ),
            target=e @ cz,  # pi(U_CZ (x) |0><0| + arbitrary)
            piecewise_steps=pieces,
            seed=3,
            manifold=Stiefel(dim=cls.N, frame=cls.M, base_frame=e),
        )

    def test_geope_synthesises_cz_on_the_vacuum_subspace(self):
        p = self._problem()
        Geope(p).optimize(max_steps=300)
        assert float(p.fidelity) > 0.9999

    def test_the_boson_returns_to_vacuum_unasked(self):
        """The payoff: nothing in the objective mentions the boson population,
        yet it starts and ends in vacuum, because leaving it is not free."""
        p = self._problem()
        Geope(p).optimize(max_steps=300)
        frame = p.manifold.compute_U(p.free())
        occupied = [
            float(
                jnp.sum(jnp.abs(frame[self.FOCK * s + 1 : self.FOCK * (s + 1), s]) ** 2)
            )
            for s in range(self.M)
        ]
        assert max(occupied) < 1e-6, occupied


# ===================================================================
# Tests — the group's fidelity formulas, unbound
# ===================================================================


class TestGroupFidelityFormulas:
    """The free functions behind the group hooks (and `geope.fidelity`).

    These take no manifold: which one a problem uses is decided by which group
    it is bound to, so they are tested here rather than as hook contracts.
    """

    I2 = np.eye(2, dtype=complex)
    I4 = np.eye(4, dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    def test_identity_with_itself(self):
        assert jnp.isclose(fidelity(self.I2, self.I2), 1.0, atol=1e-12)
        assert jnp.isclose(fidelity(self.I4, self.I4), 1.0, atol=1e-12)

    def test_distinct_unitaries_score_below_one(self):
        X = jnp.array([[0, 1], [1, 0]], dtype=complex)
        assert fidelity(X, jnp.eye(2, dtype=complex)) < 1.0

    def test_same_unitary_scores_one(self):
        assert jnp.isclose(fidelity(self.H, self.H), 1.0, atol=1e-12)
        assert jnp.isclose(fidelity(CNOT, CNOT), 1.0, atol=1e-12)

    def test_range_0_to_1(self):
        assert 0 <= fidelity(self.I4, CNOT) <= 1.0

    def test_is_symmetric(self):
        assert jnp.isclose(
            fidelity(self.I2, self.H), fidelity(self.H, self.I2), atol=1e-12
        )

    def test_global_phase_invariance(self):
        """The projective formula ignores a global phase; the full one does not."""
        phase = jnp.exp(1j * 0.3)
        assert jnp.isclose(fidelity(self.H, phase * self.H), 1.0, atol=1e-12)
        assert not jnp.isclose(fidelity_full(self.H, phase * self.H), 1.0, atol=1e-3)

    def test_infidelities_are_one_minus_their_fidelity(self):
        assert jnp.isclose(
            infidelity(self.I2, self.H), 1.0 - fidelity(self.I2, self.H), atol=1e-12
        )
        assert jnp.isclose(
            infidelity_full(self.I2, self.H),
            1.0 - fidelity_full(self.I2, self.H),
            atol=1e-12,
        )
