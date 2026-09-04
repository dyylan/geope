"""
Tests for geope/geometry.py.

Tested items:
  Classes:
    - TangentBundle  (restrict, embed, hvp)
    - Manifold      (log, distance2, fidelity, infidelity,
                     hessian_quadratic_form, bind, is_bound)
                     — the hook contracts themselves live in test_manifolds.py
    - UnitaryGroup           (phase-sensitive, full u(d))
    - SpecialUnitaryGroup    (phase-invariant, traceless su(d))
    - GeometricContext       (the four cost tiers, their laziness, and the
                              one-propagator/one-Jacobian/one-logarithm budget)
"""

import collections
import dataclasses
from dataclasses import FrozenInstanceError

import pytest
import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

jax.config.update("jax_enable_x64", True)

from geope.geometry import (
    Manifold,
    SpecialUnitaryGroup,
    TangentBundle,
    UnitaryGroup,
)
from geope.geope import linear_comb_projected_coeffs_multigate
from geope.geometry.basis import get_project_omegas_fn
from geope.parameters import Parameters
from geope.utils import (
    construct_full_pauli_basis,
    construct_Heisenberg_pauli_basis,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


@pytest.fixture
def basis_2q():
    """Full 2-qubit Pauli basis: 15 = dim su(4) traceless Hermitian elements."""
    return construct_full_pauli_basis(2)


@pytest.fixture
def tangent_2q(basis_2q):
    return TangentBundle(frame=basis_2q)


@pytest.fixture
def bind_kwargs(basis_2q):
    """What `bind` needs: the chart's generators and the ambient frame.

    It builds the chart, its Jacobian and its HVP from the generators — there is
    no chart to hand it — and stores the frame as data.
    """
    return dict(generators=basis_2q, frame=basis_2q)


@pytest.fixture
def su4():
    return SpecialUnitaryGroup(4)


@pytest.fixture
def u4():
    return UnitaryGroup(4)


def _random_unitary(d, seed):
    """A Haar-ish random unitary via the exponential of a random Hermitian."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    h = (a + a.conj().T) / 2
    return np.asarray(jsla.expm(1j * jnp.asarray(h)))


# ===================================================================
# Tests — TangentBundle
# ===================================================================


class TestTangentBundleColumns:
    @pytest.fixture
    def masked(self, basis_2q):
        columns = np.array([True, False, True, False])
        return TangentBundle(frame=basis_2q, columns=columns)

    def test_restrict_selects_the_masked_columns(self, masked):
        coeffs = jnp.asarray(np.arange(2 * 4 * 15, dtype=float).reshape(2, 4, 15))
        out = masked.restrict(coeffs)
        assert out.shape == (2, 2, 15)
        np.testing.assert_allclose(np.asarray(out), np.asarray(coeffs)[:, [0, 2], :])

    def test_embed_inverts_restrict(self, masked):
        sol = jnp.asarray(np.arange(2 * 2, dtype=float).reshape(2, 2))
        out = masked.embed(sol)
        assert out.shape == (2, 4)
        np.testing.assert_allclose(np.asarray(out)[:, [0, 2]], np.asarray(sol))
        np.testing.assert_allclose(np.asarray(out)[:, [1, 3]], 0.0)

    def test_none_columns_is_a_no_op_both_ways(self, tangent_2q):
        # This is the param_transform path: every column is free, so neither the
        # omega restriction nor the coefficient scatter does anything.
        coeffs = jnp.asarray(np.arange(2 * 4 * 15, dtype=float).reshape(2, 4, 15))
        assert tangent_2q.restrict(coeffs) is coeffs
        sol = jnp.zeros((2, 4))
        assert tangent_2q.embed(sol) is sol


# ===================================================================
# Tests — Manifold: the pure group maths (no chart, no target)
# ===================================================================


class TestManifoldIsUsableUnbound:
    def test_constructed_from_dimension_alone(self):
        m = SpecialUnitaryGroup(4)
        assert m.dim == 4
        assert not m.is_bound
        assert m.target is None and m.compute_point is None and m.tangent is None

    def test_flags_and_manifold_dims(self):
        assert SpecialUnitaryGroup(4).projective is True
        assert UnitaryGroup(4).projective is False
        assert SpecialUnitaryGroup(4).manifold_dim == 15  # dim su(4) = d^2 - 1
        assert UnitaryGroup(4).manifold_dim == 16  # dim u(4)  = d^2

    def test_is_frozen(self, su4):
        with pytest.raises(FrozenInstanceError):
            su4.dim = 2

    def test_is_abstract(self):
        with pytest.raises(TypeError):
            Manifold(4)


class TestManifoldLog:
    def test_log_of_a_point_with_itself_vanishes(self, su4):
        a = su4.log(jnp.asarray(CNOT), jnp.asarray(CNOT))
        np.testing.assert_allclose(np.asarray(a), 0.0, atol=1e-12)

    def test_log_is_skew_hermitian(self, su4):
        a = np.asarray(su4.log(jnp.asarray(_random_unitary(4, 12)), jnp.asarray(CNOT)))
        np.testing.assert_allclose(a, -a.conj().T, atol=1e-12)

    def test_su_log_is_traceless(self, su4):
        a = su4.log(jnp.asarray(_random_unitary(4, 13)), jnp.asarray(CNOT))
        assert abs(complex(jnp.trace(a))) < 1e-12

    def test_u_log_keeps_the_global_phase(self, u4):
        # A target that differs from x by a pure phase has a *traceless* SU log
        # (nothing to do) but a non-traceless U log (the phase is a direction).
        x = jnp.asarray(_random_unitary(4, 14))
        y = jnp.exp(1j * 0.3) * x
        assert abs(complex(jnp.trace(u4.log(x, y)))) > 1e-6
        assert abs(complex(jnp.trace(SpecialUnitaryGroup(4).log(x, y)))) < 1e-12

    def test_u_log_exponentiates_back_exactly(self, u4):
        # On U(d) the projection is the identity, so x exp(Log_x(y)) == y exactly.
        x = jnp.asarray(_random_unitary(4, 15))
        y = jnp.asarray(_random_unitary(4, 16))
        np.testing.assert_allclose(
            np.asarray(x @ jsla.expm(u4.log(x, y))), np.asarray(y), atol=1e-10
        )

    def test_su_log_exponentiates_back_up_to_phase(self, su4):
        # SU quotients the phase out, so the reconstruction is exact only
        # projectively -- which is precisely what its fidelity measures.
        x = jnp.asarray(_random_unitary(4, 17))
        y = jnp.asarray(_random_unitary(4, 18))
        recovered = x @ jsla.expm(su4.log(x, y))
        assert np.isclose(float(su4.fidelity(recovered, y)), 1.0, atol=1e-10)

    def test_hermitian_target_on_the_branch_cut(self):
        # Hadamard is a Hermitian unitary: its -1 eigenvalue sits exactly on the
        # principal branch cut, the case that makes taking a second log in the
        # conjugate order unsafe. One log, resolved to machine precision.
        su2 = SpecialUnitaryGroup(2)
        a = su2.log(jnp.eye(2, dtype=complex), jnp.asarray(HADAMARD))
        recovered = jsla.expm(a)
        assert np.isclose(float(su2.fidelity(recovered, jnp.asarray(HADAMARD))), 1.0)

    def test_accepts_and_ignores_a_key(self, su4):
        x, y = jnp.asarray(_random_unitary(4, 19)), jnp.asarray(CNOT)
        np.testing.assert_allclose(
            np.asarray(su4.log(x, y, jax.random.key(0))),
            np.asarray(su4.log(x, y, jax.random.key(7))),
            atol=1e-14,
        )

    def test_jittable(self, su4):
        a = jax.jit(su4.log)(jnp.asarray(_random_unitary(4, 20)), jnp.asarray(CNOT))
        assert a.shape == (4, 4)


class TestManifoldDistance:
    def test_distance_to_self_is_zero(self, su4):
        assert float(su4.distance2(jnp.asarray(CNOT), jnp.asarray(CNOT))) < 1e-20

    def test_distance_is_symmetric(self, su4):
        x, y = jnp.eye(4, dtype=complex), jnp.asarray(CNOT)
        assert np.isclose(float(su4.distance2(x, y)), float(su4.distance2(y, x)))

    def test_distance_is_half_the_squared_log_norm(self, su4):
        x, y = jnp.asarray(_random_unitary(4, 21)), jnp.asarray(CNOT)
        a = su4.log(x, y)
        assert np.isclose(
            float(su4.distance2(x, y)), 0.5 * float(su4.norm2(x, a)), atol=1e-12
        )

    def test_distance_is_positive(self, su4):
        assert float(su4.distance2(jnp.eye(4, dtype=complex), jnp.asarray(CNOT))) > 0


class TestManifoldFidelity:
    def test_self_fidelity_is_one(self, su4, u4):
        x = jnp.asarray(_random_unitary(4, 22))
        assert np.isclose(float(su4.fidelity(x, x)), 1.0)
        assert np.isclose(float(u4.fidelity(x, x)), 1.0)

    def test_su_is_phase_invariant_and_u_is_not(self, su4, u4):
        x = jnp.asarray(_random_unitary(4, 23))
        phased = jnp.exp(1j * jnp.pi) * x
        assert np.isclose(float(su4.fidelity(phased, x)), 1.0)
        assert np.isclose(float(u4.fidelity(phased, x)), -1.0)

    def test_su_fidelity_is_in_the_unit_interval(self, su4):
        f = float(su4.fidelity(jnp.eye(4, dtype=complex), jnp.asarray(CNOT)))
        assert 0.0 <= f <= 1.0

    def test_fidelity_is_order_agnostic(self, su4, u4):
        x, y = jnp.asarray(_random_unitary(4, 24)), jnp.asarray(CNOT)
        assert np.isclose(float(su4.fidelity(x, y)), float(su4.fidelity(y, x)))
        assert np.isclose(float(u4.fidelity(x, y)), float(u4.fidelity(y, x)))

    def test_infidelity_complements_fidelity(self, su4, u4):
        x, y = jnp.asarray(_random_unitary(4, 25)), jnp.asarray(CNOT)
        for m in (su4, u4):
            assert np.isclose(
                float(m.infidelity(x, y)), 1.0 - float(m.fidelity(x, y)), atol=1e-14
            )


class TestManifoldHessianQuadraticForm:
    def test_radial_direction_reduces_to_the_surrogate(self, su4, tangent_2q):
        # K_A A = A, so the exact intrinsic term equals ||A||^2 on the radial
        # direction -- the regime in which QuadraticArmijo's substitution is exact.
        a = su4.log(jnp.asarray(_random_unitary(4, 26)), jnp.asarray(CNOT))
        value, rho = su4.hessian_quadratic_form(CNOT, a, a)
        assert np.isclose(float(value), float(su4.norm2(CNOT, a)), rtol=1e-10)
        assert float(rho) > 0

    def test_is_even_in_a(self, su4):
        # The kernel h(delta) is even, so either sign convention for A gives the
        # same form -- which is what lets the context keep a single log.
        a = su4.log(jnp.asarray(_random_unitary(4, 27)), jnp.asarray(CNOT))
        omega = 1j * jnp.asarray(construct_full_pauli_basis(2).basis[0])
        v_pos, rho_pos = su4.hessian_quadratic_form(CNOT, a, omega)
        v_neg, rho_neg = su4.hessian_quadratic_form(CNOT, -a, omega)
        assert np.isclose(float(v_pos), float(v_neg), rtol=1e-12)
        assert np.isclose(float(rho_pos), float(rho_neg), rtol=1e-12)

    def test_never_exceeds_the_surrogate(self, su4, tangent_2q):
        # K_A <= I on the convex region, so the exact term is an under-estimate.
        a = 0.3 * su4.log(jnp.asarray(_random_unitary(4, 28)), jnp.asarray(CNOT))
        omega = jnp.asarray(_random_unitary(4, 29))
        omega = omega - omega.conj().T
        value, _ = su4.hessian_quadratic_form(CNOT, a, omega)
        assert float(value) <= float(su4.norm2(CNOT, omega)) + 1e-9


# ===================================================================
# Tests — Manifold.bind
# ===================================================================


class TestBind:
    def test_builds_the_chart_and_attaches_the_target(self, su4, bind_kwargs):
        bound = su4.bind(target=CNOT, **bind_kwargs)
        assert bound.is_bound
        assert isinstance(bound, SpecialUnitaryGroup)
        np.testing.assert_allclose(np.asarray(bound.target), CNOT)
        # the chart and both differentials come out of the one call
        assert bound.tangent.jacobian is not None
        assert bound.tangent.hvp is not None
        assert bound.tangent.generators is bind_kwargs["generators"]
        assert bound.compute_point(jnp.zeros((2, 15), jnp.complex128)).shape == (4, 4)

    def test_leaves_the_unbound_manifold_alone(self, su4, bind_kwargs):
        su4.bind(target=CNOT, **bind_kwargs)
        assert not su4.is_bound

    def test_rejects_a_mis_shaped_target(self, su4, bind_kwargs):
        with pytest.raises(ValueError, match=r"\(4, 4\) target"):
            su4.bind(target=HADAMARD, **bind_kwargs)

    def test_rejects_an_already_bound_manifold(self, su4, bind_kwargs):
        bound = su4.bind(target=CNOT, **bind_kwargs)
        with pytest.raises(ValueError, match="already bound"):
            bound.bind(target=CNOT, **bind_kwargs)

    def test_a_none_target_still_binds_the_chart(self, su4, bind_kwargs):
        """A target-less problem gets a working chart, but nothing that needs a target."""
        bound = su4.bind(target=None, **bind_kwargs)
        assert not bound.is_bound
        assert bound.compute_point(jnp.zeros((2, 15), jnp.complex128)).shape == (4, 4)
        with pytest.raises(ValueError, match="needs a bound manifold"):
            bound.context(jnp.zeros((2, 15), jnp.complex128))

    def test_bound_manifold_keeps_the_pure_maths(self, su4, bind_kwargs):
        bound = su4.bind(target=CNOT, **bind_kwargs)
        x = jnp.asarray(_random_unitary(4, 30))
        np.testing.assert_allclose(
            np.asarray(bound.log(x, jnp.asarray(CNOT))),
            np.asarray(su4.log(x, jnp.asarray(CNOT))),
            atol=1e-14,
        )


# ===================================================================
# Tests — GeometricContext
# ===================================================================


def _spy_manifold(params):
    """A copy of ``params.manifold`` counting chart / Jacobian / vjp / Hessian / log calls.

    The counts are of *trace-time* calls, which is what matters: the context is
    built inside a jitted update, so one Python call is one evaluation in the
    compiled step.
    """
    m = params.manifold
    counts = collections.Counter()

    def wrap(name, fn):
        def wrapped(*args, **kwargs):
            counts[name] += 1
            return fn(*args, **kwargs)

        return wrapped

    class _Spy(type(m)):
        def log(self, x, y, key=None):
            counts["log"] += 1
            return super().log(x, y, key)

    # Copy every field the manifold declares rather than naming the group's, so
    # this works on the sphere and Stiefel too — they carry `base_point`,
    # `frame` and `projective` that a hard-coded `dim=`/`target=` would drop.
    fields = {f.name: getattr(m, f.name) for f in dataclasses.fields(m)}
    fields["compute_point"] = wrap("compute_point", m.compute_point)
    # `vjp` and `hessian` are wrapped too, or the cost tier's tests would be
    # vacuous: a `value_and_grad` step touches neither `compute_point` nor
    # `jacobian`, so without these it would register as costing nothing at all.
    fields["tangent"] = dataclasses.replace(
        m.tangent,
        jacobian=wrap("jacobian", m.tangent.jacobian),
        vjp=wrap("vjp", m.tangent.vjp),
        hessian=(
            None
            if m.tangent.hessian is None
            else wrap("chart_hessian", m.tangent.hessian)
        ),
    )
    return _Spy(**fields), counts


def _params(**kwargs):
    """A small 2-qubit CNOT problem: 9 controllable of 15 basis elements."""
    return Parameters(
        basis=construct_full_pauli_basis(2),
        projected_basis=construct_Heisenberg_pauli_basis(2),
        target=CNOT,
        piecewise_steps=2,
        seed=0,
        **kwargs,
    )


@pytest.fixture
def problem():
    """``(params, free_params, direction)`` for one GEOPE-style step."""
    p = _params()
    k = p.proj_drift_basis.lie_algebra_dim
    free = jax.random.normal(jax.random.key(1), (2, k)).astype(jnp.complex128) * 0.4
    coeffs = np.zeros((2, k))
    coeffs[:, p.proj_indices_projdrift_basis] = 0.3
    return p, free, jnp.asarray(coeffs)


class TestContextCost:
    def test_one_propagator_one_jacobian_one_log_per_step(self, problem):
        # The headline invariant: a full geodesic step — both least-squares
        # operands, the direction, the slope and the curvature — costs one
        # propagator, one Jacobian and one logarithm at the base point.
        p, free, coeffs = problem
        spy, counts = _spy_manifold(p)
        ctx = spy.context(free)

        _ = ctx.gammas
        _ = ctx.omegas
        ctx.set_direction(coeffs)
        _ = ctx.F0, ctx.velocity, ctx.q, ctx.q_exact, ctx.xi_rel

        assert dict(counts) == {"compute_point": 1, "jacobian": 1, "log": 1}

    def test_each_ray_point_costs_one_propagator(self, problem):
        p, free, coeffs = problem
        spy, counts = _spy_manifold(p)
        ctx = spy.context(free)
        ctx.set_direction(coeffs)
        base = dict(counts)
        assert base == {}  # nothing evaluated yet: every tier is lazy

        _ = ctx.infidelity_at(-0.1)
        assert dict(counts) == {"compute_point": 1}
        _ = ctx.distance_at(-0.1)
        assert dict(counts) == {"compute_point": 2, "log": 1}

    def test_omegas_never_traces_the_logarithm(self, problem):
        # What makes Gecko cheap: it reads `omegas` and nothing else, so the
        # matrix logarithm is never traced into its null-space loop.
        p, free, _ = problem
        spy, counts = _spy_manifold(p)
        _ = spy.context(free).omegas
        assert counts["log"] == 0
        assert dict(counts) == {"compute_point": 1, "jacobian": 1}

    def test_gammas_alone_never_traces_the_jacobian(self, problem):
        p, free, _ = problem
        spy, counts = _spy_manifold(p)
        _ = spy.context(free).gammas
        assert counts["jacobian"] == 0
        assert dict(counts) == {"compute_point": 1, "log": 1}

    def test_gradient_step_traces_no_logarithm_and_no_jacobian(self, problem):
        # What makes GRAPE cheap, and the cost tier's headline invariant: a whole
        # first-order step reads only the pullback and the ray, so neither the
        # matrix logarithm nor the Jacobian is ever traced into it.
        p, free, coeffs = problem
        spy, counts = _spy_manifold(p)
        ctx = spy.context(free)

        _value, _grad = ctx.value_and_grad
        ctx.set_direction(coeffs)
        _ = ctx.slope
        _ = ctx.infidelity_at(-0.1)

        assert counts["log"] == 0
        assert counts["jacobian"] == 0
        # One pullback pass for the value and gradient, one propagator for the ray.
        assert dict(counts) == {"vjp": 1, "compute_point": 1}

    def test_slope_is_free_once_the_gradient_exists(self, problem):
        p, free, coeffs = problem
        spy, counts = _spy_manifold(p)
        ctx = spy.context(free)
        ctx.set_direction(coeffs)
        _ = ctx.gradient
        base = dict(counts)
        _ = ctx.slope
        assert dict(counts) == base  # a contraction, nothing more

    def test_value_and_grad_does_not_memoise_the_point(self, problem):
        # The documented trap, pinned so nobody "optimises" it away by aliasing
        # `point` onto the vjp: the two routes to the base-point infidelity cost
        # a pass each, and a consumer is expected to pick one.
        p, free, _ = problem
        spy, counts = _spy_manifold(p)
        ctx = spy.context(free)
        _ = ctx.value_and_grad
        assert dict(counts) == {"vjp": 1}
        _ = ctx.infidelity
        assert dict(counts) == {"vjp": 1, "compute_point": 1}

    def test_the_two_routes_to_the_infidelity_agree(self, problem):
        p, free, _ = problem
        ctx = p.manifold.context(free)
        assert jnp.allclose(ctx.value_and_grad[0], ctx.infidelity)

    def test_cost_hessian_costs_one_chart_hessian(self, problem):
        p, free, _ = problem
        spy, counts = _spy_manifold(p)
        h = spy.context(free).cost_hessian
        assert h.shape == (free.size, free.size)
        assert counts["log"] == 0
        assert counts["chart_hessian"] == 1

    def test_jittable_end_to_end(self, problem):
        # The context is a trace-time object: it must compose inside jit, and
        # only arrays may cross the boundary.
        p, free, coeffs = problem

        @jax.jit
        def step(fp, c):
            ctx = p.manifold.context(fp)
            ctx.set_direction(c)
            return ctx.gammas, ctx.omegas, ctx.velocity, ctx.q, ctx.distance_at(-0.1)

        gammas, omegas, s, q, d = step(free, coeffs)
        assert gammas.shape == (p.basis.lie_algebra_dim,)
        assert all(bool(jnp.isfinite(x)) for x in (s, q, d))


class TestContextDirection:
    def test_direction_dependent_properties_raise_without_one(self, problem):
        p, free, _ = problem
        ctx = p.manifold.context(free)
        for name in (
            "V",
            "Omega",
            "omega_norm2",
            "velocity",
            "xi_rel",
            "slope",
            "W",
            "q",
            "q_exact",
        ):
            with pytest.raises(ValueError, match="direction-dependent"):
                getattr(ctx, name)
        for method in ("point_at", "infidelity_at", "distance_at"):
            with pytest.raises(ValueError, match="direction-dependent"):
                getattr(ctx, method)(-0.1)

    def test_direction_free_properties_do_not_raise(self, problem):
        p, free, _ = problem
        ctx = p.manifold.context(free)
        for name in ("point", "jacobian", "A", "F0", "gammas", "omegas", "fidelity"):
            assert getattr(ctx, name) is not None

    def test_set_direction_twice_raises(self, problem):
        p, free, coeffs = problem
        ctx = p.manifold.context(free)
        ctx.set_direction(coeffs)
        with pytest.raises(ValueError, match="already has a direction"):
            ctx.set_direction(coeffs)

    def test_a_raising_property_memoises_nothing(self, problem):
        # The one rule that replaces an invalidation list: a property read too
        # early must not have cached anything.
        p, free, coeffs = problem
        ctx = p.manifold.context(free)
        with pytest.raises(ValueError):
            ctx.velocity
        ctx.set_direction(coeffs)
        assert bool(jnp.isfinite(ctx.velocity))


class TestContextValues:
    def test_distance_at_zero_is_F0(self, problem):
        p, free, coeffs = problem
        ctx = p.manifold.context(free)
        ctx.set_direction(coeffs)
        assert np.isclose(float(ctx.distance_at(0.0)), float(ctx.F0), atol=1e-12)

    def test_infidelity_at_zero_is_the_base_point_infidelity(self, problem):
        p, free, coeffs = problem
        ctx = p.manifold.context(free)
        ctx.set_direction(coeffs)
        assert np.isclose(
            float(ctx.infidelity_at(0.0)), float(ctx.infidelity), atol=1e-12
        )

    def test_fidelity_complements_infidelity(self, problem):
        p, free, _ = problem
        ctx = p.manifold.context(free)
        assert np.isclose(float(ctx.fidelity), 1.0 - float(ctx.infidelity), atol=1e-14)

    def test_F0_is_half_the_squared_log_norm(self, problem):
        p, free, _ = problem
        ctx = p.manifold.context(free)
        assert np.isclose(float(ctx.F0), 0.5 * float(ctx.A_norm2))

    def test_V_from_the_jacobian_matches_the_hvp(self, problem):
        # Tier 1 contracts tier 0's Jacobian instead of asking the HVP, which is
        # what makes the slope free (and available under param_transform). The
        # two must agree.
        p, free, coeffs = problem
        ctx = p.manifold.context(free)
        ctx.set_direction(coeffs)
        _, v_hvp, _ = p.manifold.tangent.hvp(jnp.real(free), coeffs)
        np.testing.assert_allclose(np.asarray(ctx.V), np.asarray(v_hvp), atol=1e-12)

    def test_omega_is_the_left_trivialised_velocity(self, problem):
        p, free, coeffs = problem
        ctx = p.manifold.context(free)
        ctx.set_direction(coeffs)
        np.testing.assert_allclose(
            np.asarray(ctx.Omega),
            np.asarray(ctx.point).conj().T @ np.asarray(ctx.V),
            atol=1e-12,
        )

    def test_slope_is_positive_on_the_solved_direction(self, problem):
        # The sign convention: the solve matches Omega to A, which points away
        # from the target, so the slope is positive and the useful step negative.
        p, free, _ = problem
        ctx = p.manifold.context(free)
        sol = linear_comb_projected_coeffs_multigate(ctx.omegas, ctx.gammas, None)
        ctx.set_direction(p.manifold.tangent.embed(sol))
        assert float(ctx.velocity) > 0

    def test_q_exact_never_exceeds_q(self, problem):
        p, free, coeffs = problem
        ctx = p.manifold.context(free)
        ctx.set_direction(coeffs)
        assert float(ctx.q_exact) <= float(ctx.q) + 1e-9
        assert 0.0 <= float(ctx.xi_rel) <= 1.0


class TestContextUnderParamTransform:
    @pytest.fixture
    def transformed(self):
        def transform(phi):
            out = jnp.zeros(15)
            out = out.at[0].set(jnp.cos(phi[0]))
            out = out.at[1].set(jnp.sin(phi[1]))
            out = out.at[2].set(phi[2] ** 2)
            return out

        p = _params(param_transform=transform, n_experimental_params=3)
        free = jnp.asarray(np.full((2, 3), 0.3))
        return p, free, jnp.asarray(np.full((2, 3), 0.2))

    def test_tier_zero_and_one_work(self, transformed):
        # No exponential-product structure, so no HVP -- but the slope comes
        # from the Jacobian contraction, so first-order information survives.
        p, free, coeffs = transformed
        ctx = p.manifold.context(free)
        ctx.set_direction(coeffs)
        assert p.manifold.tangent.hvp is None
        for value in (ctx.F0, ctx.velocity, ctx.xi_rel, ctx.infidelity):
            assert bool(jnp.isfinite(value))

    def test_curvature_raises(self, transformed):
        p, free, coeffs = transformed
        ctx = p.manifold.context(free)
        ctx.set_direction(coeffs)
        for name in ("W", "q", "q_exact"):
            with pytest.raises(NotImplementedError, match="param_transform"):
                getattr(ctx, name)

    def test_the_ray_works(self, transformed):
        p, free, coeffs = transformed
        ctx = p.manifold.context(free)
        ctx.set_direction(coeffs)
        assert bool(jnp.isfinite(ctx.distance_at(-0.1)))

    def test_every_column_is_solvable(self, transformed):
        # columns=None: the omega restriction and the coefficient scatter are
        # both no-ops in experimental space.
        p, free, coeffs = transformed
        assert p.manifold.tangent.columns is None
        assert p.manifold.context(free).omegas.shape == (2, 3, 15)


class TestContextRequiresBinding:
    def test_unbound_manifold_cannot_open_a_context(self):
        with pytest.raises(ValueError, match="needs a bound manifold"):
            SpecialUnitaryGroup(4).context(jnp.zeros((1, 15)))

    def test_unbound_manifold_has_no_objectives(self):
        for name in ("fidelity_at", "infidelity_at"):
            with pytest.raises(ValueError, match="needs a bound manifold"):
                getattr(SpecialUnitaryGroup(4), name)(jnp.zeros((1, 15)))


class TestTangentBundleHvp:
    def test_built_from_the_generators(self, problem):
        p, free, coeffs = problem
        tangent = p.manifold.tangent
        assert tangent.generators is p.proj_drift_basis
        x, v, w = tangent.hvp(jnp.real(free), coeffs)
        assert x.shape == v.shape == w.shape == (4, 4)

    def test_absent_without_generators(self, basis_2q):
        tangent = TangentBundle(frame=basis_2q)
        assert tangent.hvp is None

    def test_generators_without_an_hvp_is_rejected(self, basis_2q):
        """The analytic HVP *is* the recursion in those generators: both or neither."""
        with pytest.raises(ValueError, match="must be given together"):
            TangentBundle(frame=basis_2q, generators=basis_2q)

    def test_an_hvp_without_generators_is_rejected(self, basis_2q):
        with pytest.raises(ValueError, match="must be given together"):
            TangentBundle(frame=basis_2q, hvp=lambda phi, p: None)


class TestTangentBundleFrame:
    """The frame is *data*, and it is optional for a real reason.

    A manifold whose fibre coordinates are a real/imaginary split of the ambient
    array needs no matrix frame — and is cheaper for it, $2Nm$ coefficients
    against a matrix frame's $d^2$. `bind` must therefore accept ``None`` and
    build no projector at all, not fall back to one.
    """

    def test_bind_stores_the_frame_verbatim(self, su4, basis_2q, bind_kwargs):
        bound = su4.bind(target=CNOT, **bind_kwargs)
        assert bound.tangent.frame is basis_2q

    def test_binds_without_a_frame(self, su4, basis_2q):
        """The chart and its differentials do not depend on the frame."""
        bound = su4.bind(target=CNOT, generators=basis_2q, frame=None)
        assert bound.is_bound
        assert bound.tangent.frame is None
        assert bound.compute_point(jnp.zeros((2, 15), jnp.complex128)).shape == (4, 4)
        assert bound.tangent.jacobian is not None

    def test_a_group_without_a_frame_says_so(self, su4, basis_2q):
        """The group *does* coordinatise through a frame, so it must not guess one."""
        bound = su4.bind(target=CNOT, generators=basis_2q, frame=None)
        with pytest.raises(ValueError, match="ambient Hermitian frame"):
            bound.coefficients(jnp.asarray(CNOT), jnp.zeros((4, 4), jnp.complex128))

    def test_the_projector_is_built_once_and_inside_a_trace(self, su4, bind_kwargs):
        """`_project` is memoised and first read from inside the jitted update.

        Safe because the `Basis` it closes over is numpy and ``frame.n`` is an
        ``int``, so nothing it captures can be a tracer — but that is exactly the
        kind of thing that breaks silently, so pin it.
        """
        bound = su4.bind(target=CNOT, **bind_kwargs)
        point = jnp.asarray(CNOT)

        @jax.jit
        def coeffs_of(tangent):
            return bound.coefficients(point, tangent)

        first = coeffs_of(jnp.zeros((4, 4), jnp.complex128))
        assert "_project" in bound.__dict__  # memoised by the trace
        projector = bound.__dict__["_project"]
        second = coeffs_of(jnp.eye(4, dtype=jnp.complex128) * 1j)
        assert bound.__dict__["_project"] is projector  # not rebuilt
        assert first.shape == second.shape == (15,)


# ===================================================================
# Tests — the pipeline takes no autodiff transform
# ===================================================================


# Every JAX differentiation entry point the pipeline could reach for. Patched on
# the `jax` module itself, which is where every call site resolves them at call
# time (`jax.jacobian(...)`, `jax.value_and_grad(...)`, ...).
_AUTODIFF_TRANSFORMS = (
    "grad",
    "value_and_grad",
    "jacobian",
    "jacfwd",
    "jacrev",
    "jvp",
    "vjp",
    "hessian",
    "linearize",
)


class _AutodiffTaken(AssertionError):
    """Raised in place of a JAX differentiation transform."""


@pytest.fixture
def no_autodiff(monkeypatch):
    """Make any JAX differentiation transform an immediate failure."""

    def forbidden(name):
        def _raise(*args, **kwargs):
            raise _AutodiffTaken(f"jax.{name} was traced")

        return _raise

    for name in _AUTODIFF_TRANSFORMS:
        monkeypatch.setattr(jax, name, forbidden(name))
    return None


class TestNoAutodiffInThePipeline:
    """The propagator recursions supply the whole jet — nothing differentiates.

    The sibling of ``test_omegas_never_traces_the_logarithm``: a laziness/routing
    invariant that no value assertion can see, because both paths agree. Here the
    statement is stronger than "which function was chosen" — it is that tracing
    the chosen one reaches no JAX derivative at all.
    """

    def test_the_guard_catches_the_autodiff_path(self, problem, no_autodiff):
        """The fixture is only meaningful if it would fire. Prove it does."""
        p, free, _ = problem
        with pytest.raises(_AutodiffTaken):
            p.manifold.value_and_grad_autodiff(free)

    def test_chart_jacobian_takes_none(self, problem, no_autodiff):
        p, free, _ = problem
        jac = p.manifold.tangent.jacobian(free)
        assert jac.shape[:2] == tuple(p.manifold.ambient_shape)

    def test_gradient_takes_none(self, problem, no_autodiff):
        p, free, _ = problem
        _, grad = p.manifold.value_and_grad(free)
        assert grad.shape == free.shape

    def test_hessian_takes_none(self, problem, no_autodiff):
        p, free, _ = problem
        assert p.manifold.hessian(free).shape == (free.size, free.size)

    def test_a_whole_geodesic_step_takes_none(self, problem, no_autodiff):
        """The geodesic update's tiers 0-3, end to end."""
        p, free, coeffs = problem
        ctx = p.manifold.context(free)
        _ = ctx.gammas, ctx.omegas, ctx.fidelity
        ctx.set_direction(coeffs)
        _ = ctx.velocity, ctx.q, ctx.q_exact, ctx.infidelity_at(-0.1)
