"""
Tests for geope/units.py and the physical-units integration (issue #22).

Covers the units module on its own (Unit, Quantity, Conversion), the
units-off path staying unchanged, and strict mode across Parameters, Geope,
Grape and Gecko: invariance of the forward map under a change of units,
bit-identical equivalence to the dimensionless problem, every documented
misuse raising ``UnitsError`` naming its argument, the ``delta_t`` setter,
the ``transform_unit`` wrapper, and the ``params.physical`` readouts.

Tested items:
  Module (geope.units):
    - Unit, Quantity, UnitsError, Conversion
    - resolve_unit, contains_quantity, reject_quantity, convert_if_quantity
  Integration:
    - Parameters (delta_t trigger, strict mode, delta_t property, physical)
    - Geope.init / add_parameters, Grape.init
    - Gecko.bound / robust / subdivision
    - History.best_basis_coefficients under a step-dependent transform
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants

jax.config.update("jax_enable_x64", True)

from geope import Gecko, Geope, Grape, History, Parameters, construct_full_pauli_basis
from geope.units import (
    AMPLITUDE,
    TIME,
    Conversion,
    GHz,
    MHz,
    Quantity,
    UnitsError,
    contains_quantity,
    convert_if_quantity,
    ns,
    rad_per_ns,
    rad_per_s,
    rad_per_us,
    resolve_unit,
    s,
    ueV,
    us,
)

CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
CONTROL = {1: ["x", "y"], 2: ["x", "y"]}
DRIFT = {(1, 2): ["zz"]}
N_STEPS = 4


def tfim(delta_t, drift_values, **kw):
    """CNOT on a two-qubit TFIM: local X/Y controls, ZZ drift."""
    return Parameters(
        basis=construct_full_pauli_basis(2),
        control=CONTROL,
        drift=DRIFT,
        drift_values=drift_values,
        target=CNOT,
        piecewise_steps=N_STEPS,
        delta_t=delta_t,
        seed=3,
        **kw,
    )


def physical_tfim(**kw):
    """The units-on problem used throughout: 25 ns segments, 5 MHz ZZ drift."""
    return tfim(25 * ns, {(1, 2): {"zz": 5 * MHz}}, **kw)


def unitary(params):
    return np.asarray(
        params.manifold.compute_point(
            params.free(), jnp.asarray(params.delta_t, jnp.float64)
        )
    )


# ---------------------------------------------------------------------------
# 1. Units and constants
# ---------------------------------------------------------------------------


class TestUnitsAndConstants:
    @pytest.mark.parametrize("value", [0.3, 7.0, 1e-3, 4e9])
    def test_round_trips_are_exact_to_1e15(self, value):
        for a, b in [(MHz, ueV), (rad_per_ns, GHz), (ueV, rad_per_s), (ns, us)]:
            got = float((value * a).to(b).to(a).value)
            assert got == pytest.approx(value, rel=1e-15)

    def test_hz_family_carries_two_pi(self):
        got = float((1 * MHz).to(rad_per_s).value)
        assert got == pytest.approx(2 * np.pi * 1e6, rel=1e-15)

    def test_energy_map_matches_scipy_constants(self):
        e, h, hbar = scipy.constants.e, scipy.constants.h, scipy.constants.hbar
        assert float((1.0 * ueV).to(MHz).value) == pytest.approx(
            1e-6 * e / h / 1e6, rel=1e-14
        )
        assert float((1.0 * ueV).to(rad_per_ns).value) == pytest.approx(
            1e-6 * e / hbar / 1e9, rel=1e-14
        )
        assert float((5.0 * GHz).to(ueV).value) == pytest.approx(
            h * 5e9 / e * 1e6, rel=1e-14
        )
        # The worked numbers quoted in the docs.
        assert float((1.0 * ueV).to(MHz).value) == pytest.approx(241.799, abs=1e-3)
        assert float((1.0 * ueV).to(rad_per_ns).value) == pytest.approx(
            1.5193, abs=1e-4
        )
        assert float((5.0 * GHz).to(ueV).value) == pytest.approx(20.678, abs=1e-3)

    @pytest.mark.parametrize(
        "value, shape",
        [
            (5, ()),
            (5.0, ()),
            (np.float64(5), ()),
            (np.array([1.0, 2.0]), (2,)),
            ((1, 2), (2,)),
            ([1, 2], (2,)),
            (jnp.array([1.0, 2.0]), (2,)),
        ],
        ids=["int", "float", "np scalar", "ndarray", "tuple", "list", "jax array"],
    )
    def test_construction_forms(self, value, shape):
        for q in (value * MHz, MHz * value):
            assert isinstance(q, Quantity)
            assert q.unit is MHz
            assert q.value.dtype == np.float64
            assert q.value.shape == shape

    def test_construction_by_name_and_micro_spellings(self):
        assert Quantity(5.0, "MHz").unit is MHz
        assert Quantity(5.0, "ueV").unit is ueV
        assert Quantity(5.0, "µeV").unit is ueV  # MICRO SIGN
        assert Quantity(5.0, "μeV").unit is ueV  # GREEK SMALL LETTER MU
        assert resolve_unit("rad/ns") is rad_per_ns
        assert resolve_unit("rad_per_ns") is rad_per_ns

    def test_unknown_name_raises(self):
        with pytest.raises(UnitsError, match="parsec"):
            Quantity(1.0, "parsec")
        with pytest.raises(UnitsError, match="Unit"):
            resolve_unit(3)

    def test_time_and_amplitude_are_incommensurable(self):
        with pytest.raises(UnitsError, match="incommensurable"):
            (5 * MHz).to(ns)
        with pytest.raises(UnitsError, match="incommensurable"):
            (5 * ns).to("MHz")

    def test_numpy_and_jax_refuse_a_quantity(self):
        q = 5 * MHz
        with pytest.raises(UnitsError, match="bare array"):
            np.array(q)
        with pytest.raises(UnitsError, match="bare array"):
            np.asarray([q, q])
        with pytest.raises(UnitsError, match="bare array"):
            jnp.asarray(q)

    def test_a_tracer_times_a_unit_raises(self):
        with pytest.raises(UnitsError, match="jitted"):
            jax.jit(lambda x: (x * MHz).value)(2.0)

    def test_a_quantity_is_a_tag_not_an_algebra(self):
        q = 5 * MHz
        with pytest.raises(TypeError):
            q * 2
        with pytest.raises(TypeError):
            2 * q
        with pytest.raises(TypeError):
            np.float64(2) * q
        with pytest.raises(UnitsError, match="already a quantity"):
            q * ns

    def test_quantity_rejects_non_real_values(self):
        with pytest.raises(UnitsError, match="real"):
            Quantity(1 + 1j, MHz)
        with pytest.raises(UnitsError, match="numeric"):
            Quantity("five", MHz)
        # A complex value with zero imaginary part is real: readouts of
        # complex128 internal arrays go through here.
        assert float(Quantity(np.complex128(2.0), MHz).value) == 2.0

    def test_repr(self):
        assert repr(25 * ns) == "25.0 ns"
        assert repr(MHz) == "Unit('MHz')"
        assert str(MHz) == "MHz"
        assert "MHz" in repr(np.array([1.0, 2.0]) * MHz)

    def test_structure_helpers(self):
        assert contains_quantity({(1, 2): {"zz": 5 * MHz}})
        assert contains_quantity([0.1, (1 * MHz, 2 * MHz)])
        assert not contains_quantity({(1, 2): {"zz": 0.5}})
        assert convert_if_quantity(None, np.ones(3), AMPLITUDE, "x") is not None
        with pytest.raises(UnitsError, match="`x`"):
            convert_if_quantity(None, 1 * MHz, AMPLITUDE, "x")


class TestConversion:
    def test_scale_is_one_over_delta_t(self):
        c = Conversion.from_delta_t(25 * ns)
        assert c.internal_amplitude.factor == pytest.approx(1 / 25e-9)
        assert c.internal_time.factor == pytest.approx(25e-9)
        assert c.time_unit is ns
        assert c.amplitude_unit is rad_per_ns
        # 5 MHz over 25 ns is 2 pi * 5e6 * 25e-9 radians per segment.
        assert c.to_internal(5 * MHz, AMPLITUDE, "x") == pytest.approx(
            2 * np.pi * 5e6 * 25e-9
        )
        assert c.to_internal(25 * ns, TIME, "x") == pytest.approx(1.0)

    def test_to_internal_walks_the_input_shapes(self):
        c = Conversion.from_delta_t(25 * ns)
        got = c.to_internal({(1, 2): {"zz": 5 * MHz}}, AMPLITUDE, "d")
        assert isinstance(got[(1, 2)]["zz"], float)
        lo, hi = c.to_internal((-20 * MHz, 20 * MHz), AMPLITUDE, "b")
        assert lo == -hi
        both = c.to_internal((-20, 20) * MHz, AMPLITUDE, "b")
        assert isinstance(both, np.ndarray) and both.shape == (2,)

    def test_to_internal_names_the_argument(self):
        c = Conversion.from_delta_t(25 * ns)
        with pytest.raises(UnitsError, match="`drift_values`.*bare"):
            c.to_internal(0.5, AMPLITUDE, "drift_values")
        with pytest.raises(UnitsError, match="`drift_values`.*time"):
            c.to_internal(5 * ns, AMPLITUDE, "drift_values")
        with pytest.raises(UnitsError, match="`bounds`"):
            c.to_internal({"x": (-1.0, 1.0)}, AMPLITUDE, "bounds")

    def test_to_physical_inverts_to_internal(self):
        c = Conversion.from_delta_t(25 * ns)
        x = np.linspace(-3, 3, 7) * MHz
        back = c.to_physical(c.to_internal(x, AMPLITUDE, "x"), AMPLITUDE).to(MHz)
        np.testing.assert_allclose(back.value, x.value, rtol=1e-15)

    def test_from_delta_t_validation(self):
        with pytest.raises(UnitsError, match="time quantity"):
            Conversion.from_delta_t(5 * MHz)
        with pytest.raises(UnitsError, match="scalar"):
            Conversion.from_delta_t((1, 2) * ns)
        with pytest.raises(ValueError, match="positive"):
            Conversion.from_delta_t(0 * ns)
        with pytest.raises(UnitsError, match="transform_unit"):
            Conversion.from_delta_t(25 * ns, transform_unit=ns)
        assert Conversion.from_delta_t(25 * ns, "MHz").transform_unit is MHz
        with pytest.raises(UnitsError, match="transform_unit"):
            Conversion.from_delta_t(25 * ns).transform_factor()


# ---------------------------------------------------------------------------
# 2. Units off: unchanged, and quantities are refused
# ---------------------------------------------------------------------------


class TestUnitsOff:
    def test_no_conversion_object_is_built(self):
        p = tfim(1.0, {(1, 2): {"zz": 0.5}})
        assert p.conversion is None
        assert isinstance(p.delta_t, float)

    @pytest.mark.parametrize(
        "kw, name",
        [
            ({"init_values": {1: {"x": 1 * MHz}}}, "init_values"),
            ({"bounds": {"x": (-1 * MHz, 1 * MHz)}}, "bounds"),
        ],
    )
    def test_a_quantity_with_bare_delta_t_raises(self, kw, name):
        with pytest.raises(UnitsError, match=f"`{name}`.*delta_t"):
            tfim(1.0, {(1, 2): {"zz": 0.5}}, **kw)

    def test_a_quantity_drift_with_bare_delta_t_raises(self):
        with pytest.raises(UnitsError, match="`drift_values`.*delta_t"):
            tfim(1.0, {(1, 2): {"zz": 5 * MHz}})

    def test_transform_unit_with_bare_delta_t_raises(self):
        with pytest.raises(UnitsError, match="transform_unit"):
            tfim(
                1.0,
                {(1, 2): {"zz": 0.5}},
                param_transform=lambda phi: phi,
                transform_unit=MHz,
            )

    def test_physical_raises_with_the_fix(self):
        with pytest.raises(UnitsError, match="delta_t=25 \\* ns"):
            _ = tfim(1.0, {(1, 2): {"zz": 0.5}}).physical

    def test_delta_t_property_keeps_todays_behaviour(self):
        p = tfim(2.0, {(1, 2): {"zz": 0.5}})
        assert p.delta_t == 2.0 and p.total_time == pytest.approx(8.0)
        p.delta_t = 0.5
        assert p.delta_t == 0.5
        with pytest.raises(ValueError, match="positive"):
            p.delta_t = 0.0
        with pytest.raises(UnitsError, match="delta_t"):
            p.delta_t = 30 * ns

    def test_optimiser_entry_points_refuse_a_quantity(self):
        p = tfim(1.0, {(1, 2): {"zz": 0.5}})
        g = Geope(p)
        with pytest.raises(UnitsError, match="init_parameters"):
            g.init(init_parameters=np.zeros((N_STEPS, 15)) * MHz)
        with pytest.raises(UnitsError, match="`params`"):
            g.add_parameters(np.zeros((N_STEPS, 15)) * MHz)
        gk = Gecko(p)
        with pytest.raises(UnitsError, match="parameter_bounds"):
            gk.bound({"x": (-1 * MHz, 1 * MHz)}, max_bounding_steps=1)
        with pytest.raises(UnitsError, match="`delta`"):
            gk.robust(parameter_labels=["XI"], delta=1 * MHz, max_optimization_steps=1)


# ---------------------------------------------------------------------------
# 3. Unit invariance of the forward map
# ---------------------------------------------------------------------------


class TestInvariance:
    def test_the_same_physical_problem_in_different_units_gives_the_same_U(self):
        init_mhz = {1: {"x": 2.0, "y": -1.0}, 2: {"x": 0.5, "y": 1.5}}

        def build(time_unit, amp_unit):
            dt = (25 * ns).to(time_unit)
            init = {
                site: {k: (v * MHz).to(amp_unit) for k, v in ops.items()}
                for site, ops in init_mhz.items()
            }
            return tfim(dt, {(1, 2): {"zz": (5 * MHz).to(amp_unit)}}, init_values=init)

        reference = unitary(build(ns, MHz))
        for time_unit, amp_unit in [(us, rad_per_us), (ns, ueV), (s, rad_per_s)]:
            np.testing.assert_allclose(
                unitary(build(time_unit, amp_unit)), reference, atol=1e-12
            )


# ---------------------------------------------------------------------------
# 4. Equivalence to the dimensionless problem
# ---------------------------------------------------------------------------


class TestEquivalence:
    def test_units_on_run_is_bit_identical_to_the_bare_run(self):
        p_units = physical_tfim()
        conv = p_units.conversion
        p_bare = tfim(1.0, {(1, 2): {"zz": conv.to_internal(5 * MHz, AMPLITUDE, "zz")}})
        assert p_units.delta_t == 1.0
        np.testing.assert_array_equal(p_units.parameters, p_bare.parameters)

        h_units, h_bare = History(), History()
        Geope(p_units, history=h_units).optimize(max_steps=15)
        Geope(p_bare, history=h_bare).optimize(max_steps=15)
        assert [float(f) for f in h_units.fidelities] == [
            float(f) for f in h_bare.fidelities
        ]
        np.testing.assert_array_equal(p_units.parameters, p_bare.parameters)


# ---------------------------------------------------------------------------
# 5. Thorough checking: every misuse names its argument
# ---------------------------------------------------------------------------


class TestStrictMode:
    @pytest.mark.parametrize(
        "kw, match",
        [
            ({"init_values": {1: {"x": 0.1}}}, "`init_values`.*bare"),
            ({"init_values": {1: {"x": 2 * ns}}}, "`init_values`.*time"),
            ({"bounds": {"x": (-1.0, 1.0)}}, "`bounds`.*bare"),
            ({"param_transform": lambda phi: phi}, "transform_unit.*required"),
            (
                {"param_transform": lambda phi: phi, "transform_unit": ns},
                "transform_unit.*amplitude",
            ),
            ({"transform_unit": MHz}, "no `param_transform`"),
            (
                {
                    "param_transform": lambda phi: phi,
                    "transform_unit": MHz,
                    "init_values": np.zeros((N_STEPS, 4)) * MHz,
                },
                "`init_values`.*experimental",
            ),
        ],
        ids=[
            "bare init_values",
            "time init_values",
            "bare bounds",
            "transform without unit",
            "transform_unit is a time",
            "transform_unit without transform",
            "quantity init_values under transform",
        ],
    )
    def test_parameters_misuse(self, kw, match):
        with pytest.raises(UnitsError, match=match):
            physical_tfim(**kw)

    def test_non_time_delta_t(self):
        with pytest.raises(UnitsError, match="`delta_t`.*time"):
            tfim(5 * MHz, {(1, 2): {"zz": 5 * MHz}})

    def test_drift_values_bare_or_missing(self):
        with pytest.raises(UnitsError, match="`drift_values`.*bare"):
            tfim(25 * ns, {(1, 2): {"zz": 0.5}})
        with pytest.raises(UnitsError, match="`drift_values` is required"):
            tfim(25 * ns, None)

    def test_no_drift_basis_needs_no_drift_values(self):
        p = Parameters(
            basis=construct_full_pauli_basis(2),
            control=CONTROL,
            target=CNOT,
            piecewise_steps=2,
            delta_t=25 * ns,
        )
        assert p.drift_parameters is None and p.conversion is not None

    def test_gecko_misuse(self):
        p = physical_tfim()
        gk = Gecko(p)
        with pytest.raises(UnitsError, match="`parameter_bounds`.*bare"):
            gk.bound({"x": (-1.0, 1.0)}, max_bounding_steps=1)
        with pytest.raises(UnitsError, match="`delta`.*bare"):
            gk.robust(parameter_labels=["XI"], delta=0.01, max_optimization_steps=1)

    def test_experimental_space_stays_bare(self):
        p = physical_tfim(
            param_transform=lambda phi: phi,
            transform_unit=MHz,
            init_values=np.full((N_STEPS, 4), 0.2),
        )
        gk = Gecko(p)
        with pytest.raises(UnitsError, match="no unit in this release"):
            gk.bound({0: (-1 * MHz, 1 * MHz)}, max_bounding_steps=1)
        with pytest.raises(UnitsError, match="no unit in this release"):
            gk.robust(parameter_indices=(0,), delta=1 * MHz, max_optimization_steps=1)


# ---------------------------------------------------------------------------
# 6. The delta_t setter
# ---------------------------------------------------------------------------


class TestDeltaTSetter:
    def test_assignment_converts_and_a_bare_float_raises(self):
        p = physical_tfim()
        p.delta_t = 30 * ns
        assert p.delta_t == pytest.approx(1.2)
        assert float(p.physical.delta_t.value) == pytest.approx(30.0)
        with pytest.raises(UnitsError, match="`delta_t`.*bare"):
            p.delta_t = 1.5
        with pytest.raises(UnitsError, match="`delta_t`.*time"):
            p.delta_t = 5 * MHz
        with pytest.raises(ValueError, match="positive"):
            p.delta_t = -1 * ns

    def test_a_duration_sweep_recompiles_nothing(self):
        p = physical_tfim()
        fid = p.manifold.fidelity_at
        fid(p.free(), jnp.asarray(p.delta_t, jnp.float64))
        size = fid._cache_size()
        for dt in (20 * ns, 30 * ns, 0.04 * us):
            p.delta_t = dt
            fid(p.free(), jnp.asarray(p.delta_t, jnp.float64))
        assert fid._cache_size() == size

    def test_subdivision_halves_the_internal_value_and_keeps_total_time(self):
        p = physical_tfim()
        total = float(p.physical.total_time.value)
        assert total == pytest.approx(100.0)  # 4 x 25 ns
        Gecko(p).smooth(piecewise_steps_multiplier=2, max_smoothing_steps=0)
        assert p.delta_t == pytest.approx(0.5)
        assert float(p.physical.delta_t.value) == pytest.approx(12.5)
        assert float(p.physical.total_time.value) == pytest.approx(total)


# ---------------------------------------------------------------------------
# 7. The transform output unit
# ---------------------------------------------------------------------------


class TestTransformUnit:
    N_EXP = 4  # the projected basis: XI, YI, IX, IY

    def _init(self):
        return np.linspace(-1, 1, N_STEPS * self.N_EXP).reshape(N_STEPS, self.N_EXP)

    @pytest.mark.parametrize("step_dependent", [False, True])
    def test_wrapped_transform_is_the_hand_scaled_one(self, step_dependent):
        def tau(phi):
            return jnp.sin(phi)

        def tau_step(phi, step):
            return jnp.sin(phi) * (1 + step)

        transform = tau_step if step_dependent else tau
        p = physical_tfim(
            param_transform=transform, transform_unit=MHz, init_values=self._init()
        )
        c = p.conversion.transform_factor()
        phi = jnp.array([0.1, -0.2, 0.3, 0.4])
        extra = (2,) if step_dependent else ()
        np.testing.assert_allclose(
            p.param_transform(phi, *extra), c * transform(phi, *extra), rtol=1e-15
        )

        def f_wrapped(x):
            return jnp.sum(p.param_transform(x, *extra))

        def f_hand(x):
            return jnp.sum(c * transform(x, *extra))

        np.testing.assert_allclose(
            jax.grad(f_wrapped)(phi), jax.grad(f_hand)(phi), rtol=1e-15
        )
        assert p.param_transform.physical_transform is transform

    def test_declared_unit_and_rewritten_transform_agree(self):
        k = float((1 * MHz).to(rad_per_us).value)
        p_mhz = physical_tfim(
            param_transform=lambda phi: phi,
            transform_unit=MHz,
            init_values=self._init(),
        )
        p_rad = physical_tfim(
            param_transform=lambda phi: k * phi,
            transform_unit=rad_per_us,
            init_values=self._init(),
        )
        np.testing.assert_allclose(unitary(p_mhz), unitary(p_rad), atol=1e-12)

    def test_physical_basis_coefficients_read_back_in_the_declared_unit(self):
        init = self._init()
        p = physical_tfim(
            param_transform=lambda phi: 2.0 * phi, transform_unit=MHz, init_values=init
        )
        got = p.physical.basis_coefficients.to(MHz).value
        # The transform returns projected-basis coefficients, shape (N_g, 4).
        np.testing.assert_allclose(got, 2.0 * init, rtol=1e-14)


# ---------------------------------------------------------------------------
# 8. No leakage into jit: the optimisers run in strict mode
# ---------------------------------------------------------------------------


class TestOptimisersInStrictMode:
    def test_geope_grape_and_gecko_complete(self):
        p = physical_tfim(bounds={"x": (-50 * MHz, 50 * MHz)})
        Geope(p).optimize(max_steps=10)
        assert p.fidelity is not None
        gk = Gecko(p)
        gk.smooth(max_smoothing_steps=2)
        gk.bound({"x": (-20 * MHz, 20 * MHz)}, max_bounding_steps=2)
        gk.robust(parameter_labels=["XI"], delta=0.5 * MHz, max_optimization_steps=2)
        assert np.all(np.isfinite(p.parameters))

        p2 = physical_tfim()
        Grape(p2).optimize(max_steps=5)
        assert p2.fidelity is not None

    def test_init_and_add_parameters_accept_quantities(self):
        p = physical_tfim()
        g = Geope(p)
        full = np.zeros((N_STEPS, 15))
        full[:, p.projected_indices] = 0.25
        g.init(init_parameters=full * MHz)
        expected = p.conversion.to_internal(0.25 * MHz, AMPLITUDE, "x")
        np.testing.assert_allclose(p.parameters[:, p.projected_indices], expected)
        # The strict-mode drift is the configured one, not ones.
        np.testing.assert_allclose(
            p.parameters[:, p.drift_indices], np.tile(p.drift_parameters, (N_STEPS, 1))
        )
        g.add_parameters(2 * full * MHz)
        np.testing.assert_allclose(p.parameters[:, p.projected_indices], 2 * expected)

        gr = Grape(p)
        gr.init(init_parameters=full * MHz)
        np.testing.assert_allclose(p.parameters[:, p.projected_indices], expected)
        np.testing.assert_allclose(gr.drift_parameters, p.drift_parameters)


# ---------------------------------------------------------------------------
# 9. Readouts
# ---------------------------------------------------------------------------


class TestReadouts:
    def test_readout_units_and_to_dict(self):
        p = physical_tfim(init_values={1: {"x": 3 * MHz}})
        assert p.physical.delta_t.unit is ns
        assert p.physical.total_time.unit is ns
        assert p.physical.basis_coefficients.unit is rad_per_ns
        assert p.physical.drift_parameters.unit is rad_per_ns
        assert float(p.physical.drift_parameters.to(MHz).value[0]) == pytest.approx(5.0)
        assert float(p.physical.drift_parameters.to(ueV).value[0]) == pytest.approx(
            float((5 * MHz).to(ueV).value)
        )
        bare, phys = p.to_dict(), p.physical.to_dict()
        assert {k: set(v) for k, v in bare.items()} == {
            k: set(v) for k, v in phys.items()
        }
        assert float(phys[1]["x"].to(MHz).value) == pytest.approx(3.0)
        assert float(phys[1]["y"].to(MHz).value) == pytest.approx(0.0)

    def test_step_dependent_transform_readouts(self):
        def tau(phi, step):
            return phi * (1.0 + 0.1 * step)

        init = np.full((N_STEPS, 4), 0.3)
        p = physical_tfim(param_transform=tau, transform_unit=MHz, init_values=init)
        h = History()
        Geope(p, history=h).optimize(max_steps=3)
        coeffs = p.basis_coefficients
        assert coeffs.shape == (N_STEPS, 4)
        best = h.best_basis_coefficients
        assert best.shape == (N_STEPS, 4)
        # Both apply the same wrapped transform to their own parameter snapshot.
        idx = int(np.argmax(h.fidelities))
        expected = np.array(
            [
                p.param_transform(jnp.asarray(h.parameters[idx][g]), g)
                for g in range(N_STEPS)
            ]
        )
        np.testing.assert_allclose(best, expected, rtol=1e-14)
