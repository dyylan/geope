"""
Tests for geope/objectives.py.

The objectives are cost specs for ``Gecko``; these tests need no optimisation.
They pin that every cost ported out of the old per-objective step functions
still computes the same number (against inline copies of the old formulas), the
two ``BoundMidpoint`` fixes, and the weighted-sum algebra.

Tested items:
  Classes (geope.objectives):
    - Smoothness, SpectralPower, FrequencyFilter, PeakAmplitude, PulseLength,
      Robustness, BoundViolation, BoundMidpoint, CustomObjective
    - WeightedSum (construction, algebra, names, build)
  Functions:
    - resolve_parameter_indices, resolve_bounds, as_weighted_sum
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from geope.objectives import (
    BoundMidpoint,
    BoundViolation,
    CustomObjective,
    FrequencyFilter,
    Objective,
    PeakAmplitude,
    PulseLength,
    Robustness,
    Smoothness,
    SpectralPower,
    WeightedSum,
    as_weighted_sum,
    resolve_bounds,
    resolve_parameter_indices,
)
from geope.parameters import Parameters
from geope.utils import (
    construct_full_pauli_basis,
    construct_Heisenberg_pauli_basis,
    construct_restricted_pauli_basis,
)

N_GATES = 5
XX = 4  # column of "XX" in the Heisenberg projected basis


def _params(**kwargs):
    cnot = np.eye(4)[[0, 1, 3, 2]].astype(complex)
    return Parameters(
        basis=construct_full_pauli_basis(2),
        projected_basis=construct_Heisenberg_pauli_basis(2),
        target=cnot,
        piecewise_steps=N_GATES,
        seed=3,
        **kwargs,
    )


@pytest.fixture(scope="module")
def params():
    return _params()


@pytest.fixture(scope="module")
def drift_params():
    # Disjoint from the Heisenberg projected basis.
    return _params(
        drift_basis=construct_restricted_pauli_basis(2, {(1, 2): ["xy", "yx"]}),
        drift_values=[0.7, -0.3],
    )


@pytest.fixture(scope="module")
def exp_params():
    n_exp = construct_Heisenberg_pauli_basis(2).lie_algebra_dim
    return _params(param_transform=lambda phi: phi, n_experimental_params=n_exp)


@pytest.fixture
def phi():
    return jnp.asarray(np.random.default_rng(0).normal(size=(N_GATES, 9)))


# ---------------------------------------------------------------------------
# Ported costs agree with the formulas of the step functions they replace
# ---------------------------------------------------------------------------


class TestPortedCosts:
    def test_smoothness_matches_the_dense_difference_matrix(self, params, phi):
        # The old piecewise_smoothing cost, ||D phi||^2 with its explicit D.
        n_params, k = phi.size, phi.shape[1]
        D = jnp.eye(n_params, k=0) - jnp.eye(n_params, k=k)
        D = jnp.vstack([jnp.eye(k, D.shape[1]), D])
        expected = jnp.linalg.norm(D @ phi.flatten()) ** 2
        assert Smoothness().build(params)(phi) == pytest.approx(float(expected))

    def test_smoothness_penalises_only_the_endpoints_of_a_constant_pulse(self, params):
        row = jnp.arange(1.0, 10.0)
        flat = jnp.tile(row, (N_GATES, 1))
        expected = 2 * float(jnp.sum(row**2))
        assert Smoothness().build(params)(flat) == pytest.approx(expected)

    def test_spectral_power(self, params, phi):
        expected = jnp.mean(jnp.abs(jnp.fft.rfft(phi, axis=0)[1:]) ** 2)
        assert SpectralPower().build(params)(phi) == pytest.approx(float(expected))

    def test_spectral_power_of_a_single_segment_is_zero(self, params):
        # The old mean over an empty slice was NaN.
        assert SpectralPower().build(params)(jnp.ones((1, 9))) == 0.0

    def test_frequency_filter(self, params, phi):
        def lowpass(spectrum):
            return spectrum.at[2:].set(0.0)

        spectrum = jnp.fft.rfft(phi, axis=0)
        expected = jnp.mean(jnp.abs(spectrum - lowpass(spectrum)) ** 2)
        cost = FrequencyFilter(lowpass).build(params)
        assert cost(phi) == pytest.approx(float(expected))

    def test_peak_amplitude_matches_the_flat_indexing(self, params, phi):
        indices = (1, XX)
        flat, n_proj = phi.flatten(), phi.shape[1]
        sel = jnp.concatenate([jnp.arange(N_GATES) * n_proj + p for p in indices])
        expected = jnp.max(jnp.abs(flat[sel]))
        cost = PeakAmplitude(parameter_indices=indices).build(params)
        assert cost(phi) == pytest.approx(float(expected))

    def test_peak_amplitude_labels_and_indices_agree(self, params, phi):
        by_label = PeakAmplitude(parameter_labels=["XX"]).build(params)
        by_index = PeakAmplitude(parameter_indices=(XX,)).build(params)
        assert by_label(phi) == by_index(phi)

    def test_pulse_length_includes_the_drift(self, drift_params, phi):
        drift_sq = 0.7**2 + 0.3**2
        expected = jnp.sum(jnp.sqrt(jnp.sum(phi**2, axis=1) + drift_sq + 1e-30))
        cost = PulseLength().build(drift_params)
        assert cost(phi) == pytest.approx(float(expected))

    def test_pulse_length_ignores_the_drift_in_experimental_space(
        self, exp_params, phi
    ):
        expected = jnp.sum(jnp.sqrt(jnp.sum(phi[:, :2] ** 2, axis=1) + 1e-30))
        cost = PulseLength(parameter_indices=(0, 1)).build(exp_params)
        assert cost(phi) == pytest.approx(float(expected))

    def test_robustness_is_one_minus_the_worst_grid_fidelity(self, params, phi):
        objective = Robustness(parameter_indices=(0, XX), delta=0.05, num_samples=3)
        samples = np.linspace(-0.05, 0.05, 3)
        delta_t = jnp.asarray(params.delta_t, jnp.float64)
        worst = np.inf
        for d0, d1 in itertools.product(samples, samples):
            shifted = phi.at[:, 0].add(d0).at[:, XX].add(d1)
            free = jnp.zeros((N_GATES, 9), complex)
            free = free.at[:, params.proj_indices_projdrift_basis].set(shifted)
            worst = min(worst, float(params.manifold.fidelity_at(free, delta_t)))
        assert objective.build(params)(phi) == pytest.approx(1.0 - worst)

    def test_bound_violation_matches_the_per_gate_formula(self, params, phi):
        cost = BoundViolation({"xx": (-0.2, 0.3)}).build(params)
        lower = np.full((N_GATES, 9), -np.inf)
        upper = np.full((N_GATES, 9), np.inf)
        lower[:, XX], upper[:, XX] = -0.2, 0.3
        expected = jnp.max(
            jnp.clip(phi - upper, min=0.0) + jnp.clip(lower - phi, min=0.0)
        )
        assert cost(phi) == pytest.approx(float(expected))

    def test_bound_violation_is_zero_inside_the_box(self, params):
        cost = BoundViolation({"xx": (-1.0, 1.0)}).build(params)
        assert cost(jnp.full((N_GATES, 9), 0.5)) == 0.0


class TestBoundMidpoint:
    def test_minimiser_is_the_midpoint_of_an_asymmetric_box(self, params):
        # The old mid-point step drove phi toward 2 x the midpoint.
        cost = BoundMidpoint({"xx": (0.0, 2.0)}).build(params)
        at_mid = jnp.zeros((N_GATES, 9)).at[:, XX].set(1.0)
        assert cost(at_mid) == 0.0
        np.testing.assert_allclose(jax.grad(cost)(at_mid), 0.0)
        assert cost(at_mid.at[:, XX].set(2.0)) > 0.0

    def test_unbounded_columns_give_a_finite_cost(self, params, phi):
        # The old mid-point step was NaN whenever any column was unbounded.
        cost = BoundMidpoint({"xx": (0.0, 2.0)}).build(params)
        value = cost(phi)
        assert np.isfinite(float(value))
        assert np.all(np.isfinite(jax.grad(cost)(phi)))
        # Only the bounded column takes part.
        assert float(value) == pytest.approx(float(jnp.sum((phi[:, XX] - 1.0) ** 2)))

    def test_ranges_are_normalised_by_the_largest(self, params):
        cost = BoundMidpoint({"xx": (-2.0, 2.0), "x": (-1.0, 1.0)}).build(params)
        # "x" matches IX and XI; their range 2 is half the largest, 4.
        phi = jnp.zeros((1, 9)).at[0, 0].set(1.0)
        assert cost(phi) == pytest.approx(4.0)

    def test_a_pinned_column_stays_finite(self, params, phi):
        cost = BoundMidpoint({"xx": (0.5, 0.5)}).build(params)
        assert np.isfinite(float(cost(phi)))


class TestResolution:
    def test_bounds_are_per_column(self, params):
        lower, upper = resolve_bounds(params, {"xx": (0.0, 2.0)})
        assert lower.shape == upper.shape == (9,)
        assert (lower[XX], upper[XX]) == (0.0, 2.0)
        assert np.isinf(np.delete(lower, XX)).all()

    def test_experimental_bounds_are_index_keyed(self, exp_params):
        lower, upper = resolve_bounds(exp_params, {2: (-1.0, 1.0)})
        assert (lower[2], upper[2]) == (-1.0, 1.0)
        with pytest.raises(ValueError, match="integer-indexed"):
            resolve_bounds(exp_params, {"xx": (-1.0, 1.0)})

    def test_labels_resolve_to_projected_columns(self, params):
        assert resolve_parameter_indices(params, ["XX", "IX"], None) == (XX, 0)

    def test_resolution_errors(self, params, exp_params):
        with pytest.raises(ValueError, match="not both"):
            resolve_parameter_indices(params, ["XX"], (0,))
        with pytest.raises(ValueError, match="must be provided"):
            resolve_parameter_indices(params, None, None)
        with pytest.raises(ValueError, match="param_transform"):
            resolve_parameter_indices(exp_params, ["XX"], None)
        assert resolve_parameter_indices(exp_params, None, None, True) == tuple(
            range(9)
        )


# ---------------------------------------------------------------------------
# Weighted sums
# ---------------------------------------------------------------------------


class TestAlgebra:
    def test_scaling_and_adding(self):
        a, b = Smoothness(), SpectralPower()
        assert (2 * a + b).terms == ((2.0, a), (1.0, b))
        assert (a * 2).terms == ((2.0, a),)
        assert (a + b).terms == ((1.0, a), (1.0, b))
        assert (a - 0.5 * b).terms == ((1.0, a), (-0.5, b))
        assert (-(a + b)).terms == ((-1.0, a), (-1.0, b))

    def test_builtin_sum_and_nesting_flatten(self):
        a, b = Smoothness(), SpectralPower()
        assert sum([a, b]).terms == ((1.0, a), (1.0, b))
        assert (3 * (a + 2 * b)).terms == ((3.0, a), (6.0, b))
        nested = WeightedSum(((2.0, a + b),))
        assert nested.terms == ((2.0, a), (2.0, b))

    def test_invalid_weights_raise(self):
        a = Smoothness()
        with pytest.raises(ValueError, match="finite"):
            float("nan") * a
        with pytest.raises(ValueError, match="finite"):
            np.inf * a
        with pytest.raises(TypeError, match="real"):
            (1 + 2j) * a
        with pytest.raises(TypeError, match="real"):
            WeightedSum((("2", a),))

    def test_invalid_terms_raise(self):
        with pytest.raises(ValueError, match="at least one"):
            WeightedSum(())
        with pytest.raises(TypeError, match="Objective"):
            WeightedSum(((1.0, lambda phi: 0.0),))
        with pytest.raises(TypeError):
            Smoothness() * Smoothness()
        with pytest.raises(TypeError):
            Smoothness() + 1.0

    def test_as_weighted_sum(self):
        a = Smoothness()
        assert as_weighted_sum(a).terms == ((1.0, a),)
        assert as_weighted_sum(a + a) == a + a
        with pytest.raises(TypeError, match="Objective"):
            as_weighted_sum(lambda phi: 0.0)

    def test_names_are_deduplicated_and_label_reads_as_a_formula(self):
        objective = Smoothness() + 0.1 * Smoothness() + CustomObjective(jnp.sum, "e")
        assert objective.names == ("smoothness", "smoothness_2", "e")
        assert objective.label == "smoothness + 0.1·smoothness_2 + e"

    def test_specs_are_values(self):
        assert Smoothness() == Smoothness()
        assert PeakAmplitude(parameter_labels=["XX"]) == PeakAmplitude(
            parameter_labels=("XX",)
        )
        bounds = BoundViolation({"xx": (0.0, 1.0)})
        assert bounds == BoundViolation({"xx": (0.0, 1.0)})
        hash(bounds)  # immutable, so hashable
        assert isinstance(bounds, Objective)


class TestBuild:
    def test_total_is_the_raw_linear_combination(self, params, phi):
        objective = 2.0 * Smoothness() - 0.5 * SpectralPower()
        total, values = objective.build(params)(phi)
        smooth = Smoothness().build(params)(phi)
        power = SpectralPower().build(params)(phi)
        np.testing.assert_allclose(values, [smooth, power])
        assert total == pytest.approx(float(2.0 * smooth - 0.5 * power))

    def test_zero_weight_terms_report_but_carry_no_gradient(self, params, phi):
        monitored = Smoothness() + 0 * BoundMidpoint({"xx": (0.0, 2.0)})
        (total, values), grad = jax.value_and_grad(
            monitored.build(params), has_aux=True
        )(phi)
        alone = jax.grad(Smoothness().build(params))(phi)
        np.testing.assert_array_equal(grad, alone)
        assert float(values[1]) > 0.0
        assert total == values[0]

    def test_non_scalar_cost_raises(self, params, phi):
        objective = as_weighted_sum(CustomObjective(lambda phi: phi[0], "row"))
        with pytest.raises(ValueError, match="'row' must return a scalar"):
            objective.build(params)(phi)
