"""Composable null-space objectives for :class:`geope.Gecko`.

An objective is a scalar **cost of the projected pulse** that ``Gecko`` lowers
while moving only inside the Jacobian null space, so that fidelity is preserved
to first order. Objectives are values, not procedures: each is a
``@dataclass(frozen=True)`` spec (value ``__eq__`` and ``__repr__`` for free,
like the line searches) that knows how to turn itself into a cost function, and
nothing about how that cost is minimised. The minimiser is ``Gecko``'s one
projected-gradient step, which is what lets objectives be added together.

**Weighted sums.** Objectives combine linearly with ``*``, ``+`` and ``-``::

    objective = 1.0 * Smoothness() + 0.1 * PeakAmplitude(parameter_labels=["XI"])
    gecko.optimize(objective)

which builds a :class:`WeightedSum` whose cost is the raw linear combination
$C(\\phi) = \\sum_i w_i c_i(\\phi)$. Nested sums flatten, a scalar distributes over
a sum, and the builtin ``sum`` works (``sum([a, b])``). Because ``Gecko`` takes a
fixed-length step along the null-space projection of $\\nabla C$, only the
*ratios* of the weights shape the trajectory; their overall scale changes only
what ``diff_tol`` means. A **zero** weight is allowed and useful: the term is
still evaluated and reported in ``gecko.costs`` but contributes no gradient (its
input is wrapped in ``stop_gradient``, so its backward pass is never traced).
A **negative** weight is allowed too and *maximises* that term.

**The build contract.** ``objective.build(params) -> cost`` resolves the spec
against a :class:`~geope.Parameters` (labels to indices, bounds to arrays, the
drift norm, the fidelity) and returns ``cost(phi)``, where ``phi`` is the **real
float64 projected pulse** of shape ``(N_g, K_P)`` — ``K_P`` is the projected
basis dimension, or ``n_experimental_params`` under ``param_transform`` — and
the result is a real scalar. The cost must be traceable and differentiable
almost everywhere. ``Gecko`` builds an objective *after* any subdivision, so a
cost may close over the subdivided ``params.delta_t``; every cost here is
otherwise independent of ``piecewise_steps``. Subclass :class:`Objective` (or
wrap a function in :class:`CustomObjective`) to add a new one.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

if TYPE_CHECKING:
    from .parameters import Parameters


def _is_zero(value) -> bool:
    """Whether ``value`` is the numeric ``0`` the builtin ``sum`` starts from."""
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0
    )


def _as_weight(weight) -> float:
    """Coerce an objective weight to a finite real ``float``.

    Raises:
        TypeError: If ``weight`` is not a real number.
        ValueError: If ``weight`` is not finite.
    """
    if isinstance(weight, (str, bytes, bool)) or np.iscomplexobj(weight):
        raise TypeError(f"Objective weights must be real numbers, got {weight!r}.")
    try:
        value = float(weight)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"Objective weights must be real numbers, got {weight!r}."
        ) from error
    if not math.isfinite(value):
        raise ValueError(f"Objective weights must be finite, got {weight!r}.")
    return value


class Objective:
    """Base class for a null-space objective.

    Subclasses are ``@dataclass(frozen=True)`` specs that set the class attribute
    ``name`` (the key the term is reported under in ``gecko.costs``) and
    implement :meth:`build`. Multiplying by a scalar or adding another objective
    gives a :class:`WeightedSum`.
    """

    name = "objective"

    def build(self, params: Parameters) -> Callable[[Array], Array]:
        """Resolve the spec against ``params`` and return its cost function.

        Args:
            params: The `Parameters` whose pulse is being post-processed, already
                subdivided.

        Returns:
            A callable ``cost(phi)`` mapping the real ``(N_g, K_P)`` projected
            pulse to a real scalar.
        """
        raise NotImplementedError

    def __mul__(self, weight) -> WeightedSum:
        if isinstance(weight, (Objective, WeightedSum)):
            return NotImplemented
        return WeightedSum(((weight, self),))

    __rmul__ = __mul__

    def __add__(self, other) -> WeightedSum:
        if not isinstance(other, (Objective, WeightedSum)):
            return NotImplemented
        return WeightedSum(((1.0, self), (1.0, other)))

    def __radd__(self, other) -> WeightedSum:
        # Only ``0 + objective`` reaches here, which is how the builtin ``sum``
        # starts; an objective on the left dispatches to ``__add__`` instead.
        if _is_zero(other):
            return WeightedSum(((1.0, self),))
        return NotImplemented

    def __neg__(self) -> WeightedSum:
        return WeightedSum(((-1.0, self),))

    def __sub__(self, other) -> WeightedSum:
        if not isinstance(other, (Objective, WeightedSum)):
            return NotImplemented
        return self + (-1.0) * other


@dataclass(frozen=True)
class WeightedSum:
    """A linear combination $\\sum_i w_i c_i$ of objectives.

    Usually built with ``*`` and ``+`` rather than directly. Nested sums are
    flattened on construction, so ``terms`` holds only plain objectives.

    Attributes:
        terms: ``((weight, objective), ...)``; weights are coerced to finite
            ``float``.

    Raises:
        TypeError: If a term is not a ``(weight, objective)`` pair or a weight is
            not a real number.
        ValueError: If there are no terms or a weight is not finite.
    """

    terms: tuple[tuple[float, Objective], ...]

    def __post_init__(self) -> None:
        flat = []
        for term in self.terms:
            try:
                weight, objective = term
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"WeightedSum terms must be (weight, objective) pairs, got {term!r}."
                ) from error
            weight = _as_weight(weight)
            if isinstance(objective, WeightedSum):
                flat.extend((weight * w, o) for w, o in objective.terms)
            elif isinstance(objective, Objective):
                flat.append((weight, objective))
            else:
                raise TypeError(
                    f"WeightedSum terms must hold Objective instances, got {objective!r}."
                )
        if not flat:
            raise ValueError("A WeightedSum needs at least one term.")
        object.__setattr__(self, "terms", tuple(flat))

    @property
    def names(self) -> tuple[str, ...]:
        """Each term's ``name``, de-duplicated as ``name``, ``name_2``, ...."""
        counts: dict[str, int] = {}
        names = []
        for _, objective in self.terms:
            counts[objective.name] = counts.get(objective.name, 0) + 1
            n = counts[objective.name]
            names.append(objective.name if n == 1 else f"{objective.name}_{n}")
        return tuple(names)

    @property
    def label(self) -> str:
        """A readable formula, e.g. ``smoothness + 0.1·peak_amplitude``."""
        return " + ".join(
            name if weight == 1.0 else f"{weight:g}·{name}"
            for (weight, _), name in zip(self.terms, self.names)
        )

    def build(self, params: Parameters) -> Callable[[Array], tuple[Array, Array]]:
        """Build every term and return the combined cost.

        This is the single place where terms are combined.

        Args:
            params: The `Parameters` whose pulse is being post-processed, already
                subdivided.

        Returns:
            A callable ``cost(phi) -> (total, values)``: ``total`` is
            $\\sum_i w_i c_i(\\phi)$ and ``values`` the ``(n_terms,)`` array of the
            unweighted $c_i(\\phi)$. Zero-weight terms are evaluated on a
            ``stop_gradient``-ed pulse, so they report without contributing.
        """
        weights = tuple(weight for weight, _ in self.terms)
        costs = tuple(objective.build(params) for _, objective in self.terms)
        names = self.names

        def cost(phi):
            values = []
            total = jnp.zeros((), dtype=jnp.float64)
            for weight, term_cost, name in zip(weights, costs, names):
                value = term_cost(phi if weight != 0.0 else jax.lax.stop_gradient(phi))
                if jnp.shape(value) != ():
                    raise ValueError(
                        f"Objective {name!r} must return a scalar, got shape "
                        f"{jnp.shape(value)}."
                    )
                values.append(value)
                if weight != 0.0:
                    total = total + weight * value
            return total, jnp.stack(values)

        return cost

    def __add__(self, other) -> WeightedSum:
        if isinstance(other, Objective):
            other = WeightedSum(((1.0, other),))
        if not isinstance(other, WeightedSum):
            return NotImplemented
        return WeightedSum(self.terms + other.terms)

    def __radd__(self, other) -> WeightedSum:
        if _is_zero(other):
            return self
        return NotImplemented

    def __mul__(self, weight) -> WeightedSum:
        if isinstance(weight, (Objective, WeightedSum)):
            return NotImplemented
        weight = _as_weight(weight)
        return WeightedSum(tuple((weight * w, o) for w, o in self.terms))

    __rmul__ = __mul__

    def __neg__(self) -> WeightedSum:
        return (-1.0) * self

    def __sub__(self, other) -> WeightedSum:
        if not isinstance(other, (Objective, WeightedSum)):
            return NotImplemented
        return self + (-1.0) * other


def as_weighted_sum(objective: Objective | WeightedSum) -> WeightedSum:
    """Wrap a bare `Objective` as ``1.0 * objective``; pass a sum through.

    Raises:
        TypeError: If ``objective`` is neither.
    """
    if isinstance(objective, WeightedSum):
        return objective
    if isinstance(objective, Objective):
        return WeightedSum(((1.0, objective),))
    raise TypeError(
        "Expected an Objective or a WeightedSum of objectives "
        f"(e.g. 1.0 * Smoothness() + 0.1 * PulseLength()), got {objective!r}."
    )


# ---------------------------------------------------------------------------
# Resolving a spec against a Parameters
# ---------------------------------------------------------------------------


def _as_tuple(values) -> tuple | None:
    return None if values is None else tuple(values)


def resolve_parameter_indices(
    params: Parameters,
    parameter_labels: tuple[str, ...] | None,
    parameter_indices: tuple[int, ...] | None,
    default_all: bool = False,
) -> tuple[int, ...]:
    """Resolve ``parameter_labels`` / ``parameter_indices`` into column indices.

    Args:
        params: The `Parameters` the indices refer to.
        parameter_labels: Optional projected-basis label strings.
        parameter_indices: Optional integer indices into the projected pulse's
            columns.
        default_all: If both are ``None`` and this is ``True``, return every
            column.

    Returns:
        A tuple of integer indices into the projected pulse's columns.

    Raises:
        ValueError: If both are provided, if neither is and ``default_all`` is
            ``False``, or if labels are passed with ``param_transform`` active.
    """
    if parameter_labels is not None and parameter_indices is not None:
        raise ValueError("Specify parameter_labels or parameter_indices, not both")
    if params.param_transform is not None:
        if parameter_labels is not None:
            raise ValueError(
                "parameter_labels not supported with param_transform; "
                "use parameter_indices instead"
            )
        if parameter_indices is None:
            if default_all:
                return tuple(range(params.n_experimental_params))
            raise ValueError("parameter_indices required with param_transform")
        return tuple(parameter_indices)
    if parameter_indices is not None:
        return tuple(parameter_indices)
    proj_labels = list(params.projected_basis.labels)
    if parameter_labels is None:
        if default_all:
            return tuple(range(len(proj_labels)))
        raise ValueError(
            "Either parameter_labels or parameter_indices must be provided"
        )
    return tuple(proj_labels.index(label) for label in parameter_labels)


def resolve_bounds(params: Parameters, bounds) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a bounds map into per-column ``(K_P,)`` lower and upper bounds.

    The bounds are constant in time, so they are returned per column and
    broadcast over the gates by the costs — which keeps them valid across a
    subdivision.

    Args:
        params: The `Parameters` the bounds refer to.
        bounds: In projected space, a mapping (or iterable of pairs) from
            interaction labels to ``(min, max)``; under ``param_transform``, from
            integer experimental-parameter indices. Unlisted columns are
            unbounded.

    Returns:
        ``(lower, upper)``, float arrays of shape ``(K_P,)``.

    Raises:
        ValueError: If label keys are passed under ``param_transform``.
    """
    bounds = dict(bounds)
    if params.param_transform is not None:
        if any(isinstance(key, str) for key in bounds):
            raise ValueError(
                "Under `param_transform`, `bound` takes integer-indexed "
                "experimental-parameter bounds ({index: (lo, hi)}, the "
                "same convention as `pulse_constraints`); interaction "
                "labels have no meaning in experimental space."
            )
        n_exp = params.n_experimental_params
        lower = np.full(n_exp, -np.inf)
        upper = np.full(n_exp, np.inf)
        for index, (low, high) in bounds.items():
            lower[index] = low
            upper[index] = high
        return lower, upper
    lower, upper = params.proj_drift_basis.generate_bounds(bounds, 1)
    mask = params.proj_indices_projdrift_basis
    return (
        np.asarray(lower[0], dtype=np.float64)[mask],
        np.asarray(upper[0], dtype=np.float64)[mask],
    )


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Smoothness(Objective):
    """Penalise jumps between consecutive segments, with zero endpoints.

    Cost: $\\|\\phi_0\\|^2 + \\sum_g \\|\\phi_{g+1} - \\phi_g\\|^2 +
    \\|\\phi_{N_g-1}\\|^2$ — the squared differences of the pulse padded with a
    zero segment at each end, so a pulse that switches on or off abruptly is
    penalised too.
    """

    name = "smoothness"

    def build(self, params: Parameters) -> Callable[[Array], Array]:
        def cost(phi):
            jumps = jnp.diff(jnp.pad(phi, ((1, 1), (0, 0))), axis=0)
            return jnp.sum(jumps**2)

        return cost


@dataclass(frozen=True)
class SpectralPower(Objective):
    """Penalise high-frequency spectral power.

    Cost: $\\frac{1}{(N_r-1)K}\\sum_{m \\ge 1, k}|\\widehat{\\phi_k}(m)|^2$ over the
    $N_r$ bins of the ``rfft`` along time. The DC bin $m=0$ is excluded so the
    mean level of each pulse is not penalised (a single segment has no other
    bin, and costs 0).
    """

    name = "spectral_power"

    def build(self, params: Parameters) -> Callable[[Array], Array]:
        def cost(phi):
            power = jnp.abs(jnp.fft.rfft(phi, axis=0)[1:]) ** 2
            return jnp.sum(power) / max(power.size, 1)

        return cost


@dataclass(frozen=True)
class FrequencyFilter(Objective):
    """Drive the pulse toward a filtered version of itself.

    Cost: $\\mathrm{mean}\\,|\\widehat\\phi - \\mathcal F(\\widehat\\phi)|^2$, the
    squared Fourier-space distance to the filtered spectrum — by Parseval, the
    time-domain $L^2$ distance up to a constant.

    Attributes:
        filter_fn: Maps the complex ``(N_r, K)`` ``rfft`` of the pulse to a
            same-shape filtered spectrum.
    """

    filter_fn: Callable[[Array], Array]

    name = "frequency_filter"

    def build(self, params: Parameters) -> Callable[[Array], Array]:
        filter_fn = self.filter_fn

        def cost(phi):
            spectrum = jnp.fft.rfft(phi, axis=0)
            return jnp.mean(jnp.abs(spectrum - filter_fn(spectrum)) ** 2)

        return cost


@dataclass(frozen=True)
class PeakAmplitude(Objective):
    """Minimise the peak amplitude of selected parameters.

    Cost: $\\max_{g,k\\in P}|\\phi_{g,k}|$. Lowering it raises the gate-speed
    limit.

    Attributes:
        parameter_labels: Projected-basis labels $P$ (not under
            ``param_transform``).
        parameter_indices: Column indices $P$, the alternative to labels. One of
            the two is required.
    """

    parameter_labels: tuple[str, ...] | None = None
    parameter_indices: tuple[int, ...] | None = None

    name = "peak_amplitude"

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_labels", _as_tuple(self.parameter_labels))
        object.__setattr__(self, "parameter_indices", _as_tuple(self.parameter_indices))

    def build(self, params: Parameters) -> Callable[[Array], Array]:
        columns = np.array(
            resolve_parameter_indices(
                params, self.parameter_labels, self.parameter_indices
            )
        )

        def cost(phi):
            return jnp.max(jnp.abs(phi[:, columns]))

        return cost


@dataclass(frozen=True)
class PulseLength(Objective):
    """Minimise the total pulse length.

    Cost: $\\sum_g\\sqrt{\\sum_{k\\in P}\\phi_{g,k}^2 + \\|d\\|^2 + \\varepsilon}$,
    where $d$ is the per-segment drift (zero under ``param_transform``).

    Attributes:
        parameter_labels: Projected-basis labels $P$ (not under
            ``param_transform``).
        parameter_indices: Column indices $P$, the alternative to labels. With
            neither, every column is used.
    """

    parameter_labels: tuple[str, ...] | None = None
    parameter_indices: tuple[int, ...] | None = None

    name = "pulse_length"

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_labels", _as_tuple(self.parameter_labels))
        object.__setattr__(self, "parameter_indices", _as_tuple(self.parameter_indices))

    def build(self, params: Parameters) -> Callable[[Array], Array]:
        columns = np.array(
            resolve_parameter_indices(
                params, self.parameter_labels, self.parameter_indices, default_all=True
            )
        )
        drift_sq_norm = 0.0
        if (
            params.param_transform is None
            and params.drift_basis is not None
            and params.drift_parameters is not None
        ):
            drift_sq_norm = float(np.sum(np.array(params.drift_parameters) ** 2))

        def cost(phi):
            selected = phi[:, columns]
            return jnp.sum(
                jnp.sqrt(jnp.sum(selected**2, axis=1) + drift_sq_norm + 1e-30)
            )

        return cost


@dataclass(frozen=True)
class Robustness(Objective):
    """Maximise the worst-case fidelity under uniform parameter offsets.

    Cost: $1 - \\min_{\\boldsymbol\\delta} F\\bigl(U(\\phi + \\sum_{k\\in P}\\delta_k
    e_k)\\bigr)$ over a Cartesian grid of ``num_samples`` evenly spaced values in
    $[-\\Delta, \\Delta]$ per parameter in $P$; each $\\delta_k$ is applied to
    every segment. Costs ``num_samples ** len(P)`` fidelity evaluations.

    Attributes:
        parameter_labels: Projected-basis labels $P$ (not under
            ``param_transform``).
        parameter_indices: Column indices $P$, the alternative to labels. One of
            the two is required.
        delta: Half-width $\\Delta$ of the perturbation box.
        num_samples: Samples per parameter.
    """

    parameter_labels: tuple[str, ...] | None = None
    parameter_indices: tuple[int, ...] | None = None
    delta: float = 0.01
    num_samples: int = 5

    name = "robustness"

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_labels", _as_tuple(self.parameter_labels))
        object.__setattr__(self, "parameter_indices", _as_tuple(self.parameter_indices))

    def build(self, params: Parameters) -> Callable[[Array], Array]:
        columns = resolve_parameter_indices(
            params, self.parameter_labels, self.parameter_indices
        )
        if params.param_transform is not None:
            # Experimental space: every column is a free parameter.
            n_projdrift = params.n_experimental_params
            proj_mask = np.ones(n_projdrift, dtype=bool)
            drift_mask = np.zeros(n_projdrift, dtype=bool)
            drift_params = None
        else:
            n_projdrift = params.proj_drift_basis.lie_algebra_dim
            proj_mask = params.proj_indices_projdrift_basis
            drift_mask = params.drift_indices_projdrift_basis
            drift_params = (
                None if params.drift_basis is None else params.drift_parameters
            )
        fidelity_at = params.manifold.fidelity_at
        delta_t = jnp.asarray(params.delta_t, dtype=jnp.float64)
        samples = jnp.linspace(-self.delta, self.delta, self.num_samples)
        grids = jnp.meshgrid(*([samples] * len(columns)), indexing="ij")
        offsets = jnp.stack([grid.ravel() for grid in grids], axis=-1)

        def free_params(proj):
            n_steps = proj.shape[0]
            free = jnp.zeros((n_steps, n_projdrift), dtype=jnp.complex128)
            free = free.at[:, proj_mask].set(proj.astype(jnp.complex128))
            if drift_params is not None:
                drift = jnp.array(drift_params, dtype=jnp.complex128)
                free = free.at[:, drift_mask].set(jnp.tile(drift, (n_steps, 1)))
            return free

        def cost(phi):
            def fidelity_at_offset(offset):
                perturbation = jnp.zeros_like(phi)
                for k, column in enumerate(columns):
                    perturbation = perturbation.at[:, column].set(offset[k])
                return fidelity_at(free_params(phi + perturbation), delta_t)

            fidelities = jax.vmap(fidelity_at_offset)(offsets)
            return jnp.real(1.0 - jnp.min(fidelities))

        return cost


@dataclass(frozen=True)
class BoundViolation(Objective):
    """Penalise the worst violation of a box constraint.

    Cost: $\\max_{g,k}\\bigl(\\max(\\phi_{g,k} - u_k, 0) + \\max(l_k - \\phi_{g,k},
    0)\\bigr)$ — zero once every parameter is inside its bounds.

    Attributes:
        bounds: In projected space, ``{label: (min, max)}``; under
            ``param_transform``, ``{index: (min, max)}`` (the ``pulse_constraints``
            convention). Unlisted columns are unbounded. Stored as a tuple of
            pairs, so the spec stays immutable.
    """

    bounds: tuple

    name = "bound_violation"

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", tuple(dict(self.bounds).items()))

    def build(self, params: Parameters) -> Callable[[Array], Array]:
        lower, upper = resolve_bounds(params, self.bounds)

        def cost(phi):
            violation = jnp.clip(phi - upper, min=0.0) + jnp.clip(lower - phi, min=0.0)
            return jnp.max(violation)

        return cost


@dataclass(frozen=True)
class BoundMidpoint(Objective):
    """Pull bounded parameters toward the centre of their box.

    Cost: $\\sum_{g,\\,k \\text{ bounded}} \\bigl((\\phi_{g,k} - m_k)/\\hat r_k
    \\bigr)^2$, with midpoint $m_k = (l_k + u_k)/2$ and range $r_k = u_k - l_k$
    normalised by the largest, $\\hat r_k = r_k/\\max_j r_j$. Only columns with
    both bounds finite take part; a pinned column ($l_k = u_k$) uses
    $\\hat r_k = 1$.

    Attributes:
        bounds: As for :class:`BoundViolation`.
    """

    bounds: tuple

    name = "bound_midpoint"

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", tuple(dict(self.bounds).items()))

    def build(self, params: Parameters) -> Callable[[Array], Array]:
        lower, upper = resolve_bounds(params, self.bounds)
        bounded = np.isfinite(lower) & np.isfinite(upper)
        columns = np.flatnonzero(bounded)
        midpoint = (lower[bounded] + upper[bounded]) / 2.0
        span = upper[bounded] - lower[bounded]
        largest = span.max(initial=0.0)
        scale = np.where(span > 0, span / largest, 1.0) if largest > 0 else span + 1.0

        def cost(phi):
            return jnp.sum(((phi[:, columns] - midpoint) / scale) ** 2)

        return cost


@dataclass(frozen=True)
class CustomObjective(Objective):
    """Wrap a user cost function as an objective.

    Attributes:
        fn: ``fn(phi) -> scalar`` on the real ``(N_g, K_P)`` projected pulse;
            must be traceable and differentiable almost everywhere.
        name: Key the term is reported under in ``gecko.costs``.
    """

    fn: Callable[[Array], Array]
    name: str = "custom"

    def build(self, params: Parameters) -> Callable[[Array], Array]:
        return self.fn
