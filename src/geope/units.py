r"""Physical units for GEOPE: a tag at the boundary, nothing inside.

GEOPE's numerics are dimensionless. Every optimiser, chart factory and JAX
kernel consumes and produces bare floats and arrays, and keeps doing so. This
module adds a thin layer *around* them:

- A `Unit` is a name, a dimension and a factor to that dimension's canonical
  unit. Multiplying a number or array by one builds a `Quantity`:
  ``25 * ns``, ``5 * MHz``, ``np.linspace(0, 1, 20) * MHz``.
- A `Quantity` is a value tagged with a unit. It converts with ``.to(unit)``
  and does nothing else: no arithmetic, no derived dimensions. It can never
  reach jitted code — ``np.asarray(quantity)`` and ``jnp.asarray(quantity)``
  raise `UnitsError` rather than silently absorbing it.
- A `Conversion` maps physical quantities to GEOPE's internal dimensionless
  values and back. `Parameters` builds one when ``delta_t`` is a time
  quantity and exposes it as ``params.conversion``; it is never constructed
  by users.

**Two dimensions only.** GEOPE writes $H = \hbar \sum_k c_k G_k$ and
$U_g = \exp(-i\,\Delta T\,H_g/\hbar)$, so the exponent $\Delta T\,c$ is an
angle: $c$ is an *amplitude* — an angular frequency, canonical ``rad/s`` — and
$\Delta T$ a *time*, canonical ``s``. An energy $E$ is the same quantity as
the angular frequency $E/\hbar$, and a cycle frequency $f$ is $2\pi f$, so
angular-frequency, cycle-frequency and energy units form one equivalence
class and convert freely: ``(1.0 * ueV).to(MHz)`` is $E/h = 241.799$ MHz.
Hz-family units are *cyclic* and carry the $2\pi$; rad/s-family units are
angular and carry none; energies carry $1/\hbar$. A bare float cannot say
which was meant; a quantity can.

**The one factor the tag cannot catch is generator normalisation.** The
Pauli generators are unnormalised (``X = [[0, 1], [1, 0]]``,
$\mathrm{Tr}(G_j G_k) = d\,\delta_{jk}$), so $H = \hbar\,c\,X$ rotates the
Bloch vector at angular frequency $2c$: a coefficient of 5 MHz on ``X`` is a
10 MHz Rabi frequency, and a $\pi$ pulse takes
$1/(4 \cdot 5\,\mathrm{MHz}) = 50$ ns. The convention is that *the
coefficient multiplies $G_k$ as the basis defines it*. A custom basis
(``projected_basis=``) carries whatever normalisation it was built with.

**The scale.** With ``delta_t`` a quantity, `Parameters` nondimensionalises
with the reference angular frequency $\omega_0 = 1/\Delta T$, so the
internal segment duration is exactly ``1.0`` and each internal coefficient is
the rotation angle per segment — numerically the same problem as the default
``delta_t = 1.0``. The scale is fixed at construction; later ``delta_t``
assignments and `Gecko` subdivision change the internal duration, not the
scale.

**Strict when on, invisible when off.** With a bare ``delta_t`` nothing is
checked and no unit objects are built. With a quantity ``delta_t`` every
physical input must be a quantity of the right dimension, and each misuse
raises `UnitsError` naming the argument. A bare number is always internal; a
physical number always carries its unit.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np
import scipy.constants

_TWO_PI = 2.0 * np.pi
_HBAR = scipy.constants.hbar
_E = scipy.constants.e

AMPLITUDE = "amplitude"
TIME = "time"


class UnitsError(ValueError):
    """Raised on any unit misuse.

    Wrong dimension, a bare number where a quantity is required, a quantity
    while units are off, an unknown unit name, a missing declaration, or a
    quantity reaching numpy or JAX.
    """


# ---------------------------------------------------------------------------
# Unit and Quantity
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Unit:
    """A named unit: a dimension and a factor to the dimension's canonical unit.

    Multiplying a number or array by a `Unit` builds a `Quantity`::

        25 * ns  # Quantity(25.0, ns)
        (-20, 20) * MHz  # Quantity(array([-20., 20.]), MHz)

    Attributes:
        name: Display name, e.g. ``"MHz"`` or ``"rad/ns"``.
        dimension: ``"amplitude"`` or ``"time"``.
        factor: Multiplier to the dimension's canonical unit (``rad/s`` or
            ``s``): one of this unit is ``factor`` canonical units.
    """

    name: str
    dimension: str
    factor: float

    # numpy defers ``ndarray * Unit`` to ``Unit.__rmul__`` instead of
    # broadcasting the Unit into an object array.
    __array_ufunc__ = None

    def __rmul__(self, value: Any) -> Quantity:
        return Quantity(value, self)

    __mul__ = __rmul__

    def __repr__(self) -> str:
        return f"Unit({self.name!r})"

    def __str__(self) -> str:
        return self.name


@dataclasses.dataclass(frozen=True, eq=False)
class Quantity:
    """A value tagged with a unit: a scalar or an array, plus a `Unit`.

    A tag, not an algebra: `to` is the only operation. Equality and hashing
    are disabled because they fail on array values; compare through
    ``.to(unit).value``.

    A `Quantity` never crosses into jitted code. Its value is coerced with
    ``np.asarray`` at construction (so one cannot be built from a tracer), and
    ``__array__`` raises, so ``np.asarray(quantity)`` and
    ``jnp.asarray(quantity)`` fail loudly instead of yielding an object array.

    Attributes:
        value: The numeric value as a float64 ``np.ndarray`` (0-d for scalars).
        unit: The `Unit`. May be given by name (``"MHz"``, ``"rad/ns"``,
            ``"ueV"``) at construction.
    """

    value: np.ndarray
    unit: Unit

    __array_ufunc__ = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit", resolve_unit(self.unit))
        object.__setattr__(self, "value", _as_real_array(self.value, self.unit))

    def to(self, unit: Unit | str) -> Quantity:
        """Convert to another unit of the same dimension.

        For the ``"amplitude"`` dimension this spans angular frequency, cycle
        frequency and energy: ``(1.0 * ueV).to(MHz)`` is ``241.799 MHz``.

        Args:
            unit: Target `Unit` or registered unit name.

        Returns:
            A new `Quantity` in ``unit``.

        Raises:
            UnitsError: If ``unit`` is of a different dimension.
        """
        unit = resolve_unit(unit)
        if unit.dimension != self.unit.dimension:
            raise UnitsError(
                f"Cannot convert {self!r} ({self.unit.dimension}) to "
                f"{unit.name!r} ({unit.dimension}): time and amplitude are "
                "incommensurable."
            )
        return Quantity(self.value * (self.unit.factor / unit.factor), unit)

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        raise UnitsError(
            f"A Quantity ({self!r}) cannot be used as a bare array. Convert it "
            "explicitly with `.to(unit).value`, or pass it to a Parameters "
            "whose `delta_t` is a time quantity so it is converted for you."
        )

    def __repr__(self) -> str:
        if self.value.ndim == 0:
            return f"{float(self.value)} {self.unit.name}"
        return f"{np.array2string(self.value)} {self.unit.name}"


def _as_real_array(value: Any, unit: Unit) -> np.ndarray:
    """Coerce ``value`` to a float64 array, rejecting anything non-numeric."""
    if isinstance(value, Quantity):
        raise UnitsError(
            f"{value!r} is already a quantity; quantities do not multiply "
            f"(tried to apply {unit.name!r})."
        )
    try:
        arr = np.asarray(value)
    except TypeError as err:  # a JAX tracer, among others
        raise UnitsError(
            f"Cannot build a {unit.name!r} quantity from {type(value).__name__}: "
            "units are boundary code and never enter jitted functions."
        ) from err
    if arr.dtype.kind == "c":
        if np.any(arr.imag != 0):
            raise UnitsError(f"A {unit.name!r} quantity must be real; got {arr!r}.")
        arr = arr.real
    if arr.dtype.kind not in "iuf":
        raise UnitsError(
            f"A {unit.name!r} quantity needs a number or a numeric array; got "
            f"{type(value).__name__} with dtype {arr.dtype}."
        )
    return np.array(arr, dtype=np.float64)


# ---------------------------------------------------------------------------
# The registry: library-defined units over SI prefixes
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Unit] = {}


def _register(name: str, dimension: str, factor: float) -> Unit:
    unit = Unit(name, dimension, float(factor))
    _REGISTRY[name] = unit
    return unit


def _normalise_name(name: str) -> str:
    # Both micro signs (U+00B5 MICRO SIGN, U+03BC GREEK SMALL LETTER MU)
    # spell "u"; identifier spellings such as ``rad_per_ns`` map to ``rad/ns``.
    return name.replace("µ", "u").replace("μ", "u").replace("_per_", "/")


def resolve_unit(unit: Unit | str) -> Unit:
    """Return ``unit`` as a `Unit`, looking a name up in the registry.

    Args:
        unit: A `Unit`, or a registered name such as ``"MHz"``, ``"rad/ns"``
            or ``"ueV"`` (``"µeV"`` and ``"rad_per_ns"`` are accepted too).

    Returns:
        The `Unit`.

    Raises:
        UnitsError: If the name is unknown or ``unit`` is neither.
    """
    if isinstance(unit, Unit):
        return unit
    if isinstance(unit, str):
        try:
            return _REGISTRY[_normalise_name(unit)]
        except KeyError:
            raise UnitsError(
                f"Unknown unit {unit!r}. Known units: {sorted(_REGISTRY)}."
            ) from None
    raise UnitsError(f"Expected a Unit or a unit name, got {type(unit).__name__}.")


# Amplitude (canonical rad/s). Angular frequency: 1 / time-prefix.
rad_per_s = _register("rad/s", AMPLITUDE, 1.0)
rad_per_ms = _register("rad/ms", AMPLITUDE, 1e3)
rad_per_us = _register("rad/us", AMPLITUDE, 1e6)
rad_per_ns = _register("rad/ns", AMPLITUDE, 1e9)
rad_per_ps = _register("rad/ps", AMPLITUDE, 1e12)
rad_per_fs = _register("rad/fs", AMPLITUDE, 1e15)
# Cycle frequency: 2 pi * prefix.
Hz = _register("Hz", AMPLITUDE, _TWO_PI)
kHz = _register("kHz", AMPLITUDE, _TWO_PI * 1e3)
MHz = _register("MHz", AMPLITUDE, _TWO_PI * 1e6)
GHz = _register("GHz", AMPLITUDE, _TWO_PI * 1e9)
THz = _register("THz", AMPLITUDE, _TWO_PI * 1e12)
# Energy: prefix * e / hbar (J: 1 / hbar).
J = _register("J", AMPLITUDE, 1.0 / _HBAR)
eV = _register("eV", AMPLITUDE, _E / _HBAR)
meV = _register("meV", AMPLITUDE, 1e-3 * _E / _HBAR)
ueV = _register("ueV", AMPLITUDE, 1e-6 * _E / _HBAR)
neV = _register("neV", AMPLITUDE, 1e-9 * _E / _HBAR)
# Time (canonical s).
s = _register("s", TIME, 1.0)
ms = _register("ms", TIME, 1e-3)
us = _register("us", TIME, 1e-6)
ns = _register("ns", TIME, 1e-9)
ps = _register("ps", TIME, 1e-12)
fs = _register("fs", TIME, 1e-15)


# ---------------------------------------------------------------------------
# Structure walking
# ---------------------------------------------------------------------------


def contains_quantity(value: Any) -> bool:
    """Whether ``value`` is a `Quantity` or a dict/list/tuple holding one."""
    if isinstance(value, Quantity):
        return True
    if isinstance(value, Mapping):
        return any(contains_quantity(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_quantity(v) for v in value)
    return False


def reject_quantity(value: Any, name: str) -> None:
    """Raise `UnitsError` if ``value`` carries a quantity while units are off.

    Args:
        value: The argument to check (walked through dicts, lists, tuples).
        name: The argument's name, for the message.

    Raises:
        UnitsError: If a `Quantity` is found.
    """
    if contains_quantity(value):
        raise UnitsError(
            f"`{name}` carries a physical unit, but units are on only when "
            "`delta_t` is a time quantity, e.g. `Parameters(..., "
            "delta_t=25 * ns)`. Either pass `delta_t` as a quantity or give "
            f"`{name}` as a bare (internal) number."
        )


def convert_if_quantity(
    conversion: Conversion | None, value: Any, dimension: str, name: str
) -> Any:
    """Convert ``value`` if it carries a quantity; pass a bare value through.

    For the entry points the optimiser loop also calls with internal arrays
    (`Geope.init`, `Geope.add_parameters`, `Grape.init`): a user may hand them
    a quantity instead, and it is converted; a bare array is internal.

    Args:
        conversion: ``params.conversion`` — ``None`` when units are off.
        value: The argument.
        dimension: ``"amplitude"`` or ``"time"``.
        name: The argument's name, for error messages.

    Returns:
        ``value`` converted to internal values if it carried a quantity,
        otherwise ``value`` unchanged.

    Raises:
        UnitsError: If ``value`` carries a quantity while units are off.
    """
    if not contains_quantity(value):
        return value
    if conversion is None:
        reject_quantity(value, name)
    return conversion.to_internal(value, dimension, name)


# ---------------------------------------------------------------------------
# Conversion: the per-problem boundary
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Conversion:
    """Per-problem conversion between physical quantities and internal values.

    Built by `Parameters` when ``delta_t`` is a time quantity and read back as
    ``params.conversion`` by `Gecko`, `Geope` and `Grape`. Not constructed by
    users.

    The scale is held as two `Unit` objects, so every conversion is a
    `Quantity.to` call: this class has no formulas of its own.

    Attributes:
        internal_amplitude: The internal amplitude unit, $\\omega_0$ rad/s
            with $\\omega_0 = 1/\\Delta T$.
        internal_time: The internal time unit, $1/\\omega_0$ s, so that the
            internal ``delta_t`` is exactly ``1.0`` at construction.
        time_unit: Readout unit for times — the unit ``delta_t`` was given in.
        amplitude_unit: Readout unit for amplitudes — the angular partner of
            ``time_unit`` (``rad/ns`` for ``ns``).
        transform_unit: The unit of ``param_transform``'s output, or ``None``.
    """

    internal_amplitude: Unit
    internal_time: Unit
    time_unit: Unit
    amplitude_unit: Unit
    transform_unit: Unit | None

    @classmethod
    def from_delta_t(
        cls, delta_t: Quantity, transform_unit: Unit | str | None = None
    ) -> Conversion:
        """Build the conversion for a problem whose segment duration is ``delta_t``.

        Args:
            delta_t: The segment duration, a scalar time quantity.
            transform_unit: Unit of ``param_transform``'s output (amplitude),
                or ``None`` when there is no transform.

        Returns:
            The `Conversion`.

        Raises:
            UnitsError: If ``delta_t`` is not a scalar time quantity, or
                ``transform_unit`` is not an amplitude unit.
        """
        if not isinstance(delta_t, Quantity) or delta_t.unit.dimension != TIME:
            raise UnitsError(
                "`delta_t` must be a time quantity, e.g. `delta_t=25 * ns`; got "
                f"{delta_t!r}."
            )
        if delta_t.value.ndim != 0:
            raise UnitsError(f"`delta_t` must be a scalar; got {delta_t!r}.")
        seconds = float(delta_t.to(s).value)
        if not seconds > 0:
            raise ValueError(f"delta_t must be positive, got {delta_t!r}.")
        omega_0 = 1.0 / seconds
        time_unit = delta_t.unit
        amplitude_unit = _REGISTRY.get(f"rad/{time_unit.name}", rad_per_s)
        if transform_unit is not None:
            transform_unit = resolve_unit(transform_unit)
            if transform_unit.dimension != AMPLITUDE:
                raise UnitsError(
                    "`transform_unit` must be an amplitude unit (e.g. MHz, "
                    f"rad_per_ns, ueV); got {transform_unit.name!r} "
                    f"({transform_unit.dimension})."
                )
        return cls(
            internal_amplitude=Unit("internal amplitude", AMPLITUDE, omega_0),
            internal_time=Unit("internal time", TIME, seconds),
            time_unit=time_unit,
            amplitude_unit=amplitude_unit,
            transform_unit=transform_unit,
        )

    def _internal_unit(self, dimension: str) -> Unit:
        if dimension == AMPLITUDE:
            return self.internal_amplitude
        if dimension == TIME:
            return self.internal_time
        raise UnitsError(f"Unknown dimension {dimension!r}.")

    def _display_unit(self, dimension: str) -> Unit:
        if dimension == AMPLITUDE:
            return self.amplitude_unit
        if dimension == TIME:
            return self.time_unit
        raise UnitsError(f"Unknown dimension {dimension!r}.")

    def to_internal(self, value: Any, dimension: str, name: str) -> Any:
        """Convert a physical input to internal values: the strict-mode gate.

        Walks the container shapes inputs come in — a single quantity, a
        ``control``-format dict with quantity leaves, a ``(lo, hi)`` pair, a
        list — and returns the same structure with plain floats and arrays.

        Args:
            value: A `Quantity`, or a dict / list / tuple whose leaves are
                quantities.
            dimension: ``"amplitude"`` or ``"time"``.
            name: The argument's name, used in error messages.

        Returns:
            ``value`` with every quantity replaced by its internal value: a
            ``float`` for scalars, an ``np.ndarray`` otherwise.

        Raises:
            UnitsError: On a bare number, or a quantity of the wrong dimension.
        """
        example = "5 * MHz" if dimension == AMPLITUDE else "25 * ns"
        if isinstance(value, Quantity):
            if value.unit.dimension != dimension:
                raise UnitsError(
                    f"`{name}` must be a {dimension} quantity (e.g. {example}); "
                    f"got {value!r}, a {value.unit.dimension}."
                )
            internal = value.to(self._internal_unit(dimension)).value
            return float(internal) if internal.ndim == 0 else internal
        if isinstance(value, Mapping):
            return {k: self.to_internal(v, dimension, name) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self.to_internal(v, dimension, name) for v in value)
        raise UnitsError(
            f"`{name}` must be a {dimension} quantity (e.g. {example}) because "
            "units are on (`delta_t` is a time quantity); got a bare "
            f"{type(value).__name__}. A bare number is always internal — "
            "multiply it by its unit."
        )

    def to_physical(self, x: Any, dimension: str) -> Quantity:
        """Convert an internal value to a physical readout quantity.

        Args:
            x: An internal float or array.
            dimension: ``"amplitude"`` or ``"time"``.

        Returns:
            A `Quantity` in the readout unit for ``dimension``
            (``time_unit`` or ``amplitude_unit``).
        """
        return (x * self._internal_unit(dimension)).to(self._display_unit(dimension))

    def transform_factor(self) -> float:
        """The multiplier taking ``param_transform``'s output to internal values.

        Returns:
            ``float((1.0 * transform_unit).to(internal_amplitude).value)``.

        Raises:
            UnitsError: If no ``transform_unit`` was declared.
        """
        if self.transform_unit is None:
            raise UnitsError("No `transform_unit` was declared.")
        return float((1.0 * self.transform_unit).to(self.internal_amplitude).value)
