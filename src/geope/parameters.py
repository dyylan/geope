from __future__ import annotations

from functools import cached_property, partial
from typing import Callable

import inspect

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .geometry.lie import Basis

from .geometry import (
    Manifold,
    SpecialUnitaryGroup,
)
from .utils import (
    construct_restricted_pauli_basis,
    filter_basis_by_control,
    control_to_indices,
    prepare_random_parameters,
    merge_constraints,
)


class Parameters:
    """Central state object for the parameters of the system.

    Holds the system description (basis, control/drift Hamiltonians, target),
    optimisation config (constraints, bounds, ``param_transform``), and the
    live optimisation state (``parameters``, ``fidelity``) that an optimiser
    such as `Geope` updates in place.

    Attributes:
        basis: Full ``Basis`` for the system.
        projected_basis: The controllable sub-``Basis``.
        drift_basis: The drift sub-``Basis``, or ``None``.
        target: Target unitary as ``np.ndarray``.
        piecewise_steps: Number of piecewise-constant gate segments.
        fixed_drift: Whether the drift contribution is held fixed.
        control: The control dict used to build ``projected_basis``.
        drift_config: The dict used to build ``drift_basis``.
        pulse_constraints: Optional pulse-shape constraint config.
        param_transform: Optional callable mapping experimental params
            to basis coefficients.
        manifold: The `geope.geometry.Manifold` this problem lives on.
        projective: Whether the geometry is the projective (SU) one.
            Read-only; delegates to ``manifold``.
        n_experimental_params: Length of the experimental-parameter
            vector when ``param_transform`` is set.
        constraint_arrays: List of linear-equality constraint vectors,
            after merging.
        constraint_expander: Expansion matrix that maps free parameters
            into the projected basis under the constraints.
        bounds: Pre-built bounds, or ``None``.
        drift_parameters: Drift parameter ``np.ndarray``, or ``None``.
        seed: Optional random seed.
        init_spread: Half-width of the uniform initial-parameter sampling.
        parameters: Current parameter ``np.ndarray`` (full-basis), seeded to
            the initial guess and updated in place by an optimiser.
        fidelity: Current fidelity value, or ``None`` before a run.
        infidelity: ``1 - fidelity`` (``None`` before a run).
    """

    def __init__(
        self,
        basis: Basis | None = None,
        control: dict | None = None,
        drift: dict | None = None,
        projected_basis: Basis | None = None,
        drift_basis: Basis | None = None,
        init_values: dict | np.ndarray | None = None,
        drift_values: dict | np.ndarray | None = None,
        target: np.ndarray | None = None,
        piecewise_steps: int = 1,
        fixed_drift: bool = True,
        constraints: list | None = None,
        pulse_constraints: dict | list | None = None,
        bounds: dict | None = None,
        init_spread: float = 0.1,
        seed: int | jax.Array | None = None,
        param_transform: Callable | None = None,
        n_experimental_params: int | None = None,
        manifold: Manifold | None = None,
    ) -> None:
        """Initialise a Parameters bundle.

        Args:
            basis: Full ``Basis``. Required when constructing without
                an explicit external basis. If ``None``, a default
                two-qubit Pauli basis is built.
            control: Dict of allowed controllable interactions, e.g.
                ``{1: ['x', 'y'], (1, 2): ['xx']}``. Mutually exclusive
                with ``projected_basis``.
            drift: Dict of fixed drift interactions, same format.
                Mutually exclusive with ``drift_basis``.
            projected_basis: Pre-built projected ``Basis``. Used as an
                escape hatch when the projected subset can't be expressed
                as a ``control`` dict. Mutually exclusive with ``control``.
            drift_basis: Pre-built drift ``Basis``. Mutually exclusive
                with ``drift``.
            init_values: Initial parameter values. May be a dict in the
                same format as ``control``, or an ``np.ndarray``.
            drift_values: Drift parameter values. May be a dict or array.
            target: Target unitary.
            piecewise_steps: Number of piecewise-constant gate segments.
                Defaults to 1.
            fixed_drift: Whether the drift contribution is held fixed.
                Defaults to ``True``.
            constraints: Optional list of linear-equality constraints,
                each either an ``np.ndarray`` of size
                ``projected_basis.lie_algebra_dim`` or a dict in the
                ``control`` format.
            pulse_constraints: Optional pulse-shape constraints whose
                temporal profile is frozen during optimisation. In
                projected space, a control-format dict
                ``{qubit_index_or_tuple: [lowercase op labels]}`` (the
                same format as ``control``), e.g.
                ``{1: ['x'], (1, 2): ['zz']}``. In experimental space
                (``param_transform`` set), a list of integer parameter
                indices. Forwarded to ``Geope``. A dict that names an
                interaction absent from the projected basis raises
                ``ValueError``.
            bounds: Optional dict mapping interaction label to
                ``(min, max)`` bound tuples.
            init_spread: Half-width of uniform initialisation. Defaults
                to 0.1.
            seed: Optional random seed. Defaults to ``jax.random.key(0)``.
            param_transform: Optional callable mapping experimental
                params to basis coefficients. May take
                ``(phi,)`` or ``(phi, step_index)``.
            n_experimental_params: Number of experimental parameters
                when ``param_transform`` is set. Defaults to
                ``projected_basis.lie_algebra_dim``.
            manifold: The `geope.geometry.Manifold` to synthesise on, as a
                constructed instance: `geope.geometry.SpecialUnitaryGroup`
                (the default, ``dim`` taken from ``basis``) for the
                projective fidelity, `geope.geometry.UnitaryGroup` for the
                phase-sensitive one, or a `geope.geometry.StateSphere` /
                `geope.geometry.Stiefel` for state and subspace problems.
                The manifold carries the fidelity, the tangent projection
                and the geodesic with it, so it is the only place that
                choice is made.

        Raises:
            ValueError: If ``control`` and ``projected_basis`` (or ``drift``
                and ``drift_basis``) are both given, or if the control and
                drift bases overlap.
        """
        # --- Basis ---
        if basis is None:
            from .utils import construct_full_pauli_basis

            basis = construct_full_pauli_basis(2)
        self.basis = basis

        # --- Projected (control) basis ---
        if control is not None and projected_basis is not None:
            raise ValueError("Pass either `control` or `projected_basis`, not both.")
        if projected_basis is not None:
            self.projected_basis = projected_basis
        elif control is not None:
            if basis.dim != 2**basis.n:
                self.projected_basis = filter_basis_by_control(basis, control)
            else:
                self.projected_basis = construct_restricted_pauli_basis(
                    basis.n, control
                )
        else:
            self.projected_basis = basis

        # --- Drift basis ---
        if drift is not None and drift_basis is not None:
            raise ValueError("Pass either `drift` or `drift_basis`, not both.")
        if drift_basis is not None:
            self.drift_basis = drift_basis
        elif drift is not None:
            if basis.dim != 2**basis.n:
                self.drift_basis = filter_basis_by_control(basis, drift)
            else:
                self.drift_basis = construct_restricted_pauli_basis(basis.n, drift)
        else:
            self.drift_basis = None

        # Control and drift must address disjoint basis elements.
        if self.drift_basis is not None:
            shared = self.projected_indices & self.drift_indices
            if shared.any():
                shared_labels = [str(l) for l in np.array(self.basis.labels)[shared]]
                raise ValueError(
                    f"Control and drift bases overlap on {shared_labels}; they "
                    "must be disjoint. Drift values are written after control "
                    "values on shared basis elements, which would silently "
                    "overwrite the control coefficient (and zero its gradient "
                    "under `param_transform`).\n\n"
                    "To control a basis element that also carries a constant "
                    "offset, leave it out of the drift basis and add the "
                    "constant through `param_transform`:\n\n"
                    "def param_transform(x):\n"
                    f'    idx = list(basis.labels).index("{shared_labels[0]}")\n'
                    f"    x = x.at[idx].add(offset)  # the {shared_labels[0]} "
                    "drift value\n"
                    "    return x\n\n"
                    "Repeat the two body lines for each shared element listed "
                    "above, then pass `param_transform=param_transform` to "
                    "`Parameters` and drop those elements from the drift "
                    "basis. See the `Parameters` section of docs/user_guide.md "
                    "for a worked example."
                )

        # --- Immutable config ---
        self.target = np.array(target) if target is not None else None
        self.piecewise_steps = piecewise_steps
        self.fixed_drift = fixed_drift
        self.control = control
        self.drift_config = drift
        self.pulse_constraints = pulse_constraints
        # Fail fast on a pulse-shape constraint that names an interaction
        # absent from the projected basis (typo, wrong qubit, etc.). The
        # experimental-space form (a list of integer indices) is left alone.
        if isinstance(pulse_constraints, dict):
            control_to_indices(
                list(self.projected_basis.labels), pulse_constraints, strict=True
            )
        self.seed = seed
        self.init_spread = init_spread
        self.param_transform = param_transform
        # Default manifold is SU(N)
        if manifold is None:
            manifold = SpecialUnitaryGroup(basis.dim)
        # Validate target
        if self.target is not None:
            manifold.validate_point(self.target, "target")
        self.n_experimental_params = (
            n_experimental_params
            if n_experimental_params is not None
            else self.projected_basis.lie_algebra_dim
        )

        # --- Constraints ---
        self.constraint_arrays = None
        self.constraint_expander = None
        if constraints is not None:
            constraint_arrays = []
            for c in constraints:
                if isinstance(c, dict):
                    constraint_arrays.append(
                        self.projected_basis.generate_parameter_list(c)
                    )
                else:
                    constraint_arrays.append(c)
            merged = merge_constraints(constraint_arrays)
            self.constraint_arrays = [np.array(c) for c in merged]

            expander = np.eye(self.projected_basis.lie_algebra_dim)
            del_indices = []
            for c in self.constraint_arrays:
                c_proj_indices = c.astype(bool)
                idx = np.where(c_proj_indices)[0]
                expander[:, idx[0]] = c
                del_indices.append(idx[1:])
            expander = np.delete(expander, del_indices, axis=1)
            expander = expander / expander.max()
            self.constraint_expander = expander

        # --- Bounds ---
        self.bounds = None
        if bounds is not None:
            self.bounds = self.projected_basis.generate_bounds(bounds, piecewise_steps)

        # --- Live state: current parameters, seeded to the initial guess ---
        proj_indices = self.projected_indices

        if init_values is not None:
            if isinstance(init_values, dict):
                param_list = self.projected_basis.generate_parameter_list(init_values)
                init_params = np.zeros(self.basis.lie_algebra_dim)
                init_params[proj_indices] = param_list
                self.parameters = np.array([init_params] * piecewise_steps)
            else:
                self.parameters = np.array(init_values)
        else:
            if isinstance(seed, int):
                key = jax.random.key(seed)
            elif isinstance(seed, jax.Array):
                key = seed
            else:
                key = jax.random.key(0)
            keys = jax.random.split(key, piecewise_steps)
            self.parameters = np.array(
                [
                    prepare_random_parameters(
                        proj_indices,
                        expander=self.constraint_expander,
                        spread=init_spread,
                        key=keys[i],
                    )
                    for i in range(piecewise_steps)
                ]
            )

        # --- Drift parameters ---
        if drift_values is not None and self.drift_basis is not None:
            if isinstance(drift_values, dict):
                self.drift_parameters = np.array(
                    self.drift_basis.generate_parameter_list(drift_values)
                )
            else:
                self.drift_parameters = np.array(drift_values)
        elif self.drift_basis is not None:
            self.drift_parameters = np.ones(self.drift_basis.lie_algebra_dim)
        else:
            self.drift_parameters = None

        # --- Live state: current fidelity (set once a run computes it) ---
        # TODO: Turn this into a settable property.
        self.fidelity = None

        # --- The bound manifold: the one handle every optimiser reads -------
        # Last, because everything `bind` reads is settled by now: the index
        # masks, the drift values and the transform. What a `Parameters` alone
        # knows is exactly the four arguments below — which frame the
        # coefficients resolve against, which generators the pulse drives, which
        # columns the solve may move, and the reparametrisation. The chart and
        # both its differentials are the geometry layer's, so no mathematics
        # crosses this line.
        #
        # Eager rather than a `cached_property`: binding traces nothing (every
        # factory is a partial or a closure), and doing it here means no
        # `jax.jit` can ever be the first thing to touch it.
        self.manifold = manifold.bind(
            target=self.target,
            generators=self.proj_drift_basis,
            frame=self.basis,
            columns=self.proj_indices_projdrift_basis,
            wrap_chart=(
                None
                if param_transform is None
                else partial(wrap_compute_point_param_transform, self)
            ),
        )

    @property
    def infidelity(self) -> float | None:
        """``1 - fidelity``, or ``None`` before a run has computed it."""
        return None if self.fidelity is None else 1 - self.fidelity

    @property
    def projective(self) -> bool:
        """Whether the geometry is the projective (SU) one.

        Delegates to the manifold, which is the single place the choice lives.
        """
        return self.manifold.projective

    def free(self, parameters: np.ndarray | Array | None = None) -> Array:
        """The free (proj+drift) parameter columns, in the pipeline's dtype.

        The single owner of the "which columns does the optimiser move, and in
        what dtype" question that `Geope`, `Gecko` and `Grape` all need. Identity
        in experimental (``param_transform``) mode, where every column is already
        free and the dtype is real; otherwise selects the proj+drift columns of
        the full-basis array and promotes to ``complex128`` for the holomorphic
        Jacobian.

        Args:
            parameters: A full-basis parameter array. Defaults to the live
                ``self.parameters``.

        Returns:
            An ``Array`` of shape ``(piecewise_steps, K_free)``.
        """
        parameters = self.parameters if parameters is None else parameters
        if self.param_transform is not None:
            return jnp.real(jnp.asarray(parameters)).astype(jnp.float64)
        return jnp.asarray(parameters)[:, self.proj_drift_indices].astype(
            jnp.complex128
        )

    @property
    def basis_coefficients(self) -> np.ndarray | None:
        """Current parameters mapped through ``param_transform`` if set.

        Returns the induced basis coefficients corresponding to the
        current ``self.parameters``. If ``param_transform`` is ``None``
        this is just the current parameters.
        """
        if self.param_transform is not None:
            import jax

            return np.array(jax.vmap(self.param_transform)(self.parameters))
        return self.parameters

    # --- Derived algebraic metadata -------------------------------------
    # These index masks and the combined projected+drift basis are pure
    # functions of ``basis`` / ``projected_basis`` / ``drift_basis`` and stay fixed throughout.

    @cached_property
    def projected_indices(self) -> np.ndarray:
        """Boolean mask for the projected basis within the full basis."""
        return np.array(self.projected_basis.overlap(self.basis), dtype=bool)

    @cached_property
    def drift_indices(self) -> np.ndarray:
        """Boolean mask for the drift basis within the full basis.

        All-``False`` when there is no drift basis.
        """
        if self.drift_basis is None:
            return np.full(self.basis.lie_algebra_dim, False)
        return np.array(self.drift_basis.overlap(self.basis), dtype=bool)

    @cached_property
    def proj_drift_indices(self) -> np.ndarray:
        """Combined boolean mask for projected and drift elements."""
        return self.projected_indices + self.drift_indices

    @cached_property
    def proj_drift_basis(self) -> Basis:
        """Combined projected-and-drift ``Basis`` object."""
        mask = self.proj_drift_indices
        return Basis(
            self.basis.basis[mask], labels=list(np.array(self.basis.labels)[mask])
        )

    @cached_property
    def proj_indices_projdrift_basis(self) -> np.ndarray:
        """Projected indices expressed within the combined proj+drift basis."""
        return np.delete(self.projected_indices, ~self.proj_drift_indices)

    @cached_property
    def drift_indices_projdrift_basis(self) -> np.ndarray:
        """Drift indices expressed within the combined proj+drift basis."""
        return np.delete(self.drift_indices, ~self.proj_drift_indices)

    def to_dict(self) -> dict:
        """Export the current basis coefficients as a control-style dict.

        Returns a dict keyed by qubit index (or qubit-index tuple) whose
        values are dicts mapping lower-case interaction labels to real
        coefficient values.
        """
        coeffs = self.basis_coefficients
        if coeffs is None:
            return {}
        proj_indices = self.projected_indices
        proj_coeffs = (
            coeffs[0][proj_indices] if coeffs.ndim > 1 else coeffs[proj_indices]
        )

        result: dict = {}
        for label, value in zip(self.projected_basis.labels, proj_coeffs):
            new_label = ""
            qubits = []
            for i, c in enumerate(label):
                if c != "I":
                    new_label += c.lower()
                    qubits.append(i + 1)
            key = tuple(qubits) if len(qubits) > 1 else qubits[0]
            if key not in result:
                result[key] = {}
            result[key][new_label] = float(np.real(value))
        return result


def wrap_compute_point_param_transform(
    params: "Parameters", raw_compute_point: Callable[[Array], Array]
) -> Callable[[Array], Array]:
    r"""Wrap the chart to honour ``params.param_transform``.

    The user-facing experimental parameters $\phi^{\mathrm{exp}}$ are mapped to
    projected-basis coefficients via ``params.param_transform`` (possibly
    step-dependent), embedded into the proj+drift basis, and combined with the
    drift before the original ``raw_compute_point`` is called — so the whole pipeline
    then runs uniformly on $\phi^{\mathrm{exp}}$.

    It lives here rather than with the chart because it is the one piece of chart
    plumbing that reads a `Parameters`: the index masks, the drift values and the
    transform itself all come off it.

    Returned un-jitted so it fuses into the enclosing ``@jax.jit`` update step
    on first ``optimize()``.

    Args:
        params: The `Parameters` object carrying ``param_transform``.
        raw_compute_point: The projected-basis chart.

    Returns:
        The wrapped experimental-space chart.
    """
    n_exp = params.n_experimental_params
    n_proj_drift = params.proj_drift_basis.lie_algebra_dim
    proj_idx_pd = params.proj_indices_projdrift_basis
    drift_idx_pd = params.drift_indices_projdrift_basis

    # Detect step-dependence: tau(phi) vs tau(phi, step_index)
    _step_dependent = len(inspect.signature(params.param_transform).parameters) >= 2

    # Detect whether transform outputs full-basis or projected-basis coefficients
    _test_out = (
        params.param_transform(jnp.zeros(n_exp), 0)
        if _step_dependent
        else params.param_transform(jnp.zeros(n_exp))
    )
    tf_out_dim = _test_out.shape[0]
    n_proj = params.projected_basis.lie_algebra_dim
    if tf_out_dim != n_proj:
        _extract = jnp.array(
            np.where(np.array(params.projected_basis.overlap(params.basis)))[0]
        )
    else:
        _extract = None

    if params.drift_parameters is not None:
        _drift = jnp.array(params.drift_parameters, dtype=jnp.float64)
    else:
        _drift = None

    def _wrapped_compute_point(
        exp_params,
        _raw=raw_compute_point,
        _tf=params.param_transform,
        _pi=proj_idx_pd,
        _di=drift_idx_pd,
        _npd=n_proj_drift,
        _dr=_drift,
        _ext=_extract,
        _step_dep=_step_dependent,
    ):
        if _step_dep:
            ctrl = jax.vmap(_tf)(exp_params, jnp.arange(exp_params.shape[0]))
        else:
            ctrl = jax.vmap(_tf)(exp_params)
        if _ext is not None:
            ctrl = ctrl[:, _ext]
        # Promote dtype so complex tracing through real intermediates works
        _dtype = jnp.result_type(ctrl.dtype, exp_params.dtype)
        ctrl = ctrl.astype(_dtype)
        full = jnp.zeros((exp_params.shape[0], _npd), dtype=_dtype)
        full = full.at[:, _pi].set(ctrl)
        if _dr is not None:
            full = full.at[:, _di].set(
                jnp.broadcast_to(
                    _dr.astype(_dtype), (exp_params.shape[0], _dr.shape[0])
                )
            )
        return _raw(full)

    return _wrapped_compute_point
