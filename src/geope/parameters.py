from __future__ import annotations

from functools import cached_property
from typing import Callable

import jax
import numpy as np

from .lie import Basis
from .lie.pauli_projector import get_project_omegas_fn, get_project_omegas_fn_otf
from .engine import (
    get_compute_matrices_params_list_fn,
    get_fidelity_fn,
    get_infidelity_fn,
    get_fidelity_full_fn,
    get_infidelity_full_fn,
    get_geodesic_hamiltonian_fn,
    get_jacobian_fn,
    get_split_jacobian_fn,
    get_gammas_and_omegas_fn,
    get_hessian_fn,
    get_hessian_manual_fn,
    wrap_compute_U_param_transform,
)
from .utils import (
    construct_restricted_pauli_basis,
    filter_basis_by_control,
    control_to_indices,
    prepare_random_parameters,
    merge_constraints,
)


class Parameters:
    """Central state object for the Basis -> Parameters -> Optimizer pipeline.

    Holds the system description (basis, control/drift Hamiltonians, target),
    optimisation config (constraints, bounds, ``param_transform``), and the
    *live* optimisation state (``parameters``, ``fidelity``) that an optimiser
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
        projective: Whether to optimise with the projective (SU)
            fidelity (``True``) or the phase-sensitive (U) fidelity.
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
        projective: bool = True,
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
            projective: If ``True`` (default), use the projective
                (SU) fidelity. If ``False``, use phase-sensitive
                (U) fidelity.
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
        self.projective = projective
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
        self.fidelity = None

    @property
    def infidelity(self) -> float | None:
        """``1 - fidelity``, or ``None`` before a run has computed it."""
        return None if self.fidelity is None else 1 - self.fidelity

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
    # functions of ``basis`` / ``projected_basis`` / ``drift_basis`` (none of
    # which change during a run), so they are cached on first access. They are
    # the single source of truth previously computed in ``Engine.__init__``;
    # the optimisers read them off the shared ``Parameters`` object.

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

    # --- Derived optimisation functions ---------------------------------
    # The un-jitted GEOPE/GRAPE callables are pure functions of this object's
    # (immutable-after-construction) configuration, so each is built lazily by
    # a compact factory and cached here on first access. The optimisers read
    # them directly off the (shared) ``Parameters`` object — there is no shared
    # engine — and only the functions actually used by a given optimiser are
    # ever built. Caching keeps the callable identity stable, so JAX reuses the
    # compiled traces across a ``Geope`` and a ``Gecko`` sharing this object.
    # No eager JIT happens here; compilation occurs once when the enclosing
    # ``@jax.jit`` update step is first traced in ``optimize()``.

    @cached_property
    def compute_U_fn(self) -> Callable:
        """Unitary-from-parameters function (wrapped when ``param_transform`` set)."""
        base = get_compute_matrices_params_list_fn(self.proj_drift_basis.basis)
        if self.param_transform is None:
            return base
        return wrap_compute_U_param_transform(self, base)

    @cached_property
    def fid_U_fn(self) -> Callable:
        """Fidelity-of-unitary function bound to ``target``."""
        if self.projective:
            return get_fidelity_fn(self.target)
        return get_fidelity_full_fn(self.target)

    @cached_property
    def infid_U_fn(self) -> Callable:
        """Infidelity-of-unitary function bound to ``target``."""
        if self.projective:
            return get_infidelity_fn(self.target)
        return get_infidelity_full_fn(self.target)

    @cached_property
    def infid_fn(self) -> Callable:
        """Infidelity as a function of the free parameters."""
        compute_U = self.compute_U_fn
        infid_U = self.infid_U_fn
        return lambda x: infid_U(compute_U(x))

    @cached_property
    def grad_fn(self) -> Callable:
        """Value-and-gradient of the infidelity (used by GRAPE)."""
        return jax.value_and_grad(self.infid_fn)

    @cached_property
    def hess_fn_autodiff(self) -> Callable:
        """Hessian of the infidelity (used by GRAPE)."""
        return get_hessian_fn(self.infid_fn)

    @cached_property
    def hess_fn(self) -> Callable:
        """Manual (Goodwin–Kuprov) Hessian of the infidelity.

        Analytic drop-in for `hess_fn`, built from the manual propagator
        derivatives instead of autodiff. Only available without a
        ``param_transform`` (the manual derivatives operate directly on the
        proj+drift basis coefficients).
        """
        if self.param_transform is not None: # manual path not available.
            return hess_fn_autodiff()
        return get_hessian_manual_fn(
            self.proj_drift_basis.basis, self.target, projective=self.projective
        )

    @cached_property
    def jac_fn(self) -> Callable:
        """Jacobian of the unitary w.r.t. the free parameters.

        Holomorphic autodiff in projected-basis mode; a real/imag-split
        Jacobian when ``param_transform`` is set (so the imaginary part is not
        discarded through the real-valued user transform).
        """
        if self.param_transform is None:
            return get_jacobian_fn(self.compute_U_fn)
        return get_split_jacobian_fn(self.compute_U_fn)

    @cached_property
    def geo_fn(self) -> Callable:
        """Geodesic-Hamiltonian function bound to ``target`` (used by GEOPE)."""
        return get_geodesic_hamiltonian_fn(self.target, projective=self.projective)

    @cached_property
    def project_omegas_fn(self) -> Callable:
        """Projection of matrices onto the Lie-algebra basis (used by GEOPE)."""
        if self.basis.n > 5:
            return get_project_omegas_fn_otf(self.basis, batch_size=None)
        return get_project_omegas_fn(self.basis)

    @cached_property
    def gammas_and_omegas(self) -> Callable:
        """Combined gammas-and-omegas function for the GEOPE update step.

        In ``param_transform`` (experimental) mode the omega projection has one
        free column per experimental parameter, so the projected-index
        restriction selects all of them.
        """
        if self.param_transform is not None:
            omega_proj_indices = np.ones(self.n_experimental_params, dtype=bool)
        else:
            omega_proj_indices = self.proj_indices_projdrift_basis
        # Mirrors the legacy ``np.any(proj_drift_basis)`` gate (truthy for any
        # non-empty basis, including the param_transform case).
        has_proj_drift = self.proj_drift_basis.lie_algebra_dim > 0
        return get_gammas_and_omegas_fn(
            self.compute_U_fn,
            self.jac_fn,
            self.geo_fn,
            self.project_omegas_fn,
            omega_proj_indices,
            has_proj_drift,
        )

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
