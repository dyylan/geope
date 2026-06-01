from __future__ import annotations

from typing import Callable

import numpy as np

from .lie import Basis
from .utils import (
    construct_restricted_pauli_basis,
    filter_basis_by_control,
    prepare_random_parameters,
    merge_constraints,
)


class Parameters:
    """Central state object for the Basis -> Parameters -> Optimizer pipeline.

    Holds the system description (basis, control/drift Hamiltonians, target),
    optimisation config (constraints, bounds, ``param_transform``), and the
    mutable history (parameters, fidelities) that is written back by
    optimisers such as `Geope`.

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
        init_parameters: Initial parameter ``np.ndarray``.
        drift_parameters: Drift parameter ``np.ndarray``, or ``None``.
        seed: Optional random seed.
        init_spread: Half-width of the uniform initial-parameter sampling.
        parameters: Mutable list of parameter arrays appended by an optimiser.
        fidelities: Mutable list of fidelity values appended by an optimiser.
        infidelities: Mutable list of infidelity values.
        step_sizes: Mutable list of optimiser step sizes.
        steps: Mutable list of step counters.
    """

    def __init__(self,
                 basis: Basis | None = None,
                 control: dict | None = None,
                 drift: dict | None = None,
                 init_values: dict | np.ndarray | None = None,
                 drift_values: dict | np.ndarray | None = None,
                 target: np.ndarray | None = None,
                 piecewise_steps: int = 1,
                 fixed_drift: bool = True,
                 constraints: list | None = None,
                 pulse_constraints: dict | list | None = None,
                 bounds: dict | None = None,
                 init_spread: float = 0.1,
                 seed: int | None = None,
                 param_transform: Callable | None = None,
                 n_experimental_params: int | None = None,
                 projective: bool = True) -> None:
        """Initialise a Parameters bundle.

        Args:
            basis: Full ``Basis``. Required when constructing without
                an explicit external basis. If ``None``, a default
                two-qubit Pauli basis is built.
            control: Dict of allowed controllable interactions, e.g.
                ``{1: ['x', 'y'], (1, 2): ['xx']}``.
            drift: Dict of fixed drift interactions, same format.
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
            pulse_constraints: Optional pulse-shape constraints config
                (forwarded to ``Geope``).
            bounds: Optional dict mapping interaction label to
                ``(min, max)`` bound tuples.
            init_spread: Half-width of uniform initialisation. Defaults
                to 0.1.
            seed: Optional random seed.
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
        if control is not None:
            if basis.dim != 2 ** basis.n:
                self.projected_basis = filter_basis_by_control(basis, control)
            else:
                self.projected_basis = construct_restricted_pauli_basis(basis.n, control)
        else:
            self.projected_basis = basis

        # --- Drift basis ---
        if drift is not None:
            if basis.dim != 2 ** basis.n:
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
        self.seed = seed
        self.init_spread = init_spread
        self.param_transform = param_transform
        self.projective = projective
        self.n_experimental_params = (n_experimental_params
                                      if n_experimental_params is not None
                                      else self.projected_basis.lie_algebra_dim)

        # --- Constraints ---
        self.constraint_arrays = None
        self.constraint_expander = None
        if constraints is not None:
            constraint_arrays = []
            for c in constraints:
                if isinstance(c, dict):
                    constraint_arrays.append(
                        self.projected_basis.generate_parameter_list(c))
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

        # --- Init parameters ---
        proj_indices = np.array(self.projected_basis.overlap(self.basis), dtype=bool)

        if init_values is not None:
            if isinstance(init_values, dict):
                param_list = self.projected_basis.generate_parameter_list(init_values)
                init_params = np.zeros(self.basis.lie_algebra_dim)
                init_params[proj_indices] = param_list
                self.init_parameters = np.array([init_params] * piecewise_steps)
            else:
                self.init_parameters = np.array(init_values)
        else:
            self.init_parameters = np.array([
                prepare_random_parameters(proj_indices,
                                          expander=self.constraint_expander,
                                          spread=init_spread,
                                          seed=seed)
                for _ in range(piecewise_steps)])

        # --- Drift parameters ---
        if drift_values is not None and self.drift_basis is not None:
            if isinstance(drift_values, dict):
                self.drift_parameters = np.array(
                    self.drift_basis.generate_parameter_list(drift_values))
            else:
                self.drift_parameters = np.array(drift_values)
        elif self.drift_basis is not None:
            self.drift_parameters = np.ones(self.drift_basis.lie_algebra_dim)
        else:
            self.drift_parameters = None

        # --- Mutable history (written by optimisers) ---
        self.parameters: list = []
        self.fidelities: list = []
        self.infidelities: list = []
        self.step_sizes: list = []
        self.steps: list = []

    @property
    def best_fidelity(self) -> float | None:
        """Maximum recorded fidelity, or ``None`` if no run has happened."""
        if not self.fidelities:
            return None
        return max(self.fidelities)

    @property
    def best_parameters(self) -> np.ndarray | None:
        """Parameter array at the step of maximum fidelity, or ``None``."""
        if not self.fidelities:
            return None
        idx = int(np.argmax(self.fidelities))
        return self.parameters[idx]

    @property
    def best_basis_coefficients(self) -> np.ndarray | None:
        """Best parameters mapped through ``param_transform`` if set.

        Returns the induced basis coefficients corresponding to
        ``self.best_parameters``. If ``param_transform`` is ``None``
        this is just the best parameters.
        """
        bp = self.best_parameters
        if bp is None:
            return None
        if self.param_transform is not None:
            import jax
            return np.array(jax.vmap(self.param_transform)(bp))
        return bp

    def to_dict(self) -> dict:
        """Export the best basis coefficients as a control-style dict.

        Returns a dict keyed by qubit index (or qubit-index tuple) whose
        values are dicts mapping lower-case interaction labels to real
        coefficient values.
        """
        coeffs = self.best_basis_coefficients
        if coeffs is None:
            return {}
        proj_indices = np.array(self.projected_basis.overlap(self.basis), dtype=bool)
        proj_coeffs = coeffs[0][proj_indices] if coeffs.ndim > 1 else coeffs[proj_indices]

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

    def reset_history(self) -> None:
        """Clear all mutable run history."""
        self.parameters = []
        self.fidelities = []
        self.infidelities = []
        self.step_sizes = []
        self.steps = []
