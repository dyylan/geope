from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_enable_x64", True)

from functools import partial
from typing import Callable, TYPE_CHECKING

from .geope import GeopeEngine
from .parameters import Parameters
from .history import History

if TYPE_CHECKING:
    from .geope import Geope


class Gecko:
    """Null-space ("auxiliary cost") optimiser for GEOPE solutions.

    ``Gecko`` post-processes an already-found GEOPE solution: it moves the
    parameters within the Jacobian null space so that fidelity is preserved
    while an auxiliary cost is improved — smoothing, frequency shaping,
    pulse length, gate speed, robustness, or parameter bounds.

    A ``Gecko`` always needs a built ``GeopeEngine``. Constructing one is
    expensive (JIT compilation), so a ``Gecko`` can either build its own
    engine from a `Parameters` object, or borrow a `Geope`'s already-built
    engine:

    - ``Gecko(params=p)`` — build a fresh engine from ``p``.
    - ``Gecko(geope=g)`` — reuse ``g.engine`` and ``g.params`` directly.
    - ``Gecko(params=p, geope=g)`` — reuse ``g.engine`` but first verify it
      is compatible with the separately-supplied ``p`` (raises ``ValueError``
      on mismatch).
    - ``Gecko()`` — raises ``ValueError``.

    Attributes:
        params: The bound `Parameters` object (live optimisation state).
        engine: The `GeopeEngine` (reused from a `Geope` or freshly built).
        history: Optional `History` logger (``None`` unless supplied).
        step_size: Transient last optimisation rate.
        pulse_constraints: Optional pulse-shape constraint config.
        constraint_expander: Linear-equality constraint expander (from params).
        drift_parameters: Drift parameter ``np.ndarray`` (``None`` in
            experimental / ``param_transform`` mode).

    .. note::
        In the ``geope`` and ``params+geope`` reuse modes the engine and
        `Parameters` are **shared** with the source `Geope`. A null-space
        pass with ``piecewise_steps_multiplier > 1`` subdivides the pulse and
        advances ``params.parameters`` / ``params.piecewise_steps`` /
        ``engine.piecewise_steps`` together — so the shared `Geope`'s state
        moves forward too, and a later ``geope.optimize()`` continues from
        the subdivided pulse.
    """

    def __init__(self,
                 params: Parameters | None = None,
                 geope: "Geope | None" = None,
                 history: History | None = None,
                 verbose: bool = False) -> None:
        """Initialise the Gecko optimiser.

        Args:
            params: A `Parameters` instance. Required unless ``geope`` is
                given (a `Geope` already carries its own ``geope.params``).
            geope: An optional `Geope` whose engine (and, when ``params`` is
                omitted, whose `Parameters`) is reused instead of rebuilding.
            history: Optional `History` logger. When supplied, the run
                trajectory is recorded into it.
            verbose: Whether to print progress. Defaults to False.

        Raises:
            ValueError: If neither ``params`` nor ``geope`` is given, or if
                both are given but the engine is incompatible with ``params``.
        """
        if params is None and geope is None:
            raise ValueError(
                "Gecko requires either `params` or `geope` (or both). "
                "Pass a Parameters object, or a Geope instance whose engine "
                "and params should be reused."
            )

        if geope is not None and params is None:
            # Reuse a Geope's engine and its Parameters (self-consistent).
            self.engine = geope.engine
            self.params = geope.params
            self._real_params = geope._real_params
        elif geope is None:
            # Build a fresh engine from the supplied Parameters.
            engine = GeopeEngine(
                target_unitary=params.target,
                full_basis=params.basis,
                projected_basis=params.projected_basis,
                drift_basis=params.drift_basis,
                piecewise_steps=params.piecewise_steps,
                projective=params.projective,
            )
            if params.param_transform is not None:
                engine.wrap_param_transform(params)
            self.engine = engine
            self.params = params
            self._real_params = params.param_transform is not None
        else:
            # Both supplied: reuse the engine but verify it matches params.
            self._assert_engine_compatible(geope.engine, params)
            self.engine = geope.engine
            self.params = params
            self._real_params = params.param_transform is not None

        # Compute a baseline fidelity if the params have never been evaluated
        # (e.g. a fresh Parameters that has not been through Geope.optimize).
        if self.params.fidelity is None:
            _dtype = jnp.float64 if self._real_params else jnp.complex128
            free_params = jnp.array(
                [p[self.engine.proj_drift_indices] for p in self.params.parameters]
            )
            free_params = (
                jnp.real(free_params) if self._real_params else free_params
            ).astype(_dtype)
            self.params.fidelity = self.engine.fid_U_fn(self.engine.compute_U_fn(free_params))

        self.history = history
        if self.history is not None:
            self.history.params = self.params
        self.step_size = 0
        self.verbose = verbose
        self.pulse_constraints = self.params.pulse_constraints
        self.constraint_expander = self.params.constraint_expander
        self.drift_parameters = None if self._real_params else self.params.drift_parameters
        # Bounding-only state, set lazily in bound().
        self.parameter_bounds = None
        self.lower_bounds = None
        self.upper_bounds = None

    @staticmethod
    def _assert_engine_compatible(engine: GeopeEngine, params: Parameters) -> None:
        """Verify a reused engine is compatible with a supplied Parameters.

        Compares the engine's bound target, projective flag, segment count,
        and basis structure (or experimental dimensionality under
        ``param_transform``) against ``params``. Raises naming the first
        mismatch found.

        Args:
            engine: The `GeopeEngine` to validate.
            params: The `Parameters` it must be compatible with.

        Raises:
            ValueError: On the first incompatibility found.
        """
        if bool(engine.projective) != bool(params.projective):
            raise ValueError(
                f"Gecko: engine.projective={engine.projective} does not match "
                f"params.projective={params.projective}.")

        t_engine = np.asarray(engine.target_unitary)
        t_params = np.asarray(params.target)
        if t_engine.shape != t_params.shape or not np.allclose(t_engine, t_params):
            raise ValueError(
                "Gecko: the engine's target unitary does not match params.target.")

        if engine.piecewise_steps != params.piecewise_steps:
            raise ValueError(
                f"Gecko: engine.piecewise_steps={engine.piecewise_steps} does not "
                f"match params.piecewise_steps={params.piecewise_steps}.")

        if params.param_transform is not None:
            # Engine must already be wrapped into experimental space.
            if getattr(engine, "drift_basis", None) is not None:
                raise ValueError(
                    "Gecko: params has a param_transform but the engine was not "
                    "wrapped into experimental space (drift_basis still present).")
            if engine.proj_drift_basis.lie_algebra_dim != params.n_experimental_params:
                raise ValueError(
                    f"Gecko: engine experimental dimension "
                    f"{engine.proj_drift_basis.lie_algebra_dim} does not match "
                    f"params.n_experimental_params={params.n_experimental_params}.")
        else:
            e_labels = engine.projected_basis.labels
            p_labels = params.projected_basis.labels
            if (e_labels is not None and p_labels is not None
                    and list(e_labels) != list(p_labels)):
                raise ValueError(
                    "Gecko: engine.projected_basis labels do not match "
                    "params.projected_basis.")
            if (engine.projected_basis.lie_algebra_dim
                    != params.projected_basis.lie_algebra_dim):
                raise ValueError(
                    "Gecko: engine.projected_basis dimension does not match "
                    "params.projected_basis.")
            e_drift = getattr(engine, "drift_basis", None)
            p_drift = params.drift_basis
            if (e_drift is None) != (p_drift is None):
                raise ValueError(
                    "Gecko: engine.drift_basis presence does not match "
                    "params.drift_basis.")
            if e_drift is not None and p_drift is not None:
                ed_labels = e_drift.labels
                pd_labels = p_drift.labels
                if (ed_labels is not None and pd_labels is not None
                        and list(ed_labels) != list(pd_labels)):
                    raise ValueError(
                        "Gecko: engine.drift_basis labels do not match "
                        "params.drift_basis.")
                if e_drift.lie_algebra_dim != p_drift.lie_algebra_dim:
                    raise ValueError(
                        "Gecko: engine.drift_basis dimension does not match "
                        "params.drift_basis.")

    def smooth(
        self,
        piecewise_steps_multiplier: int = 1,
        smoothing_rate: float = 0.01,
        max_smoothing_steps: int = 100,
        diff_tol: float = 0.1,
    ) -> tuple[bool, int]:
        """Smooth the piecewise-constant pulse by null-space optimisation.

        Minimises the differences between consecutive gate segments
        while remaining in the null space of the Jacobian so that
        fidelity is preserved.

        Args:
            piecewise_steps_multiplier: Factor by which to increase the
                number of gate segments. Defaults to 1.
            smoothing_rate: Learning rate for the null-space update.
                Defaults to 0.01.
            max_smoothing_steps: Maximum smoothing iterations.
                Defaults to 100.
            diff_tol: Convergence tolerance on the smoothing cost.
                Defaults to 0.1.

        Returns:
            A tuple ``(success, iters)`` where `success` is ``True`` if
            `diff_tol` was reached.
        """
        success, iters = self._null_space_optimisation(piecewise_smoothing,
                                                        piecewise_steps_multiplier=piecewise_steps_multiplier,
                                                        max_steps=max_smoothing_steps,
                                                        rate=smoothing_rate,
                                                        diff_tol=diff_tol,
                                                        label="Smoothing")
        return success, iters

    def smooth_frequency(
        self,
        piecewise_steps_multiplier: int = 1,
        smoothing_rate: float = 0.01,
        max_smoothing_steps: int = 100,
        diff_tol: float = 0.1,
    ) -> tuple[bool, int]:
        """Suppress high-frequency spectral power in the pulse.

        Args:
            piecewise_steps_multiplier: Factor by which to subdivide
                gate segments. Defaults to 1.
            smoothing_rate: Learning rate. Defaults to 0.01.
            max_smoothing_steps: Maximum iterations. Defaults to 100.
            diff_tol: Convergence tolerance on the spectral cost.
                Defaults to 0.1.

        Returns:
            A tuple ``(success, iters)``.
        """
        success, iters = self._null_space_optimisation(
            piecewise_smoothing_frequency,
            piecewise_steps_multiplier=piecewise_steps_multiplier,
            max_steps=max_smoothing_steps,
            rate=smoothing_rate,
            diff_tol=diff_tol,
            label="Smoothing (freq)",
        )
        return success, iters

    def filter_frequency(
        self,
        filter_fn: Callable[[Array], Array],
        piecewise_steps_multiplier: int = 1,
        smoothing_rate: float = 0.01,
        max_smoothing_steps: int = 100,
        diff_tol: float = 0.1,
    ) -> tuple[bool, int]:
        """Drive the pulse toward ``filter_fn(rfft(pulse))``.

        Args:
            filter_fn: Complex-array filter applied to the rfft of phi.
            piecewise_steps_multiplier: Factor by which to subdivide
                gate segments. Defaults to 1.
            smoothing_rate: Learning rate. Defaults to 0.01.
            max_smoothing_steps: Maximum iterations. Defaults to 100.
            diff_tol: Convergence tolerance. Defaults to 0.1.

        Returns:
            A tuple ``(success, iters)``.
        """
        success, iters = self._null_space_optimisation(
            partial(piecewise_smoothing_frequency_filter, filter_fn=filter_fn),
            piecewise_steps_multiplier=piecewise_steps_multiplier,
            max_steps=max_smoothing_steps,
            rate=smoothing_rate,
            diff_tol=diff_tol,
            label="Smoothing (freq filter)",
        )
        return success, iters

    def _resolve_parameter_indices(
        self,
        parameter_labels: list[str] | None,
        parameter_indices: tuple[int, ...] | None,
        default_all: bool = False,
    ) -> tuple[int, ...]:
        """Resolve ``parameter_labels`` / ``parameter_indices`` into integer indices.

        Args:
            parameter_labels: Optional list of projected-basis label strings.
            parameter_indices: Optional tuple of integer indices.
            default_all: If both are ``None`` and this is ``True``, return
                the full range of projected indices.

        Returns:
            A tuple of integer indices into the projected basis.

        Raises:
            ValueError: If both are provided, or labels are passed with
                ``param_transform`` active.
        """
        if parameter_labels is not None and parameter_indices is not None:
            raise ValueError("Specify parameter_labels or parameter_indices, not both")
        real = getattr(self, "_real_params", False)
        if real:
            if parameter_labels is not None:
                raise ValueError(
                    "parameter_labels not supported with param_transform; "
                    "use parameter_indices instead"
                )
            if parameter_indices is None:
                if default_all:
                    return tuple(range(self.params.n_experimental_params))
                raise ValueError("parameter_indices required with param_transform")
            return tuple(parameter_indices)
        if parameter_indices is not None:
            return tuple(parameter_indices)
        proj_labels = list(self.engine.projected_basis.labels)
        if parameter_labels is None:
            if default_all:
                return tuple(range(len(proj_labels)))
            raise ValueError("Either parameter_labels or parameter_indices must be provided")
        return tuple(proj_labels.index(label) for label in parameter_labels)

    def speed(
        self,
        parameter_labels: list[str] | None = None,
        parameter_indices: tuple[int, ...] | None = None,
        piecewise_steps_multiplier: int = 1,
        optimization_rate: float = 0.01,
        max_optimization_steps: int = 100,
        diff_tol: float = 0.1,
    ) -> tuple[bool, int]:
        """Minimise the peak amplitude of selected projected parameters.

        Cost: $\\max_{g,k\\in P}|\\phi_k(g)|$. Minimising it raises the
        gate-speed limit.

        Args:
            parameter_labels: Projected-basis labels to minimise.
            parameter_indices: Integer indices (alternative to labels).
            piecewise_steps_multiplier: Subdivision factor. Defaults to 1.
            optimization_rate: Learning rate. Defaults to 0.01.
            max_optimization_steps: Maximum iterations. Defaults to 100.
            diff_tol: Convergence tolerance. Defaults to 0.1.

        Returns:
            A tuple ``(success, iters)``.
        """
        parameter_indices = self._resolve_parameter_indices(
            parameter_labels, parameter_indices, default_all=False
        )
        real = getattr(self, "_real_params", False)
        n_proj = (self.params.n_experimental_params if real
                  else self.engine.projected_basis.lie_algebra_dim)
        speed_fn = get_speed_null_space_fn(n_proj, parameter_indices)
        success, iters = self._null_space_optimisation(
            speed_fn,
            piecewise_steps_multiplier=piecewise_steps_multiplier,
            max_steps=max_optimization_steps,
            rate=optimization_rate,
            diff_tol=diff_tol,
            label="Speed Optimization",
        )
        return success, iters

    def length(
        self,
        parameter_labels: list[str] | None = None,
        parameter_indices: tuple[int, ...] | None = None,
        piecewise_steps_multiplier: int = 1,
        optimization_rate: float = 0.01,
        max_optimization_steps: int = 100,
        diff_tol: float = 0.1,
    ) -> tuple[bool, int]:
        """Minimise the total pulse length.

        Cost: $\\sum_g\\sqrt{\\sum_{k\\in P}\\phi_k(g)^2 + \\|d_g\\|^2}$
        where $d_g$ is the per-segment drift contribution.

        Args:
            parameter_labels: Projected-basis labels to minimise.
                If ``None``, all projected parameters are used.
            parameter_indices: Integer indices (alternative to labels).
            piecewise_steps_multiplier: Subdivision factor. Defaults to 1.
            optimization_rate: Learning rate. Defaults to 0.01.
            max_optimization_steps: Maximum iterations. Defaults to 100.
            diff_tol: Convergence tolerance. Defaults to 0.1.

        Returns:
            A tuple ``(success, iters)``.
        """
        parameter_indices = self._resolve_parameter_indices(
            parameter_labels, parameter_indices, default_all=True
        )
        real = getattr(self, "_real_params", False)
        n_proj = (self.params.n_experimental_params if real
                  else self.engine.projected_basis.lie_algebra_dim)
        drift_sq_norm = 0.0
        if not real and self.engine.drift_basis is not None and getattr(self, "drift_parameters", None) is not None:
            drift_per_gate = np.array(self.drift_parameters) / piecewise_steps_multiplier
            drift_sq_norm = float(np.sum(drift_per_gate ** 2))
        length_fn = get_length_null_space_fn(n_proj, parameter_indices, drift_sq_norm=drift_sq_norm)
        success, iters = self._null_space_optimisation(
            length_fn,
            piecewise_steps_multiplier=piecewise_steps_multiplier,
            max_steps=max_optimization_steps,
            rate=optimization_rate,
            diff_tol=diff_tol,
            label="Length Optimization",
        )
        return success, iters

    def robust(
        self,
        parameter_labels: list[str] | None = None,
        parameter_indices: tuple[int, ...] | None = None,
        delta: float = 0.01,
        num_samples: int = 5,
        piecewise_steps_multiplier: int = 1,
        optimization_rate: float = 0.01,
        max_optimization_steps: int = 100,
        diff_tol: float = 0.1,
    ) -> tuple[bool, int]:
        """Maximise the worst-case fidelity over $\\delta$ perturbations.

        Cost: $1 - \\min_{\\boldsymbol\\delta} F(U(\\phi+\\sum_k \\delta_k e_k))$
        over a Cartesian grid of ``num_samples`` evenly-spaced values
        in $[-\\delta, +\\delta]$ for each $k\\in P$.

        Args:
            parameter_labels: Projected-basis labels to make robust.
            parameter_indices: Integer indices (alternative to labels).
            delta: Half-width of the perturbation box. Defaults to 0.01.
            num_samples: Samples per parameter. Defaults to 5.
            piecewise_steps_multiplier: Subdivision factor. Defaults to 1.
            optimization_rate: Learning rate. Defaults to 0.01.
            max_optimization_steps: Maximum iterations. Defaults to 100.
            diff_tol: Convergence tolerance. Defaults to 0.1.

        Returns:
            A tuple ``(success, iters)``.
        """
        parameter_indices = self._resolve_parameter_indices(
            parameter_labels, parameter_indices, default_all=False
        )
        real = getattr(self, "_real_params", False)
        if real:
            n_proj = self.params.n_experimental_params
            n_projdrift = n_proj
            drift_params = None
        else:
            n_proj = self.engine.projected_basis.lie_algebra_dim
            n_projdrift = self.engine.proj_drift_basis.lie_algebra_dim
            drift_params = (
                self.drift_parameters
                if hasattr(self, "drift_parameters") and self.engine.drift_basis is not None
                else None
            )
        robustness_fn = get_robustness_null_space_fn(
            self.engine.fid_U_fn,
            self.engine.compute_U_fn,
            self.engine.proj_indices_projdrift_basis,
            self.engine.drift_indices_projdrift_basis,
            drift_params,
            n_proj,
            n_projdrift,
            parameter_indices,
            delta,
            num_samples,
        )
        success, iters = self._null_space_optimisation(
            robustness_fn,
            piecewise_steps_multiplier=piecewise_steps_multiplier,
            max_steps=max_optimization_steps,
            rate=optimization_rate,
            diff_tol=diff_tol,
            label="Robustness Optimization",
        )
        return success, iters

    def bound(
        self,
        parameter_bounds: dict[str, tuple[float, float]],
        method: str = "projected_gradient",
        bounding_rate: float = 0.01,
        max_bounding_steps: int = 100,
        diff_tol: float = 0.1,
    ) -> tuple[bool, int]:
        """Enforce parameter bounds via null-space optimisation.

        Projects the parameters into the feasible box defined by
        `parameter_bounds` while staying in the Jacobian null space.

        Args:
            parameter_bounds: Dictionary mapping interaction labels to
                ``(min, max)`` tuples.
            method: Bounding strategy — ``'projected_gradient'`` /
                ``'pg'`` or ``'mid_point'`` / ``'mp'``.
                Defaults to ``'projected_gradient'``.
            bounding_rate: Learning rate. Defaults to 0.01.
            max_bounding_steps: Maximum iterations. Defaults to 100.
            diff_tol: Convergence tolerance. Defaults to 0.1.

        Returns:
            A tuple ``(success, iters)`` where `success` is ``True`` if
            `diff_tol` was reached.

        Raises:
            ValueError: If an unsupported `method` is provided.
        """
        self.parameter_bounds = parameter_bounds
        bounds = self.engine.proj_drift_basis.generate_bounds(self.parameter_bounds,
                                                              self.engine.piecewise_steps)
        self.lower_bounds = jnp.array(bounds[0], dtype=jnp.float64)
        self.upper_bounds = jnp.array(bounds[1], dtype=jnp.float64)

        if method == "projected_gradient" or method == "pg":
            piecewise_bounding = piecewise_bounding_pg
        elif method == "mid_point" or method == "mp":
            piecewise_bounding = piecewise_bounding_mp
        else:
            raise ValueError(f"Bounding method {method} not implemented.")

        success, iters = self._null_space_optimisation(piecewise_bounding,
                                                        max_steps=max_bounding_steps,
                                                        rate=bounding_rate,
                                                        diff_tol=diff_tol,
                                                        label="Bounding",
                                                        lower_bounds=self.lower_bounds[:,self.engine.proj_indices_projdrift_basis],
                                                        upper_bounds=self.upper_bounds[:,self.engine.proj_indices_projdrift_basis],
                                                        )
        return success, iters

    def get_free_params_update_smoothing(self) -> Callable[[Array, np.ndarray], Array]:
        """Build a JIT-compiled function to reconstruct free parameters.

        Combines projected and drift parameters into the full
        free-parameter array during smoothing.

        Returns:
            A JIT-compiled callable
            ``update_free_params_smoothing(proj_params, params)``.
        """

        _dtype = jnp.float64 if self._real_params else jnp.complex128

        @jax.jit
        def update_free_params_smoothing(proj_params, params):
            free_params = jnp.zeros((self.engine.piecewise_steps, self.engine.proj_drift_basis.lie_algebra_dim),
                                    dtype=_dtype)
            free_params = free_params.at[:, self.engine.proj_indices_projdrift_basis].set(proj_params)
            free_params = free_params.at[:, self.engine.drift_indices_projdrift_basis].set(
                params[:, self.engine.drift_indices])
            return free_params

        return update_free_params_smoothing

    def _null_space_optimisation(self,
                                 null_space_function: Callable[..., tuple[Array, Array]],
                                 *,
                                 piecewise_steps_multiplier: int = 1,
                                 rate: float = 0.01,
                                 max_steps: int = 100,
                                 diff_tol: float = 0.1,
                                 label: str | None = None,
                                 **kwargs) -> tuple[bool, int]:
        """Run a generic null-space optimisation loop.

        Iteratively applies `null_space_function` to move parameters
        within the Jacobian null space, preserving fidelity while
        optimising an auxiliary cost (smoothing or bounding).

        Args:
            null_space_function: Callable implementing the null-space
                update rule.
            piecewise_steps_multiplier: Factor by which to subdivide
                existing gate segments. Defaults to 1.
            rate: Learning rate. Defaults to 0.01.
            max_steps: Maximum iterations. Defaults to 100.
            diff_tol: Convergence tolerance. Defaults to 0.1.
            label: Label printed during progress logging.
            **kwargs: Extra keyword arguments forwarded to
                `null_space_function`.

        Returns:
            A tuple ``(success, iters)`` where `success` is ``True`` if
            `diff_tol` was reached.
        """
        # Update the number of piecewise steps and initialise new parameters.
        # Keep engine, params.parameters length, and params.piecewise_steps in sync.
        new_count = self.engine.piecewise_steps * piecewise_steps_multiplier
        self.engine.set_piecewise_steps(new_count)
        self.params.piecewise_steps = new_count

        new_parameters = [list(np.copy(self.params.parameters)) for _ in range(piecewise_steps_multiplier)]
        self.params.parameters = (
            np.array([x for group in zip(*new_parameters) for x in group]) / piecewise_steps_multiplier)

        _dtype = jnp.float64 if self._real_params else jnp.complex128
        _drift = self.params.parameters[:, self.engine.proj_drift_indices]
        _proj = self.params.parameters[:, self.engine.projected_indices]
        if self._real_params:
            _drift, _proj = jnp.real(_drift), jnp.real(_proj)
        free_params = _drift.astype(_dtype)
        proj_params = _proj.astype(_dtype)

        # Record the initial subdivided parameters (fidelity carried over unchanged)
        self.step_size = 0
        if self.history is not None:
            self.history.record(self)

        params_update = self.get_free_params_update_smoothing()

        c = 0
        diff = np.inf
        pulse_templates = None
        if self.pulse_constraints is not None:
            E_pulse, pulse_templates = self.engine.build_pulse_expander(
                self.pulse_constraints, self._real_params,
                self.params.n_experimental_params, np.array(proj_params).real)
            if self.constraint_expander is not None:
                E_gate = np.kron(np.eye(self.engine.piecewise_steps), self.constraint_expander)
                expander = jnp.array(E_gate @ np.linalg.pinv(E_gate) @ E_pulse)
            else:
                expander = jnp.array(E_pulse)
        elif self.constraint_expander is not None:
            expander = jnp.kron(jnp.eye(self.engine.piecewise_steps), jnp.array(self.constraint_expander))
        else:
            expander = None
        fid=0
        while (diff > diff_tol) and (c < max_steps):
            _, omegas_steps_phis = self.engine.gammas_and_omegas(free_params)
            vh, num = find_null_space(omegas_steps_phis, expander)

            assert num > 0, "Nullspace is empty!"
            null_space = vh[num:, :].T.conj()

            proj_params, diff = null_space_function(proj_params, null_space, expander, rate, **kwargs)

            if pulse_templates is not None:
                proj_params = np.array(proj_params)
                for k, tmpl in pulse_templates.items():
                    scale = float(np.dot(proj_params[:, k].real, tmpl))
                    proj_params[:, k] = scale * tmpl

            free_params = params_update(proj_params, self.params.parameters)

            fid = self.engine.fid_U_fn(self.engine.compute_U_fn(free_params))

            c += 1
            print(
                f"[{c}/{max_steps}] [Fidelity = {fid}] {label} : cost = {diff} (aim = {diff_tol})                      ",
                end="\r")
        print(f"[{c}/{max_steps}] [Fidelity = {fid}] {label} : cost = {diff} (aim = {diff_tol})                        ")
        success = diff_tol >= diff
        new_params = np.zeros_like(self.params.parameters)
        new_params[:, self.engine.proj_drift_indices] = [p.real for p in free_params]
        self.params.parameters = new_params
        self.params.fidelity = fid
        self.step_size = rate
        if self.history is not None:
            self.history.record(self)
        return success, c


@partial(jax.jit, static_argnames=("rcond"))
def find_null_space(
    omegas_steps_phis: Array,
    expander: Array | None,
    rcond: float | None = None,
) -> tuple[Array, int]:
    """Find the null space of the Jacobian omega matrix.

    Performs an SVD and identifies the right singular vectors
    whose singular values fall below a tolerance.

    Args:
        omegas_steps_phis: Omega ``Array`` matrices from the Jacobian,
            one per gate segment.
        expander: Optional constraint expansion ``Array``, or ``None``.
        rcond: Relative condition number cutoff. Defaults to machine
            epsilon times the larger matrix dimension.

    Returns:
        A tuple ``(vh, num)`` where ``vh`` is the right singular
        vectors ``Array`` and ``num`` is the rank (number of
        non-null singular values).
    """
    comb_vecs_T = jnp.concatenate(omegas_steps_phis, axis=0).T
    comb_vecs_T = comb_vecs_T @ expander if expander is not None else comb_vecs_T
    u, s, vh = jax.scipy.linalg.svd(comb_vecs_T, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = jnp.finfo(s.dtype).eps * max(M, N)
    tol = jnp.amax(s) * rcond
    num = jnp.sum(s > tol, dtype=int)
    return vh, num


@partial(jax.jit, static_argnames=("smoothing_rate"))
def piecewise_smoothing_frequency(
    phi: Array,
    null_space: Array,
    expander: Array | None,
    smoothing_rate: float = 0.01,
) -> tuple[Array, Array]:
    """Null-space update that suppresses high-frequency spectral power.

    Minimises $\\frac{1}{(N_g/2)K}\\sum_{m \\ge 1, k}|\\widehat{\\phi_k}(m)|^2$
    via gradient descent projected onto the Jacobian null space, so
    fidelity is preserved. The DC bin $m=0$ is excluded so the mean
    level of each pulse is not penalised.

    Args:
        phi: Current projected parameter ``Array`` of shape ``(N_g, K)``.
        null_space: Null-space basis ``Array``.
        expander: Optional constraint expansion ``Array``, or ``None``.
        smoothing_rate: Learning rate. Defaults to 0.01.

    Returns:
        A tuple ``(updated_phi, cost)``.
    """
    null_space = expander @ null_space if expander is not None else null_space
    phi = jnp.real(phi).astype(jnp.float64)
    phi_flat = phi.flatten()

    def minimize_power(params_flat):
        params_2d = params_flat.reshape(phi.shape)
        phi_rfft = jnp.fft.rfft(params_2d, axis=0)
        return jnp.mean(jnp.abs(phi_rfft[1:]) ** 2)

    val, grad = jax.value_and_grad(minimize_power)(phi_flat)
    x, _, _, _ = jnp.linalg.lstsq(null_space, -grad)
    sol = null_space @ (smoothing_rate * x / (jnp.linalg.norm(phi_flat) * jnp.linalg.norm(x) + 1e-12))
    sol = phi + sol.reshape(phi.shape)
    return sol, val


@partial(jax.jit, static_argnames=("smoothing_rate", "filter_fn"))
def piecewise_smoothing_frequency_filter(
    phi: Array,
    null_space: Array,
    expander: Array | None,
    smoothing_rate: float = 0.01,
    filter_fn: Callable[[Array], Array] | None = None,
) -> tuple[Array, Array]:
    """Null-space update that drives $\\phi$ toward ``filter_fn(rfft(phi))``.

    Minimises the squared distance, in Fourier space, between the
    current pulse and its filtered version. By Parseval's theorem
    this equals the time-domain $L^2$ distance up to a constant.

    Args:
        phi: Current projected parameter ``Array`` of shape ``(N_g, K)``.
        null_space: Null-space basis ``Array``.
        expander: Optional constraint expansion ``Array``, or ``None``.
        smoothing_rate: Learning rate. Defaults to 0.01.
        filter_fn: Callable mapping a complex ``(N_rfft, K)`` rfft array
            to a same-shape filtered array.

    Returns:
        A tuple ``(updated_phi, cost)``.
    """
    null_space = expander @ null_space if expander is not None else null_space
    phi = jnp.real(phi).astype(jnp.float64)
    phi_flat = phi.flatten()

    def distance_to_filtered(params_flat):
        params_2d = params_flat.reshape(phi.shape)
        phi_rfft = jnp.fft.rfft(params_2d, axis=0)
        phi_rfft_filtered = filter_fn(phi_rfft)
        diff = phi_rfft - phi_rfft_filtered
        return jnp.mean(jnp.abs(diff) ** 2)

    val, grad = jax.value_and_grad(distance_to_filtered)(phi_flat)
    x, _, _, _ = jnp.linalg.lstsq(null_space, -grad)
    sol = null_space @ (smoothing_rate * x / (jnp.linalg.norm(phi_flat) * jnp.linalg.norm(x) + 1e-12))
    sol = phi + sol.reshape(phi.shape)
    return sol, val


def get_speed_null_space_fn(n_proj: int, parameter_indices: tuple[int, ...]) -> Callable:
    """Build a JIT-compiled peak-amplitude minimisation step.

    Cost: $\\max_{g,k\\in P}|\\phi_k(g)|$.

    Args:
        n_proj: Number of projected basis parameters per step.
        parameter_indices: Indices within the projected basis to minimise.

    Returns:
        A callable ``(phi, null_space, expander, optimization_rate) ->
        (new_phi, cost)``.
    """
    _pi = tuple(parameter_indices)

    def cost(phi_flat):
        n_steps = phi_flat.size // n_proj
        sel = jnp.concatenate([jnp.arange(n_steps) * n_proj + p for p in _pi])
        return jnp.max(jnp.abs(phi_flat[sel]))

    cost_vg = jax.jit(jax.value_and_grad(cost))

    @partial(jax.jit, static_argnames=("optimization_rate",))
    def step(phi, null_space, expander, optimization_rate=0.01):
        ns = expander @ null_space if expander is not None else null_space
        phi_flat = jnp.real(phi).flatten().astype(jnp.float64)
        val, grad = cost_vg(phi_flat)
        x, _, _, _ = jnp.linalg.lstsq(ns, -grad)
        sol = ns @ (optimization_rate * x / (jnp.linalg.norm(x) + 1e-12))
        sol = phi + sol.reshape(phi.shape)
        return sol, val

    return step


def get_length_null_space_fn(n_proj: int, parameter_indices: tuple[int, ...], drift_sq_norm: float = 0.0) -> Callable:
    """Build a JIT-compiled pulse-length minimisation step.

    Cost: $\\sum_g \\sqrt{\\sum_{k\\in P}\\phi_k(g)^2 + \\|d_g\\|^2 + \\varepsilon}$
    where $d_g$ is the constant drift contribution.

    Args:
        n_proj: Number of projected basis parameters per step.
        parameter_indices: Indices within the projected basis to minimise.
        drift_sq_norm: Sum of squared drift parameters per gate.
            Added inside each per-step norm.

    Returns:
        A callable ``(phi, null_space, expander, optimization_rate) ->
        (new_phi, cost)``.
    """
    _pi = jnp.array(parameter_indices, dtype=jnp.int32)
    _dsq = jnp.float64(drift_sq_norm)

    def cost(phi_flat):
        n_steps = phi_flat.size // n_proj
        phi_mat = phi_flat.reshape(n_steps, n_proj)
        selected = phi_mat[:, _pi]
        per_step_norms = jnp.sqrt(jnp.sum(selected ** 2, axis=1) + _dsq + 1e-30)
        return jnp.sum(per_step_norms)

    cost_vg = jax.jit(jax.value_and_grad(cost))

    @partial(jax.jit, static_argnames=("optimization_rate",))
    def step(phi, null_space, expander, optimization_rate=0.01):
        ns = expander @ null_space if expander is not None else null_space
        phi_flat = jnp.real(phi).flatten().astype(jnp.float64)
        val, grad = cost_vg(phi_flat)
        x, _, _, _ = jnp.linalg.lstsq(ns, -grad)
        sol = ns @ (optimization_rate * x / (jnp.linalg.norm(x) + 1e-12))
        sol = phi + sol.reshape(phi.shape)
        return sol, val

    return step


def get_robustness_null_space_fn(
    fid_U_fn: Callable,
    compute_U_fn: Callable,
    proj_indices: np.ndarray,
    drift_indices: np.ndarray,
    drift_params: np.ndarray | None,
    n_proj: int,
    n_projdrift: int,
    parameter_indices: tuple[int, ...],
    delta: float,
    num_samples: int = 5,
) -> Callable:
    """Build a JIT-compiled worst-case-fidelity null-space step.

    Cost: $1 - \\min_{\\boldsymbol\\delta\\in[-\\Delta,\\Delta]^{|P|}}
    F\\bigl(U(\\phi + \\sum_{k\\in P}\\delta_k e_k)\\bigr)$, sampled on a
    Cartesian grid of $\\mathtt{num\\_samples}^{|P|}$ points. Each
    perturbation $\\delta_k$ is applied uniformly to all gate segments.

    Args:
        fid_U_fn: JIT-compiled fidelity function taking a unitary.
        compute_U_fn: JIT-compiled function computing the unitary
            from free parameters.
        proj_indices: Boolean mask of projected positions within the
            proj+drift basis.
        drift_indices: Boolean mask of drift positions within the
            proj+drift basis.
        drift_params: Drift values, or ``None``.
        n_proj: Number of projected parameters per gate.
        n_projdrift: Number of proj+drift parameters per gate.
        parameter_indices: Indices within the projected basis to make
            robust.
        delta: Half-width $\\Delta$ of the perturbation box.
        num_samples: Number of samples per parameter. Defaults to 5.

    Returns:
        A callable ``(phi, null_space, expander, optimization_rate) ->
        (new_phi, cost)``.
    """
    sample_deltas = jnp.linspace(-delta, delta, num_samples)
    n_robust = len(parameter_indices)
    grids = jnp.meshgrid(*([sample_deltas] * n_robust), indexing='ij')
    delta_combinations = jnp.stack([g.ravel() for g in grids], axis=-1)
    _pi = tuple(parameter_indices)

    def _make_free_params(proj_flat_real):
        n_steps = proj_flat_real.size // n_proj
        proj_params = proj_flat_real.reshape(n_steps, n_proj)
        free_params = jnp.zeros((n_steps, n_projdrift), dtype=jnp.complex128)
        free_params = free_params.at[:, proj_indices].set(proj_params.astype(jnp.complex128))
        if drift_params is not None:
            free_params = free_params.at[:, drift_indices].set(
                jnp.tile(jnp.array(drift_params, dtype=jnp.complex128), (n_steps, 1)))
        return free_params

    def min_fidelity_cost(proj_flat_real):
        n_steps = proj_flat_real.size // n_proj

        def fid_at_deltas(delta_vec):
            perturbation = jnp.zeros_like(proj_flat_real)
            for k, pidx in enumerate(_pi):
                gate_idxs = jnp.arange(n_steps) * n_proj + pidx
                perturbation = perturbation.at[gate_idxs].set(delta_vec[k])
            return fid_U_fn(compute_U_fn(_make_free_params(proj_flat_real + perturbation)))

        fidelities = jax.vmap(fid_at_deltas)(delta_combinations)
        return jnp.real(1.0 - jnp.min(fidelities))

    cost_vg = jax.jit(jax.value_and_grad(min_fidelity_cost))

    @partial(jax.jit, static_argnames=("optimization_rate",))
    def step(phi, null_space, expander, optimization_rate=0.01):
        ns = expander @ null_space if expander is not None else null_space
        phi_flat = jnp.real(phi).flatten().astype(jnp.float64)
        val, grad = cost_vg(phi_flat)
        x, _, _, _ = jnp.linalg.lstsq(ns, -grad)
        sol = ns @ (optimization_rate * x / (jnp.linalg.norm(x) + 1e-12))
        sol = phi + sol.reshape(phi.shape)
        return sol, val

    return step


@partial(jax.jit, static_argnames=("smoothing_rate"))
def piecewise_smoothing(
    phi: Array,
    null_space: Array,
    expander: Array | None,
    smoothing_rate: float = 0.01,
) -> tuple[Array, Array]:
    """Null-space update that smooths consecutive gate segments.

    Minimises the squared differences between adjacent parameter
    blocks by projecting a least-squares solution onto the null space.

    Args:
        phi: Current projected parameter ``Array`` of shape
            ``(piecewise_steps, K_proj)``.
        null_space: Null-space basis ``Array``.
        expander: Optional constraint expansion ``Array``, or ``None``.
        smoothing_rate: Learning rate scaling the update. Defaults to 0.01.

    Returns:
        A tuple ``(updated_phi, cost)`` where ``cost`` is the squared
        norm of the difference vector.
    """
    indep_params = phi.shape[1] # size of lie algebra of projected basis
    null_space = expander @ null_space if expander is not None else null_space
    phi_flat = phi.flatten()
    phi_flat = jnp.real(phi_flat).astype(jnp.float64)
    n_params = phi_flat.size  # phi = (piecewise_step_multiplier * K, K_non_drift)
    # Difference matrix
    D = jnp.eye(n_params, k=0) - jnp.eye(n_params, k=indep_params)
    D = jnp.vstack([jnp.eye(indep_params, D.shape[1]), D])
    # We have D (phi + Nullspace @ x) as difference vector
    A = D @ null_space  # A = (piecewise_step_multiplier * K + K_non_drift, dim(ker(J)))
    b = D @ phi_flat  # b = (piecewise_step_multiplier * K + K_non_drift,)
    x, _, _, _ = jnp.linalg.lstsq(A, -b)
    sol = null_space @ (smoothing_rate * x / (jnp.linalg.norm(phi_flat) * jnp.linalg.norm(x)))
    sol = phi + sol.reshape(phi.shape)
    return sol, jnp.linalg.norm(b) ** 2  # Difference is given by phi @ D.T @ D @ phi


@partial(jax.jit, static_argnames=("bounding_rate"))
def piecewise_bounding_mp(
    phi: Array,
    null_space: Array,
    expander: Array | None,
    bounding_rate: float = 0.01,
    lower_bounds: Array | None = None,
    upper_bounds: Array | None = None,
) -> tuple[Array, Array]:
    """Null-space bounding update using the mid-point method.

    Moves parameters towards the centre of the feasible box via a
    least-squares projection onto the null space.

    Args:
        phi: Current projected parameter ``Array``.
        null_space: Null-space basis ``Array``.
        expander: Optional constraint expansion ``Array``, or ``None``.
        bounding_rate: Learning rate. Defaults to 0.01.
        lower_bounds: Lower bound ``Array``. Defaults to ``None``.
        upper_bounds: Upper bound ``Array``. Defaults to ``None``.

    Returns:
        A tuple ``(updated_phi, cost)``.
    """
    null_space = expander @ null_space if expander is not None else null_space
    # Flatten parameters
    phi_flat = phi.flatten()
    phi_flat = jnp.real(phi_flat).astype(jnp.float64)
    n_params = phi_flat.size
    r,c = null_space.shape

    # Prepare bounds in flattened form
    lower_flat = lower_bounds.flatten()
    upper_flat = upper_bounds.flatten()
    mid_point = (lower_flat + upper_flat) / 2.0
    range = upper_flat - lower_flat
    range = range / jnp.max(range)  # Normalize range to avoid scaling issues

    phi_mid = jnp.concatenate((phi_flat, mid_point))
    zero_mid = jnp.concatenate((jnp.zeros(c), mid_point))

    zero = jnp.zeros((n_params, n_params))
    zero_r = jnp.zeros((r, n_params))
    zero_c = jnp.zeros((n_params, c))
    eye_n = jnp.eye(n_params)
    eye_d = jnp.diag(1/range)
    eye_c = jnp.eye(c)
    D = jnp.block([[eye_d, -eye_d],[zero, zero]])
    N = jnp.block([[null_space, zero_r],[zero_c, eye_n]])
    E = jnp.block([eye_c, zero_c.T]).T

    # We have D (phi + Nullspace @ x) as difference vector
    A = D @ N @ E  # A = (piecewise_step_multiplier * K + K_non_drift, dim(ker(J)))
    b = D @ phi_mid + D @ N @ zero_mid  # b = (piecewise_step_multiplier * K + K_non_drift,)
    x, _, _, _ = jnp.linalg.lstsq(A, -b)
    sol = null_space @ (bounding_rate * x / (jnp.linalg.norm(phi_flat) * jnp.linalg.norm(x)))
    sol = phi + sol.reshape(phi.shape)
    return sol, jnp.linalg.norm(b) ** 2  # Difference is given by phi @ D.T @ D @ phi


@partial(jax.jit, static_argnames=("bounding_rate"))
def piecewise_bounding_pg(
    phi: Array,
    null_space: Array,
    expander: Array | None,
    bounding_rate: float = 0.01,
    lower_bounds: Array | None = None,
    upper_bounds: Array | None = None,
) -> tuple[Array, Array]:
    """Null-space bounding update using projected gradient.

    Computes the gradient of the maximum bound violation and projects
    it onto the null space to find a feasibility-improving direction.

    Args:
        phi: Current projected parameter ``Array``.
        null_space: Null-space basis ``Array``.
        expander: Optional constraint expansion ``Array``, or ``None``.
        bounding_rate: Learning rate. Defaults to 0.01.
        lower_bounds: Lower bound ``Array``. Defaults to ``None``.
        upper_bounds: Upper bound ``Array``. Defaults to ``None``.

    Returns:
        A tuple ``(updated_phi, cost)`` where `cost` is the maximum
        constraint violation.
    """
    null_space = expander @ null_space if expander is not None else null_space
    # Flatten parameters
    phi_flat = phi.flatten()
    phi_flat = jnp.real(phi_flat).astype(jnp.float64)

    # Prepare bounds in flattened form
    lower_flat = lower_bounds.flatten()
    upper_flat = upper_bounds.flatten()

    # Cost: sum of squared distances outside the box
    def cost_function(x):
        upper_violation = jnp.clip(x - upper_flat, min=0.0)
        lower_violation = jnp.clip(lower_flat - x, min=0.0)
        # return jnp.mean(upper_violation**2 + lower_violation**2)
        return jnp.max(upper_violation + lower_violation)

    val, grad = jax.value_and_grad(cost_function)(phi_flat)

    x, _, _, _ = jnp.linalg.lstsq(null_space, -grad)

    sol = null_space @ (bounding_rate * x / (jnp.linalg.norm(x)+1e-12))
    sol = phi + sol.reshape(phi.shape)

    return sol, val
