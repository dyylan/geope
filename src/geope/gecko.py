from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

jax.config.update("jax_enable_x64", True)

from collections.abc import Callable
from functools import partial

from .geope import build_pulse_expander
from .objectives import (
    BoundMidpoint,
    BoundViolation,
    FrequencyFilter,
    Objective,
    PeakAmplitude,
    PulseLength,
    Robustness,
    Smoothness,
    SpectralPower,
    WeightedSum,
    as_weighted_sum,
)
from .parameters import Parameters
from .utils.callbacks import normalize_callbacks, run_callbacks
from .utils.history import History


class Gecko:
    """Null-space optimiser for GEOPE solutions.

    ``Gecko`` post-processes an optimized Pulse solution: it moves the
    parameters within the Jacobian null space so that fidelity is preserved
    while an auxiliary cost is improved.

    ``Gecko`` is constructed from a `Parameters` object.

        g = Geope(p); g.optimize()
        Gecko(g.params).smooth(...)

    The cost is an `Objective` from `geope.objectives`, or a weighted sum of
    them, passed to :meth:`optimize`::

        Gecko(g.params).optimize(Smoothness() + 0.1 * PulseLength())

    The single-objective methods (`smooth`, `speed`, `bound`, ...) are
    shorthands for one term.

    Because the functions are cached on ``params``, `Gecko` reuses the
    `Geope`'s already-compiled traces rather than recompiling.

    Attributes:
        params: The bound `Parameters` object.
        history: Optional `History` logger. None by default.
        #TODO: do we need this to be public? Only used in optimize
        step_size: Transient last optimisation rate.
        costs: The unweighted value of each objective term at the last
            iteration, keyed by term name (``WeightedSum.names``). Refreshed
            every iteration, so callbacks can read it.
        pulse_constraints: Optional pulse-shape constraint config.
        constraint_expander: Linear-equality constraint expander (from params).

    .. note::
        The `Parameters` object is **shared** with the source `Geope`. A
        null-space pass with ``piecewise_steps_multiplier > 1`` subdivides the
        pulse and advances ``params.parameters`` / ``params.piecewise_steps``
        in place — so the shared `Geope`'s state moves forward too, and a later
        ``geope.optimize()`` continues from the subdivided pulse.
    """

    def __init__(
        self, params: Parameters, history: History | None = None, verbose: bool = False
    ) -> None:
        """Initialise the Gecko optimiser.

        Args:
            params: A `Parameters` instance — the same object a `Geope` uses.
                Pass a `Geope`'s ``geope.params`` to post-process its solution;
                the cached optimisation functions are reused.
            history: Optional `History` logger. When supplied, the run
                trajectory is recorded into it.
            verbose: Whether to print progress. Defaults to False.

        Raises:
            TypeError: If ``params`` is not a `Parameters` instance.
        """
        if not isinstance(params, Parameters):
            raise TypeError(
                "Gecko requires a Parameters object as its first argument. "
                "Pass a Parameters object, or a Geope's `geope.params`."
            )

        self.params = params
        self._real_params = params.param_transform is not None

        # Compute a baseline fidelity if the params have never been evaluated
        # (e.g. a fresh Parameters that has not been through Geope.optimize).
        if self.params.fidelity is None:
            self.params.fidelity = self.params.manifold.fidelity_at(
                self.params.free(), jnp.asarray(self.params.delta_t, jnp.float64)
            )

        self.history = history
        if self.history is not None:
            self.history.params = self.params
        self.step_size = 0
        self.costs: dict[str, float] = {}
        self.verbose = verbose
        self.pulse_constraints = self.params.pulse_constraints
        self.constraint_expander = self.params.constraint_expander

    def optimize(
        self,
        objective: Objective | WeightedSum,
        *,
        piecewise_steps_multiplier: int = 1,
        rate: float = 0.01,
        max_steps: int = 100,
        diff_tol: float = 0.1,
        callbacks: Callable | list[Callable] | tuple[Callable, ...] | None = None,
    ) -> tuple[bool, int]:
        """Minimise an objective, or a weighted sum of objectives, in the null space.

        Minimises the raw linear combination $C(\\phi) = \\sum_i w_i c_i(\\phi)$
        by fixed-length steps along the null-space projection of
        $-\\nabla C$, so fidelity is preserved to first order. Only the ratios
        of the weights shape the trajectory; their overall scale changes what
        ``diff_tol`` means. Each term's unweighted value is reported every
        iteration in ``self.costs``.

        Args:
            objective: An `Objective` (e.g. ``Smoothness()``) or a `WeightedSum`
                built with ``*`` and ``+``, e.g.
                ``Smoothness() + 0.1 * PeakAmplitude(parameter_labels=["XI"])``.
            piecewise_steps_multiplier: Factor by which to subdivide the gate
                segments before the pass. Defaults to 1.
            rate: Length of each step in parameter space. Defaults to 0.01.
            max_steps: Maximum iterations. Defaults to 100.
            diff_tol: Convergence tolerance on the combined cost $C$.
                Defaults to 0.1.
            callbacks: Optional callback(s) ``callback(step, history, gecko)``
                run at the end of every iteration; a falsy return stops the
                loop. See :meth:`_null_space_optimisation`.

        Returns:
            A tuple ``(success, iters)`` where `success` is ``True`` if
            `diff_tol` was reached.
        """
        return self._null_space_optimisation(
            objective,
            piecewise_steps_multiplier=piecewise_steps_multiplier,
            rate=rate,
            max_steps=max_steps,
            diff_tol=diff_tol,
            callbacks=callbacks,
        )

    def smooth(
        self,
        piecewise_steps_multiplier: int = 1,
        smoothing_rate: float = 0.01,
        max_smoothing_steps: int = 100,
        diff_tol: float = 0.1,
        callbacks: Callable | list[Callable] | tuple[Callable, ...] | None = None,
    ) -> tuple[bool, int]:
        """Smooth the piecewise-constant pulse by null-space optimisation.

        Minimises the differences between consecutive gate segments
        (`Smoothness`) while remaining in the null space of the Jacobian so
        that fidelity is preserved.

        Args:
            piecewise_steps_multiplier: Factor by which to increase the
                number of gate segments. Defaults to 1.
            smoothing_rate: Step length for the null-space update.
                Defaults to 0.01.
            max_smoothing_steps: Maximum smoothing iterations.
                Defaults to 100.
            diff_tol: Convergence tolerance on the smoothing cost.
                Defaults to 0.1.
            callbacks: Optional callback(s) ``callback(step, history, gecko)``
                run at the end of every iteration; a falsy return stops the
                loop. See :meth:`_null_space_optimisation`.

        Returns:
            A tuple ``(success, iters)`` where `success` is ``True`` if
            `diff_tol` was reached.
        """
        return self._null_space_optimisation(
            Smoothness(),
            piecewise_steps_multiplier=piecewise_steps_multiplier,
            max_steps=max_smoothing_steps,
            rate=smoothing_rate,
            diff_tol=diff_tol,
            label="Smoothing",
            callbacks=callbacks,
        )

    def smooth_frequency(
        self,
        piecewise_steps_multiplier: int = 1,
        smoothing_rate: float = 0.01,
        max_smoothing_steps: int = 100,
        diff_tol: float = 0.1,
        callbacks: Callable | list[Callable] | tuple[Callable, ...] | None = None,
    ) -> tuple[bool, int]:
        """Suppress high-frequency spectral power in the pulse (`SpectralPower`).

        Args:
            piecewise_steps_multiplier: Factor by which to subdivide
                gate segments. Defaults to 1.
            smoothing_rate: Step length. Defaults to 0.01.
            max_smoothing_steps: Maximum iterations. Defaults to 100.
            diff_tol: Convergence tolerance on the spectral cost.
                Defaults to 0.1.
            callbacks: Optional callback(s) ``callback(step, history, gecko)``
                run at the end of every iteration; a falsy return stops the
                loop. See :meth:`_null_space_optimisation`.

        Returns:
            A tuple ``(success, iters)``.
        """
        return self._null_space_optimisation(
            SpectralPower(),
            piecewise_steps_multiplier=piecewise_steps_multiplier,
            max_steps=max_smoothing_steps,
            rate=smoothing_rate,
            diff_tol=diff_tol,
            label="Smoothing (freq)",
            callbacks=callbacks,
        )

    def filter_frequency(
        self,
        filter_fn: Callable[[Array], Array],
        piecewise_steps_multiplier: int = 1,
        smoothing_rate: float = 0.01,
        max_smoothing_steps: int = 100,
        diff_tol: float = 0.1,
        callbacks: Callable | list[Callable] | tuple[Callable, ...] | None = None,
    ) -> tuple[bool, int]:
        """Drive the pulse toward ``filter_fn(rfft(pulse))`` (`FrequencyFilter`).

        Args:
            filter_fn: Complex-array filter applied to the rfft of phi.
            piecewise_steps_multiplier: Factor by which to subdivide
                gate segments. Defaults to 1.
            smoothing_rate: Step length. Defaults to 0.01.
            max_smoothing_steps: Maximum iterations. Defaults to 100.
            diff_tol: Convergence tolerance. Defaults to 0.1.
            callbacks: Optional callback(s) ``callback(step, history, gecko)``
                run at the end of every iteration; a falsy return stops the
                loop. See :meth:`_null_space_optimisation`.

        Returns:
            A tuple ``(success, iters)``.
        """
        return self._null_space_optimisation(
            FrequencyFilter(filter_fn),
            piecewise_steps_multiplier=piecewise_steps_multiplier,
            max_steps=max_smoothing_steps,
            rate=smoothing_rate,
            diff_tol=diff_tol,
            label="Smoothing (freq filter)",
            callbacks=callbacks,
        )

    def speed(
        self,
        parameter_labels: list[str] | None = None,
        parameter_indices: tuple[int, ...] | None = None,
        piecewise_steps_multiplier: int = 1,
        optimization_rate: float = 0.01,
        max_optimization_steps: int = 100,
        diff_tol: float = 0.1,
        callbacks: Callable | list[Callable] | tuple[Callable, ...] | None = None,
    ) -> tuple[bool, int]:
        """Minimise the peak amplitude of selected projected parameters.

        Cost (`PeakAmplitude`): $\\max_{g,k\\in P}|\\phi_k(g)|$. Minimising it
        raises the gate-speed limit.

        Args:
            parameter_labels: Projected-basis labels to minimise.
            parameter_indices: Integer indices (alternative to labels).
            piecewise_steps_multiplier: Subdivision factor. Defaults to 1.
            optimization_rate: Step length. Defaults to 0.01.
            max_optimization_steps: Maximum iterations. Defaults to 100.
            diff_tol: Convergence tolerance. Defaults to 0.1.
            callbacks: Optional callback(s) ``callback(step, history, gecko)``
                run at the end of every iteration; a falsy return stops the
                loop. See :meth:`_null_space_optimisation`.

        Returns:
            A tuple ``(success, iters)``.
        """
        return self._null_space_optimisation(
            PeakAmplitude(
                parameter_labels=parameter_labels, parameter_indices=parameter_indices
            ),
            piecewise_steps_multiplier=piecewise_steps_multiplier,
            max_steps=max_optimization_steps,
            rate=optimization_rate,
            diff_tol=diff_tol,
            label="Speed Optimization",
            callbacks=callbacks,
        )

    def length(
        self,
        parameter_labels: list[str] | None = None,
        parameter_indices: tuple[int, ...] | None = None,
        piecewise_steps_multiplier: int = 1,
        optimization_rate: float = 0.01,
        max_optimization_steps: int = 100,
        diff_tol: float = 0.1,
        callbacks: Callable | list[Callable] | tuple[Callable, ...] | None = None,
    ) -> tuple[bool, int]:
        """Minimise the total pulse length.

        Cost (`PulseLength`): $\\sum_g\\sqrt{\\sum_{k\\in P}\\phi_k(g)^2 +
        \\|d_g\\|^2}$ where $d_g$ is the per-segment drift contribution.

        Args:
            parameter_labels: Projected-basis labels to minimise.
                If ``None``, all projected parameters are used.
            parameter_indices: Integer indices (alternative to labels).
            piecewise_steps_multiplier: Subdivision factor. Defaults to 1.
            optimization_rate: Step length. Defaults to 0.01.
            max_optimization_steps: Maximum iterations. Defaults to 100.
            diff_tol: Convergence tolerance. Defaults to 0.1.
            callbacks: Optional callback(s) ``callback(step, history, gecko)``
                run at the end of every iteration; a falsy return stops the
                loop. See :meth:`_null_space_optimisation`.

        Returns:
            A tuple ``(success, iters)``.
        """
        return self._null_space_optimisation(
            PulseLength(
                parameter_labels=parameter_labels, parameter_indices=parameter_indices
            ),
            piecewise_steps_multiplier=piecewise_steps_multiplier,
            max_steps=max_optimization_steps,
            rate=optimization_rate,
            diff_tol=diff_tol,
            label="Length Optimization",
            callbacks=callbacks,
        )

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
        callbacks: Callable | list[Callable] | tuple[Callable, ...] | None = None,
    ) -> tuple[bool, int]:
        """Maximise the worst-case fidelity over $\\delta$ perturbations.

        Cost (`Robustness`): $1 - \\min_{\\boldsymbol\\delta}
        F(U(\\phi+\\sum_k \\delta_k e_k))$ over a Cartesian grid of
        ``num_samples`` evenly-spaced values in $[-\\delta, +\\delta]$ for each
        $k\\in P$.

        Args:
            parameter_labels: Projected-basis labels to make robust.
            parameter_indices: Integer indices (alternative to labels).
            delta: Half-width of the perturbation box. Defaults to 0.01.
            num_samples: Samples per parameter. Defaults to 5.
            piecewise_steps_multiplier: Subdivision factor. Defaults to 1.
            optimization_rate: Step length. Defaults to 0.01.
            max_optimization_steps: Maximum iterations. Defaults to 100.
            diff_tol: Convergence tolerance. Defaults to 0.1.
            callbacks: Optional callback(s) ``callback(step, history, gecko)``
                run at the end of every iteration; a falsy return stops the
                loop. See :meth:`_null_space_optimisation`.

        Returns:
            A tuple ``(success, iters)``.
        """
        return self._null_space_optimisation(
            Robustness(
                parameter_labels=parameter_labels,
                parameter_indices=parameter_indices,
                delta=delta,
                num_samples=num_samples,
            ),
            piecewise_steps_multiplier=piecewise_steps_multiplier,
            max_steps=max_optimization_steps,
            rate=optimization_rate,
            diff_tol=diff_tol,
            label="Robustness Optimization",
            callbacks=callbacks,
        )

    def bound(
        self,
        parameter_bounds: dict[str, tuple[float, float]],
        method: str = "projected_gradient",
        bounding_rate: float = 0.01,
        max_bounding_steps: int = 100,
        diff_tol: float = 0.1,
        callbacks: Callable | list[Callable] | tuple[Callable, ...] | None = None,
    ) -> tuple[bool, int]:
        """Enforce parameter bounds via null-space optimisation.

        Moves the parameters toward the feasible box defined by
        `parameter_bounds` while staying in the Jacobian null space.

        Args:
            parameter_bounds: In projected space, a dictionary mapping
                interaction labels to ``(min, max)`` tuples. In experimental
                space (``param_transform`` set), a dictionary mapping integer
                parameter indices to ``(min, max)`` tuples — the same
                index convention ``pulse_constraints`` uses there; unbounded
                parameters default to ``(-inf, inf)``.
            method: Bounding strategy — ``'projected_gradient'`` / ``'pg'``
                minimises the worst violation (`BoundViolation`);
                ``'mid_point'`` / ``'mp'`` pulls bounded parameters toward the
                centre of their box (`BoundMidpoint`).
                Defaults to ``'projected_gradient'``.
            bounding_rate: Step length. Defaults to 0.01.
            max_bounding_steps: Maximum iterations. Defaults to 100.
            diff_tol: Convergence tolerance. Defaults to 0.1.
            callbacks: Optional callback(s) ``callback(step, history, gecko)``
                run at the end of every iteration; a falsy return stops the
                loop. See :meth:`_null_space_optimisation`.

        Returns:
            A tuple ``(success, iters)`` where `success` is ``True`` if
            `diff_tol` was reached.

        Raises:
            ValueError: If an unsupported `method` is provided, or if a
                label-keyed dict is passed under ``param_transform``.
        """
        if method == "projected_gradient" or method == "pg":
            objective = BoundViolation(parameter_bounds)
        elif method == "mid_point" or method == "mp":
            objective = BoundMidpoint(parameter_bounds)
        else:
            raise ValueError(f"Bounding method {method} not implemented.")

        return self._null_space_optimisation(
            objective,
            max_steps=max_bounding_steps,
            rate=bounding_rate,
            diff_tol=diff_tol,
            label="Bounding",
            callbacks=callbacks,
        )

    def get_free_params_update_smoothing(self) -> Callable[[Array, np.ndarray], Array]:
        """Build a JIT-compiled function to reconstruct free parameters.

        Combines projected and drift parameters into the full
        free-parameter array during smoothing.

        Returns:
            A JIT-compiled callable
            ``update_free_params_smoothing(proj_params, params)``.
        """
        _dtype = jnp.float64 if self._real_params else jnp.complex128

        if self._real_params:
            # Experimental space: every column is free, so the projected
            # parameters are already the full free-parameter array.
            @jax.jit
            def update_free_params_smoothing(proj_params, params):
                return proj_params.astype(_dtype)

            return update_free_params_smoothing

        @jax.jit
        def update_free_params_smoothing(proj_params, params):
            free_params = jnp.zeros(
                (
                    self.params.piecewise_steps,
                    self.params.proj_drift_basis.lie_algebra_dim,
                ),
                dtype=_dtype,
            )
            free_params = free_params.at[
                :, self.params.proj_indices_projdrift_basis
            ].set(proj_params)
            free_params = free_params.at[
                :, self.params.drift_indices_projdrift_basis
            ].set(params[:, self.params.drift_indices])
            return free_params

        return update_free_params_smoothing

    def _subdivide(self, multiplier: int) -> None:
        """Split every segment into ``multiplier`` copies and divide ``delta_t``.

        Exact for any ``param_transform``, since
        $\\exp(\\Delta t\\,A) = \\exp((\\Delta t/m)A)^m$. Keeps
        ``params.parameters``, ``params.piecewise_steps`` and
        ``params.delta_t`` in sync; a no-op for ``multiplier == 1``.
        """
        if multiplier == 1:
            return
        self.params.piecewise_steps = self.params.piecewise_steps * multiplier
        self.params.delta_t = self.params.delta_t / multiplier
        self.params.parameters = np.repeat(
            np.asarray(self.params.parameters), multiplier, axis=0
        )

    def _null_space_optimisation(
        self,
        objective: Objective | WeightedSum,
        *,
        piecewise_steps_multiplier: int = 1,
        rate: float = 0.01,
        max_steps: int = 100,
        diff_tol: float = 0.1,
        label: str | None = None,
        callbacks: Callable | list[Callable] | tuple[Callable, ...] | None = None,
    ) -> tuple[bool, int]:
        """Run the null-space optimisation loop for one objective.

        Subdivides the pulse, builds ``objective`` against the subdivided
        `Parameters`, then iteratively takes a fixed-length step of ``rate``
        along the null-space projection of the negative cost gradient
        (:func:`_null_space_gradient_step`), preserving fidelity to first order.

        Args:
            objective: An `Objective` or a `WeightedSum` of objectives.
            piecewise_steps_multiplier: Factor by which to subdivide
                existing gate segments. Defaults to 1.
            rate: Step length. Defaults to 0.01.
            max_steps: Maximum iterations. Defaults to 100.
            diff_tol: Convergence tolerance on the combined cost.
                Defaults to 0.1.
            label: Label printed during progress logging. Defaults to the
                objective's formula (``WeightedSum.label``).
            callbacks: Optional callback, or list/tuple of callbacks, invoked at
                the end of every iteration with the signature
                ``callback(step, history, gecko) -> bool``. All callbacks run
                each iteration; the loop stops early if any returns a falsy
                value (so a pure logging callback must ``return True``).
                ``step`` is the 1-based iteration index, ``history`` is
                ``gecko.history`` (may be ``None``), and ``gecko`` is this
                optimiser. During the loop the running geometry is visible via
                ``gecko.params.parameters`` and the per-term costs via
                ``gecko.costs``; ``gecko.params.fidelity`` and
                ``gecko.step_size`` are only finalised after the loop returns.

        Returns:
            A tuple ``(success, iters)`` where `success` is ``True`` if
            `diff_tol` was reached.

        Raises:
            TypeError: If ``objective`` is not an `Objective` or `WeightedSum`.
        """
        objective = as_weighted_sum(objective)
        label = objective.label if label is None else label

        # Subdivide first: an objective may close over the subdivided duration.
        # If it fails to build (a bad label, label-keyed bounds under
        # ``param_transform``), undo the subdivision so the error leaves the
        # shared Parameters as it found them.
        previous = (
            self.params.parameters,
            self.params.piecewise_steps,
            self.params.delta_t,
        )
        self._subdivide(piecewise_steps_multiplier)
        try:
            cost_fn = objective.build(self.params)
        except Exception:
            (
                self.params.parameters,
                self.params.piecewise_steps,
                self.params.delta_t,
            ) = previous
            raise
        names = objective.names
        step = _null_space_gradient_step(cost_fn)

        _dtype = jnp.float64 if self._real_params else jnp.complex128
        if self._real_params:
            # Experimental space: every column is both a free and a projected
            # parameter.
            _drift = jnp.real(self.params.parameters)
            _proj = jnp.real(self.params.parameters)
        else:
            _drift = self.params.parameters[:, self.params.proj_drift_indices]
            _proj = self.params.parameters[:, self.params.projected_indices]
        free_params = _drift.astype(_dtype)
        proj_params = _proj.astype(_dtype)

        # Record the initial subdivided parameters (fidelity carried over unchanged)
        self.step_size = 0
        if self.history is not None:
            self.history.record(self)

        params_update = self.get_free_params_update_smoothing()

        cbs = normalize_callbacks(callbacks)

        c = 0
        diff = np.inf
        pulse_templates = None
        if self.pulse_constraints is not None:
            E_pulse, pulse_templates = build_pulse_expander(
                self.params.piecewise_steps,
                self.params.projected_basis,
                self.pulse_constraints,
                self._real_params,
                self.params.n_experimental_params,
                np.array(proj_params).real,
            )
            if self.constraint_expander is not None:
                E_gate = np.kron(
                    np.eye(self.params.piecewise_steps), self.constraint_expander
                )
                expander = jnp.array(E_gate @ np.linalg.pinv(E_gate) @ E_pulse)
            else:
                expander = jnp.array(E_pulse)
        elif self.constraint_expander is not None:
            expander = jnp.kron(
                jnp.eye(self.params.piecewise_steps),
                jnp.array(self.constraint_expander),
            )
        else:
            expander = None
        fid = self.params.fidelity
        # A ``GeometricContext`` is trace-time only, so the context-reading
        # closure has to be jitted here rather than memoised on the manifold;
        # un-jitted, each iteration would re-lower its inner ``lax.scan``s.
        # ``manifold.fidelity_at`` is already compiled and memoised.
        #
        # Reading ``omegas`` off the context and nothing else is what keeps the
        # matrix logarithm out of this loop entirely: every context quantity is
        # lazy, and the geodesic tangent is never asked for.
        manifold = self.params.manifold
        delta_t = jnp.asarray(self.params.delta_t, dtype=jnp.float64)
        omegas_fn = jax.jit(lambda fp, dt: manifold.context(fp, dt).omegas)
        fid_of_params = manifold.fidelity_at
        while (diff > diff_tol) and (c < max_steps):
            vh, num = find_null_space(omegas_fn(free_params, delta_t), expander)

            assert num > 0, "Nullspace is empty!"
            null_space = vh[num:, :].T.conj()

            proj_params, total, values = step(proj_params, null_space, expander, rate)
            diff = float(total)
            self.costs = dict(zip(names, np.asarray(values).tolist()))

            if pulse_templates is not None:
                proj_params = np.array(proj_params)
                for k, tmpl in pulse_templates.items():
                    scale = float(np.dot(proj_params[:, k].real, tmpl))
                    proj_params[:, k] = scale * tmpl

            free_params = params_update(proj_params, self.params.parameters)

            fid = fid_of_params(free_params, delta_t)

            c += 1
            print(
                f"[{c}/{max_steps}] [Fidelity = {fid}] {label} : cost = {diff} (aim = {diff_tol})                      ",
                end="\r",
            )

            # Run user callbacks at the end of the iteration; stop early if any
            # requests it.
            if not run_callbacks(cbs, c, self.history, self):
                break
        print(
            f"[{c}/{max_steps}] [Fidelity = {fid}] {label} : cost = {diff} (aim = {diff_tol})                        "
        )
        success = diff_tol >= diff
        if self._real_params:
            new_params = np.array([np.real(p) for p in free_params])
        else:
            new_params = np.zeros_like(self.params.parameters)
            new_params[:, self.params.proj_drift_indices] = [
                p.real for p in free_params
            ]
        self.params.parameters = new_params
        self.params.fidelity = fid
        self.step_size = rate
        if self.history is not None:
            self.history.record(self)
        return success, c


def _null_space_gradient_step(
    cost_fn: Callable[[Array], tuple[Array, Array]],
) -> Callable[..., tuple[Array, Array, Array]]:
    """Build the one null-space step every objective shares.

    The step solves ``x = lstsq(N, -grad C)`` — the null-space coordinates of
    the projected negative gradient — and moves a fixed length ``rate`` along
    ``N x``. The direction is linear in the gradient, so for a weighted sum it
    is the normalised projection of $\\sum_i w_i \\nabla c_i$.

    Args:
        cost_fn: ``cost_fn(phi) -> (total, values)`` on the real projected
            pulse, as returned by `WeightedSum.build`.

    Returns:
        A JIT-compiled callable ``step(phi, null_space, expander, rate) ->
        (new_phi, total, values)``, with ``total`` and ``values`` evaluated at
        the incoming ``phi``. ``rate`` is traced, so changing it does not
        recompile.
    """
    value_and_grad = jax.value_and_grad(cost_fn, has_aux=True)

    @jax.jit
    def step(phi, null_space, expander, rate):
        ns = expander @ null_space if expander is not None else null_space
        (total, values), grad = value_and_grad(jnp.real(phi).astype(jnp.float64))
        x, _, _, _ = jnp.linalg.lstsq(ns, -grad.reshape(-1))
        update = ns @ (rate * x / (jnp.linalg.norm(x) + 1e-12))
        return phi + update.reshape(phi.shape), total, values

    return step


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
