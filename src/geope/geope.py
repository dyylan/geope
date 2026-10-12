from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

jax.config.update("jax_enable_x64", True)

from collections.abc import Callable

from .geometry.basis import Basis
from .line_searches import (
    Armijo,
    LineSearch,
)
from .parameters import Parameters
from .utils import (
    control_to_indices,
    merge_constraints,
    prepare_random_parameters,
)
from .utils.callbacks import normalize_callbacks, run_callbacks
from .utils.history import History

if TYPE_CHECKING:
    from .geometry.context import GeometricContext

# Default settings.
DEFAULT_PRECISION = 0.9999999
DEFAULT_MAX_STEP_SIZE = 1.0
DEFAULT_GRAM_SCHMIDT_STEP_SIZE = 1.3

# Relative threshold to assert if step made progress.
PROGRESS_RTOL = 1e-5

# The least-squares methods `least_squares` accepts.
LEAST_SQUARES_METHODS = ("svd", "lm")
# Factor the Levenberg-Marquardt damping grows by on a rejected trial.
LM_LAMBDA_FACTOR = 10.0


class Geope:
    """Top-level GEOPE optimiser for quantum gate synthesis.

    Orchestrates the geodesic-based optimisation of Lie algebra
    parameters ($\\phi$) to synthesise a target unitary from a
    controllable subalgebra.

    Attributes:
        params: The bound `Parameters` object (the single source of truth
            for all configuration, the live optimisation state, and the
            lazily-built/cached optimisation functions).
        #TODO: do we need this to be public? Only used in optimize
        precision: Target fidelity threshold. Set by :meth:`optimize` (the
            ``__init__``-body default holds until the first run).
        #TODO: do we need this to be public? Only used in optimize
        max_step_size: Maximum line-search step size. Set by :meth:`optimize`
            (baked into the jitted update, so it joins the compile memo).
        #TODO: do we need this to be public? Only used in optimize
        gram_schmidt_step_size: Step size for Gram-Schmidt fallback moves. Set
            by :meth:`optimize` (host-side only).
        #TODO: do we need this to be public? Only used in optimize
        line_search: The active :class:`~geope.line_searches.LineSearch` object
            from the most recent :meth:`optimize` call. ``None`` until
            :meth:`optimize` is first called.
        #TODO: do we need this to be public? Only used in optimize
        line_search_state: The current line-search state pytree (re-``init()``d
            per run); ``None`` until :meth:`optimize` is first called.
        #TODO: do we need this to be public? Only used in optimize
        step_size: Transient last line-search step size.
        least_squares_args: The resolved :func:`least_squares` configuration
            from the most recent :meth:`optimize` call (its defaults before
            then).
        ls_diagnostics: Diagnostics of the *most recent* geodesic least-squares
            solve, as a dict of host-side scalars with keys ``residual``,
            ``residual_rel``, ``rank``, ``cond``, ``damping`` and ``lm_iter``
            (see :func:`least_squares`). Seeded with ``nan`` / ``-1``
            sentinels before the first solve of a run. Not logged by default —
            pass a ``logging_fn`` to `History` (or read it from a callback) to
            record it.
        history: Optional `History` logger (``None`` unless supplied),
            holding the full run trajectory.
    """

    def __init__(
        self,
        params: Parameters,
        verbose: bool = False,
        history: History | None = None,
    ) -> None:
        """Initialise the Geope optimiser.

        ``Geope`` requires a `Parameters` object — the engine, initial
        parameters, drift, constraints, pulse constraints, seed,
        initialisation spread, projective flag and ``param_transform`` are
        all read from it. To construct one, use :class:`Parameters`.

        Args:
            params: A `Parameters` instance bundling every input the
                optimiser needs.
            verbose: Whether to print progress. Defaults to False.
            history: Optional `History` logger. When supplied, the full run
                trajectory is recorded into it (``geope.history``); when
                ``None`` (default), no history is kept.

        Raises:
            TypeError: If ``params`` is not a `Parameters` instance.
                The legacy ``Geope(engine=...)`` call site is no longer
                supported; build a ``Parameters`` object instead.
        """
        if not isinstance(params, Parameters):
            raise TypeError(
                "Geope requires a Parameters object as its first argument. "
                "Build a Parameters object with `geope.Parameters(basis=..., "
                "control=..., target=..., ...)` and pass that in."
            )

        self.params = params
        seed = params.seed
        if isinstance(seed, int):
            self._key = jax.random.key(seed)
        elif isinstance(seed, jax.Array):
            self._key = seed  # already a jax.Array key
        else:
            # TODO: we should have a random seed as default.
            self._key = jax.random.key(0)
        self._real_params = params.param_transform is not None
        # The optimisation functions and algebraic metadata are read directly
        # off ``params``.

        if self._real_params:
            init_parameters = self._init_for_param_transform(params)
            drift_parameters = None
            constraints = None
        else:
            init_parameters = params.parameters
            drift_parameters = params.drift_parameters
            constraints = params.constraint_arrays

        self.history = history
        if self.history is not None:
            self.history.params = params
        self.step_size = 0

        # The run-control knobs are optimize() arguments now; initialise them
        # to the module defaults so the object is valid before the first run
        # (e.g. a direct add_parameters / gram_schmidt call reads
        # self.max_step_size / self.gram_schmidt_step_size). optimize()
        # overwrites them per run.
        self.precision = DEFAULT_PRECISION
        self.max_step_size = DEFAULT_MAX_STEP_SIZE
        self.gram_schmidt_step_size = DEFAULT_GRAM_SCHMIDT_STEP_SIZE
        self.init_parameters_spread = params.init_spread
        self.pulse_constraints = params.pulse_constraints

        # The line search and its hyperparameters are arguments of optimize(),
        # not the constructor. The JIT-compiled update_step bakes the line search
        # and max_step_size into its closure, so it is built lazily by optimize()
        # (via _configure_line_search) and rebuilt only when that configuration
        # changes. The line-search object and its threaded state stay unset until
        # the first optimize() call.
        self.line_search = None
        self.line_search_state = None
        self.least_squares_args = _check_least_squares_args(None)
        self._linesearch_config = None
        self.update_step = None

        self.verbose = verbose
        # Initialize parameters
        self.init(init_parameters, drift_parameters, constraints, params.seed)

    def _init_for_param_transform(self, params: Parameters) -> np.ndarray:
        """Compute initial parameters in experimental-parameter space.

        If ``params.parameters`` is shaped ``(piecewise_steps, n_exp)``,
        use it directly; otherwise sample uniformly in
        $[-\\text{init\\_spread}\\,\\pi, +\\text{init\\_spread}\\,\\pi]$.

        Args:
            params: The ``Parameters`` object.

        Returns:
            An ``np.ndarray`` of shape ``(piecewise_steps, n_exp)``.
        """
        n_exp = params.n_experimental_params
        _user_init = np.array(params.parameters)
        if _user_init.shape == (params.piecewise_steps, n_exp):
            return _user_init
        return np.array(
            jax.random.uniform(
                self._split_key(),
                shape=(params.piecewise_steps, n_exp),
                minval=-params.init_spread * np.pi,
                maxval=params.init_spread * np.pi,
            )
        )

    def init(
        self,
        init_parameters: np.ndarray | None = None,
        drift_parameters: np.ndarray | None = None,
        constraints: list[np.ndarray] | np.ndarray | None = None,
        seed: int | jax.Array | None = None,
    ) -> None:
        """(Re-)initialise optimiser state.

        Sets up constraints, initial parameters, drift parameters and the
        live state (``params.parameters`` / ``params.fidelity``), and records
        step 0 into ``history`` when one is attached.

        Args:
            init_parameters: Initial parameter array. Defaults to random.
            drift_parameters: Fixed drift parameter values. Defaults to ones.
            constraints: Linear equality constraints.
            seed: Random seed (int) or JAX key for reproducibility.
        """
        if isinstance(seed, int):
            self._key = jax.random.key(seed)
        elif isinstance(seed, jax.Array):
            self._key = seed  # already a jax.Array key
        # else: keep existing self._key unchanged

        # Set constraints
        self.constraint_expander = None
        if constraints is not None:
            expander = np.eye(self.params.projected_basis.lie_algebra_dim)
            constraints = (
                constraints if isinstance(constraints, list) else [constraints]
            )
            self.constraints = [np.array(c) for c in merge_constraints(constraints)]
            del_indices = []
            for c in self.constraints:
                c_proj_indices = c.astype(bool)
                idx = np.where(c_proj_indices)[0]
                expander[:, idx[0]] = c
                del_indices.append(idx[1:])

            expander = np.delete(expander, del_indices, axis=1)
            expander = expander / expander.max()
            self.constraint_expander = expander

        self.parameter_bounds = None

        # Initialize variables
        if self._real_params:
            # param_transform mode: init_parameters are already in experimental space
            if init_parameters is not None:
                self.init_parameters = np.array(init_parameters)
            else:
                n_exp = self.params.n_experimental_params
                self.init_parameters = np.array(
                    jax.random.uniform(
                        self._split_key(),
                        shape=(self.params.piecewise_steps, n_exp),
                        minval=-self.init_parameters_spread * np.pi,
                        maxval=self.init_parameters_spread * np.pi,
                    )
                )
        elif init_parameters is None:
            self.init_parameters = np.array(
                [
                    prepare_random_parameters(
                        self.params.projected_indices,
                        expander=self.constraint_expander,
                        spread=self.init_parameters_spread,
                        key=self._split_key(),
                    )
                    for _ in range(self.params.piecewise_steps)
                ]
            )
        else:
            if np.array(init_parameters).shape == (self.params.basis.lie_algebra_dim,):
                self.init_parameters = np.array(
                    [init_parameters] * self.params.piecewise_steps
                )
            elif np.array(init_parameters).shape == (
                self.params.piecewise_steps,
                self.params.basis.lie_algebra_dim,
            ):
                self.init_parameters = np.array(init_parameters)
            elif np.array(init_parameters).shape == (
                self.params.projected_basis.lie_algebra_dim,
            ) or np.array(init_parameters).shape == (
                self.params.piecewise_steps,
                self.params.projected_basis.lie_algebra_dim,
            ):
                self.init_parameters = np.zeros(
                    (self.params.piecewise_steps, self.params.basis.lie_algebra_dim)
                )
                self.init_parameters[:, self.params.projected_indices] = np.array(
                    init_parameters
                )
            else:
                raise ValueError(
                    "Initial parameters must be of shape (full_basis.lie_algebra_dim,) or (full_basis.lie_algebra_dim, piecewise_steps) or (projected_basis.lie_algebra_dim,) or (piecewise_steps, projected_basis.lie_algebra_dim)"
                )
        if not self._real_params:
            if drift_parameters is not None and self.params.drift_basis is None:
                raise ValueError(
                    "Drift parameters are set but no drift basis is defined."
                )
            if self.params.drift_basis is not None:
                if drift_parameters is None:
                    self.drift_parameters = np.ones(
                        self.params.drift_basis.lie_algebra_dim
                    )
                else:
                    self.drift_parameters = np.array(drift_parameters)
                    assert (
                        self.params.drift_basis.lie_algebra_dim
                        == self.drift_parameters.shape[0]
                    ), (
                        "Drift parameters must be the same length as the size of the drift basis."
                    )
                self.init_parameters[:, self.params.drift_indices] = np.tile(
                    self.drift_parameters, (self.params.piecewise_steps, 1)
                )
            else:
                self.drift_parameters = None
        else:
            self.drift_parameters = None
        self.params.parameters = np.array(self.init_parameters)
        self.params.fidelity = self.params.manifold.fidelity_at(
            self.params.free(), self._delta_t()
        )
        self.step_size = 0
        # least-squares diagnostics
        self.ls_diagnostics = {
            "residual": float("nan"),
            "residual_rel": float("nan"),
            "rank": -1,
            "cond": float("nan"),
            "damping": float("nan"),
            "lm_iter": -1,
        }

        if self.history is not None:
            self.history.reset()
            self.history.record(self)  # step 0

    def _configure_line_search(
        self,
        line_search: LineSearch,
        max_step_size: float,
        least_squares_args: dict | None = None,
    ) -> None:
        """Select the line search and least squares and (re)build the update.
        If line_search, max_step or the least-squares configuration changes, we
        have to rebuild these object to force retracing.

        Args:
            line_search: The :class:`~geope.line_searches.LineSearch` object.
            max_step_size: Maximum line-search step; baked into the jitted
                closure, so it is part of the compile memo key.
            least_squares_args: Keyword arguments of :func:`least_squares`
                (``None`` for its defaults). Baked into the jitted closure, so
                part of the compile memo key — as a frozen copy, so mutating the
                caller's dict later cannot reuse a stale compiled step.

        Raises:
            TypeError: If ``line_search`` is not a
                :class:`~geope.line_searches.LineSearch`. Worth checking because
                `geope.optimizers` is a second family of frozen dataclasses
                passed to an ``optimize`` keyword, and one of them is called
                ``Adam`` — as a line search of that name once was.
            TypeError: If ``least_squares_args`` has a key :func:`least_squares`
                does not take.
            ValueError: If ``least_squares_args`` holds an invalid value (see
                :func:`least_squares`).
        """
        if not isinstance(line_search, LineSearch):
            raise TypeError(
                "Geope.optimize(line_search=...) takes a "
                "geope.line_searches.LineSearch — GoldenSection, Armijo, "
                "QuadraticArmijo or ApproximateQuadraticArmijo — not "
                f"{type(line_search).__name__}."
            )
        least_squares_args = _check_least_squares_args(least_squares_args)
        config = (
            line_search,
            max_step_size,
            tuple(sorted(least_squares_args.items())),
        )
        # Set the instance attributes before the memo check so self.<attr>
        # always tracks the latest values — safe because an equal config means
        # identical trace-time behaviour.
        self.line_search = line_search
        self.max_step_size = max_step_size
        self.least_squares_args = least_squares_args
        # TODO: Can we do this without the extra self._linsearch_config?
        if self._linesearch_config == config:
            return
        self.update_step = self.get_update_step()
        self._linesearch_config = config

    def optimize(
        self,
        max_steps: int = 1000,
        line_search: LineSearch | None = None,
        precision: float = DEFAULT_PRECISION,
        max_step_size: float = DEFAULT_MAX_STEP_SIZE,
        gram_schmidt_step_size: float = DEFAULT_GRAM_SCHMIDT_STEP_SIZE,
        callbacks: Callable | list[Callable] | tuple[Callable, ...] | None = None,
        least_squares_args: dict | None = None,
    ) -> Parameters:
        """Run the GEOPE optimisation loop.

        Iterates geodesic update steps until the fidelity exceeds
        ``precision`` or ``max_steps`` is reached.

        Args:
            max_steps: Maximum number of optimisation steps. Defaults to 1000.
            line_search: The :class:`~geope.line_searches.LineSearch` object
                tuning the geodesic step size each step (e.g. ``Armijo()``).
                Defaults to :class:`~geope.line_searches.Armijo`.
            precision: Target fidelity threshold. Defaults to 0.9999999.
                Host-side only (loop control) — zero compile impact.
            max_step_size: Maximum line-search step. Normalised (the default),
                the longest step has length ``max_step_size / (sqrt(G) *
                delta_t)`` in parameter space, i.e. it is per unit of total
                duration, whatever the scale of the least-squares solution; with
                ``least_squares_args={"normalized": False}`` the bracket is
                ``max_step_size`` itself, where ``1`` is the full Gauss-Newton
                (or LM) step. Defaults to 1.0. Baked into the jitted update, so
                changing it triggers a one-off recompile.
            gram_schmidt_step_size: Step size for the Gram-Schmidt fallback, per
                unit of total duration like ``max_step_size``. Defaults to 1.3.
                Host-side only — zero compile impact. A falsy value disables the
                fallback.
            callbacks: Optional callback, or list/tuple of callbacks, invoked at
                the end of every step with the signature
                ``callback(step, history, geope) -> bool``. All callbacks run
                each step; the loop stops early if any returns a falsy value
                (so a pure logging callback must ``return True``). ``step`` is
                the 1-based index of the step just completed, ``history`` is
                ``geope.history`` (may be ``None``), and ``geope`` is this
                optimiser.
            least_squares_args: Keyword arguments of :func:`least_squares`,
                which solves for the search direction each step: ``method``
                (``"svd"``, the default, or ``"lm"``), ``svd_cutoff``,
                ``lm_maxiter`` and ``lm_lambda_init``, plus ``normalized``
                (default ``True``), which picks the bracket described under
                ``max_step_size``. ``None`` (default) uses the defaults. Baked
                into the jitted update, so changing it triggers a one-off
                recompile. ``step_size`` is in units of the least-squares
                solution either way.

        A step is kept when it reduced the line search's own objective by more
        than ``PROGRESS_RTOL`` relatively (`GoldenSection` minimises
        the infidelity, the Armijo family the squared geodesic distance);
        otherwise the Gram-Schmidt fallback replaces it. Convergence is always
        tested on the fidelity.

        Returns:
            The bound `Parameters` instance, carrying the final
            ``parameters`` / ``fidelity``. Read the result via
            ``params.parameters``, ``params.fidelity`` or ``params.to_dict()``;
            the full trajectory and ``best_*`` live on ``geope.history`` when
            a `History` was supplied.
        """
        # Host-side run-control knobs (read by the loop / the non-jitted
        # gram_schmidt) — outside the compile memo entirely.
        self.precision = precision
        self.gram_schmidt_step_size = gram_schmidt_step_size

        # Select the line search and (re)build the JIT-compiled update functions
        # if this configuration differs from the last run. max_step_size is
        # passed in because it is baked into the jitted closure and so guarded
        # by the compile memo (it must not just be set as self.max_step_size
        # after a memo hit, or a reused closure would keep the old value).
        line_search = line_search or Armijo()
        self._configure_line_search(line_search, max_step_size, least_squares_args)

        # Reset the per-run line-search state. Doing this here — not inside
        # _configure_line_search — decouples fresh run state from compile reuse:
        # the memo can reuse the compiled update_step while any state a search
        # carries across steps still restarts on every optimize() call.
        self.line_search_state = self.line_search.init()

        # Build pulse-constrained update step if needed
        pulse_templates = None
        if self.pulse_constraints is not None and self.params.piecewise_steps > 1:
            combined_expander, pulse_templates = self._build_optimize_expander()
            update_step = self.get_update_step(expander_override=combined_expander)
        else:
            update_step = self.update_step

        cbs = normalize_callbacks(callbacks)

        step = 0
        while (self.params.fidelity < self.precision) and (step < max_steps):
            step += 1
            (
                coeffs,
                new_params_update,
                fidelity,
                value0,
                value,
                step_size,
                self.line_search_state,
                ls_diagnostics,
            ) = update_step(self.params.free(), self._delta_t(), self.line_search_state)
            # Pull the diagnostics to the host
            self.ls_diagnostics = {
                key: int(value) if key in ("rank", "lm_iter") else float(value)
                for key, value in ls_diagnostics.items()
            }

            # Accept on the objective the line search actually minimised
            progressed = value0 - value > PROGRESS_RTOL * abs(value0)

            if fidelity > self.precision:
                if self.verbose:
                    print(
                        f"[{step}/{max_steps}] [Fidelity = {fidelity}] A solution!                                                                     ",
                        end="\r",
                    )
            elif progressed:
                if self.verbose:
                    print(
                        f"[{step}/{max_steps}] [Fidelity = {fidelity}] Omega geodesic reduced the {self.line_search.objective} for this step...                 ",
                        end="\r",
                    )
            else:
                if self.verbose:
                    print(
                        f"[{step}/{max_steps}] [Fidelity = {self.params.fidelity}] Omega geodesic stalled on the {self.line_search.objective} for this step. Moving phi away...    ",
                        end="\r",
                    )
                # TODO: Can we just guarantees this is always a float by making it a property?
                if self.gram_schmidt_step_size:
                    new_params_update, fidelity, step_size = self.gram_schmidt(coeffs)

            self.add_parameters(new_params_update, fidelity, step_size)

            # Enforce pulse template constraints if applicable
            if pulse_templates is not None:
                self._enforce_pulse_template(pulse_templates)

            # Run user callbacks at the end of the step; stop early if any
            # requests it.
            if not run_callbacks(cbs, step, self.history, self):
                break
        if self.verbose:
            print()
        return self.params

    def add_parameters(
        self,
        params: np.ndarray | Array,
        fidelity: float | Array | None = None,
        step_size: float | None = None,
    ) -> float | Array:
        """Update the live parameters and fidelity, logging a step.

        Handles different parameter shapes by mapping them back to the
        full basis. Computes the fidelity if not provided, sets
        ``params.parameters`` / ``params.fidelity``, and records a row into
        ``history`` when one is attached.

        A recorded row reads :attr:`ls_diagnostics` as-is. Inside
        :meth:`optimize` that holds the current step's solve; called directly
        from user code it holds whatever the last solve left there (or the
        sentinels, if there has been none).

        Args:
            params: Parameter array (full, projected+drift, or projected shape).
            fidelity: Pre-computed fidelity value. If ``None``, it is
                computed from the parameters.
            step_size: Step size used. Defaults to ``self.max_step_size``.

        Returns:
            The fidelity of the new parameter set.
        """
        # Parameters are real by construction; the update arrives in the
        # holomorphic-Jacobian dtype (complex128, imaginary part exactly zero),
        # so drop it here rather than letting numpy discard it with a warning.
        params = np.real(np.asarray(params))
        if self._real_params:
            # Experimental space: only (piecewise_steps, n_exp) is valid
            new_params = np.array(params)
        elif params.shape == (
            self.params.piecewise_steps,
            self.params.basis.lie_algebra_dim,
        ):
            new_params = np.zeros(
                (self.params.piecewise_steps, self.params.basis.lie_algebra_dim)
            )
            new_params = params
        elif params.shape == (
            self.params.piecewise_steps,
            self.params.proj_drift_basis.lie_algebra_dim,
        ):
            new_params = np.zeros(
                (self.params.piecewise_steps, self.params.basis.lie_algebra_dim)
            )
            new_params[:, self.params.proj_drift_indices] = params
        elif params.shape == (
            self.params.piecewise_steps,
            self.params.projected_basis.lie_algebra_dim,
        ):
            new_params = np.zeros(
                (self.params.piecewise_steps, self.params.basis.lie_algebra_dim)
            )
            new_params[:, self.params.projected_indices] = params
            if self.params.drift_basis is not None:
                new_params[:, self.params.drift_indices] = jnp.tile(
                    self.drift_parameters, (self.params.piecewise_steps, 1)
                )
        else:
            raise ValueError(
                "Parameter shape does not match with full basis, projected & drift basis, or projected basis."
            )
        if fidelity is None:
            fidelity = self.params.manifold.fidelity_at(
                self.params.free(new_params), self._delta_t()
            )
        if step_size is None:
            step_size = self.max_step_size
        self.params.parameters = new_params
        self.params.fidelity = fidelity
        self.step_size = step_size
        if self.history is not None:
            self.history.record(self)
        return fidelity

    def _split_key(self) -> jax.Array:
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def _delta_t(self) -> Array:
        """``params.delta_t`` as a JAX array, ready for the jitted callables."""
        return jnp.asarray(self.params.delta_t, dtype=jnp.float64)

    def gram_schmidt(self, coeffs: Array) -> tuple[Array, Array, float]:
        """Generate a Gram-Schmidt orthogonal fallback direction.

        When the geodesic step fails to improve fidelity, this method
        constructs a random direction orthogonal to the previous
        coefficients and performs a signed step.

        Args:
            coeffs: The previous coefficient array from the geodesic step.

        Returns:
            A tuple ``(new_params, fidelity, step_size)``.
        """
        if self._real_params:
            n_exp = self.params.n_experimental_params
            proj_c = (
                np.array(
                    jax.random.normal(
                        self._split_key(), shape=(self.params.piecewise_steps, n_exp)
                    )
                )
                * 0.1
            )
        else:
            proj_c = np.array(
                [
                    prepare_random_parameters(
                        self.params.projected_indices,
                        self.constraint_expander,
                        key=self._split_key(),
                    )[self.params.proj_drift_indices]
                    for _ in range(self.params.piecewise_steps)
                ]
            )
            # The drift is deliberately absent from the direction: it is a
            # displacement added to parameters that already carry the drift.
        proj_c_con = np.concatenate(proj_c, axis=0)
        coeffs_con = np.concatenate(coeffs, axis=0)

        # Use the Gram-Schmidt procedure to generate a perpendicular vector to the previous coefficients.
        proj_c = jnp.reshape(
            proj_c_con
            - (((proj_c_con @ coeffs_con) / (coeffs_con @ coeffs_con)) * coeffs_con),
            coeffs.shape,
        )
        return self._update_parameters_gram_schmidt(proj_c)

    def _update_parameters_gram_schmidt(
        self, coeffs: np.ndarray
    ) -> tuple[np.ndarray, Array, float]:
        """Evaluate the best signed Gram-Schmidt step.

        Tests both positive and negative directions and returns the
        one that yields higher fidelity.

        Args:
            coeffs: Orthogonalised coefficient array.

        Returns:
            A tuple ``(new_parameters, fidelity, step_size)``.
        """
        fids = {}
        # Per unit of total duration, like the line-search bracket: a Gecko
        # subdivision (G -> mG, delta_t -> delta_t / m) leaves it unchanged.
        scaled_gs_step = self.gram_schmidt_step_size / self.params.total_time
        if self._real_params:
            current_params = self.params.free()
            for sign in [1, -1]:
                new_exp = current_params + sign * scaled_gs_step * coeffs
                fids[sign] = self.params.manifold.fidelity_at(
                    jnp.asarray(new_exp, dtype=jnp.float64), self._delta_t()
                )
            sign = 1 if fids[1] > fids[-1] else -1
            fidelity = fids[sign]
            new_parameters = current_params + sign * scaled_gs_step * coeffs
        else:
            # Move along the projected columns only. The drift is already in
            # ``self.params.parameters``, to which this direction is added, so
            # letting it leak into ``direction`` would displace the drift and
            # make the evaluated pulse differ from the one returned below.
            direction = np.zeros_like(coeffs)
            direction[:, self.params.proj_indices_projdrift_basis] = coeffs[
                :, self.params.proj_indices_projdrift_basis
            ]
            current_params = np.array(self.params.parameters)[
                :, self.params.proj_drift_indices
            ]
            # Evaluate the candidate arrays themselves, so the reported fidelity
            # is by construction the fidelity of the parameters returned.
            for sign in [1, -1]:
                trial = current_params + sign * scaled_gs_step * direction
                fids[sign] = self.params.manifold.fidelity_at(
                    jnp.asarray(trial), self._delta_t()
                )
            sign = 1 if fids[1] > fids[-1] else -1
            fidelity = fids[sign]
            new_parameters = current_params + sign * scaled_gs_step * direction

        # if self.parameter_bounds is not None:
        #     new_parameters, fidelity = self.bound_parameters(new_parameters, scaled_gs_step)

        return new_parameters, fidelity, sign * scaled_gs_step

    def _build_optimize_expander(self) -> tuple[Array, dict[int, np.ndarray]]:
        """Build the combined pulse + linear-constraint expander.

        Combines the pulse-shape expander with ``self.constraint_expander``
        via ``E_combined = C_g @ pinv(C_g) @ E_pulse`` where ``C_g`` is
        the Kronecker'd linear-constraint expander.

        Returns:
            A tuple ``(combined_expander_array, templates_dict)`` ready
            to pass into ``get_update_step(expander_override=...)``.
        """
        if getattr(self, "_real_params", False):
            proj_params = self.params.parameters
        else:
            proj_params = self.params.parameters[:, self.params.projected_indices]
        E_pulse, pulse_templates = build_pulse_expander(
            self.params.piecewise_steps,
            self.params.projected_basis,
            self.pulse_constraints,
            getattr(self, "_real_params", False),
            self.params.n_experimental_params,
            np.array(proj_params).real,
        )
        if self.constraint_expander is not None:
            E_gate = np.kron(
                np.eye(self.params.piecewise_steps), self.constraint_expander
            )
            combined = E_gate @ np.linalg.pinv(E_gate) @ E_pulse
        else:
            combined = E_pulse
        return jnp.array(combined), pulse_templates

    def _enforce_pulse_template(self, pulse_templates: dict[int, np.ndarray]) -> None:
        """Re-project constrained parameter columns onto their templates.

        Updates the live ``self.params.parameters`` so that for every
        constrained index $k$, the time profile of $\\phi_k$ is
        proportional to the stored unit template $t_k$. Then recomputes
        the fidelity and patches the last logged row when logging.

        Args:
            pulse_templates: Dict mapping projected (or experimental)
                index to the unit-norm template vector.
        """
        params = np.array(self.params.parameters)
        if getattr(self, "_real_params", False):
            for k, tmpl in pulse_templates.items():
                col = params[:, k].real
                scale = float(np.dot(col, tmpl))
                params[:, k] = scale * tmpl
        else:
            proj_where = np.where(self.params.projected_indices)[0]
            for k, tmpl in pulse_templates.items():
                full_idx = proj_where[k]
                col = params[:, full_idx].real
                scale = float(np.dot(col, tmpl))
                params[:, full_idx] = scale * tmpl
        self.params.parameters = params
        # Recompute fidelity after enforcement
        fid = float(
            self.params.manifold.fidelity_at(self.params.free(params), self._delta_t())
        )
        self.params.fidelity = fid
        if self.history is not None:
            if "parameters" in self.history.logs:
                self.history.logs["parameters"][-1] = np.array(params)
            if "fidelities" in self.history.logs:
                self.history.logs["fidelities"][-1] = fid
            if "infidelities" in self.history.logs:
                self.history.logs["infidelities"][-1] = 1 - fid

    def get_update_step(
        self, expander_override: Array | None = None
    ) -> Callable[..., tuple]:
        r"""Build the JIT-compiled geodesic update step.

        One jitted function per line-search and least-squares configuration, and
        one `geope.geometry.GeometricContext` per call of it. The step is: open
        the context, solve the least-squares problem for the linear combination
        of omegas that best matches the geodesic direction (:func:`least_squares`),
        attach its negation to the context, and line-search along it. The
        direction keeps its least-squares scale; ``normalized`` acts on the
        bracket instead.

        The context is built *inside* this function and never leaves it — it is a
        trace-time object, not a pytree. The solve and the line search read the
        same context, so the base point's propagator, Jacobian and matrix
        logarithm are each evaluated once per step however geometry-hungry the
        line search is. (``method="lm"`` adds its own trial evaluations, one
        propagator and one logarithm each, at points off the ray.)

        Args:
            expander_override: Optional constraint expander to use in
                place of ``self.constraint_expander``'s Kronecker'd
                version. Used by pulse-shape constraints.

        Returns:
            A JIT-compiled callable ``update_step(free_params, delta_t, ls_state)``
            returning ``(coeffs, new_params, fidelity, value0, value, step_size,
            new_ls_state, ls_diagnostics)``.

            ``value`` is the line search's own objective at the accepted step and
            ``value0`` the same objective at ``t = 0``, which is what
            :meth:`optimize` tests for progress. ``ls_diagnostics`` describes the
            least-squares solve for this step (see :func:`least_squares`).

            ``coeffs`` is the negated least-squares solution at its own scale.
            With ``normalized=True`` the bracket is
            $[0, \texttt{max\_step\_size}/(\sqrt{N_g}\,\lVert\text{sol}\rVert\,
            \Delta t)]$, so its far end is a step of fixed length
            $\texttt{max\_step\_size}/(\sqrt{N_g}\,\Delta t)$ in parameter space
            (an empty bracket if the solution vanishes); otherwise it is
            $[0, \texttt{max\_step\_size}]$, where a unit step is the full
            Gauss-Newton (or LM) step. ``step_size`` is in units of ``coeffs``
            either way.
        """
        manifold = self.params.manifold
        line_search = self.line_search
        # Bracket half-width and the solve's configuration: baked in, so they
        # join the compile memo.
        max_step_size = self.max_step_size
        least_squares_args = dict(self.least_squares_args)
        # Not the solve's but the bracket's: see below.
        normalized = least_squares_args.pop("normalized")

        @jax.jit
        def update_step(free_params, delta_t, ls_state):
            ctx = manifold.context(free_params, delta_t)

            if expander_override is not None:
                expander_gates = expander_override
            elif self.constraint_expander is not None:
                expander_gates = jnp.kron(
                    jnp.eye(self.params.piecewise_steps), self.constraint_expander
                )
            else:
                expander_gates = None

            # The solve reads only the context's direction-free tier 0, which the
            # line search below reuses from the same memo.
            sol, ls_diagnostics = least_squares(
                ctx, expander_gates, **least_squares_args
            )

            # Scatter the solved columns back over the chart's parameters (a
            # no-op under param_transform, where every column is solvable) and
            # negate: the solve matches the tangent to A, which points uphill.
            coeffs = -manifold.tangent.embed(sol)
            ctx.set_direction(coeffs)

            # One-sided bracket on the descent side: alpha = 0 is "don't move" and
            # a useful step is positive (see `GeometricContext.A`).
            if normalized:
                # Normalised through the bracket rather than the direction: its
                # far end is a step of length max_step_size / (sqrt(G) * delta_t)
                # in parameter space, whatever the least-squares scale of `sol`.
                # The 1 / (sqrt(G) * delta_t) keeps the step's effect on the
                # unitary independent of how the duration is split into
                # segments (the chart scales the pulse by delta_t). A vanishing
                # solution gets an empty bracket rather than an infinite one.
                norm = jnp.linalg.norm(sol)
                nonzero = norm > 0
                alpha_max = jnp.where(
                    nonzero,
                    max_step_size
                    / (
                        jnp.sqrt(free_params.shape[0])
                        * jnp.where(nonzero, norm, 1.0)
                        * delta_t
                    ),
                    0.0,
                )
            else:
                # The solution keeps its least-squares scale, which already
                # carries that factor (the omegas contain the chart's -delta_t),
                # so alpha = 1 is the full Gauss-Newton (or LM) step.
                alpha_max = jnp.asarray(max_step_size, jnp.float64)
            result = line_search(
                ctx, jnp.asarray(0.0, jnp.float64), alpha_max, ls_state
            )
            new_params = free_params + result.step_size * coeffs

            if line_search.objective == "infidelity":
                # The search already evaluated the infidelity at the accepted
                # step, and fidelity = 1 - infidelity exactly: no extra propagator.
                value0 = ctx.infidelity
                fidelity = 1.0 - result.value
            else:
                value0 = ctx.F0
                fidelity = manifold.fidelity_at(new_params, delta_t)

            return (
                coeffs,
                new_params,
                fidelity,
                value0,
                result.value,
                result.step_size,
                result.state,
                ls_diagnostics,
            )

        return update_step


def build_pulse_expander(
    piecewise_steps: int,
    projected_basis: Basis,
    pulse_constraints: dict | list,
    real_params: bool,
    n_exp: int,
    proj_params: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Build a flat-space expander enforcing pulse-shape constraints.

    For each constrained parameter index $k$ the time profile $\\phi_k(g)$ is
    constrained to a fixed unit-norm template $t_k\\in\\mathbb{R}^{N_g}$,
    leaving only the scalar amplitude free. Unconstrained parameters retain one
    free variable per gate.

    Args:
        piecewise_steps: Number of gate segments ``N_g``.
        projected_basis: The projected (controllable) ``Basis`` (used for the
            label lookup in projected space).
        pulse_constraints: The pulse-shape constraint config. In experimental
            space, an iterable of integer parameter indices; in projected
            space, a control-format dict ``{qubit_index_or_tuple: [labels]}``.
        real_params: Whether the optimiser operates in experimental
            (``param_transform``) space.
        n_exp: Number of experimental parameters (used when ``real_params``).
        proj_params: Current projected parameter ``np.ndarray`` used as the
            shape template.

    Returns:
        A tuple ``(E, templates)`` where ``E`` is the
        ``(N_g * n_proj, n_free)`` expander matrix and ``templates`` maps each
        constrained integer index to its unit-norm template vector.
    """
    L = piecewise_steps
    if real_params:
        n_proj = n_exp
        pulse_indices = list(pulse_constraints)
    else:
        n_proj = projected_basis.lie_algebra_dim
        proj_labels = list(projected_basis.labels)
        if not isinstance(pulse_constraints, dict):
            raise TypeError(
                "pulse_constraints must be a control-format dict in "
                "projected space, e.g. {(1, 2): ['zz']}."
            )
        pulse_indices = control_to_indices(proj_labels, pulse_constraints, strict=True)
    pulse_set = set(pulse_indices)
    non_pulse = [k for k in range(n_proj) if k not in pulse_set]

    n_free = L * len(non_pulse) + len(pulse_indices)
    E = np.zeros((L * n_proj, n_free))

    col = 0
    for g in range(L):
        for k in non_pulse:
            E[k + g * n_proj, col] = 1.0
            col += 1

    templates = {}
    for k in pulse_indices:
        template = np.array(proj_params[:, k]).real
        norm_t = np.linalg.norm(template)
        if norm_t < 1e-12:
            template = np.ones(L) / np.sqrt(L)
        else:
            template = template / norm_t
        templates[k] = template
        for g in range(L):
            E[k + g * n_proj, col] = float(template[g])
        col += 1

    return E, templates


def least_squares(
    ctx: GeometricContext,
    expander: Array | None,
    *,
    method: str = "svd",
    svd_cutoff: float | None = None,
    lm_maxiter: int = 10,
    lm_lambda_init: float = 1e-3,
) -> tuple[Array, dict[str, Array]]:
    r"""Solve the geodesic least-squares problem off a step's context.

    Finds the linear combination of the context's omegas that best reproduces
    its gammas, optionally through a constraint expander. Writing
    $A=\omega^\intercal E$ (or $\omega^\intercal$ when there is no expander) and
    $b=\gamma$, the two methods solve

    $$x_{\text{svd}} = \arg\min_x \lVert A x - b\rVert_2, \qquad
      x_{\text{lm}}(\lambda) = \arg\min_x \lVert A x - b\rVert_2^2
        + \lambda\lVert x\rVert_2^2 .$$

    ``"lm"`` is Levenberg damping of that solve, read as a Gauss-Newton step on
    the squared geodesic distance $F_0$: one SVD of $A$ gives $x(\lambda)$ for
    every $\lambda$ through the filter factors $\sigma_i/(\sigma_i^2+\lambda)$.
    Starting from $\lambda_0 = \texttt{lm\_lambda\_init}\cdot\sigma_0^2$ (scaled
    by the largest singular value, so the setting is scale-free), it accepts the
    first $\lambda$ whose full step $\phi - x(\lambda)$ lowers the distance below
    the context's own `F0`, and multiplies $\lambda$ by `LM_LAMBDA_FACTOR`
    otherwise. $\lambda$ restarts at $\lambda_0$ on every call.

    **Context discipline.** ``ctx`` is the same object the line search receives
    next, so this function only *reads* its direction-free tier 0 (``omegas``,
    ``gammas`` and, for ``"lm"``, ``F0``), all of which the line search reuses
    from the memo. It never calls ``ctx.set_direction`` (a context takes one
    direction, and the caller sets it), and the ``"lm"`` loop touches the
    context only through values read *before* the loop, because a
    ``cached_property`` first read inside ``lax`` control flow would memoise a
    tracer of the loop body.

    Args:
        ctx: The step's :class:`~geope.geometry.GeometricContext`, with no
            direction set.
        expander: Optional constraint expansion ``Array``, or ``None``.
        method: ``"svd"`` (default) for ``jnp.linalg.lstsq``, ``"lm"`` for
            Levenberg damping with accept/reject on the geodesic distance.
        svd_cutoff: Condition-number cap of the SVD: singular values below
            ``s[0] / svd_cutoff`` are discarded. ``None`` keeps the
            ``jnp.linalg.lstsq`` default, $\varepsilon\max(M, N)$ relative.
        lm_maxiter: Maximum number of trial $\lambda$ values for ``"lm"``,
            each costing one propagator and one logarithm. If none lowers the
            distance, the last (most damped) solution is returned.
        lm_lambda_init: Initial damping for ``"lm"``, relative to $\sigma_0^2$.

    Returns:
        A tuple ``(sol, diagnostics)``. ``sol`` has shape
        ``(piecewise_steps, K_solvable)``, in the solvable columns (scatter it
        with ``manifold.tangent.embed``), pointing *uphill* — the caller
        negates it — and at its least-squares scale, where a unit step is the
        full Gauss-Newton (or LM) step. ``Geope`` normalises through the
        line-search bracket rather than by rescaling it. ``diagnostics`` holds
        scalars

        * ``residual`` — $\lVert A x - b\rVert_2$.
        * ``residual_rel`` — $\lVert A x-b\rVert_2/\lVert b\rVert_2$, in
          $[0, 1]$; ``0.0`` when $\lVert b\rVert_2$ vanishes. This is the
          headline number: it measures how much of the geodesic direction lies
          outside the controllable subspace, and is comparable across steps and
          across problems.
        * ``rank`` — the numerical rank of $A$ under the cutoff.
        * ``cond`` — $\sigma_0/\sigma_{\text{rank}-1}$, the condition number
          over the singular values retained above the cutoff, i.e. the
          effective conditioning of the solve.
        * ``damping`` — the $\lambda$ that produced $x$; ``0.0`` for ``"svd"``.
        * ``lm_iter`` — the number of trial $\lambda$ evaluated; ``0`` for
          ``"svd"``.

    Raises:
        ValueError: If ``method`` is not ``"svd"`` or ``"lm"``.
    """
    omegas = ctx.omegas
    gammas = ctx.gammas
    sol_shape = omegas.shape[:2]

    comb_vecs = jnp.concatenate(omegas, axis=0)
    comb_vecs_T = comb_vecs.T @ expander if expander is not None else comb_vecs.T
    rcond = None if svd_cutoff is None else 1.0 / svd_cutoff
    diagnostics = {}

    if method == "svd":
        x, residual2, rank, singular_values = jnp.linalg.lstsq(
            comb_vecs_T, gammas, rcond=rcond
        )
        residual = jnp.sqrt(residual2)
    elif method == "lm":
        x, rank, singular_values, damping, lm_iter = _levenberg_marquardt(
            ctx,
            comb_vecs_T,
            gammas,
            expander,
            sol_shape,
            lm_maxiter,
            lm_lambda_init,
        )
        residual = jnp.linalg.norm(comb_vecs_T @ x - gammas)
        diagnostics["damping"] = damping
        diagnostics["lm_iter"] = lm_iter
    else:
        raise ValueError(
            f"Unknown least-squares method {method!r}; expected one of "
            f"{LEAST_SQUARES_METHODS}."
        )

    sol = expander @ x if expander is not None else x
    sol = sol.reshape(sol_shape)
    # residual and relative residual
    diagnostics["residual"] = residual
    residual_rel = residual / jnp.linalg.norm(gammas)
    diagnostics["residual_rel"] = residual_rel
    # Smallest singular value
    smallest_kept = singular_values[rank - 1]
    positive = smallest_kept > 0
    safe_smallest: Array = jnp.where(positive, smallest_kept, 1.0)
    # Add diagnostics
    diagnostics["cond"] = jnp.where(
        positive, singular_values[0] / safe_smallest, jnp.inf
    )
    diagnostics["rank"] = rank

    return sol, diagnostics


def _levenberg_marquardt(
    ctx: GeometricContext,
    lhs: Array,
    rhs: Array,
    expander: Array | None,
    sol_shape: tuple[int, int],
    maxiter: int,
    lambda_init: float,
) -> tuple[Array, Array, Array, Array, Array]:
    """Levenberg-damped solve of ``lhs @ x = rhs``, accepting on the distance.

    See :func:`least_squares`. Returns ``(x, rank, singular_values, damping,
    n_trials)``, where ``damping`` is the $\\lambda$ that produced ``x``.
    """
    # Every context read happens here, before the loop.
    F0 = ctx.F0
    manifold = ctx.manifold
    free_params = ctx.free_params
    delta_t = ctx.delta_t
    target = ctx.target

    u, s, vh = jnp.linalg.svd(lhs, full_matrices=False)
    rcond = float(jnp.finfo(jnp.result_type(lhs, rhs)).eps) * max(lhs.shape)
    keep = s >= rcond * s[0]
    rank = keep.sum()
    v = vh.conj().T
    u_b = u.conj().T @ rhs

    def solve(lam):
        denom = jnp.where(keep, s**2 + lam, 1.0)
        return v @ (jnp.where(keep, s / denom, 0.0) * u_b)

    def distance(x):
        sol = (expander @ x if expander is not None else x).reshape(sol_shape)
        point = manifold.compute_point(
            free_params - manifold.tangent.embed(sol), delta_t
        )
        return manifold.distance2(point, target)

    def cond(carry):
        n_trials, _, _, _, accepted = carry
        return (~accepted) & (n_trials < maxiter)

    def body(carry):
        n_trials, lam, _, _, _ = carry
        x = solve(lam)
        # A NaN distance compares False, so it is rejected like an increase.
        accepted = distance(x) < F0
        return n_trials + 1, lam * LM_LAMBDA_FACTOR, lam, x, accepted

    lam0 = jnp.asarray(lambda_init * s[0] ** 2, jnp.float64)
    x0 = jnp.zeros(v.shape[0], jnp.result_type(v, u_b))
    n_trials, _, damping, x, _ = jax.lax.while_loop(
        cond,
        body,
        (jnp.asarray(0, jnp.int32), lam0, lam0, x0, jnp.asarray(False)),
    )
    return x, rank, s, damping, n_trials


def _check_least_squares_args(args: dict | None) -> dict:
    """Fill ``least_squares_args`` with the defaults of :func:`least_squares`.

    Host-side, so a bad configuration fails in ``optimize`` before anything is
    compiled. Returns a fresh dict, never the caller's. Besides the keyword
    arguments of :func:`least_squares` it carries ``normalized``, which is not
    the solve's but the bracket's: ``Geope``'s update step takes it off before
    calling the solve.

    Raises:
        TypeError: On a key that is neither a keyword of :func:`least_squares`
            nor ``normalized``.
        ValueError: On an unknown ``method``, a non-positive ``svd_cutoff`` or
            ``lm_lambda_init``, or ``lm_maxiter < 1``.
    """
    defaults = {
        name: p.default
        for name, p in inspect.signature(least_squares).parameters.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    }
    defaults["normalized"] = True
    args = dict(args or {})
    unknown = set(args) - set(defaults)
    if unknown:
        raise TypeError(
            f"Unknown least_squares_args {sorted(unknown)}; expected a subset "
            f"of {sorted(defaults)}."
        )
    resolved = {**defaults, **args}
    if resolved["method"] not in LEAST_SQUARES_METHODS:
        raise ValueError(
            f"Unknown least-squares method {resolved['method']!r}; expected one "
            f"of {LEAST_SQUARES_METHODS}."
        )
    if resolved["svd_cutoff"] is not None and not resolved["svd_cutoff"] > 0:
        raise ValueError("svd_cutoff must be positive (a condition-number cap).")
    if not resolved["lm_maxiter"] >= 1:
        raise ValueError("lm_maxiter must be at least 1.")
    if not resolved["lm_lambda_init"] > 0:
        raise ValueError("lm_lambda_init must be positive.")
    return resolved
