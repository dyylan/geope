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

# Levenberg-Marquardt trust-region controller, Moré (1978), Algorithm (7.1).
# Relative tolerance on matching the step length to the radius:
# | ||h|| - radius | <= LM_SIGMA * radius.
LM_SIGMA = 0.1
# Gain ratio above which a trial step is accepted.
LM_ACCEPT_RHO = 1e-4
# Gain ratios at or below / at or above which the radius shrinks / grows.
LM_SHRINK_RHO = 0.25
LM_GROW_RHO = 0.75
# Bounds on the radius contraction factor mu.
LM_MU_MIN = 0.1
LM_MU_MAX = 0.5
# Iteration cap on the Hebden iteration matching the damping to the radius.
LM_HEBDEN_MAXITER = 10


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
            ``residual_rel``, ``rank`` and ``cond``, plus ``damping``,
            ``lm_iter``, ``lm_rho``, ``lm_radius`` and ``lm_hit`` for
            ``method="lm"`` only (see :func:`least_squares`); :meth:`optimize`
            re-keys it for its method. Seeded with ``nan`` / ``-1`` / ``False``
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
        # least-squares diagnostics, keyed for the configured method
        self.ls_diagnostics = _ls_diagnostic_sentinels(
            self.least_squares_args["method"]
        )

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
            max_step_size: Maximum line-search step. For the normalised
                ``"svd"`` solve (the default), the longest step has length
                ``max_step_size / (sqrt(G) * delta_t)`` in parameter space, i.e.
                it is per unit of total duration, whatever the scale of the
                least-squares solution. With ``"normalized": False``, and always
                for ``"lm"``, the bracket is ``max_step_size`` itself, where
                ``1`` is the full Gauss-Newton step (for ``"lm"``, the accepted
                trust-region step). Defaults to 1.0. Baked into the jitted
                update, so changing it triggers a one-off recompile.
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
                ``lm_maxiter`` and ``lm_radius_init``, plus ``normalized``
                (default ``True``), which picks the ``"svd"`` bracket described
                under ``max_step_size`` and is ignored by ``"lm"``. ``None``
                (default) uses the defaults. Baked into the jitted update, so
                changing it triggers a one-off recompile. ``step_size`` is in
                units of the least-squares solution either way. The ``"lm"``
                trust-region radius is carried across steps in
                :attr:`line_search_state` (``"lm_radius"``) and restarts with it
                on every call.

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

        # The diagnostics' keys follow the method: "lm" reports its trust-region
        # keys and "svd" none of them. Keep what the keys both share, seed the
        # new ones, and drop the other method's.
        self.ls_diagnostics = {
            key: self.ls_diagnostics.get(key, sentinel)
            for key, sentinel in _ls_diagnostic_sentinels(
                self.least_squares_args["method"]
            ).items()
        }

        # Build pulse-constrained update step if needed
        pulse_templates = None
        combined_expander = None
        if self.pulse_constraints is not None and self.params.piecewise_steps > 1:
            combined_expander, pulse_templates = self._build_optimize_expander()
            update_step = self.get_update_step(expander_override=combined_expander)
        else:
            update_step = self.update_step

        # The solve's own state ("lm": its trust-region radius and Moré
        # scaling) rides in the same pytree and restarts with it. It is sized
        # by the solve's unknowns, hence after the expander is known.
        self.line_search_state |= self._least_squares_state(combined_expander)

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
                key: _DIAGNOSTIC_CASTS.get(key, float)(value)
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

    def _solve_expander(self, expander_override: Array | None = None) -> Array | None:
        """The least-squares solve's constraint expander, or ``None``.

        ``expander_override`` (the pulse-constraint path's) when given, else the
        per-gate Kronecker'd ``constraint_expander``, else no expander.
        """
        if expander_override is not None:
            return expander_override
        if self.constraint_expander is not None:
            return jnp.kron(
                jnp.eye(self.params.piecewise_steps), self.constraint_expander
            )
        return None

    def _least_squares_state(self, expander_override: Array | None = None) -> dict:
        """The solve's own entries of the threaded state, fresh for a run.

        Empty for ``"svd"``. For ``"lm"``: an unset trust-region radius
        ``"lm_radius"`` (NaN, so the first step sizes it from the pulse it
        starts at) and Moré's scaling ``"lm_scale"``, the running
        maximum of the column norms of the solve's matrix — one entry per
        unknown, starting at zero so the first step's norms become it.
        """
        if self.least_squares_args["method"] != "lm":
            return {}
        expander = self._solve_expander(expander_override)
        if expander is not None:
            n_unknowns = expander.shape[1]
        else:
            columns = self.params.manifold.tangent.columns
            per_gate = (
                self.params.free().shape[1] if columns is None else int(columns.sum())
            )
            n_unknowns = self.params.piecewise_steps * per_gate
        return {
            "lm_radius": jnp.asarray(jnp.nan, jnp.float64),
            "lm_scale": jnp.zeros(n_unknowns, jnp.float64),
        }

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
        bracket instead, and only for ``"svd"``. The ``"lm"`` trust-region
        radius is threaded through ``ls_state`` under ``"lm_radius"``: the
        solve takes the state and returns it updated, and the line search's own
        state is laid over it, since a search rebuilds only its own keys.

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
            For ``"svd"`` with ``normalized=True`` the bracket is
            $[0, \texttt{max\_step\_size}/(\sqrt{N_g}\,\lVert\text{sol}\rVert\,
            \Delta t)]$, so its far end is a step of fixed length
            $\texttt{max\_step\_size}/(\sqrt{N_g}\,\Delta t)$ in parameter space
            (an empty bracket if the solution vanishes); otherwise it is
            $[0, \texttt{max\_step\_size}]$, where a unit step is the full
            Gauss-Newton step, or for ``"lm"`` the accepted trust-region step.
            ``step_size`` is in units of ``coeffs`` either way.
        """
        manifold = self.params.manifold
        line_search = self.line_search
        # Bracket half-width and the solve's configuration: baked in, so they
        # join the compile memo.
        max_step_size = self.max_step_size
        least_squares_args = dict(self.least_squares_args)
        # Not the solve's but the bracket's, and only the plain SVD solve's: an
        # LM step's length is its trust region's, which alpha = 1 takes.
        normalized = least_squares_args.pop("normalized")
        normalized = normalized and least_squares_args["method"] == "svd"
        expander_gates = self._solve_expander(expander_override)

        @jax.jit
        def update_step(free_params, delta_t, ls_state):
            ctx = manifold.context(free_params, delta_t)

            # The solve reads only the context's direction-free tier 0, which the
            # line search below reuses from the same memo. It shares the threaded
            # state with the line search, keeping its own keys in it ("lm" its
            # trust-region radius and scaling).
            sol, ls_diagnostics, ls_state = least_squares(
                ctx, expander_gates, ls_state, **least_squares_args
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
            # A line search rebuilds its state from its own keys only, so lay it
            # over the solve's rather than replacing it.
            new_ls_state = {**ls_state, **result.state}
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
                new_ls_state,
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
    ls_state: dict | None = None,
    *,
    method: str = "svd",
    svd_cutoff: float | None = None,
    lm_maxiter: int = 10,
    lm_radius_init: float = 100.0,
) -> tuple[Array, dict[str, Array], dict]:
    r"""Solve the geodesic least-squares problem off a step's context.

    Finds the linear combination of the context's omegas that best reproduces
    its gammas, optionally through a constraint expander. Writing
    $A=\omega^\intercal E$ (or $\omega^\intercal$ when there is no expander) and
    $b=\gamma$, the two methods solve

    $$x_{\text{svd}} = \arg\min_x \lVert A x - b\rVert_2, \qquad
      x_{\text{lm}} = \arg\min_{\lVert Dx\rVert_2\le\Delta}
        \lVert A x - b\rVert_2 .$$

    ``"lm"`` is the trust-region Levenberg-Marquardt step of Moré (1978), read
    as a Gauss-Newton step on the squared geodesic distance $F_0$. For a radius
    $\Delta$ it takes the undamped step if that fits within $(1+\sigma)\Delta$,
    and otherwise the damped step $x(\lambda)=(A^\dagger A+\lambda D^2)^{-1}
    A^\dagger b$ whose scaled length $\lVert Dx\rVert$ matches $\Delta$ to
    within $\sigma\Delta$, choosing $\lambda$ by Moré's safeguarded Hebden
    iteration. One SVD of $AD^{-1}$ serves every $\lambda$, through the filter
    factors $\sigma_i/(\sigma_i^2+\lambda)$, so this costs no propagator. Each
    trial $\phi - x$ is then scored by the gain ratio of actual to predicted
    reduction,

    $$\rho = \frac{1 - F(x)/F_0}
      {(\lVert Ax\rVert^2 + 2\lambda\lVert Dx\rVert^2)/\lVert b\rVert^2},$$

    both relative, so the constant of `Manifold.coefficients` cancels between
    the distance and the model. It is accepted when $\rho > 10^{-4}$; the
    radius shrinks by Moré's quadratic-fit factor $\mu\in[0.1, 0.5]$ when
    $\rho\le\tfrac14$, grows to $2\lVert Dx\rVert$ when $\rho\ge\tfrac34$ (or
    the step was undamped), and stays put otherwise. Rejected trials retry at
    the same base point with the shrunk radius, up to ``lm_maxiter`` of them.

    The scaling $D=\mathrm{diag}(d)$ is Moré's (§6), in the solve's own
    coordinates (after the expander): $d_i$ tracks the largest norm column $i$
    of $A$ has had — the square root of the Gauss-Newton curvature
    $(A^\dagger A)_{ii}$ along that unknown — and never shrinks, so a stiffer
    unknown gets a shorter axis of the trust region. Like the radius it is
    carried across steps in ``ls_state``. Only the numerical rank of $AD^{-1}$
    is cut — the damping, not a spectral truncation, regularises the solve.

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
        ls_state: The step's threaded state, ``Geope.line_search_state``; ``None``
            for an empty one. The solve reads and writes only its own keys,
            which the previous ``"lm"`` step handed on: ``"lm_radius"``, the
            trust-region radius (missing, NaN or non-positive means unset: the
            first step of a run, which starts from ``lm_radius_init``), and
            ``"lm_scale"``, the running
            maximum of the column norms of $A$, one per unknown (missing means
            none yet: the current norms). ``"svd"`` keeps no state.
        method: ``"svd"`` (default) for ``jnp.linalg.lstsq``, ``"lm"`` for the
            trust-region Levenberg-Marquardt step.
        svd_cutoff: Condition-number cap of the ``"svd"`` solve: singular
            values below ``s[0] / svd_cutoff`` are discarded. ``None`` keeps the
            ``jnp.linalg.lstsq`` default, $\varepsilon\max(M, N)$ relative.
            ``"lm"`` always uses that default, for the numerical rank only.
        lm_maxiter: Maximum number of trial steps for ``"lm"``, each costing
            one propagator and one logarithm. If none is accepted, the last
            (shortest) one is returned.
        lm_radius_init: The first radius of a run for ``"lm"``, relative to
            the scaled size $\lVert Dx_0\rVert$ of the pulse it starts from,
            $x_0$ being the pulse's solvable columns in the solve's unknowns
            (through the expander, by least squares, when there is one); for
            a zero pulse, the radius itself. MINPACK's ``factor``, and its
            default ``100``. Like MINPACK's, it counts an unknown whose column
            of $A$ has never moved at $d_i = 1$. It is deliberately not relative to the undamped
            step, which is longest where $A$ is worst conditioned: near a zero
            pulse that step can be $10^4$ long and its model wrong at a
            thousandth of it.

    Returns:
        A tuple ``(sol, diagnostics, ls_state)``. ``sol`` has shape
        ``(piecewise_steps, K_solvable)``, in the solvable columns (scatter it
        with ``manifold.tangent.embed``), pointing *uphill* — the caller
        negates it — and at its least-squares scale, where a unit step is the
        full Gauss-Newton (or LM) step. ``Geope`` normalises through the
        line-search bracket rather than by rescaling it. ``ls_state`` is a copy
        of the input state, with ``"lm_radius"`` and ``"lm_scale"`` updated for
        ``"lm"`` and otherwise unchanged. ``diagnostics`` holds scalars

        * ``residual`` — $\lVert A x - b\rVert_2$.
        * ``residual_rel`` — $\lVert A x-b\rVert_2/\lVert b\rVert_2$, in
          $[0, 1]$; ``0.0`` when $\lVert b\rVert_2$ vanishes. This is the
          headline number: it measures how much of the geodesic direction lies
          outside the controllable subspace, and is comparable across steps and
          across problems.
        * ``rank`` — the numerical rank of $A$ under the cutoff (for ``"lm"``,
          of the scaled $AD^{-1}$ it factors).
        * ``cond`` — $\sigma_0/\sigma_{\text{rank}-1}$, the condition number
          over the singular values retained above the cutoff, i.e. the
          effective conditioning of the solve (for ``"lm"``, of $AD^{-1}$).

        and, for ``"lm"`` only,

        * ``damping`` — the $\lambda$ that produced $x$; ``0.0`` for an
          undamped step.
        * ``lm_iter`` — the number of trial steps evaluated.
        * ``lm_rho`` — the gain ratio of the returned trial.
        * ``lm_radius`` — the radius for the next step's solve, in the scaled
          norm $\lVert Dx\rVert$, also written to the returned ``ls_state``.
        * ``lm_hit`` — whether the returned step lies on the trust-region
          boundary, i.e. was damped onto it ($\lambda > 0$, so
          $\lvert\lVert Dx\rVert-\Delta\rvert\le\sigma\Delta$); ``False`` when
          the undamped step fitted inside the radius, or there was nothing to
          solve.

    Raises:
        ValueError: If ``method`` is not ``"svd"`` or ``"lm"``.
    """
    omegas = ctx.omegas
    gammas = ctx.gammas
    sol_shape = omegas.shape[:2]

    comb_vecs = jnp.concatenate(omegas, axis=0)
    comb_vecs_T = comb_vecs.T @ expander if expander is not None else comb_vecs.T
    rcond = None if svd_cutoff is None else 1.0 / svd_cutoff
    ls_state = dict(ls_state or {})

    if method == "svd":
        x, residual2, rank, singular_values = jnp.linalg.lstsq(
            comb_vecs_T, gammas, rcond=rcond
        )
        # The squared residual is shaped (1,) because the target is 1-D.
        residual = jnp.sqrt(residual2[0])
        diagnostics = {}
    elif method == "lm":
        x, rank, singular_values, diagnostics, scale = _levenberg_marquardt(
            ctx,
            comb_vecs_T,
            gammas,
            expander,
            sol_shape,
            ls_state.get("lm_radius"),
            ls_state.get("lm_scale"),
            lm_maxiter,
            lm_radius_init,
        )
        ls_state["lm_radius"] = diagnostics["lm_radius"]
        ls_state["lm_scale"] = scale
        residual = jnp.linalg.norm(comb_vecs_T @ x - gammas)
    else:
        raise ValueError(
            f"Unknown least-squares method {method!r}; expected one of "
            f"{LEAST_SQUARES_METHODS}."
        )

    sol = expander @ x if expander is not None else x
    sol = sol.reshape(sol_shape)
    # residual and relative residual
    diagnostics["residual"] = residual
    # Near convergence the geodesic tangent vanishes, so the relative residual
    # is a genuine 0/0; define it as 0 there (the residual vanishes with it).
    target_norm = jnp.linalg.norm(gammas)
    nonzero = target_norm > 0
    diagnostics["residual_rel"] = jnp.where(
        nonzero, residual / jnp.where(nonzero, target_norm, 1.0), 0.0
    )
    # Smallest singular value
    smallest_kept = singular_values[rank - 1]
    positive = smallest_kept > 0
    safe_smallest: Array = jnp.where(positive, smallest_kept, 1.0)
    # Add diagnostics
    diagnostics["cond"] = jnp.where(
        positive, singular_values[0] / safe_smallest, jnp.inf
    )
    diagnostics["rank"] = rank

    return sol, diagnostics, ls_state


def _levenberg_marquardt(
    ctx: GeometricContext,
    lhs: Array,
    rhs: Array,
    expander: Array | None,
    sol_shape: tuple[int, int],
    radius: Array | None,
    scale: Array | None,
    maxiter: int,
    radius_init: float,
) -> tuple[Array, Array, Array, dict[str, Array], Array]:
    r"""Moré's trust-region Levenberg-Marquardt step for ``lhs @ x = rhs``.

    See :func:`least_squares`. Returns ``(x, rank, singular_values,
    diagnostics, scale)``: ``diagnostics`` holds ``damping``, ``lm_iter``,
    ``lm_rho``, ``lm_radius`` and ``lm_hit``, and ``scale`` is the updated
    diagonal of $D$.

    Two loops, as in Moré (1978): the outer one evaluates the distance at a
    trial step and adapts the radius (Algorithm (7.1)); the inner one picks the
    damping that fits the step to the radius (§5) from the SVD alone.

    **Scaling (§6).** $D = \mathrm{diag}(d)$ tracks, per unknown, the largest
    column norm of $A$ seen so far, $d_i \leftarrow \max(d_i,
    \lVert A_{:,i}\rVert)$ — the square root of the Gauss-Newton curvature
    $(A^\dagger A)_{ii}$ along that unknown — so it never shrinks; a column not
    yet seen to move is floored at 1, as in MINPACK. A stiffer unknown thus gets
    a shorter axis of the trust region $\lVert Dx\rVert\le\Delta$ and a heavier
    penalty $\lambda\lVert Dx\rVert^2$. Everything below works in the scaled
    unknowns $w = Dx$ on $\tilde A = AD^{-1} = U\Sigma V^\dagger$: with
    $c_i = s_i^2\lvert z_i\rvert^2$, $z = U^\dagger b$, every quantity the loops
    need is a spectral sum, $\lVert Dx(\lambda)\rVert^2 =
    \sum_i c_i/(s_i^2+\lambda)^2$ and $\lVert Ax(\lambda)\rVert^2 =
    \sum_i c_i s_i^2/(s_i^2+\lambda)^2$. The undamped step is accordingly the
    least-squares solution of least $\lVert Dx\rVert$, and ``rank`` and the
    singular values are those of $\tilde A$.
    """
    # Every context read happens here, before the loops.
    F0 = ctx.F0
    manifold = ctx.manifold
    free_params = ctx.free_params
    delta_t = ctx.delta_t
    target = ctx.target

    column_norms = jnp.linalg.norm(lhs, axis=0)
    scale = column_norms if scale is None else jnp.maximum(scale, column_norms)
    scale = jnp.where(scale > 0, scale, 1.0)

    u, s, vh = jnp.linalg.svd(lhs / scale, full_matrices=False)
    # Only the numerical rank is cut. On the kept directions the reduced problem
    # has full rank, which Moré's lower bound on the damping assumes.
    eps = float(jnp.finfo(jnp.result_type(lhs, rhs)).eps) * max(lhs.shape)
    keep = s >= eps * s[0]
    rank = keep.sum()
    s2 = jnp.where(keep, s**2, 1.0)
    z = u.conj().T @ rhs
    c = jnp.where(keep, s2 * jnp.abs(z) ** 2, 0.0)
    v = vh.conj().T
    b_norm2 = jnp.sum(jnp.abs(rhs) ** 2)
    # ||(A D^-1)^H b||: zero exactly when b has no component the controls reach.
    gradient_norm = jnp.sqrt(jnp.sum(c))
    stationary = gradient_norm == 0

    def solve(lam):
        """The step x(lam), back from the scaled unknowns w = D x."""
        return v @ (jnp.where(keep, s / (s2 + lam), 0.0) * z) / scale

    def step_norm(lam):
        """||D x(lam)||, the length the radius bounds."""
        return jnp.sqrt(jnp.sum(c / (s2 + lam) ** 2))

    def distance(x):
        sol = (expander @ x if expander is not None else x).reshape(sol_shape)
        point = manifold.compute_point(
            free_params - manifold.tangent.embed(sol), delta_t
        )
        return manifold.distance2(point, target)

    def damping_for(radius, lam):
        """The damping fitting the step to ``radius`` (Moré §5), from ``lam``."""

        def phi_and_slope(lam):
            norm = step_norm(lam)
            slope = -jnp.sum(c / (s2 + lam) ** 3) / jnp.where(stationary, 1.0, norm)
            return norm - radius, slope

        phi0, slope0 = phi_and_slope(0.0)
        # The undamped step is taken whenever it fits (Algorithm (7.1)(a)).
        fits = phi0 <= LM_SIGMA * radius
        upper = gradient_norm / radius
        lower = -phi0 / slope0

        def cond(carry):
            j, _, _, _, _, done = carry
            return (~done) & (j < LM_HEBDEN_MAXITER)

        def body(carry):
            j, lam, lower, upper, _, _ = carry
            outside = (lam <= lower) | (lam >= upper)
            lam = jnp.where(
                outside, jnp.maximum(1e-3 * upper, jnp.sqrt(lower * upper)), lam
            )
            phi, slope = phi_and_slope(lam)
            done = jnp.abs(phi) <= LM_SIGMA * radius
            upper = jnp.where(phi < 0, lam, upper)
            lower = jnp.maximum(lower, lam - phi / slope)
            # Hebden's update: Newton on 1 / ||Dx(lam)||, not on ||Dx(lam)||.
            lam_next = lam - (phi + radius) / radius * phi / slope
            return j + 1, lam_next, lower, upper, lam, done

        _, _, _, _, lam, _ = jax.lax.while_loop(
            cond, body, (jnp.asarray(0, jnp.int32), lam, lower, upper, lam, fits)
        )
        return jnp.where(fits, 0.0, lam)

    def outer_cond(carry):
        n_trials, _, _, _, _, accepted = carry
        return (~accepted) & (n_trials < maxiter) & (~stationary)

    def outer_body(carry):
        n_trials, radius, lam, _, _, _ = carry
        lam = damping_for(radius, lam)
        x = solve(lam)
        norm2 = jnp.sum(c / (s2 + lam) ** 2)
        model2 = jnp.sum(c * s2 / (s2 + lam) ** 2)
        # Moré's (||f_+|| / ||f||)^2, and the gain ratio in his normalised form
        # (4.4): relative actual over relative predicted reduction.
        ratio = distance(x) / F0
        predicted = (model2 + 2.0 * lam * norm2) / b_norm2
        rho = (1.0 - ratio) / predicted
        # A non-finite trial is rejected, and contracts the radius the most.
        finite = jnp.isfinite(ratio)
        rho = jnp.where(finite, rho, -jnp.inf)
        # Contraction factor: minimiser of the quadratic through the distance
        # at 0 and 1 and its slope at 0 (§4), clipped to [mu_min, mu_max].
        gamma = -(model2 + lam * norm2) / b_norm2
        mu = jnp.clip(0.5 * gamma / (gamma + 0.5 * (1.0 - ratio)), LM_MU_MIN, LM_MU_MAX)
        mu = jnp.where(ratio <= 1.0, LM_MU_MAX, mu)
        mu = jnp.where((ratio > 100.0) | ~finite, LM_MU_MIN, mu)
        radius_next = jnp.where(
            rho <= LM_SHRINK_RHO,
            mu * radius,
            jnp.where(
                (rho >= LM_GROW_RHO) | (lam == 0.0), 2.0 * jnp.sqrt(norm2), radius
            ),
        )
        accepted = rho > LM_ACCEPT_RHO
        return n_trials + 1, radius_next, lam, x, rho, accepted

    # An unset radius (the first step of a run) starts at radius_init times the
    # scaled size ||D x0|| of the pulse itself, x0 being its solvable part in
    # the solve's unknowns (MINPACK's `factor`), or at radius_init for a zero
    # pulse. Not relative to the undamped step: that is longest exactly where
    # A is worst conditioned, which near a zero pulse is where it is least
    # trustworthy. Behind a cond, so a set radius never pays for the mapping.
    radius = jnp.asarray(jnp.nan if radius is None else radius, jnp.float64)
    usable = jnp.isfinite(radius) & (radius > 0)
    columns = manifold.tangent.columns
    pulse = (free_params if columns is None else free_params[:, columns]).reshape(-1)

    def initial_radius():
        x0 = pulse if expander is None else jnp.linalg.lstsq(expander, pulse)[0]
        size = jnp.linalg.norm(scale * x0)
        return radius_init * jnp.where(size > 0, size, 1.0)

    radius = jax.lax.cond(usable, lambda: radius, initial_radius)

    init = (
        jnp.asarray(0, jnp.int32),
        radius,
        jnp.asarray(0.0, jnp.float64),
        jnp.zeros(v.shape[0], jnp.result_type(v, z)),
        jnp.asarray(jnp.nan, jnp.float64),
        jnp.asarray(False),
    )
    n_trials, radius, damping, x, rho, _ = jax.lax.while_loop(
        outer_cond, outer_body, init
    )
    diagnostics = {
        "damping": damping,
        "lm_iter": n_trials,
        "lm_rho": rho,
        "lm_radius": radius,
        # Damped exactly when the undamped step overshot the radius, and then
        # fitted onto its boundary.
        "lm_hit": damping > 0.0,
    }
    return x, rank, s, diagnostics, scale


# Host-side type of each least-squares diagnostic that is not a float.
_DIAGNOSTIC_CASTS = {"rank": int, "lm_iter": int, "lm_hit": bool}


def _ls_diagnostic_sentinels(method: str) -> dict:
    """``nan`` / ``-1`` / ``False`` placeholders for ``method``'s diagnostics."""
    sentinels = {
        "residual": float("nan"),
        "residual_rel": float("nan"),
        "rank": -1,
        "cond": float("nan"),
    }
    if method == "lm":
        sentinels |= {
            "damping": float("nan"),
            "lm_iter": -1,
            "lm_rho": float("nan"),
            "lm_radius": float("nan"),
            "lm_hit": False,
        }
    return sentinels


def _check_least_squares_args(args: dict | None) -> dict:
    """Fill ``least_squares_args`` with the defaults of :func:`least_squares`.

    Host-side, so a bad configuration fails in ``optimize`` before anything is
    compiled. Returns a fresh dict, never the caller's. Besides the keyword
    arguments of :func:`least_squares` it carries ``normalized``, which is not
    the solve's but the bracket's: ``Geope``'s update step takes it off before
    calling the solve. (The ``ls_state`` argument is per-step state, not
    configuration, which is why it precedes the keyword-only marker.)

    Raises:
        TypeError: On a key that is neither a keyword of :func:`least_squares`
            nor ``normalized``.
        ValueError: On an unknown ``method``, a non-positive ``svd_cutoff`` or
            ``lm_radius_init``, or ``lm_maxiter < 1``.
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
    if not resolved["lm_radius_init"] > 0:
        raise ValueError("lm_radius_init must be positive.")
    return resolved
