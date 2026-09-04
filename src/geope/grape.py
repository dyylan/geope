from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from .parameters import Parameters

jax.config.update("jax_enable_x64", True)

from .optimizers import NewtonTRM, Optimizer
from .utils import prepare_random_parameters
from .utils.history import History
from .utils.callbacks import normalize_callbacks, run_callbacks
from typing import Callable


class Grape:
    """Gradient/Hessian-based GRAPE optimiser for quantum gate synthesis.

    Mirrors the `Geope` usage pattern in every respect: it is constructed from a
    `Parameters` object (the single source of truth for all configuration and the
    live optimisation state), while the update rule and ``max_steps`` are
    arguments of :meth:`optimize`; and it builds **one** `GeometricContext` per
    step inside a single jitted ``update_step``.

    Where `Geope` walks the geodesic toward the target, `Grape` walks the
    infidelity's own gradient. The step is therefore assembled from the context's
    *cost* tier — ``value_and_grad``, and ``cost_hessian`` for the Newton rules —
    and never from ``point``, ``jacobian`` or ``A``. Because every context
    quantity is lazy, that is the whole reason **a GRAPE run traces no matrix
    logarithm and never builds the Jacobian**; see `geope.optimizers` for the rule
    an update rule has to obey to keep it that way.

    Attributes:
        params: The bound `Parameters` object (also the source of the
            lazily-built/cached optimisation functions).
        precision: Target fidelity threshold.
        optimizer: The active `geope.optimizers.Optimizer` from the most recent
            :meth:`optimize` call; ``None`` until :meth:`optimize` is first called.
        optimizer_state: The current optimiser state pytree (re-``init()``d per
            run); ``None`` until :meth:`optimize` is first called.
        method: The active rule's ``name`` (e.g. ``'newton_trm'``), or ``None``.
        step_size: Transient last accepted step, ``dt``.
        history: Optional `History` logger (``None`` unless supplied).
    """

    def __init__(
        self,
        params: Parameters,
        precision: float = 0.9999999,
        verbose: bool = False,
        history: History | None = None,
    ) -> None:
        """Initialise the Grape optimiser.

        ``Grape`` requires a `Parameters` object — the engine, initial
        parameters, drift, seed, initialisation spread, projective flag and
        ``param_transform`` are all read from it.

        Args:
            params: A `Parameters` instance bundling every input the
                optimiser needs.
            precision: Target fidelity. Defaults to 0.9999999.
            verbose: Whether to print progress. Defaults to False.
            history: Optional `History` logger. When supplied, the full run
                trajectory is recorded into it; when ``None`` (default), no
                history is kept.

        Raises:
            TypeError: If ``params`` is not a `Parameters` instance.
        """
        if not isinstance(params, Parameters):
            raise TypeError(
                "Grape requires a Parameters object as its first argument. "
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
            self._key = jax.random.key(0)
        self._real_params = params.param_transform is not None
        # The optimisation functions (compute_point/fid/infid/grad/hess) and the
        # algebraic metadata are read directly off ``params`` (built lazily and
        # cached there). No engine, no eager JIT.

        if self._real_params:
            init_parameters = self._init_for_param_transform(params)
            drift_parameters = None
        else:
            init_parameters = params.parameters
            drift_parameters = params.drift_parameters

        self.history = history
        if self.history is not None:
            self.history.params = params
        self.step_size = 0

        self.precision = precision
        self.init_parameters_spread = params.init_spread

        # The update rule is an argument of optimize(), not the constructor. The
        # JIT-compiled update_step bakes it into its closure, so it is built
        # lazily by optimize() (via _configure_optimizer) and rebuilt only when
        # the rule changes — which the frozen dataclass's value __eq__ decides,
        # exactly as Geope's line search does. They stay unset until then.
        self.optimizer = None
        self.optimizer_state = None
        self._optimizer_config = None
        self.update_step = None

        self.verbose = verbose
        # Initialize parameters
        self.init(init_parameters, drift_parameters, params.seed)

    @property
    def method(self) -> str | None:
        """The active update rule's ``name``, or ``None`` before the first run."""
        return None if self.optimizer is None else self.optimizer.name

    def _split_key(self) -> jax.Array:
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def _proj_drift_mask(self) -> np.ndarray:
        """Effective proj+drift index mask used to scatter free params back.

        All-true over the experimental parameters in ``param_transform`` mode
        (every column is free); the natural proj+drift mask otherwise.
        """
        if self._real_params:
            return np.ones(self.params.n_experimental_params, dtype=bool)
        return self.params.proj_drift_indices

    def _init_for_param_transform(self, params: Parameters) -> np.ndarray:
        """Compute initial parameters in experimental-parameter space.

        If ``params.parameters`` is shaped ``(piecewise_steps, n_exp)``,
        use it directly; otherwise sample uniformly in
        ``[-init_spread * pi, +init_spread * pi]``.

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
        seed: int | jax.Array | None = None,
    ) -> None:
        """(Re-)initialise optimiser state.

        Sets up initial parameters, drift parameters and the live state
        (``params.parameters`` / ``params.fidelity``), and records step 0
        into ``history`` when one is attached.

        Args:
            init_parameters: Initial parameter array. Defaults to random.
            drift_parameters: Fixed drift parameter values. Defaults to ones.
            seed: Random seed (int) or JAX key for reproducibility.
        """
        if isinstance(seed, int):
            self._key = jax.random.key(seed)
        elif isinstance(seed, jax.Array):
            self._key = seed  # already a jax.Array key
        # else: keep existing self._key unchanged

        # Initialize variables
        if init_parameters is None:
            self.init_parameters = np.array(
                [
                    prepare_random_parameters(
                        self.params.projected_indices,
                        expander=None,
                        spread=self.init_parameters_spread,
                        key=self._split_key(),
                    )
                    for _ in range(self.params.piecewise_steps)
                ]
            )
        else:
            if (len(init_parameters.shape) == 1) and (self.params.piecewise_steps > 1):
                self.init_parameters = np.array(
                    [init_parameters] * self.params.piecewise_steps
                )
            else:
                self.init_parameters = np.array(init_parameters)
        if not self._real_params and self.params.drift_basis is not None:
            if drift_parameters is None:
                self.drift_parameters = np.ones(self.params.drift_basis.lie_algebra_dim)
            else:
                self.drift_parameters = np.array(drift_parameters)
                assert (
                    self.params.drift_basis.lie_algebra_dim
                    == self.drift_parameters.shape[0]
                ), "Drift parameters must be the same length as the size of the drift basis."

            self.init_parameters[:, self.params.drift_indices] = np.tile(
                self.drift_parameters, (self.params.piecewise_steps, 1)
            )
        else:
            self.drift_parameters = None

        self.params.parameters = np.array(self.init_parameters)
        self.params.fidelity = self.params.manifold.fidelity_at(self.params.free())
        self.step_size = 0
        # A change of parameters invalidates any optimiser state (Adam's moments
        # are shaped like them); optimize() re-init()s it per run regardless, but
        # clear it here so the object is never left holding a stale one.
        self.optimizer_state = None
        if self.history is not None:
            self.history.reset()
            self.history.record(self)  # step 0

    def _configure_optimizer(self, optimizer: Optimizer) -> None:
        """Select the update rule and (re)build its jitted update function.

        The JIT-compiled ``update_step`` closes over the rule, so it is rebuilt
        whenever the rule changes — which the frozen dataclass's value ``__eq__``
        decides, so two equal rules reuse the compiled function. This is exactly
        `geope.Geope._configure_line_search`, with `geope.optimizers.Optimizer` in
        place of the line search.

        Args:
            optimizer: The `geope.optimizers.Optimizer` to run.

        Raises:
            TypeError: If ``optimizer`` is not an `geope.optimizers.Optimizer`.
        """
        if not isinstance(optimizer, Optimizer):
            raise TypeError(
                "Grape.optimize(optimizer=...) takes a geope.optimizers.Optimizer "
                "— GradientDescent, Adam, NewtonTRM or NewtonRFO — not "
                f"{type(optimizer).__name__}. The `method='nr-trm'` strings were "
                "replaced by these config objects; pass NewtonTRM(delta=...)."
            )
        # Set the attribute before the memo check so self.optimizer always tracks
        # the latest value — safe because an equal rule means identical trace-time
        # behaviour.
        self.optimizer = optimizer
        if self._optimizer_config == optimizer:
            return
        self.update_step = self.get_update_step()
        self._optimizer_config = optimizer

    def get_update_step(self) -> Callable[..., tuple]:
        """Build the JIT-compiled GRAPE update step.

        One jitted function per update rule, and one
        `geope.geometry.GeometricContext` per call of it. The step is: open the
        context, let the rule read the cost tier off it and choose an uphill
        direction and a (negative) step, then move.

        The context is built *inside* this function and never leaves it — it is a
        trace-time object, not a pytree. Everything the rule needs it reads off
        that one context, and everything it does *not* read costs nothing, which
        is why no matrix logarithm and no Jacobian is traced here.

        Returns:
            A JIT-compiled callable ``update_step(free_params, opt_state)``
            returning ``(new_parameters, infidelity, dt, new_opt_state)``, where
            ``infidelity`` is measured **at** ``new_parameters``.
        """
        manifold = self.params.manifold
        optimizer = self.optimizer
        proj_drift_mask = self._proj_drift_mask()
        lie_algebra_dim = len(proj_drift_mask)

        @jax.jit
        def update_step(free_params, opt_state):
            ctx = manifold.context(free_params)
            result = optimizer(ctx, opt_state)
            new_free = free_params + result.dt * result.coeffs

            # Scatter the moved columns back over the full basis.
            new_parameters = jnp.zeros(
                (free_params.shape[0], lie_algebra_dim),
                dtype=free_params.real.dtype,
            )
            new_parameters = new_parameters.at[:, proj_drift_mask].set(new_free.real)
            return new_parameters, result.value, result.dt, result.state

        return update_step

    def optimize(
        self,
        max_steps: int = 100,
        optimizer: Optimizer | None = None,
        callbacks: Callable | list[Callable] | tuple[Callable, ...] | None = None,
    ) -> Parameters:
        """Run the GRAPE optimisation loop.

        Iterates gradient/Hessian update steps until the fidelity exceeds
        ``self.precision`` or ``max_steps`` is reached.

        Args:
            max_steps: Maximum number of optimisation steps. Defaults to 100.
            optimizer: The `geope.optimizers.Optimizer` to run — `GradientDescent`,
                `Adam`, `NewtonTRM` or `NewtonRFO`, each a frozen dataclass
                carrying its own hyperparameters. Defaults to
                `geope.optimizers.NewtonTRM`.
            callbacks: Optional callback, or list/tuple of callbacks, invoked at
                the end of every step with the signature
                ``callback(step, history, grape) -> bool``. All callbacks run
                each step; the loop stops early if any returns a falsy value
                (so a pure logging callback must ``return True``). ``step`` is
                the 1-based index of the step just completed, ``history`` is
                ``grape.history`` (may be ``None``), and ``grape`` is this
                optimiser.

        Returns:
            The bound `Parameters` instance, carrying the final
            ``parameters`` (current array) and ``fidelity`` (scalar) — which
            describe the **same** pulse, since the reported infidelity is measured
            at the step just taken. The full trajectory and ``best_*`` live on
            ``grape.history`` when a `History` was supplied.

        Raises:
            TypeError: If ``optimizer`` is not an `geope.optimizers.Optimizer`.
        """
        optimizer = optimizer or NewtonTRM()
        self._configure_optimizer(optimizer)

        # Reset the per-run state, decoupled from compile reuse exactly as in
        # Geope.optimize: the memo can reuse the compiled update_step while Adam's
        # moments and the Newton warm start still restart on every call.
        self.optimizer_state = self.optimizer.init(self.params.free())

        cbs = normalize_callbacks(callbacks)

        step = 0
        while (self.params.fidelity < self.precision) and (step < max_steps):
            step += 1
            (
                new_parameters,
                infidelity,
                step_size,
                self.optimizer_state,
            ) = self.update_step(self.params.free(), self.optimizer_state)
            if self.verbose:
                if infidelity < 1 - self.precision:
                    print(
                        f"[{step}/{max_steps}] [Infidelity = {infidelity}] A solution!                                                                     ",
                        end="\r",
                    )
                else:
                    print(
                        f"[{step}/{max_steps}] Infidelity = {infidelity}                                                                                             ",
                        end="\r",
                    )
            self.params.parameters = np.array(new_parameters)
            self.params.fidelity = 1 - infidelity
            self.step_size = float(step_size)
            if self.history is not None:
                self.history.record(self)

            # Run user callbacks at the end of the step; stop early if any
            # requests it.
            if not run_callbacks(cbs, step, self.history, self):
                break
        if self.verbose:
            print("")
        return self.params
