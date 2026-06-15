from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
import optax

from .engine import (
    Engine,
    get_infidelity_fn,
    get_fidelity_full_fn,
    get_infidelity_full_fn,
)
from .parameters import Parameters

jax.config.update("jax_enable_x64", True)

from .utils import prepare_random_parameters
from .utils.history import History
from functools import partial


class GrapeEngine(Engine):
    """All jitted functions for Grape"""

    def __init__(self, target_unitary,
                 full_basis,
                 projected_basis,
                 drift_basis=None,
                 piecewise_steps=1,
                 projective: bool = True):
        super(GrapeEngine, self).__init__(target_unitary, full_basis, projected_basis, drift_basis, piecewise_steps)
        self.projective = projective
        if projective:
            self.infid_U_fn = get_infidelity_fn(target_unitary)
        else:
            self.fid_U_fn = jax.jit(get_fidelity_full_fn(target_unitary))
            self.infid_U_fn = get_infidelity_full_fn(target_unitary)
        self.infid_fn = lambda x: self.infid_U_fn(self.compute_U_fn(x))
        self.grad_fn = jax.value_and_grad(self.infid_fn)
        self.hess_fn = jax.jit(
            lambda y: jax.vmap(lambda x: hvp_forward_over_reverse(self.infid_fn, y, x))(jnp.eye(y.size, dtype=y.dtype)))

    def wrap_param_transform(self, params: Parameters) -> None:
        """Replace ``compute_U_fn`` to honour ``params.param_transform``.

        The user-facing experimental parameters are mapped to projected-basis
        coefficients via ``params.param_transform``, embedded into the
        proj+drift basis, and combined with the drift before the original
        ``compute_U_fn`` is called. The infidelity, gradient, and Hessian
        functions are re-derived from the wrapped ``compute_U_fn``, and the
        engine's indices are overridden so the rest of the pipeline operates
        in experimental space.

        Args:
            params: The ``Parameters`` object carrying ``param_transform``.
        """
        raw_compute_U = self.compute_U_fn
        n_exp = params.n_experimental_params
        n_proj_drift = self.proj_drift_basis.lie_algebra_dim
        proj_idx_pd = self.proj_indices_projdrift_basis
        drift_idx_pd = self.drift_indices_projdrift_basis

        # Determine extraction indices when transform outputs full-basis
        # coefficients rather than projected-basis coefficients.
        _test_out = params.param_transform(jnp.zeros(n_exp))
        tf_out_dim = _test_out.shape[0]
        n_proj = params.projected_basis.lie_algebra_dim
        if tf_out_dim != n_proj:
            _extract = jnp.array(np.where(
                np.array(self.projected_basis.overlap(params.basis)))[0])
        else:
            _extract = None

        # Capture drift for embedding inside compute_U
        if params.drift_parameters is not None:
            _drift = jnp.array(params.drift_parameters, dtype=jnp.float64)
        else:
            _drift = None

        def _wrapped_compute_U(exp_params, _raw=raw_compute_U,
                               _tf=params.param_transform,
                               _pi=proj_idx_pd, _di=drift_idx_pd,
                               _npd=n_proj_drift, _dr=_drift,
                               _ext=_extract):
            ctrl = jax.vmap(_tf)(exp_params)
            if _ext is not None:
                ctrl = ctrl[:, _ext]
            # Promote dtype for Jacobian tracing compatibility
            _dtype = jnp.result_type(ctrl.dtype, exp_params.dtype)
            ctrl = ctrl.astype(_dtype)
            full = jnp.zeros((exp_params.shape[0], _npd), dtype=_dtype)
            full = full.at[:, _pi].set(ctrl)
            if _dr is not None:
                full = full.at[:, _di].set(
                    jnp.broadcast_to(_dr.astype(_dtype),
                                     (exp_params.shape[0], _dr.shape[0])))
            return _raw(full)

        self.compute_U_fn = jax.jit(_wrapped_compute_U)
        # Re-derive dependent functions
        self.infid_fn = lambda x: self.infid_U_fn(self.compute_U_fn(x))
        self.grad_fn = jax.value_and_grad(self.infid_fn)
        self.hess_fn = jax.jit(
            lambda y: jax.vmap(lambda x: hvp_forward_over_reverse(self.infid_fn, y, x))(
                jnp.eye(y.size, dtype=y.dtype)))

        # Override engine indices so _init/optimize work in experimental space.
        # All-true mask of length n_exp makes extract/expand a no-op.
        self.proj_drift_indices = np.ones(n_exp, dtype=bool)
        self.drift_indices = np.full(n_exp, False)
        self.drift_basis = None


class Grape:
    """Gradient/Hessian-based GRAPE optimiser for quantum gate synthesis.

    Mirrors the `Geope` usage pattern: it is constructed from a `Parameters`
    object (the single source of truth for all configuration and the live
    optimisation state), while the optimiser ``method``, its hyperparameters
    and ``max_steps`` are arguments of :meth:`optimize`.

    Attributes:
        params: The bound `Parameters` object.
        engine: The internal `GrapeEngine` constructed from ``params``.
        precision: Target fidelity threshold.
        method: Optimiser method from the most recent :meth:`optimize` call
            (``'gd'``, ``'adam'``, ``'nr-trm'`` or ``'nr-rfo'``); ``None``
            until :meth:`optimize` is first called.
        step_size: Transient last step size (always 0 for GRAPE).
        history: Optional `History` logger (``None`` unless supplied).
    """

    def __init__(self,
                 params: Parameters,
                 precision: float = 0.9999999,
                 verbose: bool = False,
                 history: History | None = None) -> None:
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
        engine = GrapeEngine(
            target_unitary=params.target,
            full_basis=params.basis,
            projected_basis=params.projected_basis,
            drift_basis=params.drift_basis,
            piecewise_steps=params.piecewise_steps,
            projective=params.projective,
        )

        # Wrap compute_U_fn if param_transform is set
        if params.param_transform is not None:
            engine.wrap_param_transform(params)
            init_parameters = self._init_for_param_transform(engine, params)
            drift_parameters = None
        else:
            init_parameters = params.parameters
            drift_parameters = params.drift_parameters

        self.engine = engine
        self._real_params = params.param_transform is not None

        self.history = history
        if self.history is not None:
            self.history.params = params
        self.step_size = 0

        self.precision = precision
        self.init_parameters_spread = params.init_spread

        # The optimiser method and its hyperparameters are arguments of
        # optimize(), not the constructor. The JIT-compiled update_step bakes
        # the method and hyperparameters into its closure, so it is built
        # lazily by optimize() (via _configure_optimizer) and rebuilt only
        # when that configuration changes. They stay unset until then.
        self.method = None
        self._optimizer_config = None
        self.update_step = None
        self.optimizer = None
        self.optimizer_state = None

        self.verbose = verbose
        # Initialize parameters
        self.init(init_parameters, drift_parameters, params.seed)

    def _split_key(self) -> jax.Array:
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def _init_for_param_transform(self, engine: GrapeEngine, params: Parameters) -> np.ndarray:
        """Compute initial parameters in experimental-parameter space.

        If ``params.parameters`` is shaped ``(piecewise_steps, n_exp)``,
        use it directly; otherwise sample uniformly in
        ``[-init_spread * pi, +init_spread * pi]``.

        Args:
            engine: The wrapped ``GrapeEngine``.
            params: The ``Parameters`` object.

        Returns:
            An ``np.ndarray`` of shape ``(piecewise_steps, n_exp)``.
        """
        n_exp = params.n_experimental_params
        _user_init = np.array(params.parameters)
        if _user_init.shape == (params.piecewise_steps, n_exp):
            return _user_init
        return np.array(jax.random.uniform(
            self._split_key(),
            shape=(params.piecewise_steps, n_exp),
            minval=-params.init_spread * np.pi,
            maxval=params.init_spread * np.pi,
        ))

    def init(self, init_parameters=None, drift_parameters=None, seed=None) -> None:
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
            self.init_parameters = np.array([prepare_random_parameters(self.engine.projected_indices,
                                                                       expander=None,
                                                                       spread=self.init_parameters_spread,
                                                                       key=self._split_key()) for _ in range(self.engine.piecewise_steps)])
        else:
            if (len(init_parameters.shape) == 1) and (self.engine.piecewise_steps > 1):
                self.init_parameters = np.array([init_parameters] * self.engine.piecewise_steps)
            else:
                self.init_parameters = np.array(init_parameters)
        if self.engine.drift_basis is not None:
            if drift_parameters is None:
                self.drift_parameters = np.ones(self.engine.drift_basis.lie_algebra_dim)
            else:
                self.drift_parameters = np.array(drift_parameters)
                assert self.engine.drift_basis.lie_algebra_dim == self.drift_parameters.shape[0], \
                    "Drift parameters must be the same length as the size of the drift basis."

            self.init_parameters[:, self.engine.drift_indices] = np.tile(self.drift_parameters, (self.engine.piecewise_steps, 1))
        else:
            self.drift_parameters = None

        self.params.parameters = np.array(self.init_parameters)
        _dtype = np.float64 if self._real_params else np.complex128
        free_params = self.params.parameters[:, self.engine.proj_drift_indices].astype(_dtype)
        self.params.fidelity = self.engine.fid_U_fn(self.engine.compute_U_fn(free_params))
        self.step_size = 0
        # A change of parameters invalidates any optax state built for the
        # previous parameter values; force a rebuild on the next optimize().
        self.optimizer_state = None
        self._optimizer_config = None
        if self.history is not None:
            self.history.reset()
            self.history.record(self)        # step 0

    def _configure_optimizer(self, method: str, optimizer_kwargs: dict) -> None:
        """Select the optimiser method and (re)build its update function.

        The JIT-compiled ``update_step`` closes over the method and its
        hyperparameters, so it is recreated (and the optax state re-initialised
        from the current parameters) whenever the configuration changes. The
        current configuration is memoised in ``_optimizer_config`` so repeated
        ``optimize()`` calls with unchanged settings reuse the compiled
        function and continue the optimiser state.

        Args:
            method: ``'gd'``, ``'adam'``, ``'nr-trm'`` or ``'nr-rfo'``.
            optimizer_kwargs: Method hyperparameters (``learning_rate`` for
                ``'gd'``/``'adam'``, ``delta`` for ``'nr-trm'``, ``kappa`` for
                ``'nr-rfo'``).
        """
        config = (method, tuple(sorted(optimizer_kwargs.items())))
        if self._optimizer_config == config and self.optimizer_state is not None:
            return
        if method in ['gd', 'adam']:
            learning_rate = optimizer_kwargs.get('learning_rate')
            if method == 'gd':
                optimizer = optax.sgd(learning_rate=learning_rate)
            else:
                optimizer = optax.adam(learning_rate=learning_rate)
            self.update_step = get_update_step_gd(self.engine.proj_drift_indices, self.engine.grad_fn, optimizer)
        elif method in ['nr-trm', 'nr-rfo']:
            # Use backtracking for second order optimization
            optimizer = optax.scale_by_backtracking_linesearch(max_backtracking_steps=100)
            if method == 'nr-trm':
                delta = optimizer_kwargs.get('delta')
                self.update_step = get_update_step_trm(self.engine.proj_drift_indices,
                                                       self.engine.infid_fn,
                                                       self.engine.grad_fn,
                                                       self.engine.hess_fn,
                                                       optimizer,
                                                       delta)
            else:
                kappa = optimizer_kwargs.get('kappa', 100)
                self.update_step = get_update_step_rfo(self.engine.proj_drift_indices,
                                                       self.engine.infid_fn,
                                                       self.engine.grad_fn,
                                                       self.engine.hess_fn,
                                                       optimizer,
                                                       kappa)
        else:
            raise NotImplementedError(f"Method {method} not implemented")

        _dtype = np.float64 if self._real_params else np.complex128
        free_params = self.params.parameters[:, self.engine.proj_drift_indices].astype(_dtype)
        self.optimizer = optimizer
        self.optimizer_state = {"optimizer": optimizer.init(free_params)}
        self.method = method
        self._optimizer_config = config

    def optimize(self, max_steps: int = 100, method: str = 'nr-trm', **optimizer_kwargs) -> Parameters:
        """Run the GRAPE optimisation loop.

        Iterates gradient/Hessian update steps until the fidelity exceeds
        ``self.precision`` or ``max_steps`` is reached.

        Args:
            max_steps: Maximum number of optimisation steps. Defaults to 100.
            method: ``'gd'``, ``'adam'``, ``'nr-trm'`` (default) or
                ``'nr-rfo'``.
            **optimizer_kwargs: Method hyperparameters — ``learning_rate`` for
                ``'gd'``/``'adam'``, ``delta`` for ``'nr-trm'``, ``kappa`` for
                ``'nr-rfo'``.

        Returns:
            The bound `Parameters` instance, carrying the final
            ``parameters`` (current array) and ``fidelity`` (scalar). The full
            trajectory and ``best_*`` live on ``grape.history`` when a
            `History` was supplied.
        """
        self._configure_optimizer(method, optimizer_kwargs)

        step = 0
        _dtype = np.float64 if self._real_params else np.complex128
        while (self.params.fidelity < self.precision) and (step < max_steps):
            step += 1
            free_params = self.params.parameters[:, self.engine.proj_drift_indices].astype(_dtype)
            new_parameters, infidelity, self.optimizer_state = self.update_step(free_params, self.optimizer_state)
            if self.verbose:
                if infidelity < 1 - self.precision:
                    print(
                        f"[{step}/{max_steps}] [Infidelity = {infidelity}] A solution!                                                                     ",
                        end="\r")
                else:
                    print(
                        f"[{step}/{max_steps}] Infidelity = {infidelity}                                                                                             ",
                        end="\r")
            self.params.parameters = np.array(new_parameters)
            self.params.fidelity = 1 - infidelity
            self.step_size = 0
            if self.history is not None:
                self.history.record(self)
        if self.verbose:
            print("")
        return self.params


def get_update_step_gd(proj_drift_indices, grad_fn, optimizer):
    lie_algebra_dim = len(proj_drift_indices)

    @jax.jit
    def update_step(free_params, optimizer_state):
        # Get Hessian and gradients
        infidelity_new_phi, grads = grad_fn(free_params)
        # use the linesearch backtracking, make sure we pass a function that needs to get minimized.
        updates, optimizer_state["optimizer"] = optimizer.update(grads, optimizer_state["optimizer"], free_params)
        # Updates the parameters.
        free_params = optax.apply_updates(free_params, updates)
        # Expand the coefficients to the larger space
        new_parameters = jnp.zeros((free_params.shape[0], lie_algebra_dim), dtype=free_params.real.dtype)
        new_parameters = new_parameters.at[:, proj_drift_indices].set(free_params.real)
        return new_parameters, infidelity_new_phi, optimizer_state

    return update_step


def get_update_step_trm(proj_drift_indices, fid_fn, grad_fn, hess_fn, optimizer, delta):
    lie_algebra_dim = len(proj_drift_indices)

    @jax.jit
    def update_step(free_params, optimizer_state):
        # Get Hessian and gradients
        infidelity_new_phi, grads = grad_fn(free_params)
        hessian = hess_fn(free_params)
        hessian = jnp.reshape(hessian, (hessian.shape[0], hessian.shape[0]))
        # Perform newton step to get update
        grads_nr = newton_trm_step(hessian, grads.flatten(), delta).reshape(free_params.shape)
        # use the linesearch backtracking, make sure we pass a function that needs to get minimized.
        updates, optimizer_state["optimizer"] = optimizer.update(-grads_nr, optimizer_state["optimizer"], free_params,
                                                                 value=infidelity_new_phi, grad=-grads_nr,
                                                                 value_fn=fid_fn)
        # Updates the parameters.
        free_params = optax.apply_updates(free_params, updates)
        # Expand the coefficients to the larger space
        new_parameters = jnp.zeros((free_params.shape[0], lie_algebra_dim), dtype=free_params.real.dtype)
        new_parameters = new_parameters.at[:, proj_drift_indices].set(free_params.real)
        return new_parameters, infidelity_new_phi, optimizer_state

    return update_step


@partial(jax.jit, static_argnums=(2,))
def newton_trm_step(hessian, gradient, delta):
    Σ, U = jnp.linalg.eigh(hessian)
    # Shift spectrum by a delta
    sigma = jnp.max(jnp.array([0., delta - jnp.min(Σ)]))
    Σreg = Σ + sigma
    # Solve system
    cfac_reg = jax.scipy.linalg.cho_factor(U @ (jnp.diag(Σreg) @ U.conj().T))
    return jax.scipy.linalg.cho_solve(cfac_reg, gradient)
    # return gradient


def get_update_step_rfo(proj_drift_indices, fid_fn, grad_fn, hess_fn, optimizer, kappa):
    lie_algebra_dim = len(proj_drift_indices)

    @jax.jit
    def update_step(free_params, optimizer_state):
        # Get Hessian and gradients
        infidelity_new_phi, grads = grad_fn(free_params)
        hessian = hess_fn(free_params)
        hessian = jnp.reshape(hessian, (hessian.shape[0], hessian.shape[0]))
        # Perform newton step to get update
        grads_nr = newton_rfo_step(hessian, grads.flatten(), kappa)
        grads_nr = grads_nr.reshape(free_params.shape)
        # use the linesearch backtracking, make sure we pass a function that needs to get minimized.
        updates, optimizer_state["optimizer"] = optimizer.update(-grads_nr, optimizer_state["optimizer"], free_params,
                                                                 value=infidelity_new_phi, grad=-grads_nr,
                                                                 value_fn=fid_fn)
        # Updates the parameters.
        free_params = optax.apply_updates(free_params, updates)
        # Expand the coefficients to the larger space
        new_parameters = jnp.zeros((free_params.shape[0], lie_algebra_dim), dtype=free_params.real.dtype)
        new_parameters = new_parameters.at[:, proj_drift_indices].set(free_params.real)
        return new_parameters, infidelity_new_phi, optimizer_state

    return update_step


@partial(jax.jit, static_argnums=(2,))
def condition_loop(hessian, g, kappa):
    nparams = hessian.shape[0]
    phi = 0.9  # 0.9 seems to work well
    max_cond = kappa # 1e4 is from Spinach Settings
    max_iter = 300  # 0.9**300 = 1e-14
    g = jnp.expand_dims(g, axis=1)

    def body_fn(val):
        k, i, a, H = val
        # jax.debug.print("alpha {}", a)
        # jax.debug.print("i {} - kappa: {}", i, kappa)
        # jax.debug.print("max_cond {}", max_cond)
        H_aug = jnp.block([[H * a ** 2, g * a],
                           [g.T * a, 0.]])
        # Regularize
        sigma = jnp.min(jnp.array([0., jnp.min(jnp.linalg.eigvalsh(H_aug))]))
        H_aug = H_aug - jnp.eye(H_aug.shape[0]) * sigma
        # Grab original Hamiltonian
        H = H_aug[:nparams, :nparams] / a ** 2
        return jnp.linalg.cond(H), i + 1, a * phi, H

    def cond_fn(val):
        # If kappa is larger than our target condition number, stop
        cond1 = val[0] > max_cond
        # Stop at max iterations
        cond2 = val[1] < max_iter
        return jax.lax.bitwise_and(cond1, cond2)
    # set initial alpha
    alpha_0 = 1. # Other choices are possible but this seems to work well.
    return jax.lax.while_loop(cond_fn, body_fn, (jnp.inf, 0, alpha_0, hessian))


@partial(jax.jit, static_argnums=(2,))
def newton_rfo_step(hessian, gradient, phi):
    # Regularize in loop
    _, _, _, hessian = condition_loop(hessian, gradient, phi)
    # Symmetrize
    hessian = jnp.real(hessian + hessian.T) / 2
    # Cholesky solve
    cfac_reg = jax.scipy.linalg.cho_factor(hessian)
    return jax.scipy.linalg.cho_solve(cfac_reg, gradient)


def hvp_forward_over_reverse(f, params, v):
    v = v.reshape(params.shape)
    return jax.jvp(jax.grad(f), (params,), (v,))[1]
