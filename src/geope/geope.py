from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_enable_x64", True)

from .lie.pauli_projector import get_project_omegas_fn, get_project_omegas_fn_otf
from .engine import (
    Engine,
    get_infidelity_fn,
    get_fidelity_full_fn,
    get_infidelity_full_fn,
)
from .lie import Hamiltonian, Basis
from .utils import golden_section_search, adam_line_search, prepare_random_parameters, merge_constraints, control_to_indices
from .jax.logm import logm
from .jax.jacobian import get_jacobian_manual
from .parameters import Parameters
from .utils.history import History
from functools import partial
from typing import Callable
import inspect


class GeopeEngine(Engine):
    """Engine with JIT-compiled geodesic optimisation functions.

    Extends `Engine` with Jacobian, projection, geodesic, infidelity,
    and gradient functions that are JIT-compiled for use with the GEOPE
    algorithm.

    Attributes:
        project_omegas_fn: JIT-compiled function for projecting omegas.
        jac_fn: Jacobian function of the unitary with respect to parameters.
        geo_fn: JIT-compiled geodesic Hamiltonian function.
        infid_fn: Infidelity function.
        grad_fn: Value-and-gradient of the infidelity.
    """

    def __init__(self, target_unitary: np.ndarray,
                 full_basis: Basis,
                 projected_basis: Basis,
                 drift_basis: Basis | None = None,
                 piecewise_steps: int = 1,
                 batch_size: int | None = None,
                 projective: bool = True) -> None:
        """Initialise the GeopeEngine.

        Args:
            target_unitary: The target unitary ``np.ndarray``.
            full_basis: The full Lie algebra ``Basis``.
            projected_basis: The projected (controllable) subalgebra ``Basis``.
            drift_basis: The drift (uncontrollable) subalgebra ``Basis``.
                Defaults to ``None``.
            piecewise_steps: Number of piecewise-constant gate segments. Defaults to 1.
            batch_size: Optional batch size for on-the-fly omega projection
                when the number of qubits exceeds 5.
            projective: If ``True`` (default), use the projective (SU)
                geodesic and $F_{\\mathrm{proj}} = |\\mathrm{Tr}(U_T^\\dagger U)|/d$.
                If ``False``, use the full (U) geodesic and
                $F_{\\mathrm{full}} = \\mathrm{Re}\\,\\mathrm{Tr}(U_T^\\dagger U)/d$.
        """
        super(GeopeEngine, self).__init__(target_unitary, full_basis, projected_basis, drift_basis, piecewise_steps)
        self.projective = projective
        if full_basis.n > 5:
            # TODO: We will probably have to batch the jacobian function here as well
            # We should calculate the jacobians of the expms and stitch together the
            # matrices for the gradient ourselves.
            self.project_omegas_fn = jax.jit(get_project_omegas_fn_otf(self.full_basis, batch_size=batch_size))
            self.jac_fn = get_jacobian_manual(self.proj_drift_basis.basis)
            del self.full_basis
        else:
            self.project_omegas_fn = jax.jit(get_project_omegas_fn(self.full_basis))
        self.jac_fn = jax.jacobian(self.compute_U_fn, argnums=0, holomorphic=True)
        self.geo_fn = jax.jit(get_geodesic_hamiltonian_fn(target_unitary, projective=projective))
        if projective:
            self.infid_U_fn = get_infidelity_fn(target_unitary)
        else:
            self.fid_U_fn = jax.jit(get_fidelity_full_fn(target_unitary))
            self.infid_U_fn = get_infidelity_full_fn(target_unitary)
        self.infid_fn = lambda x: self.infid_U_fn(self.compute_U_fn(x))
        self.grad_fn = jax.value_and_grad(self.infid_fn)
        self.gammas_and_omegas = self._make_gammas_and_omegas()

    def _make_gammas_and_omegas(self) -> Callable[[Array, Array], tuple[Array, Array]]:
        """Build the JIT-compiled function that computes gammas and omegas.

        Gammas are the projected geodesic Hamiltonian coefficients;
        omegas encode the Jacobian of each unitary gate with respect to
        each parameter, projected onto the Pauli basis. The closure
        captures the engine's current ``compute_U_fn`` / ``jac_fn`` /
        ``geo_fn`` / ``project_omegas_fn`` and projected+drift indices,
        so it must be (re)built after any wrapping that replaces those
        functions (see :meth:`wrap_param_transform`).

        Returns:
            A JIT-compiled callable ``gammas_and_omegas(free_params, key)``.
        """
        project_omegas_fn = self.project_omegas_fn
        jac_fn = self.jac_fn
        compute_U_fn = self.compute_U_fn
        geodesic_fn = self.geo_fn
        proj_drift_basis = self.proj_drift_basis
        proj_indices = self.proj_indices_projdrift_basis

        @jax.jit
        def gammas_and_omegas(free_params, key):
            unitary = compute_U_fn(free_params)
            gammaU = geodesic_fn(unitary, key=key) # seed for logm
            gammaU_params = project_omegas_fn(jnp.expand_dims(gammaU, axis=0)).squeeze(axis=0) / (gammaU.shape[0])

            dUs = jnp.array(jac_fn(free_params))
            dUs_t = jnp.transpose(dUs, [2, 3, 0, 1])
            omegas_steps_phis = jnp.array([project_omegas_fn(1.j * omegaUs) for omegaUs in dUs_t])

            if np.any(proj_drift_basis):
                omegas_steps_phis = omegas_steps_phis.at[:, proj_indices, :].get()

            return gammaU_params, omegas_steps_phis

        return gammas_and_omegas

    def wrap_param_transform(self, params: Parameters) -> None:
        """Replace ``compute_U_fn`` / ``jac_fn`` to honour ``params.param_transform``.

        The user-facing parameters $\\phi^{\\mathrm{exp}}\\in\\mathbb{R}^{N_g\\times n_{\\mathrm{exp}}}$
        are mapped to projected-basis coefficients via ``params.param_transform``
        (possibly step-dependent), embedded into the proj+drift basis,
        and combined with the drift before the original ``compute_U_fn``
        is called. The Jacobian is replaced by a split-real-imag version
        so that complex tracing preserves the imaginary part of
        intermediates. The engine's indices are overridden so the rest
        of the pipeline operates in experimental space.

        ``gammas_and_omegas`` is rebuilt at the end so it closes over the
        wrapped functions.

        Args:
            params: The ``Parameters`` object carrying ``param_transform``.
        """
        raw_compute_U = self.compute_U_fn
        n_exp = params.n_experimental_params
        n_proj_drift = self.proj_drift_basis.lie_algebra_dim
        proj_idx_pd = self.proj_indices_projdrift_basis
        drift_idx_pd = self.drift_indices_projdrift_basis

        # Detect step-dependence: tau(phi) vs tau(phi, step_index)
        _step_dependent = len(inspect.signature(params.param_transform).parameters) >= 2

        # Detect whether transform outputs full-basis or projected-basis coefficients
        _test_out = (params.param_transform(jnp.zeros(n_exp), 0)
                     if _step_dependent
                     else params.param_transform(jnp.zeros(n_exp)))
        tf_out_dim = _test_out.shape[0]
        n_proj = params.projected_basis.lie_algebra_dim
        if tf_out_dim != n_proj:
            _extract = jnp.array(np.where(
                np.array(self.projected_basis.overlap(params.basis)))[0])
        else:
            _extract = None

        if params.drift_parameters is not None:
            _drift = jnp.array(params.drift_parameters, dtype=jnp.float64)
        else:
            _drift = None

        def _wrapped_compute_U(exp_params, _raw=raw_compute_U,
                               _tf=params.param_transform,
                               _pi=proj_idx_pd, _di=drift_idx_pd,
                               _npd=n_proj_drift, _dr=_drift,
                               _ext=_extract, _step_dep=_step_dependent):
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
                    jnp.broadcast_to(_dr.astype(_dtype),
                                     (exp_params.shape[0], _dr.shape[0])))
            return _raw(full)

        self.compute_U_fn = jax.jit(_wrapped_compute_U)
        # Split real/imag Jacobian to avoid losing imaginary parts through
        # real-valued intermediates in the user transform.
        _compute_U = self.compute_U_fn

        def _split_U(x, _cu=_compute_U):
            U = _cu(x)
            return jnp.stack([jnp.real(U), jnp.imag(U)])

        _raw_jac_split = jax.jacobian(_split_U, argnums=0)

        def _jac_fn(x, _rjs=_raw_jac_split):
            jac_split = _rjs(x)
            return jac_split[0] + 1j * jac_split[1]

        self.jac_fn = _jac_fn
        self.infid_fn = lambda x: self.infid_U_fn(self.compute_U_fn(x))
        self.grad_fn = jax.value_and_grad(self.infid_fn)

        # Override engine indices so the rest of the pipeline operates in
        # experimental space
        self.proj_drift_indices = np.arange(n_exp)
        self.drift_indices = np.array([], dtype=int)
        self.drift_basis = None
        self.proj_indices_projdrift_basis = np.ones(n_exp, dtype=bool)
        self.drift_indices_projdrift_basis = np.zeros(n_exp, dtype=bool)
        self.proj_drift_basis._lie_algebra_dim = n_exp
        self.projected_indices = np.ones(n_exp, dtype=bool)

        # Rebuild gammas/omegas so it closes over the wrapped functions and
        # the overridden indices.
        self.gammas_and_omegas = self._make_gammas_and_omegas()

    def build_pulse_expander(
        self,
        pulse_constraints: dict | list,
        real_params: bool,
        n_exp: int,
        proj_params: np.ndarray,
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Build a flat-space expander enforcing pulse-shape constraints.

        For each constrained parameter index $k$ the time profile
        $\\phi_k(g)$ is constrained to a fixed unit-norm template
        $t_k\\in\\mathbb{R}^{N_g}$, leaving only the scalar amplitude
        free. Unconstrained parameters retain one free variable per gate.

        Args:
            pulse_constraints: The pulse-shape constraint config. In
                experimental space, an iterable of integer parameter
                indices; in projected space, a control-format dict
                ``{qubit_index_or_tuple: [lowercase op labels]}`` (the
                same format as ``control``), e.g. ``{(1, 2): ['zz']}``.
            real_params: Whether the engine operates in experimental
                (``param_transform``) space.
            n_exp: Number of experimental parameters (used when
                ``real_params`` is ``True``).
            proj_params: Current projected parameter ``np.ndarray`` of
                shape ``(N_g, K_proj)`` (or ``(N_g, n_exp)`` in
                experimental mode) used as the shape template.

        Returns:
            A tuple ``(E, templates)`` where ``E`` is the
            ``(N_g * n_proj, n_free)`` expander matrix and ``templates``
            is a dict mapping the constrained integer index to its
            unit-norm template vector.
        """
        L = self.piecewise_steps
        if real_params:
            n_proj = n_exp
            pulse_indices = list(pulse_constraints)
        else:
            n_proj = self.projected_basis.lie_algebra_dim
            proj_labels = list(self.projected_basis.labels)
            if not isinstance(pulse_constraints, dict):
                raise TypeError(
                    "pulse_constraints must be a control-format dict in "
                    "projected space, e.g. {(1, 2): ['zz']}.")
            pulse_indices = control_to_indices(proj_labels, pulse_constraints,
                                               strict=True)
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


class Geope:
    """Top-level GEOPE optimiser for quantum gate synthesis.

    Orchestrates the geodesic-based optimisation of Lie algebra
    parameters ($\\phi$) to synthesise a target unitary from a
    controllable subalgebra.

    Attributes:
        params: The bound `Parameters` object (the single source of truth
            for all configuration and for the live optimisation state).
        engine: The internal `GeopeEngine` constructed from ``params``.
        precision: Target fidelity threshold.
        max_step_size: Maximum line-search step size.
        gram_schmidt_step_size: Step size for Gram-Schmidt fallback moves.
        line_search_method: Line-search strategy from the most recent
            :meth:`optimize` call (``'golden_section'``, ``'difference_step'``,
            ``'adam'``, ``'adam_fd'`` or ``'adam_grad'``; ``'adam'`` is an alias
            for ``'adam_fd'``). ``None`` until :meth:`optimize` is first called.
        adam_lr: Adam learning rate from the most recent :meth:`optimize` call
            (used by the ``'adam*'`` methods); ``None`` until then.
        adam_steps: Number of Adam iterations from the most recent
            :meth:`optimize` call (used by the ``'adam*'`` methods); ``None``
            until then.
        step_size: Transient last line-search step size.
        history: Optional `History` logger (``None`` unless supplied),
            holding the full run trajectory.
    """

    def __init__(self,
                 params: Parameters,
                 precision: float = 0.9999999,
                 max_step_size: float = 0.9,
                 gram_schmidt_step_size: float = 1.3,
                 verbose: bool = False,
                 history: History | None = None) -> None:
        """Initialise the Geope optimiser.

        ``Geope`` requires a `Parameters` object — the engine, initial
        parameters, drift, constraints, pulse constraints, seed,
        initialisation spread, projective flag and ``param_transform`` are
        all read from it. To construct one, use :class:`Parameters`.

        Args:
            params: A `Parameters` instance bundling every input the
                optimiser needs.
            precision: Target fidelity. Defaults to 0.9999999.
            max_step_size: Maximum line-search step. Defaults to 0.9.
            gram_schmidt_step_size: Step size for Gram-Schmidt moves.
                Defaults to 1.3.
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
        elif isinstance(seed,jax.Array):
            self._key = seed  # already a jax.Array key
        else:
            self._key = jax.random.key(0)
        engine = GeopeEngine(
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
            constraints = None
        else:
            init_parameters = params.parameters
            drift_parameters = params.drift_parameters
            constraints = params.constraint_arrays

        self.engine = engine
        self._real_params = params.param_transform is not None

        self.history = history
        if self.history is not None:
            self.history.params = params
        self.step_size = 0

        self.precision = precision
        self.max_step_size = max_step_size
        self.gram_schmidt_step_size = gram_schmidt_step_size
        self.init_parameters_spread = params.init_spread
        self.pulse_constraints = params.pulse_constraints

        # The line-search method and its hyperparameters are arguments of
        # optimize(), not the constructor. The JIT-compiled update_step /
        # update_linesearch bake the method and hyperparameters into their
        # closures, so they are built lazily by optimize() (via
        # _configure_line_search) and rebuilt only when that configuration
        # changes. They stay unset until the first optimize() call.
        self.line_search_method = None
        self.adam_lr = None
        self.adam_steps = None
        self._linesearch_config = None
        self.update_step = None
        self.update_linesearch = None

        self.verbose = verbose
        # Initialize parameters
        self.init(init_parameters, drift_parameters, constraints, params.seed)

    def _init_for_param_transform(self, engine: GeopeEngine, params: Parameters) -> np.ndarray:
        """Compute initial parameters in experimental-parameter space.

        If ``params.parameters`` is shaped ``(piecewise_steps, n_exp)``,
        use it directly; otherwise sample uniformly in
        $[-\\text{init\\_spread}\\,\\pi, +\\text{init\\_spread}\\,\\pi]$.

        Args:
            engine: The wrapped ``GeopeEngine``.
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
        elif isinstance(seed,jax.Array):
            self._key = seed  # already a jax.Array key
        # else: keep existing self._key unchanged

        # Set constraints
        self.constraint_expander = None
        if constraints is not None:
            expander = np.eye(self.engine.projected_basis.lie_algebra_dim)
            constraints = constraints if isinstance(constraints, list) else [constraints]
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
                self.init_parameters = np.array(jax.random.uniform(
                    self._split_key(),
                    shape=(self.engine.piecewise_steps, n_exp),
                    minval=-self.init_parameters_spread * np.pi,
                    maxval=self.init_parameters_spread * np.pi,
                ))
        elif init_parameters is None:
            self.init_parameters = np.array([prepare_random_parameters(self.engine.projected_indices,
                                                                       expander=self.constraint_expander,
                                                                       spread=self.init_parameters_spread,
                                                                       key=self._split_key()) for _ in range(self.engine.piecewise_steps)])
        else:
            if np.array(init_parameters).shape == (self.engine.full_basis.lie_algebra_dim,):
                self.init_parameters = np.array([init_parameters] * self.engine.piecewise_steps)
            elif np.array(init_parameters).shape == (self.engine.piecewise_steps, self.engine.full_basis.lie_algebra_dim):
                self.init_parameters = np.array(init_parameters)
            elif np.array(init_parameters).shape == (self.engine.projected_basis.lie_algebra_dim,):
                self.init_parameters = np.zeros((self.engine.piecewise_steps, self.engine.full_basis.lie_algebra_dim))
                self.init_parameters[:, self.engine.projected_indices] = np.array(init_parameters)
            elif np.array(init_parameters).shape == (self.engine.piecewise_steps, self.engine.projected_basis.lie_algebra_dim):
                self.init_parameters = np.zeros((self.engine.piecewise_steps, self.engine.full_basis.lie_algebra_dim))
                self.init_parameters[:, self.engine.projected_indices] = np.array(init_parameters)
            else:
                raise ValueError("Initial parameters must be of shape (full_basis.lie_algebra_dim,) or (full_basis.lie_algebra_dim, piecewise_steps) or (projected_basis.lie_algebra_dim,) or (piecewise_steps, projected_basis.lie_algebra_dim)")
        if not self._real_params:
            if drift_parameters is not None and self.engine.drift_basis is None:
                raise ValueError("Drift parameters are set but no drift basis is defined.")
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
        else:
            self.drift_parameters = None
        self.params.parameters = np.array(self.init_parameters)
        _dtype = np.float64 if self._real_params else np.complex128
        free_params = jnp.array([p[self.engine.proj_drift_indices] for p in self.params.parameters]).astype(_dtype)
        self.params.fidelity = self.engine.fid_U_fn(self.engine.compute_U_fn(free_params))
        self.step_size = 0
        if self.history is not None:
            self.history.reset()
            self.history.record(self)        # step 0

    def _configure_line_search(
        self,
        line_search_method: str,
        adam_lr: float,
        adam_steps: int,
    ) -> None:
        """Select the line-search method and (re)build its update functions.

        The JIT-compiled ``update_step`` / ``update_linesearch`` close over the
        method string and Adam hyperparameters, so they must be recreated and
        re-traced whenever the configuration changes. The current
        configuration is memoised in ``_linesearch_config`` so that repeated
        ``optimize()`` calls with unchanged settings reuse the already-compiled
        functions instead of triggering a fresh JAX recompilation.

        Args:
            line_search_method: Line-search strategy (see :meth:`optimize`).
            adam_lr: Adam learning rate (used by the ``'adam*'`` methods).
            adam_steps: Number of Adam iterations (used by the ``'adam*'``
                methods).
        """
        config = (line_search_method, adam_lr, adam_steps)
        if self._linesearch_config == config:
            return
        self.line_search_method = line_search_method
        self.adam_lr = adam_lr
        self.adam_steps = adam_steps
        self.update_linesearch = self.get_update_linesearch(
            self.engine.fid_U_fn, self.engine.compute_U_fn)
        self.update_step = self.get_update_step()
        self._linesearch_config = config

    def optimize(self, max_steps: int = 1000,
                 line_search_method: str = "golden_section",
                 adam_lr: float = 0.05,
                 adam_steps: int = 3) -> Parameters:
        """Run the GEOPE optimisation loop.

        Iterates geodesic update steps until the fidelity exceeds
        ``self.precision`` or ``max_steps`` is reached.

        Args:
            max_steps: Maximum number of optimisation steps. Defaults to 1000.
            line_search_method: ``'golden_section'``, ``'difference_step'``,
                ``'adam'``, ``'adam_fd'`` or ``'adam_grad'``. ``'adam'`` is an
                alias for ``'adam_fd'`` (Adam with a finite-difference
                gradient); ``'adam_grad'`` uses an exact autodiff gradient.
                Defaults to ``'golden_section'``.
            adam_lr: Learning rate for the Adam line-search methods.
                Defaults to 0.05. Ignored by other methods.
            adam_steps: Number of Adam iterations for the Adam line-search
                methods. Defaults to 3 — enough to resolve the 1-D step size
                over the clipped ``max_step_size`` interval at the default
                ``adam_lr``; smaller learning rates may need more. Ignored by
                other methods.

        Returns:
            The bound `Parameters` instance, carrying the final
            ``parameters`` / ``fidelity``. Read the result via
            ``params.parameters``, ``params.fidelity`` or ``params.to_dict()``;
            the full trajectory and ``best_*`` live on ``geope.history`` when
            a `History` was supplied.
        """
        # Select the line-search method and (re)build the JIT-compiled update
        # functions if this configuration differs from the last run.
        self._configure_line_search(line_search_method, adam_lr, adam_steps)

        # Build pulse-constrained update step if needed
        pulse_templates = None
        if self.pulse_constraints is not None and self.engine.piecewise_steps > 1:
            combined_expander, pulse_templates = self._build_optimize_expander()
            update_step = self.get_update_step(expander_override=combined_expander)
        else:
            update_step = self.update_step

        step = 0
        _dtype = jnp.float64 if self._real_params else jnp.complex128
        while (self.params.fidelity < self.precision) and (step < max_steps):
            step += 1
            free_params = self.params.parameters[:, self.engine.proj_drift_indices].astype(_dtype)
            coeffs, new_params_update, fidelity, step_size = update_step(free_params, self.params.parameters, self.engine.piecewise_steps, self._split_key())

            if fidelity > self.precision:
                if self.verbose:
                    print(
                        f"[{step}/{max_steps}] [Fidelity = {fidelity}] A solution!                                                                     ",
                        end="\r")
            elif (fidelity > self.params.fidelity) and not jnp.isclose(fidelity, self.params.fidelity,
                                                                       atol=(1 - self.precision) / 100):
                if self.verbose:
                    print(
                        f"[{step}/{max_steps}] [Fidelity = {fidelity}] Omega geodesic gave a positive fidelity update for this step...                 ",
                        end="\r")
            else:
                if self.verbose:
                    print(
                        f"[{step}/{max_steps}] [Fidelity = {self.params.fidelity}] Omega geodesic gave a negative fidelity update for this step. Moving phi away...    ",
                        end="\r")
                if self.gram_schmidt_step_size:
                    new_params_update, fidelity, step_size = self.gram_schmidt(coeffs)
                pass

            self.add_parameters(new_params_update, fidelity, step_size)

            # Enforce pulse template constraints if applicable
            if pulse_templates is not None:
                self._enforce_pulse_template(pulse_templates)
        if self.verbose:
            print("")
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

        Args:
            params: Parameter array (full, projected+drift, or projected shape).
            fidelity: Pre-computed fidelity value. If ``None``, it is
                computed from the parameters.
            step_size: Step size used. Defaults to ``self.max_step_size``.

        Returns:
            The fidelity of the new parameter set.
        """
        if self._real_params:
            # Experimental space: only (piecewise_steps, n_exp) is valid
            new_params = np.array(params)
        elif params.shape == (self.engine.piecewise_steps, self.engine.full_basis.lie_algebra_dim):
            new_params = np.zeros((self.engine.piecewise_steps, self.engine.full_basis.lie_algebra_dim))
            new_params = params
        elif params.shape == (self.engine.piecewise_steps, self.engine.proj_drift_basis.lie_algebra_dim):
            new_params = np.zeros((self.engine.piecewise_steps, self.engine.full_basis.lie_algebra_dim))
            new_params[:, self.engine.proj_drift_indices] = params
        elif params.shape == (self.engine.piecewise_steps, self.engine.projected_basis.lie_algebra_dim):
            new_params = np.zeros((self.engine.piecewise_steps, self.engine.full_basis.lie_algebra_dim))
            new_params[:, self.engine.projected_indices] = params
            if self.engine.drift_basis is not None:
                new_params[:, self.engine.drift_indices] = jnp.tile(self.drift_parameters, (self.engine.piecewise_steps, 1))
        else:
            ValueError("Parameter shape does not match with full basis, projected & drift basis, or projected basis.")
        if fidelity is None:
            _dtype = jnp.float64 if self._real_params else jnp.complex128
            free_params = new_params[:, self.engine.proj_drift_indices].astype(_dtype)
            fidelity = self.engine.fid_U_fn(self.engine.compute_U_fn(free_params))
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
            proj_c = np.array(jax.random.normal(self._split_key(),
                                                shape=(self.engine.piecewise_steps, n_exp))) * 0.1
        else:
            proj_c = np.array(
                [prepare_random_parameters(self.engine.projected_indices, self.constraint_expander,
                                           key=self._split_key())[
                     self.engine.proj_drift_indices] for _ in
                 range(self.engine.piecewise_steps)])
            if self.engine.drift_basis is not None:
                proj_c[:, self.engine.drift_indices_projdrift_basis] = jnp.tile(self.drift_parameters,
                                                                                (self.engine.piecewise_steps, 1))
        proj_c_con = np.concatenate(proj_c, axis=0)
        coeffs_con = np.concatenate(coeffs, axis=0)

        # Use the Gram-Schmidt procedure to generate a perpendicular vector to the previous coefficients.
        proj_c = jnp.reshape(proj_c_con - (((proj_c_con @ coeffs_con) / (coeffs_con @ coeffs_con)) * coeffs_con),
                             coeffs.shape)
        return self._update_parameters_gram_schmidt(proj_c)

    def _update_parameters_gram_schmidt(self, coeffs: np.ndarray) -> tuple[np.ndarray, Array, float]:
        """Evaluate the best signed Gram-Schmidt step.

        Tests both positive and negative directions and returns the
        one that yields higher fidelity.

        Args:
            coeffs: Orthogonalised coefficient array.

        Returns:
            A tuple ``(new_parameters, fidelity, step_size)``.
        """
        fids = {}
        scaled_gs_step = self.gram_schmidt_step_size
        if self._real_params:
            _dtype = jnp.float64
            current_params = self.params.parameters[:, self.engine.proj_drift_indices]
            for sign in [1, -1]:
                new_exp = current_params + sign * scaled_gs_step * coeffs
                fids[sign] = self.engine.fid_U_fn(
                    self.engine.compute_U_fn(jnp.array(new_exp, dtype=_dtype)))
            sign = 1 if fids[1] > fids[-1] else -1
            fidelity = fids[sign]
            new_parameters = current_params + sign * scaled_gs_step * coeffs
        else:
            for sign in [1, -1]:
                cs = np.copy(coeffs)
                cs[:, self.engine.proj_indices_projdrift_basis] = cs[:,
                                                                  self.engine.proj_indices_projdrift_basis] * sign * scaled_gs_step
                u = np.eye(self.engine.full_basis.dim)
                for i, c in enumerate(cs):
                    u = Hamiltonian(self.engine.proj_drift_basis,
                                    self.params.parameters[i][self.engine.proj_drift_indices] + c).unitary.matrix @ u
                fids[sign] = self.engine.fid_U_fn(u)

            if fids[1] > fids[-1]:
                sign = 1
                fidelity = fids[1]
            else:
                sign = -1
                fidelity = fids[-1]
            coeffs[:, self.engine.proj_indices_projdrift_basis] = coeffs[:,
                                                                  self.engine.proj_indices_projdrift_basis] * sign * scaled_gs_step
            coeffs[:, self.engine.drift_indices_projdrift_basis] = 0
            new_parameters = np.array(self.params.parameters)[:, self.engine.proj_drift_indices] + coeffs

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
            proj_params = self.params.parameters[:, self.engine.projected_indices]
        E_pulse, pulse_templates = self.engine.build_pulse_expander(
            self.pulse_constraints, getattr(self, "_real_params", False),
            self.params.n_experimental_params, np.array(proj_params).real)
        if self.constraint_expander is not None:
            E_gate = np.kron(np.eye(self.engine.piecewise_steps), self.constraint_expander)
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
            proj_where = np.where(self.engine.projected_indices)[0]
            for k, tmpl in pulse_templates.items():
                full_idx = proj_where[k]
                col = params[:, full_idx].real
                scale = float(np.dot(col, tmpl))
                params[:, full_idx] = scale * tmpl
        self.params.parameters = params
        # Recompute fidelity after enforcement
        _dtype = jnp.float64 if getattr(self, "_real_params", False) else jnp.complex128
        free_params = params[:, self.engine.proj_drift_indices].astype(_dtype)
        fid = float(self.engine.fid_U_fn(self.engine.compute_U_fn(free_params)))
        self.params.fidelity = fid
        if self.history is not None:
            if "parameters" in self.history.logs:
                self.history.logs["parameters"][-1] = np.array(params)
            if "fidelities" in self.history.logs:
                self.history.logs["fidelities"][-1] = fid
            if "infidelities" in self.history.logs:
                self.history.logs["infidelities"][-1] = 1 - fid

    def get_update_linesearch(
        self, fid_fn: Callable[..., Array], compute_U_fn: Callable[..., Array]
    ) -> Callable[..., tuple[Array, Array, Array]]:
        """Build a JIT-compiled line-search update function.

        Returns a function that, given current parameters and a search
        direction, finds the optimal step size via the configured
        line-search method.

        Args:
            fid_fn: JIT-compiled fidelity function.
            compute_U_fn: JIT-compiled unitary computation function.

        Returns:
            A callable ``update_linesearch(params, coeffs, piecewise_steps)``
            that returns ``(new_parameters, fidelity, dt)``.
        """

        infid_fn = self.engine.infid_U_fn

        def infidelity_t(t, params, coeffs):
            return infid_fn(compute_U_fn(params + t * coeffs))

        def max_t(params, coeffs, piecewise_steps):
            pos_raw = (self.upper_bounds - params) / coeffs
            neg_raw = (self.lower_bounds - params) / coeffs

            pos_coeffs_t_max = jnp.minimum(jnp.maximum(pos_raw, 0), self.max_step_size/piecewise_steps)
            neg_coeffs_t_max = jnp.minimum(jnp.maximum(neg_raw, 0), self.max_step_size/piecewise_steps)

            t_max_arr = pos_coeffs_t_max + neg_coeffs_t_max
            t_max = jnp.min(t_max_arr)
            return t_max

        @jax.jit
        def update_linesearch(params, coeffs, piecewise_steps):
            sliced_params = params.at[:, self.engine.proj_drift_indices].get()
            f = partial(infidelity_t, params=sliced_params, coeffs=coeffs)
            max_step_size = self.max_step_size/piecewise_steps
            if self.line_search_method == "golden_section":
                dt, infid = golden_section_search(f, -max_step_size, 0., tol=1e-5)
            elif self.line_search_method == "difference_step":
                tol = 0.1 * f(0)
                dt, infid = golden_section_search(f, -max_step_size, 0., tol=tol)
            elif self.line_search_method in ("adam", "adam_fd"):
                dt, infid = adam_line_search(
                    f, -max_step_size, 0.,
                    lr=self.adam_lr, num_steps=self.adam_steps,
                    finite_difference=True,
                )
            elif self.line_search_method == "adam_grad":
                dt, infid = adam_line_search(
                    f, -max_step_size, 0.,
                    lr=self.adam_lr, num_steps=self.adam_steps,
                    finite_difference=False,
                )
            else:
                raise ValueError(
                    f"Unknown line_search_method {self.line_search_method!r}; "
                    "expected 'golden_section', 'difference_step', 'adam', "
                    "'adam_fd', or 'adam_grad'."
                )
            new_parameters = sliced_params + dt * coeffs
            fidelity = 1 - infid

            return new_parameters, fidelity, dt

        return update_linesearch

    def get_update_step(self, expander_override: Array | None = None) -> Callable[..., tuple[Array, Array, Array, Array]]:
        """Build a JIT-compiled geodesic update step function.

        Computes the optimal linear combination of omegas that matches
        the geodesic direction, then performs a line search.

        Args:
            expander_override: Optional constraint expander to use in
                place of ``self.constraint_expander``'s Kronecker'd
                version. Used by pulse-shape constraints.

        Returns:
            A JIT-compiled callable
            ``update_step(free_params, params, piecewise_steps, key)``
            returning ``(coeffs, new_params, fidelity, step_size)``.
        """

        @jax.jit
        def update_step(free_params, params, piecewise_steps, key):

            gammaU_params, omegas_steps_phis = self.engine.gammas_and_omegas(free_params, key)

            if expander_override is not None:
                expander_gates = expander_override
            elif self.constraint_expander is not None:
                expander_gates = jnp.kron(jnp.eye(self.engine.piecewise_steps),
                                          self.constraint_expander)
            else:
                expander_gates = None

            sol = linear_comb_projected_coeffs_multigate(omegas_steps_phis, gammaU_params, expander_gates)

            # Expand the coefficients
            coeffs = jnp.zeros((self.engine.piecewise_steps, self.engine.proj_drift_basis.lie_algebra_dim))
            coeffs = coeffs.at[:, self.engine.proj_indices_projdrift_basis].set(sol)
            coeffs = coeffs * (jnp.sqrt(len(coeffs)) / jnp.linalg.norm(coeffs))

            new_params, fidelity_new_phi, step_size = self.update_linesearch(params, coeffs, piecewise_steps)

            return coeffs, new_params, fidelity_new_phi, step_size

        return update_step

def linear_comb_projected_coeffs_multigate(
    combination_vectors: Array, target_vector: Array, expander: Array | None
) -> Array:
    """Solve for the linear combination of omegas matching a target vector.

    Uses least-squares to find coefficients that best reproduce
    `target_vector` from the columns of the concatenated
    `combination_vectors`, optionally expanded by a constraint matrix.

    Args:
        combination_vectors: ``Array`` of omega vectors with shape
            ``(piecewise_steps, K_proj, K_full)``.
        target_vector: The target geodesic direction ``Array``.
        expander: Optional constraint expansion ``Array``, or ``None``.

    Returns:
        Solution ``Array`` of shape ``(piecewise_steps, K_proj)``.
    """
    comb_vecs = jnp.concatenate(combination_vectors, axis=0)
    comb_vecs_T = comb_vecs.T @ expander if expander is not None else comb_vecs.T

    res = jnp.linalg.lstsq(comb_vecs_T, target_vector)
    # TODO: If the residual is too large, we want to throw NaN and handle the error.

    sol = expander @ res[0] if expander is not None else res[0]
    return sol.reshape(combination_vectors.shape[0], combination_vectors.shape[1])


def geodesic_hamiltonian(unitary: Array, target_unitary: Array, projective: bool = True,
                         key: Array = jax.random.key(0)) -> Array:
    """Compute the geodesic Hamiltonian between a unitary and a target.

    Computes the generator $g = -i\\log(U^\\dagger U_T) \\in \\mathfrak{u}(d)$
    and returns $U g'$ where $g' = g - \\frac{\\mathrm{Tr}(g)}{d}\\mathbb{1}$
    (the SU part) when ``projective=True``, or $g' = g$ (full U) when
    ``projective=False``.

    Args:
        unitary: The current unitary ``Array``.
        target_unitary: The target unitary ``Array``.
        projective: If ``True``, subtract the global-phase generator
            (SU geodesic). If ``False``, keep it (U geodesic).
            Defaults to ``True``.
        key: JAX random key forwarded to ``logm``. Defaults to
            ``jax.random.key(0)``.

    Returns:
        The geodesic tangent ``Array`` $U g'$ at the current unitary.
    """
    g = -1.j * logm(jnp.einsum('ji,jk->ik', unitary.conj(), target_unitary), key=key)
    if projective:
        Id = jnp.eye(g.shape[0])
        global_phase = jnp.real(jnp.einsum('ij,ji->', Id, g)) / g.shape[0]
        g = g - global_phase * Id
    return unitary @ g


def get_geodesic_hamiltonian_fn(target_unitary: Array, projective: bool = True) -> Callable[[Array, Array], Array]:
    """Create a partial geodesic Hamiltonian function with a fixed target.

    Args:
        target_unitary: The target unitary ``Array`` to bind.
        projective: If ``True``, return the projective (SU) geodesic.
            Defaults to ``True``.

    Returns:
        A ``Callable[[Array, Array], Array]`` that accepts a unitary and a
        JAX random key and returns the geodesic Hamiltonian.
    """
    return partial(geodesic_hamiltonian, target_unitary=target_unitary, projective=projective)


def hvp_forward_over_reverse(
    f: Callable[[Array], Array], params: Array, v: Array
) -> Array:
    """Compute a Hessian-vector product via forward-over-reverse mode.

    Args:
        f: Scalar-valued callable of ``params``.
        params: Parameter ``Array`` at which to evaluate.
        v: Tangent ``Array`` for the Hessian-vector product.

    Returns:
        The Hessian-vector product $\\nabla^2 f \\cdot v$.
    """
    v = v.reshape(params.shape)
    return jax.jvp(jax.grad(f), (params,), (v,))[1]

