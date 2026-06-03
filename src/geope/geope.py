from __future__ import annotations

import numpy as np
import scipy.optimize as spo

import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_enable_x64", True)

from .pauli_projector import get_project_omegas_fn, get_project_omegas_fn_otf
from .engine import (
    Engine,
    fidelity,
    get_infidelity_fn,
    get_fidelity_full_fn,
    get_infidelity_full_fn,
)
from .lie import Hamiltonian, Basis
from .utils import golden_section_search, prepare_random_parameters, merge_constraints
from .logm import logm
from .jacobian_manual import get_jacobian_manual
from .parameters import Parameters
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


class Geope:
    """Top-level GEOPE optimiser for quantum gate synthesis.

    Orchestrates the geodesic-based optimisation of Lie-algebra
    parameters ($\\phi$) to synthesise a target unitary from a
    controllable subalgebra.

    Attributes:
        params: The bound `Parameters` object (the single source of truth
            for all configuration and the destination for the run history).
        engine: The internal `GeopeEngine` constructed from ``params``.
        max_steps: Maximum number of optimisation iterations.
        precision: Target fidelity threshold.
        max_step_size: Maximum line-search step size.
        gram_schmidt_step_size: Step size for Gram-Schmidt fallback moves.
        line_search_method: Line-search strategy (``'golden_section'`` or
            ``'difference_step'``).
        parameters: History of parameter arrays.
        fidelities: History of fidelity values.
        infidelities: History of infidelity values.
        step_sizes: History of step sizes.
        steps: History of step counts.
    """

    def __init__(self,
                 params: Parameters,
                 max_steps: int = 1000,
                 precision: float = 0.9999999,
                 max_step_size: float = 0.9,
                 gram_schmidt_step_size: float = 1.3,
                 line_search_method: str = "golden_section",
                 verbose: bool = False) -> None:
        """Initialise the Geope optimiser.

        ``Geope`` requires a `Parameters` object — the engine, initial
        parameters, drift, constraints, pulse constraints, seed,
        initialisation spread, projective flag and ``param_transform`` are
        all read from it. To construct one, use :class:`Parameters`.

        Args:
            params: A `Parameters` instance bundling every input the
                optimiser needs.
            max_steps: Maximum optimisation steps. Defaults to 1000.
            precision: Target fidelity. Defaults to 0.9999999.
            max_step_size: Maximum line-search step. Defaults to 0.9.
            gram_schmidt_step_size: Step size for Gram-Schmidt moves.
                Defaults to 1.3.
            line_search_method: ``'golden_section'`` or ``'difference_step'``.
                Defaults to ``'golden_section'``.
            verbose: Whether to print progress. Defaults to False.

        Raises:
            TypeError: If ``params`` is not a `Parameters` instance.
                The legacy ``Geope(engine=...)`` call site is no longer
                supported; build a ``Parameters`` object instead.
        """
        if not isinstance(params, Parameters):
            raise TypeError(
                "Geope requires a Parameters object as its first argument. "
                "The legacy `Geope(engine=...)` call site has been removed. "
                "Build a Parameters object with `geope.Parameters(basis=..., "
                "control=..., target=..., ...)` and pass that in."
            )

        self.params = params
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
            self._wrap_param_transform(engine, params)
            init_parameters = self._init_for_param_transform(engine, params)
            drift_parameters = None
            constraints = None
        else:
            init_parameters = params.init_parameters
            drift_parameters = params.drift_parameters
            constraints = params.constraint_arrays

        self.engine = engine
        self._real_params = params.param_transform is not None

        self.max_steps = max_steps
        self.precision = precision
        self.max_step_size = max_step_size
        self.gram_schmidt_step_size = gram_schmidt_step_size
        self.init_parameters_spread = params.init_spread
        self.line_search_method = line_search_method
        self.pulse_constraints = params.pulse_constraints

        # Get update steps
        self.gammas_and_omegas = self.get_gammas_and_omegas(self.engine.project_omegas_fn,
                                                            self.engine.jac_fn,
                                                            self.engine.compute_U_fn,
                                                            self.engine.geo_fn)
        self.update_step = self.get_update_step()
        self.update_linesearch = self.get_update_linesearch(self.engine.fid_U_fn,
                                                            self.engine.compute_U_fn)
        self.bound_parameters = self.get_bound_parameters(self.engine.fid_U_fn,
                                                          self.engine.compute_U_fn)

        self.verbose = verbose
        # Initialize parameters
        self.init(init_parameters, drift_parameters, constraints, params.seed)

    def _wrap_param_transform(self, engine: GeopeEngine, params: Parameters) -> None:
        """Replace ``engine.compute_U_fn`` and ``engine.jac_fn`` to honour
        ``params.param_transform``.

        The user-facing parameters $\\phi^{\\mathrm{exp}}\\in\\mathbb{R}^{N_g\\times n_{\\mathrm{exp}}}$
        are mapped to projected-basis coefficients via ``params.param_transform``
        (possibly step-dependent), embedded into the proj+drift basis,
        and combined with the drift before the original
        ``compute_U_fn`` is called. The Jacobian is replaced by a
        split-real-imag version so that complex tracing preserves
        the imaginary part of intermediates.

        Args:
            engine: The freshly-constructed ``GeopeEngine`` to mutate.
            params: The ``Parameters`` object carrying ``param_transform``.
        """
        raw_compute_U = engine.compute_U_fn
        n_exp = params.n_experimental_params
        n_proj_drift = engine.proj_drift_basis.lie_algebra_dim
        proj_idx_pd = engine.proj_indices_projdrift_basis
        drift_idx_pd = engine.drift_indices_projdrift_basis

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
                np.array(engine.projected_basis.overlap(params.basis)))[0])
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

        engine.compute_U_fn = jax.jit(_wrapped_compute_U)
        # Split real/imag Jacobian to avoid losing imaginary parts through
        # real-valued intermediates in the user transform.
        _compute_U = engine.compute_U_fn

        def _split_U(x, _cu=_compute_U):
            U = _cu(x)
            return jnp.stack([jnp.real(U), jnp.imag(U)])

        _raw_jac_split = jax.jacobian(_split_U, argnums=0)

        def _jac_fn(x, _rjs=_raw_jac_split):
            jac_split = _rjs(x)
            return jac_split[0] + 1j * jac_split[1]

        engine.jac_fn = _jac_fn
        engine.infid_fn = lambda x: engine.infid_U_fn(engine.compute_U_fn(x))
        engine.grad_fn = jax.value_and_grad(engine.infid_fn)

        # Override engine indices so the rest of Geope operates in experimental space
        engine.proj_drift_indices = np.arange(n_exp)
        engine.drift_indices = np.array([], dtype=int)
        engine.drift_basis = None
        engine.proj_indices_projdrift_basis = np.ones(n_exp, dtype=bool)
        engine.drift_indices_projdrift_basis = np.zeros(n_exp, dtype=bool)
        engine.proj_drift_basis._lie_algebra_dim = n_exp
        engine.projected_indices = np.ones(n_exp, dtype=bool)

    def _init_for_param_transform(self, engine: GeopeEngine, params: Parameters) -> np.ndarray:
        """Compute initial parameters in experimental-parameter space.

        If ``params.init_parameters`` is shaped ``(piecewise_steps, n_exp)``,
        use it directly; otherwise sample uniformly in
        $[-\\text{init\\_spread}\\,\\pi, +\\text{init\\_spread}\\,\\pi]$.

        Args:
            engine: The wrapped ``GeopeEngine``.
            params: The ``Parameters`` object.

        Returns:
            An ``np.ndarray`` of shape ``(piecewise_steps, n_exp)``.
        """
        n_exp = params.n_experimental_params
        _user_init = np.array(params.init_parameters)
        if _user_init.shape == (params.piecewise_steps, n_exp):
            return _user_init
        rng = np.random.default_rng(params.seed)
        return rng.uniform(
            -params.init_spread * np.pi, params.init_spread * np.pi,
            (params.piecewise_steps, n_exp))

    def init(
        self,
        init_parameters: np.ndarray | None = None,
        drift_parameters: np.ndarray | None = None,
        constraints: list[np.ndarray] | np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        """(Re-)initialise optimiser state.

        Sets up constraints, initial parameters, drift parameters,
        and resets the fidelity / step history.

        Args:
            init_parameters: Initial parameter array. Defaults to random.
            drift_parameters: Fixed drift parameter values. Defaults to ones.
            constraints: Linear equality constraints.
            seed: Random seed for reproducibility.
        """
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
                rng = np.random.default_rng(seed)
                n_exp = self.params.n_experimental_params
                self.init_parameters = rng.uniform(
                    -self.init_parameters_spread * np.pi,
                    self.init_parameters_spread * np.pi,
                    (self.engine.piecewise_steps, n_exp))
        elif init_parameters is None:
            self.init_parameters = np.array([prepare_random_parameters(self.engine.projected_indices,
                                                                       expander=self.constraint_expander,
                                                                       spread=self.init_parameters_spread,
                                                                       seed=seed) for _ in range(self.engine.piecewise_steps)])
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
        self.parameters = [self.init_parameters]
        _dtype = np.float64 if self._real_params else np.complex128
        free_params = jnp.array([p[self.engine.proj_drift_indices] for p in self.parameters[-1]]).astype(_dtype)
        self.fidelities = [self.engine.fid_U_fn(self.engine.compute_U_fn(free_params))]
        self.infidelities = [1 - self.fidelities[-1]]
        self.step_sizes = [0]
        self.steps = [0]

    def optimize(self, extra_steps: int = 0) -> Parameters:
        """Run the GEOPE optimisation loop.

        Iterates geodesic update steps until the fidelity exceeds
        ``self.precision`` or the maximum number of steps is reached.

        Args:
            extra_steps: Additional steps beyond ``self.max_steps``.
                Defaults to 0.

        Returns:
            The bound `Parameters` instance, with its mutable history
            populated. Call ``params.best_fidelity``, ``params.best_parameters``
            or ``params.to_dict()`` to read the solution.
        """
        # Build pulse-constrained update step if needed
        pulse_templates = None
        if self.pulse_constraints is not None and self.engine.piecewise_steps > 1:
            combined_expander, pulse_templates = self._build_optimize_expander()
            update_step = self.get_update_step(expander_override=combined_expander)
        else:
            update_step = self.update_step

        step = self.steps[-1]
        _dtype = jnp.float64 if self._real_params else jnp.complex128
        while (self.fidelities[-1] < self.precision) and (step < self.max_steps + extra_steps):
            step += 1
            free_params = self.parameters[-1][:, self.engine.proj_drift_indices].astype(_dtype)
            coeffs, new_params_update, fidelity, step_size = update_step(free_params, self.parameters[-1], self.engine.piecewise_steps)

            if fidelity > self.precision:
                if self.verbose:
                    print(
                        f"[{step}/{self.max_steps + extra_steps}] [Fidelity = {fidelity}] A solution!                                                                     ",
                        end="\r")
            elif (fidelity > self.fidelities[-1]) and not jnp.isclose(fidelity, self.fidelities[-1],
                                                                      atol=(1 - self.precision) / 100):
                if self.verbose:
                    print(
                        f"[{step}/{self.max_steps + extra_steps}] [Fidelity = {fidelity}] Omega geodesic gave a positive fidelity update for this step...                 ",
                        end="\r")
            else:
                if self.verbose:
                    print(
                        f"[{step}/{self.max_steps + extra_steps}] [Fidelity = {self.fidelities[-1]}] Omega geodesic gave a negative fidelity update for this step. Moving phi away...    ",
                        end="\r")
                if self.gram_schmidt_step_size:
                    new_params_update, fidelity, step_size = self.gram_schmidt(coeffs)
                pass
                
            self.add_parameters(new_params_update, fidelity, step_size)

            # Enforce pulse template constraints if applicable
            if pulse_templates is not None:
                self._enforce_pulse_template(pulse_templates)
        self.max_steps += extra_steps
        if self.verbose:
            print("")
        # Sync history to Parameters object (always)
        self.params.parameters = self.parameters
        self.params.fidelities = self.fidelities
        self.params.infidelities = self.infidelities
        self.params.step_sizes = self.step_sizes
        self.params.steps = self.steps
        return self.params

    def add_parameters(
        self,
        params: np.ndarray | Array,
        fidelity: float | Array | None = None,
        step_size: float | None = None,
    ) -> float | Array:
        """Append a new parameter set to the optimisation history.

        Handles different parameter shapes by mapping them back to the
        full basis. Computes the fidelity if not provided.

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
        self.parameters.append(new_params)

        if fidelity is None:
            _dtype = jnp.float64 if self._real_params else jnp.complex128
            free_params = self.parameters[-1][:, self.engine.proj_drift_indices].astype(_dtype)
            fidelity = self.engine.fid_U_fn(self.engine.compute_U_fn(free_params))
        self.fidelities.append(fidelity)
        self.infidelities.append(1 - fidelity)
        if step_size is None:
            step_size = self.max_step_size
        self.step_sizes.append(step_size)
        self.steps.append(self.steps[-1]+1)  
        return fidelity

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
            proj_c = np.random.randn(self.engine.piecewise_steps, n_exp) * 0.1
        else:
            proj_c = np.array(
                [prepare_random_parameters(self.engine.projected_indices, self.constraint_expander)[
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
            current_params = self.parameters[-1][:, self.engine.proj_drift_indices]
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
                                    self.parameters[-1][i][self.engine.proj_drift_indices] + c).unitary.matrix @ u
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
            new_parameters = np.array(self.parameters[-1])[:, self.engine.proj_drift_indices] + coeffs

        # if self.parameter_bounds is not None:
        #     new_parameters, fidelity = self.bound_parameters(new_parameters, scaled_gs_step)

        return new_parameters, fidelity, sign * scaled_gs_step

    def _build_pulse_expander(self, proj_params: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Build a flat-space expander enforcing pulse-shape constraints.

        For each constrained parameter index $k$ the time profile
        $\\phi_k(g)$ is constrained to a fixed unit-norm template
        $t_k\\in\\mathbb{R}^{N_g}$, leaving only the scalar amplitude
        free. Unconstrained parameters retain one free variable per gate.

        Args:
            proj_params: Current projected parameter ``np.ndarray`` of
                shape ``(N_g, K_proj)`` (or ``(N_g, n_exp)`` in
                experimental mode) used as the shape template.

        Returns:
            A tuple ``(E, templates)`` where ``E`` is the
            ``(N_g * n_proj, n_free)`` expander matrix and ``templates``
            is a dict mapping the constrained integer index to its
            unit-norm template vector.
        """
        L = self.engine.piecewise_steps
        if getattr(self, "_real_params", False):
            n_proj = self.params.n_experimental_params
            pulse_indices = list(self.pulse_constraints)
        else:
            n_proj = self.engine.projected_basis.lie_algebra_dim
            proj_labels = list(self.engine.projected_basis.labels)
            if isinstance(self.pulse_constraints, dict):
                labels = list(self.pulse_constraints.keys())
            else:
                labels = list(self.pulse_constraints)
            pulse_indices = [proj_labels.index(label) for label in labels]
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
            proj_params = self.parameters[-1]
        else:
            proj_params = self.parameters[-1][:, self.engine.projected_indices]
        E_pulse, pulse_templates = self._build_pulse_expander(np.array(proj_params).real)
        if self.constraint_expander is not None:
            E_gate = np.kron(np.eye(self.engine.piecewise_steps), self.constraint_expander)
            combined = E_gate @ np.linalg.pinv(E_gate) @ E_pulse
        else:
            combined = E_pulse
        return jnp.array(combined), pulse_templates

    def _enforce_pulse_template(self, pulse_templates: dict[int, np.ndarray]) -> None:
        """Re-project constrained parameter columns onto their templates.

        Mutates ``self.parameters[-1]`` in place so that for every
        constrained index $k$, the time profile of $\\phi_k$ is
        proportional to the stored unit template $t_k$. Then
        recomputes the trailing fidelity / infidelity.

        Args:
            pulse_templates: Dict mapping projected (or experimental)
                index to the unit-norm template vector.
        """
        params = np.array(self.parameters[-1])
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
        self.parameters[-1] = params
        # Recompute fidelity after enforcement
        _dtype = jnp.float64 if getattr(self, "_real_params", False) else jnp.complex128
        free_params = params[:, self.engine.proj_drift_indices].astype(_dtype)
        fid = float(self.engine.fid_U_fn(self.engine.compute_U_fn(free_params)))
        self.fidelities[-1] = fid
        self.infidelities[-1] = 1 - fid

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
            new_parameters = sliced_params + dt * coeffs
            fidelity = 1 - infid

            return new_parameters, fidelity, dt

        return update_linesearch

    def get_gammas_and_omegas(
        self,
        project_omegas_fn: Callable[..., Array],
        jac_fn: Callable[..., Array],
        compute_U_fn: Callable[..., Array],
        geodesic_fn: Callable[..., Array],
    ) -> Callable[[Array], tuple[Array, Array]]:
        """Build a JIT-compiled function that computes gammas and omegas.

        Gammas are the projected geodesic Hamiltonian coefficients;
        omegas encode the Jacobian of each unitary gate with respect
        to each parameter, projected onto the Pauli basis.

        Args:
            project_omegas_fn: Function to project matrices onto the basis.
            jac_fn: Jacobian function of the unitary.
            compute_U_fn: Unitary computation function.
            geodesic_fn: Geodesic Hamiltonian function.

        Returns:
            A JIT-compiled callable ``gammas_and_omegas(free_params)``.
        """

        @jax.jit
        def gammas_and_omegas(free_params):
            unitary = compute_U_fn(free_params)
            gammaU = geodesic_fn(unitary)
            gammaU_params = project_omegas_fn(jnp.expand_dims(gammaU, axis=0)).squeeze(axis=0) / (gammaU.shape[0])

            dUs = jnp.array(jac_fn(free_params))
            dUs_t = jnp.transpose(dUs, [2, 3, 0, 1])
            omegas_steps_phis = jnp.array([project_omegas_fn(1.j * omegaUs) for omegaUs in dUs_t])

            if np.any(self.engine.proj_drift_basis):
                omegas_steps_phis = omegas_steps_phis.at[:, self.engine.proj_indices_projdrift_basis, :].get()

            return gammaU_params, omegas_steps_phis

        return gammas_and_omegas

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
            ``update_step(free_params, params, piecewise_steps)``
            returning ``(coeffs, new_params, fidelity, step_size)``.
        """

        @jax.jit
        def update_step(free_params, params, piecewise_steps):

            gammaU_params, omegas_steps_phis = self.gammas_and_omegas(free_params)

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

    def get_bound_parameters(
        self, fid_fn: Callable[..., Array], compute_U_fn: Callable[..., Array]
    ) -> Callable[[Array, float], tuple[Array, Array]]:
        """Build a JIT-compiled parameter-clipping function.

        Clips parameters to the configured bounds and returns the
        resulting fidelity.

        Args:
            fid_fn: JIT-compiled fidelity function.
            compute_U_fn: JIT-compiled unitary computation function.

        Returns:
            A JIT-compiled callable
            ``bound_parameters(params, offset)`` returning
            ``(clipped_params, fidelity)``.
        """

        @jax.jit
        def bound_parameters(params, offset):
            basis = self.engine.proj_drift_basis
            bounds = basis.generate_bounds(self.parameter_bounds, self.engine.piecewise_steps)
            new_params = jnp.clip(params.astype(jnp.float64), 
                                  jnp.array(bounds[0], dtype=jnp.float64) + offset, 
                                  jnp.array(bounds[1], dtype=jnp.float64) - offset)
            fid = fid_fn(compute_U_fn(new_params))
            return new_params.astype(jnp.complex128), fid

        return bound_parameters

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
        # Update the number of piecewise steps and initialise new parameters
        self.engine.piecewise_steps = self.engine.piecewise_steps * piecewise_steps_multiplier

        new_parameters = [list(np.copy(self.parameters[-1])) for _ in range(piecewise_steps_multiplier)]
        self.parameters.append(
            np.array([x for group in zip(*new_parameters) for x in group]) / piecewise_steps_multiplier)

        _dtype = jnp.float64 if self._real_params else jnp.complex128
        free_params = self.parameters[-1][:, self.engine.proj_drift_indices].astype(_dtype)
        proj_params = self.parameters[-1][:, self.engine.projected_indices].astype(_dtype)

        # Record a step for the initial subdivided parameters so all history lists stay in sync
        self.fidelities.append(self.fidelities[-1])
        self.infidelities.append(self.infidelities[-1])
        self.step_sizes.append(0)
        self.steps.append(self.steps[-1] + 1)

        params_update = self.get_free_params_update_smoothing()

        c = 0
        diff = np.inf
        pulse_templates = None
        if self.pulse_constraints is not None:
            E_pulse, pulse_templates = self._build_pulse_expander(np.array(proj_params).real)
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
            _, omegas_steps_phis = self.gammas_and_omegas(free_params)
            vh, num = find_null_space(omegas_steps_phis, expander)

            assert num > 0, "Nullspace is empty!"
            null_space = vh[num:, :].T.conj()

            proj_params, diff = null_space_function(proj_params, null_space, expander, rate, **kwargs)

            if pulse_templates is not None:
                proj_params = np.array(proj_params)
                for k, tmpl in pulse_templates.items():
                    scale = float(np.dot(proj_params[:, k].real, tmpl))
                    proj_params[:, k] = scale * tmpl

            free_params = params_update(proj_params, self.parameters[-1])

            fid = self.engine.fid_U_fn(self.engine.compute_U_fn(free_params))

            c += 1
            print(
                f"[{c}/{max_steps}] [Fidelity = {fid}] {label} : cost = {diff} (aim = {diff_tol})                      ",
                end="\r")
        print(f"[{c}/{max_steps}] [Fidelity = {fid}] {label} : cost = {diff} (aim = {diff_tol})                        ")
        success = diff_tol >= diff
        new_params = np.zeros_like(self.parameters[-1])
        new_params[:, self.engine.proj_drift_indices] = [p.real for p in free_params]
        self.parameters.append(new_params)
        self.fidelities.append(fid)
        self.infidelities.append(1 - fid)
        self.step_sizes.append(rate)
        self.steps.append(self.steps[-1] + 1)
        return success, c


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


def geodesic_hamiltonian(unitary: Array, target_unitary: Array, projective: bool = True) -> Array:
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

    Returns:
        The geodesic tangent ``Array`` $U g'$ at the current unitary.
    """
    g = -1.j * logm(jnp.einsum('ji,jk->ik', unitary.conj(), target_unitary), key=jax.random.key(1111))
    if projective:
        Id = jnp.eye(g.shape[0])
        global_phase = jnp.real(jnp.einsum('ij,ji->', Id, g)) / g.shape[0]
        g = g - global_phase * Id
    return unitary @ g


def get_geodesic_hamiltonian_fn(target_unitary: Array, projective: bool = True) -> Callable[[Array], Array]:
    """Create a partial geodesic Hamiltonian function with a fixed target.

    Args:
        target_unitary: The target unitary ``Array`` to bind.
        projective: If ``True``, return the projective (SU) geodesic.
            Defaults to ``True``.

    Returns:
        A ``Callable[[Array], Array]`` that accepts a single unitary
        and returns the geodesic Hamiltonian.
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
    phi = phi.astype(jnp.float64)
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
    phi = phi.astype(jnp.float64)
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
        phi_flat = phi.flatten().astype(jnp.float64)
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
        phi_flat = phi.flatten().astype(jnp.float64)
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
        phi_flat = phi.flatten().astype(jnp.float64)
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
    phi_flat = phi_flat.astype(jnp.float64)
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
    phi_flat = phi_flat.astype(jnp.float64)
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
    phi_flat = phi_flat.astype(jnp.float64)

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