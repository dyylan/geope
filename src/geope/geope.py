import numpy as np
import scipy.optimize as spo

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .pauli_projector import get_project_omegas_fn, get_project_omegas_fn_otf
from .engine import Engine, fidelity
from .lie import Hamiltonian, Basis
from .utils import golden_section_search, prepare_random_parameters, merge_constraints
from .logm import logm
from .jacobian_manual import get_jacobian_manual
from functools import partial


class GeopeEngine(Engine):
    """All jitted functions for Geope"""

    def __init__(self, target_unitary: Basis,
                 full_basis: Basis,
                 projected_basis: Basis,
                 drift_basis: Basis = None,
                 gates: int = 1,
                 batch_size=None):
        super(GeopeEngine, self).__init__(target_unitary, full_basis, projected_basis, drift_basis, gates)
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
        self.geo_fn = jax.jit(get_geodesic_hamiltonian_fn(target_unitary))
        self.infid_fn = lambda x: self.infid_U_fn(self.compute_U_fn(x))
        self.grad_fn = jax.value_and_grad(self.infid_fn)


class Geope:
    """
    Handling the optimization of the phi parameters.
    """

    def __init__(self,
                 engine: GeopeEngine,
                 drift_parameters=None,
                 init_parameters=None,
                 constraints=None,
                 max_steps=1000,
                 precision=0.9999999,
                 max_step_size=0.9,
                 gram_schmidt_step_size=1.3,
                 init_parameters_spread=0.1,
                 line_search_method="golden_section",
                 verbose: bool = False,
                 seed: int = None):
        self.engine = engine
        self.max_steps = max_steps
        self.precision = precision
        self.max_step_size = max_step_size
        self.gram_schmidt_step_size = gram_schmidt_step_size
        self.init_parameters_spread = init_parameters_spread
        self.line_search_method = line_search_method

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
        self.init(init_parameters, drift_parameters, constraints, seed)

    def init(self, init_parameters=None, drift_parameters=None, constraints=None, seed=None):
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
        if init_parameters is None:
            self.init_parameters = np.array([prepare_random_parameters(self.engine.projected_indices,
                                                                       expander=self.constraint_expander,
                                                                       spread=self.init_parameters_spread,
                                                                       seed=seed) for _ in range(self.engine.gates)])
        else:
            if np.array(init_parameters).shape == (self.engine.full_basis.lie_algebra_dim,):
                self.init_parameters = np.array([init_parameters] * self.engine.gates)
            elif np.array(init_parameters).shape == (self.engine.gates, self.engine.full_basis.lie_algebra_dim):
                self.init_parameters = np.array(init_parameters)
            else:
                raise ValueError("Initial parameters must be of shape (projected_basis.lie_algebra_dim,) or (gates, projected_basis.lie_algebra_dim)")
            # assert self.engine.full_basis.lie_algebra_dim == self.init_parameters.shape[0], \
            #     "Drift parameters must be the same length as the size of the drift basis."
        if self.engine.drift_basis is not None:
            if drift_parameters is None:
                self.drift_parameters = np.ones(self.engine.drift_basis.lie_algebra_dim)
            else:
                self.drift_parameters = np.array(drift_parameters)
                assert self.engine.drift_basis.lie_algebra_dim == self.drift_parameters.shape[0], \
                    "Drift parameters must be the same length as the size of the drift basis."

            self.init_parameters[:, self.engine.drift_indices] = np.tile(self.drift_parameters, (self.engine.gates, 1))
        self.parameters = [self.init_parameters]
        free_params = jnp.array([p[self.engine.proj_drift_indices] for p in self.parameters[-1]]).astype(np.complex128)
        self.fidelities = [self.engine.fid_U_fn(self.engine.compute_U_fn(free_params))]
        self.infidelities = [1 - self.fidelities[-1]]
        self.step_sizes = [0]
        self.steps = [0]

    def optimize(self, extra_steps=0):
        step = self.steps[-1]
        while (self.fidelities[-1] < self.precision) and (step < self.max_steps + extra_steps):
            step += 1
            free_params = self.parameters[-1][:, self.engine.proj_drift_indices].astype(jnp.complex128)
            coeffs, new_params_update, fidelity, step_size = self.update_step(free_params, self.parameters[-1], self.engine.gates)

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
        self.max_steps += extra_steps
        if self.verbose:
            print("")
        if self.fidelities[-1] >= self.precision:
            return True
        else:
            return False

    def add_parameters(self, params, fidelity=None, step_size=None):
        if params.shape == (self.engine.gates, self.engine.full_basis.lie_algebra_dim):
            new_params = np.zeros((self.engine.gates, self.engine.full_basis.lie_algebra_dim))
            new_params = params
        elif params.shape == (self.engine.gates, self.engine.proj_drift_basis.lie_algebra_dim):
            new_params = np.zeros((self.engine.gates, self.engine.full_basis.lie_algebra_dim))
            new_params[:, self.engine.proj_drift_indices] = params
        elif params.shape == (self.engine.gates, self.engine.projected_basis.lie_algebra_dim):
            new_params = np.zeros((self.engine.gates, self.engine.full_basis.lie_algebra_dim))
            new_params[:, self.engine.projected_indices] = params
            if self.engine.drift_basis is not None:
                new_params[:, self.engine.drift_indices] = jnp.tile(self.drift_parameters, (self.engine.gates, 1))
        else:
            ValueError("Parameter shape does not match with full basis, projected & drift basis, or projected basis.")
        self.parameters.append(new_params)

        if fidelity is None:
            free_params = self.parameters[-1][:, self.engine.proj_drift_indices].astype(jnp.complex128)
            fidelity = self.engine.fid_U_fn(self.engine.compute_U_fn(free_params))
        self.fidelities.append(fidelity)
        self.infidelities.append(1 - fidelity)
        if step_size is None:
            step_size = self.max_step_size
        self.step_sizes.append(step_size)
        self.steps.append(self.steps[-1]+1)  
        return fidelity

    def smooth(self, piecewise_steps_multiplier=1, smoothing_rate=0.01, max_smoothing_steps=100, diff_tol=0.1):
        success, iters = self._null_space_optimisation(piecewise_smoothing, 
                                                        piecewise_steps_multiplier=piecewise_steps_multiplier,
                                                        max_steps=max_smoothing_steps, 
                                                        rate=smoothing_rate,
                                                        diff_tol=diff_tol,
                                                        label="Smoothing")
        return success, iters

    def bound(self, parameter_bounds, method="projected_gradient", bounding_rate=0.01, max_bounding_steps=100, diff_tol=0.1):
        self.parameter_bounds = parameter_bounds
        bounds = self.engine.proj_drift_basis.generate_bounds(self.parameter_bounds, 
                                                              self.engine.gates)
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
    
    def gram_schmidt(self, coeffs):
        proj_c = np.array(
            [prepare_random_parameters(self.engine.projected_indices, self.constraint_expander)[
                 self.engine.proj_drift_indices] for _ in
             range(self.engine.gates)])
        if self.engine.drift_basis is not None:
            proj_c[:, self.engine.drift_indices_projdrift_basis] = jnp.tile(self.drift_parameters,
                                                                            (self.engine.gates, 1))
        proj_c_con = np.concatenate(proj_c, axis=0)
        coeffs_con = np.concatenate(coeffs, axis=0)

        # Use the Gram-Schmidt procedure to generate a perpendicular vector to the previous coefficients.
        proj_c = jnp.reshape(proj_c_con - (((proj_c_con @ coeffs_con) / (coeffs_con @ coeffs_con)) * coeffs_con),
                             coeffs.shape)
        return self._update_parameters_gram_schmidt(proj_c)

    def _update_parameters_gram_schmidt(self, coeffs):
        fids = {}
        for sign in [1, -1]:
            cs = np.copy(coeffs)
            cs[:, self.engine.proj_indices_projdrift_basis] = cs[:,
                                                              self.engine.proj_indices_projdrift_basis] * sign * self.gram_schmidt_step_size
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
                                                              self.engine.proj_indices_projdrift_basis] * sign * self.gram_schmidt_step_size
        coeffs[:, self.engine.drift_indices_projdrift_basis] = 0
        new_parameters = np.array(self.parameters[-1])[:, self.engine.proj_drift_indices] + coeffs

        # if self.parameter_bounds is not None:
        #     new_parameters, fidelity = self.bound_parameters(new_parameters, self.gram_schmidt_step_size)

        return new_parameters, fidelity, sign * self.gram_schmidt_step_size

    def get_update_linesearch(self, fid_fn, compute_U_fn):

        def fidelity_t(t, params, coeffs):
            return fid_fn(compute_U_fn(params + t * coeffs))

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
            f = partial(fidelity_t, params=sliced_params, coeffs=coeffs)
            max_step_size = self.max_step_size/piecewise_steps
            if self.line_search_method == "golden_section":
                dt, fidelity = golden_section_search(f, -max_step_size, 0., tol=1e-5)
            elif self.line_search_method == "difference_step":
                tol = 0.1 * (1-f(0))
                dt, fidelity = golden_section_search(f, -max_step_size, 0., tol=tol)
            new_parameters = sliced_params + dt * coeffs

            return new_parameters, fidelity, dt

        return update_linesearch

    def get_gammas_and_omegas(self, project_omegas_fn, jac_fn, compute_U_fn, geodesic_fn):

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

    def get_update_step(self):

        @jax.jit
        def update_step(free_params, params, piecewise_steps):

            gammaU_params, omegas_steps_phis = self.gammas_and_omegas(free_params)

            expander_gates = jnp.kron(jnp.eye(self.engine.gates),
                                      self.constraint_expander) if self.constraint_expander is not None else None
            
            sol = linear_comb_projected_coeffs_multigate(omegas_steps_phis, gammaU_params, expander_gates)

            # Expand the coefficients
            coeffs = jnp.zeros((self.engine.gates, self.engine.proj_drift_basis.lie_algebra_dim))
            coeffs = coeffs.at[:, self.engine.proj_indices_projdrift_basis].set(sol)
            coeffs = coeffs * (jnp.sqrt(len(coeffs)) / jnp.linalg.norm(coeffs))

            new_params, fidelity_new_phi, step_size = self.update_linesearch(params, coeffs, piecewise_steps)

            return coeffs, new_params, fidelity_new_phi, step_size

        return update_step

    def get_free_params_update_smoothing(self):

        @jax.jit
        def update_free_params_smoothing(proj_params, params):
            free_params = jnp.zeros((self.engine.gates, self.engine.proj_drift_basis.lie_algebra_dim),
                                    dtype=jnp.complex128)
            free_params = free_params.at[:, self.engine.proj_indices_projdrift_basis].set(proj_params)
            free_params = free_params.at[:, self.engine.drift_indices_projdrift_basis].set(
                params[:, self.engine.drift_indices])
            return free_params

        return update_free_params_smoothing

    def get_bound_parameters(self, fid_fn, compute_U_fn):

        @jax.jit
        def bound_parameters(params, offset):
            basis = self.engine.proj_drift_basis
            bounds = basis.generate_bounds(self.parameter_bounds, self.engine.gates)
            new_params = jnp.clip(params.astype(jnp.float64), 
                                  jnp.array(bounds[0], dtype=jnp.float64) + offset, 
                                  jnp.array(bounds[1], dtype=jnp.float64) - offset)
            fid = fid_fn(compute_U_fn(new_params))
            return new_params.astype(jnp.complex128), fid

        return bound_parameters

    def _null_space_optimisation(self, 
                                 null_space_function, 
                                 *,
                                 piecewise_steps_multiplier=1,
                                 rate=0.01,
                                 max_steps=100, 
                                 diff_tol=0.1, 
                                 label=None, 
                                 **kwargs):
        # Double the number of gates and initialise new parameters
        self.engine.gates = self.engine.gates * piecewise_steps_multiplier

        new_parameters = [list(np.copy(self.parameters[-1])) for _ in range(piecewise_steps_multiplier)]
        self.parameters.append(
            np.array([x for group in zip(*new_parameters) for x in group]) / piecewise_steps_multiplier)

        free_params = self.parameters[-1][:, self.engine.proj_drift_indices].astype(jnp.complex128)
        proj_params = self.parameters[-1][:, self.engine.projected_indices].astype(jnp.complex128)
        params_update = self.get_free_params_update_smoothing()

        c = 0
        diff = np.inf
        expander = jnp.kron(jnp.eye(self.engine.gates), jnp.array(self.constraint_expander)) if self.constraint_expander is not None else None
        fid=0
        while (diff > diff_tol) and (c < max_steps):
            _, omegas_steps_phis = self.gammas_and_omegas(free_params)
            vh, num = find_null_space(omegas_steps_phis, expander)

            assert num > 0, "Nullspace is empty!"
            null_space = vh[num:, :].T.conj()

            proj_params, diff = null_space_function(proj_params, null_space, expander, rate, **kwargs)
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


def linear_comb_projected_coeffs_multigate(combination_vectors, target_vector, expander):
    comb_vecs = jnp.concatenate(combination_vectors, axis=0)
    comb_vecs_T = comb_vecs.T @ expander if expander is not None else comb_vecs.T

    res = jnp.linalg.lstsq(comb_vecs_T, target_vector)
    # TODO: If the residual is too large, we want to throw NaN and handle the error.

    sol = expander @ res[0] if expander is not None else res[0]
    return sol.reshape(combination_vectors.shape[0], combination_vectors.shape[1])


def geodesic_hamiltonian(unitary, target_unitary):
    """
    Returns the geodesic to a target unitary.
    """
    g = -1.j * logm(jnp.einsum('ji,jk->ik', unitary.conj(), target_unitary), key=jax.random.key(1111))
    Id = jnp.eye(g.shape[0])
    global_phase = jnp.real(jnp.einsum('ij,ji->', Id, g)) / g.shape[0]
    return unitary @ (g - global_phase * Id)


def get_geodesic_hamiltonian_fn(target_unitary):
    return partial(geodesic_hamiltonian, target_unitary=target_unitary)


def hvp_forward_over_reverse(f, params, v):
    v = v.reshape(params.shape)
    return jax.jvp(jax.grad(f), (params,), (v,))[1]


@partial(jax.jit, static_argnames=("rcond"))
def find_null_space(omegas_steps_phis, expander, rcond=None):
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
def piecewise_smoothing(phi, null_space, expander, smoothing_rate=0.01):
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
def piecewise_bounding_mp(phi, null_space, expander, bounding_rate=0.01, lower_bounds=None, upper_bounds=None):
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
def piecewise_bounding_pg(phi, null_space, expander, bounding_rate=0.01, lower_bounds=None, upper_bounds=None):
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