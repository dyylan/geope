r"""Matrix Lie groups: the manifolds GEOPE was built on.

`MatrixLieGroup` is the middle layer between the general
`geope.geometry.manifold.Manifold` interface and the two concrete groups. It is
where **left-trivialisation** lives, and it is worth being explicit about how much
that one structure buys, because a manifold without it (see
`geope.geometry.stiefel`) has to supply all of it by hand:

* tangent spaces at every point are identified with the Lie algebra
  $\mathfrak g = T_1 G$, so `to_tangent` is just $U^\dagger\dot X$ and one global
  Hermitian `Basis` coordinatises every fibre — hence `inner` and `coefficients`
  ignore the base point entirely;
* the metric is bi-invariant, so the geodesic logarithm is a matrix logarithm;
* the chart is a product of exponentials in the generators, which gives the
  manual propagator derivatives (Jacobian, HVP, Hessian) their structure.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, ClassVar

import jax.numpy as jnp
import numpy as np
from jax import Array

from ...jax.dexpm import (
    get_Ui_fn,
    get_d2expm,
    get_d2expm_eig,
    get_dexpm,
    get_dexpm_eig,
)
from ...jax.hessian import (
    get_hvp_propagator,
    hessian_propagator,
    su_hessian_quadratic_form,
)
from ...jax.jacobian import jacobian_propagator
from ...jax.logm import logm_unitary
from ..chart import get_compute_matrices_params_list_fn
from ..manifold import Manifold

# Loose enough for a target assembled in float32 or from a few matrix products,
# tight enough to catch a genuinely non-unitary matrix.
_UNITARY_ATOL = 1e-8


# ---------------------------------------------------------------------------
# The two fidelity formulas
# ---------------------------------------------------------------------------
# Free functions rather than methods because they are the library's oldest
# public API (``geope.fidelity`` and friends) and take no manifold: which one a
# problem uses is decided by *which group* it is bound to, below.


def fidelity(unitary: Array, target_unitary: Array) -> Array:
    r"""Projective fidelity $\lvert\mathrm{Tr}(U_T^\dagger U)\rvert / d$.

    The normalised absolute value of the Hilbert-Schmidt inner product, so a
    global phase on either argument is invisible.

    Args:
        unitary: The unitary ``Array`` to evaluate.
        target_unitary: The target unitary ``Array``.

    Returns:
        A scalar fidelity ``Array`` in the range $[0, 1]$.
    """
    return jnp.abs(jnp.einsum("ji,ji->", target_unitary.conj(), unitary)) / len(
        target_unitary[0]
    )


def infidelity(unitary: Array, target_unitary: Array) -> Array:
    r"""Projective infidelity $1 - F_{\mathrm{proj}}(U, U_T)$.

    Args:
        unitary: The unitary ``Array`` to evaluate.
        target_unitary: The target unitary ``Array``.

    Returns:
        A scalar infidelity ``Array`` in $[0, 1]$.
    """
    return 1 - jnp.abs(jnp.einsum("ji,ji->", target_unitary.conj(), unitary)) / len(
        target_unitary[0]
    )


def fidelity_full(unitary: Array, target_unitary: Array) -> Array:
    r"""Phase-sensitive (non-projective) fidelity.

    $F_{\mathrm{full}}(U, U_T) = \mathrm{Re}\,\mathrm{Tr}(U_T^\dagger U) / d$.
    Unlike the projective fidelity, this is sensitive to a global phase
    on $U$ and lies in $[-1, 1]$.

    Args:
        unitary: The unitary ``Array`` to evaluate.
        target_unitary: The target unitary ``Array``.

    Returns:
        A scalar fidelity ``Array`` in $[-1, 1]$.
    """
    return jnp.real(jnp.einsum("ji,ji->", target_unitary.conj(), unitary)) / len(
        target_unitary[0]
    )


def infidelity_full(unitary: Array, target_unitary: Array) -> Array:
    r"""Phase-sensitive infidelity $1 - F_{\mathrm{full}}(U, U_T)$.

    Args:
        unitary: The unitary ``Array`` to evaluate.
        target_unitary: The target unitary ``Array``.

    Returns:
        A scalar infidelity ``Array`` in $[0, 2]$.
    """
    return 1 - jnp.real(jnp.einsum("ji,ji->", target_unitary.conj(), unitary)) / len(
        target_unitary[0]
    )


# Private aliases: the hooks below share their names with these functions, and a
# bare ``fidelity(...)`` inside ``def fidelity(self, ...)`` reads like recursion.
_fidelity_projective = fidelity
_infidelity_projective = infidelity
_fidelity_full = fidelity_full
_infidelity_full = infidelity_full


@dataclass(frozen=True, eq=False)
class MatrixLieGroup(Manifold):
    r"""A compact matrix Lie group with a bi-invariant metric.

    Elements of the Lie algebra are **skew-Hermitian** throughout — the generator
    of $U = e^{iH}$ is $iH$, not $H$. The single factor of $i$ that turns one into
    a Hermitian matrix the basis can resolve lives in `coefficients`.

    Attributes:
        dim: The Hilbert-space dimension $d$; a point is a $d\times d$ unitary.
    """

    dim: int

    @property
    def ambient_shape(self) -> tuple[int, ...]:
        return (self.dim, self.dim)

    def validate_point(self, point: np.ndarray, what: str = "point") -> None:
        r"""Check membership of the group: $UU^\dagger = \mathbb 1$.

        Every geometric primitive here assumes it — `log` takes the logarithm
        of $U^\dagger V$ expecting a unitary (so `logm_unitary`'s Schur form is
        diagonal), and the fidelity normalisation assumes $\lvert\mathrm{Tr}
        \rvert \le d$. A non-unitary target produces plausible-looking numbers
        rather than an error, so it is worth catching at configuration time.

        Host-side and numpy; see `geope.geometry.manifold.Manifold.validate_point`.
        """
        point = np.asarray(point)
        deviation = float(np.max(np.abs(point @ np.conj(point).T - np.eye(self.dim))))
        if deviation > _UNITARY_ATOL:
            raise ValueError(
                f"{self.name} needs a unitary {what}: max|U U^dag - 1| is "
                f"{deviation:.3g}, above the {_UNITARY_ATOL:g} tolerance."
            )

    # --- what still distinguishes SU from U ---------------------------------

    @abstractmethod
    def project(self, x: Array) -> Array:
        r"""Project $\mathfrak u(d)$ onto this group's Lie algebra."""

    # --- the trivialised geometry -------------------------------------------

    def to_tangent(self, point: Array, ambient: Array) -> Array:
        r"""Left-trivialise: $\dot X \mapsto U^\dagger\dot X$, skew-Hermitian.

        Doing this *before* projecting is what makes `coefficients` lossless. The
        raw Jacobian columns $\partial_{g,k}U$ are *ambient*, of the form
        $U\cdot(\text{skew-Hermitian})$, whose skew-Hermitian part rotates with
        $U$; projecting them directly would discard most of each matrix and make
        the geodesic least squares minimise in a $U$-dependent seminorm rather
        than in $\langle\cdot,\cdot\rangle_F$. Left translation is itself an
        isometry — it is the *projection after it* that would be lossy, which is
        why the two cannot be commuted.
        """
        return jnp.einsum("ji,...jk->...ik", jnp.conj(point), ambient)

    def inner(self, point: Array, x: Array, y: Array) -> Array:
        r"""$\mathrm{Re}\,\mathrm{Tr}(x^\dagger y)$ — bi-invariant, so ``point`` is unused."""
        return self._euclidean_inner(x, y)

    def coefficients(self, point: Array, tangent: Array) -> Array:
        r"""Resolve skew-Hermitian algebra elements against the tangent bundle's basis.

        Computes $c_k = \mathrm{Tr}(B_k\,iX)/d$, i.e. the real coefficients with
        $iX = \sum_k c_k B_k$ (equivalently $X = -i\sum_k c_k B_k$), so the metric
        constant of the `Manifold.coefficients` contract is $c = 1/d$.
        ``point`` is unused: one basis serves every fibre.

        **What this assumes of the basis.** Hermitian elements, orthogonal under
        the trace inner product and normalised to
        $\mathrm{Tr}(B_j B_k) = d\,\delta_{jk}$ — which every basis
        `geope.utils` constructs satisfies. Completeness is explicitly *not*
        assumed: a basis with fewer than $\dim\mathfrak g$ elements simply
        projects onto a subspace, which the spin-boson bases rely on.

        Args:
            point: Unused.
            tangent: Skew-Hermitian ``Array`` of shape ``(..., d, d)``.

        Returns:
            A real ``Array`` of shape ``(..., K)``.
        """
        self._require_bound("coefficients")
        tangent = 1.0j * jnp.asarray(tangent)
        flat = tangent.reshape((-1,) + tangent.shape[-2:])
        coeffs = self.tangent.project(flat)
        return coeffs.reshape(tangent.shape[:-2] + (coeffs.shape[-1],))

    def log(self, x: Array, y: Array, key: Array | None = None) -> Array:
        r"""$\mathrm{Log}_x(y) = \mathrm{proj}\bigl(\log_{\min}(x^\dagger y)\bigr)$.

        The algebra element $A$ with $x\,e^{A} = y$: skew-Hermitian, and traceless
        on SU. **The pipeline's only matrix logarithm** — the geodesic search
        direction, the geodesic-distance objective and the line-search slope and
        curvature are all algebra on the single $A$ a step takes from here.

        Taking a second logarithm in the conjugate order is not a harmless
        convention: $\log(x^\dagger y) = -\log(y^\dagger x)$ fails on the
        principal branch cut, so a target with a $-1$ eigenvalue (Hadamard,
        CNOT, Toffoli) would have the direction and the objective disagree by
        $2\pi$ on that eigenphase.

        Args:
            x: The base point ``(d, d)``, unitary.
            y: The end point ``(d, d)``, unitary.
            key: Unused by `geope.jax.logm_unitary`, which needs no randomness;
                accepted so the general `geope.jax.logm` (whose randomised
                1-norm estimator is seeded by it) can be swapped in.
        """
        # x^dagger y is a product of unitaries, so the unitary-specialised log
        # applies: exact, and accurate on the branch cut where Hermitian targets
        # put their -1 eigenvalue.
        m = jnp.einsum("ji,jk->ik", jnp.conj(x), y)
        return self.project(logm_unitary(m, key))

    def tangent_acceleration(self, point: Array, v: Array, w: Array) -> Array:
        r"""$\dot\Omega(0) = V^\dagger V + U^\dagger W$.

        The derivative of $\Omega(t) = U(t)^\dagger\dot U(t)$ by the product rule
        — the *chart's* bending, as opposed to the group's own curvature.
        """
        return v.conj().T @ v + point.conj().T @ w

    def hessian_quadratic_form(
        self, point: Array, a: Array, omega: Array
    ) -> tuple[Array, Array]:
        r"""$\bigl(\langle\Omega,\mathcal K_A\Omega\rangle_F,\ \rho\bigr)$ on $\mathfrak{su}(d)$.

        $\mathcal K_A = \frac{\operatorname{ad}_A}{2}\coth(\frac{\operatorname{ad}_A}{2})$,
        evaluated without forming the operator, plus the eigenphase spread $\rho$
        from the same eigendecomposition. Bi-invariance makes it independent of
        ``point``, and the kernel is even in $A$, so either sign convention for
        ``a`` gives the same value. See
        :func:`geope.jax.su_hessian_quadratic_form`.
        """
        return su_hessian_quadratic_form(a, omega)

    def chart(self, generators) -> Callable[[Array], Array]:
        r"""The product of piecewise-constant exponentials in ``generators``.

        $\Phi(\phi) = \prod_g \exp\bigl(i\sum_k \phi_{g,k} B_k\bigr)$ — the pulse
        model, and the structure the manual propagator derivatives exploit.
        """
        return get_compute_matrices_params_list_fn(generators.basis)

    def chart_hvp(
        self, generators
    ) -> Callable[[Array, Array], tuple[Array, Array, Array]]:
        r"""The propagator HVP: $O(G)$, and no dense Hessian ever formed.

        See :func:`geope.jax.hvp_propagator`.
        """
        return get_hvp_propagator(jnp.asarray(generators.basis))

    @cached_property
    def hessian(self) -> Callable[[Array], Array]:
        """The infidelity Hessian, analytically when the chart allows it.

        The manual Goodwin–Kuprov propagator Hessian when the chart is a plain
        product of exponentials (``tangent.generators`` is set), falling back to
        `Manifold.hessian_autodiff` when it is not — i.e. under
        ``param_transform``, where the manual propagator derivatives do not apply.
        """
        self._require_bound("hessian")
        if self.tangent.generators is None:
            return self.hessian_autodiff
        return get_hessian_propagator_fn(
            self.tangent.generators.basis, self.target, projective=self.projective
        )


def get_hessian_propagator_fn(
    basis: np.ndarray,
    target: Array,
    projective: bool = True,
    method: str = "eig",
    hermitian: bool = True,
) -> Callable[[Array], Array]:
    r"""Build the infidelity Hessian manually (Goodwin–Kuprov NR-GRAPE).

    Analytic drop-in for `geope.jax.hessian.get_hessian_fn`: returns
    ``hess(y) -> (P, P)`` with ``P = y.size``, the Hessian of the same infidelity
    that `get_hessian_fn` differentiates by autodiff, but built from the manual
    propagator derivatives (`geope.jax.jacobian.jacobian_propagator`,
    `geope.jax.hessian.hessian_propagator`) rather than from forward-over-reverse
    HVPs. It is a *group* construction: it needs the generator basis the chart is
    a product of exponentials in, and it hard-codes the two unitary fidelities.

    Let $z = \mathrm{Tr}(U_T^\dagger U)$, $\partial_a z$, $\partial_a\partial_b z$
    be obtained by contracting $U$, $\partial U$, $\partial^2 U$ against
    $U_T^\dagger$. For the phase-sensitive cost $C = 1 - \mathrm{Re}(z)/d$ the
    Hessian is the linear contraction $-\mathrm{Re}(\partial_a\partial_b z)/d$;
    for the projective cost $C = 1 - |z|/d$,

    $$\partial_a\partial_b|z| =
        \frac{\mathrm{Re}(\overline{\partial_a z}\,\partial_b z)
              + \mathrm{Re}(\bar z\,\partial_a\partial_b z)}{|z|}
        - \frac{\mathrm{Re}(\bar z\,\partial_a z)\,
                \mathrm{Re}(\bar z\,\partial_b z)}{|z|^3}.$$

    Like the projective fidelity itself, this is singular as $|z| \to 0$ (the
    near-identity / traceless-target gotcha) — the autodiff Hessian shares that.

    Memory note: this materialises the dense propagator Hessian
    (`geope.jax.hessian.hessian_propagator`, $O(G^2 d^2 K^2)$); intended for the
    small systems where NR-GRAPE is used.

    Args:
        basis: Proj+drift basis ``(K, d, d)`` — the same basis the bound
            ``compute_point`` uses.
        target: Target unitary ``(d, d)``.
        projective: Match the projective (``True``) or phase-sensitive
            (``False``) infidelity.
        method: ``"eig"`` (spectral, default) or ``"block"`` (auxiliary-matrix)
            per-step derivatives.
        hermitian: For ``method="eig"``, assume real parameters (skew-Hermitian
            generators) and use the faster ``eigh``-based spectral derivatives.
            Set ``False`` for complex-valued parameters.

    Returns:
        A ``Callable[[Array], Array]`` ``hess(y)`` returning the ``(P, P)``
        infidelity Hessian. Left un-jitted so it fuses into the enclosing
        ``@jax.jit`` update step.
    """
    Ui_fn = get_Ui_fn(basis)
    if method == "eig":
        jac_step = get_dexpm_eig(basis, hermitian=hermitian)
        hess_step = get_d2expm_eig(basis, hermitian=hermitian)
    elif method == "block":
        jac_step = get_dexpm(basis)
        hess_step = get_d2expm(basis)
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'eig' or 'block'.")

    compute_point = get_compute_matrices_params_list_fn(basis)
    t_conj = jnp.asarray(target).conj()
    d = jnp.asarray(target).shape[0]

    def hess(y: Array) -> Array:
        U = compute_point(y)
        dU = jacobian_propagator(y, Ui_fn, jac_step)  # (G, d, d, K)
        H = hessian_propagator(y, Ui_fn, jac_step, hess_step)  # (G, G, d, d, K, K)

        # Contract the propagator and its derivatives with U_T^dagger.
        z = jnp.einsum("ab,ab->", t_conj, U)
        dz = jnp.einsum("ab,iabk->ik", t_conj, dU)  # (G, K)
        d2z = jnp.einsum("ab,ijabkl->ijkl", t_conj, H)  # (G, G, K, K)

        n_g, n_k = y.shape
        P = n_g * n_k
        dz_f = dz.reshape(P)
        d2z_f = jnp.transpose(d2z, (0, 2, 1, 3)).reshape(P, P)

        if not projective:
            return -jnp.real(d2z_f) / d

        r = jnp.abs(z)
        z_bar = jnp.conj(z)
        re_zdz = jnp.real(z_bar * dz_f)  # (P,)
        term1 = (
            jnp.real(jnp.outer(jnp.conj(dz_f), dz_f)) + jnp.real(z_bar * d2z_f)
        ) / r
        term2 = jnp.outer(re_zdz, re_zdz) / r**3
        return -(term1 - term2) / d

    return hess


@dataclass(frozen=True, eq=False)
class UnitaryGroup(MatrixLieGroup):
    r"""$\mathrm U(d)$ with the phase-**sensitive** fidelity.

    The Lie algebra is all of $\mathfrak u(d)$ (skew-Hermitian, trace included),
    so the global phase is a controllable direction and the fidelity
    $F_{\mathrm{full}} = \mathrm{Re}\,\mathrm{Tr}(x^\dagger y)/d \in [-1, 1]$
    scores it. This is ``projective=False``; see ``docs/user_guide.md`` for the
    gotchas with traceless targets near the identity.
    """

    projective: ClassVar[bool] = False
    name: ClassVar[str] = "U(d)"

    @property
    def manifold_dim(self) -> int:
        r"""$\dim\mathfrak u(d) = d^2$."""
        return self.dim**2

    def project(self, x: Array) -> Array:
        r"""Identity: every skew-Hermitian matrix is already in $\mathfrak u(d)$."""
        return x

    def fidelity(self, x: Array, y: Array) -> Array:
        r"""$\mathrm{Re}\,\mathrm{Tr}(x^\dagger y)/d \in [-1, 1]$."""
        return _fidelity_full(y, x)

    def infidelity(self, x: Array, y: Array) -> Array:
        r"""$1 - \mathrm{Re}\,\mathrm{Tr}(x^\dagger y)/d \in [0, 2]$."""
        return _infidelity_full(y, x)


@dataclass(frozen=True, eq=False)
class SpecialUnitaryGroup(MatrixLieGroup):
    r"""$\mathrm{SU}(d)$ with the phase-**invariant** fidelity (the default).

    The Lie algebra is $\mathfrak{su}(d)$ (skew-Hermitian *and* traceless), so the
    global phase is quotiented out: `project` removes it from every tangent vector
    and the fidelity $F_{\mathrm{proj}} = |\mathrm{Tr}(x^\dagger y)|/d \in [0, 1]$
    ignores it. This is ``projective=True``.

    Deliberately a *sibling* of `UnitaryGroup` rather than a subclass: it would
    override every member anyway, and $\mathrm{SU}(d)\subset\mathrm U(d)$ as
    groups would make ``isinstance(m, UnitaryGroup)`` a misleading test for
    "phase-sensitive". Test ``m.projective``.
    """

    projective: ClassVar[bool] = True
    name: ClassVar[str] = "SU(d)"

    @property
    def manifold_dim(self) -> int:
        r"""$\dim\mathfrak{su}(d) = d^2 - 1$."""
        return self.dim**2 - 1

    def project(self, x: Array) -> Array:
        r"""Remove the global-phase generator: $x - \tfrac{\mathrm{Tr}x}{d}\mathbb 1$.

        Only the imaginary part of the trace is subtracted, so a skew-Hermitian
        argument stays exactly skew-Hermitian however much float error its trace
        has picked up.
        """
        phase = 1.0j * jnp.imag(jnp.trace(x)) / self.dim
        return x - phase * jnp.eye(self.dim, dtype=x.dtype)

    def fidelity(self, x: Array, y: Array) -> Array:
        r"""$|\mathrm{Tr}(x^\dagger y)|/d \in [0, 1]$."""
        return _fidelity_projective(y, x)

    def infidelity(self, x: Array, y: Array) -> Array:
        r"""$1 - |\mathrm{Tr}(x^\dagger y)|/d \in [0, 1]$."""
        return _infidelity_projective(y, x)
