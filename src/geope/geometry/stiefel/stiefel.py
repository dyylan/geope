r"""The complex Stiefel manifold $\mathrm{St}_m(\mathbb C^N)$ with the canonical metric.

Orthonormal $m$-frames in $\mathbb C^N$, i.e. the quotient
$\mathrm{SU}(N)/\mathrm{SU}(N-m)$. This is the manifold to synthesise on when only
an $m$-dimensional subspace of the target matters and the rest of the unitary is
**redundancy**: a spin-boson gate whose boson sector is arbitrary, a subspace
encoding, a state preparation. Quotienting that freedom out is cheaper than
optimising over it.

The three regimes, all one class:

* $m = N$ — no redundancy. Reduces to the group, but through an iterative
  logarithm; use `geope.geometry.lie.groups.SpecialUnitaryGroup` instead.
* $1 < m < N$ — the interesting case, and what this module exists for.
* $m = 1$ — state preparation. With ``projective=True`` this reproduces
  `geope.geometry.stiefel.sphere.StateSphere` exactly (pinned by a test), and
  that class is preferable: its logarithm is closed-form rather than iterative.

**The canonical metric, not the embedded one.** $\langle\Delta,\Upsilon\rangle_Q =
\mathrm{Tr}(\Delta^\dagger(\mathbb 1 - \tfrac12 QQ^\dagger)\Upsilon)$ weights the
directions that rotate *within* the frame at half the directions that leak out of
it, which is what makes it the metric of the quotient rather than of the
embedding. It has different geodesics from the Frobenius metric, so the two are
genuinely different Riemannian manifolds, and every primitive here — `log`,
`distance2`, `exp` — belongs to this one.

References:
    A. Edelman, T. Arias and S. Smith, *The geometry of algorithms with
    orthogonality constraints*, SIAM J. Matrix Anal. Appl. **20**, 303 (1998) —
    the canonical metric, the tangent space and the exponential.

    R. Zimmermann and K. Hüper, *Computing the Riemannian logarithm on the
    Stiefel manifold*, SIAM J. Matrix Anal. Appl. **43**, 953 (2022) — the
    iterative logarithm implemented in `_stiefel_log`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ...jax.hessian import stiefel_hessian_quadratic_form
from ...jax.logm import logm_unitary
from ..manifold import Manifold

# Membership tolerance on Q^dag Q = 1, matching the group's unitarity tolerance.
_FRAME_ATOL = 1e-8

# Below this overlap the global phase is undefined and alignment is skipped.
_TINY = 1e-12

# 1 - 1/sqrt(2): the coefficient in W_Q^{1/2} = 1 - (1 - 1/sqrt(2)) Q Q^dag.
_HALF_SQRT = 1.0 - 1.0 / np.sqrt(2.0)


def _sym(a: Array) -> Array:
    r"""$\tfrac12(A + A^\dagger)$, batched over leading axes."""
    return 0.5 * (a + jnp.conj(jnp.swapaxes(a, -1, -2)))


def _stiefel_log(
    q: Array, q_star: Array, tol: float, max_iter: int
) -> tuple[Array, Array]:
    r"""The Zimmermann–Hüper iterative Riemannian logarithm.

    Finds $\Delta$ with $\mathrm{Exp}_Q(\Delta) = Q_\star$ by building a
    $2m\times2m$ unitary whose logarithm's lower-right block measures the failure
    to be a Stiefel tangent, then rotating that block away:

    1. $M = Q^\dagger Q_\star$, and $Q_\perp N_0 = Q_\star - QM$ by QR.
    2. $V_0 \in \mathrm U(2m)$ with first block-column $(M; N_0)$ — already
       orthonormal, since $M^\dagger M + N_0^\dagger N_0 = Q_\star^\dagger Q_\star
       = \mathbb 1_m$ — completed by a full QR.
    3. Iterate $\log V_i = \bigl(\begin{smallmatrix} A_i & -B_i^\dagger\\ B_i &
       C_i\end{smallmatrix}\bigr)$ and $V_{i+1} = V_i\,\mathrm{diag}(\mathbb 1_m,
       e^{-C_i})$ until $\lVert C_i\rVert_F \le \kappa$.
    4. Return $\Delta = QA + Q_\perp B$.

    Each $V_i$ is unitary, so step 3 uses `geope.jax.logm_unitary` — a plain
    $2m\times2m$ Schur decomposition, exact for normal input — rather than the
    general `geope.jax.logm` and its scaling-and-squaring machinery.

    The QR in step 1 is degenerate when $Q_\star \to Q$, which is exactly where a
    converging optimisation spends its time. It needs no guard: $B \to 0$ there,
    which annihilates whatever arbitrary basis the QR returns for a vanishing
    matrix, and the tests pin finiteness down to $\lVert\Delta\rVert \sim
    10^{-14}$ and at $Q_\star = Q$ exactly.

    Args:
        q: The base frame, ``(N, m)``.
        q_star: The end frame, ``(N, m)``.
        tol: Stop once $\lVert C_i\rVert_F$ falls to this.
        max_iter: Hard iteration cap, so the loop always terminates.

    Returns:
        A tuple ``(delta, n_iter)`` — the tangent at ``q``, and the iterations
        spent (``max_iter`` means it did not converge).
    """
    m = q.shape[1]
    dtype = jnp.result_type(q, q_star, jnp.complex128)
    q, q_star = q.astype(dtype), q_star.astype(dtype)

    overlap = jnp.conj(q).T @ q_star
    q_perp, n_0 = jnp.linalg.qr(q_star - q @ overlap)

    # (M; N_0) has orthonormal columns already; a complete QR supplies a basis
    # for its orthogonal complement, which fills the remaining m columns.
    m_n = jnp.concatenate([overlap, n_0], axis=0)
    completion = jnp.linalg.qr(m_n, mode="complete")[0][:, m:]
    v_0 = jnp.concatenate([m_n, completion], axis=1)

    eye = jnp.eye(m, dtype=dtype)
    zero = jnp.zeros((m, m), dtype=dtype)

    def blocks(v: Array) -> tuple[Array, Array, Array]:
        log_v = logm_unitary(v)
        return log_v[:m, :m], log_v[m:, :m], log_v[m:, m:]

    def cond(state):
        _, _, _, c, i = state
        return jnp.logical_and(jnp.linalg.norm(c) > tol, i < max_iter)

    def body(state):
        v, _, _, c, i = state
        rotation = jnp.block([[eye, zero], [zero, jax.scipy.linalg.expm(-c)]])
        v_next = v @ rotation
        a_next, b_next, c_next = blocks(v_next)
        return (v_next, a_next, b_next, c_next, i + 1)

    a_0, b_0, c_0 = blocks(v_0)
    _, a, b, _, n_iter = jax.lax.while_loop(
        cond, body, (v_0, a_0, b_0, c_0, jnp.array(0, dtype=jnp.int32))
    )
    return q @ a + q_perp @ b, n_iter


@dataclass(frozen=True, eq=False)
class Stiefel(Manifold):
    r"""$\mathrm{St}_m(\mathbb C^N) = \{Q \in \mathbb C^{N\times m} : Q^\dagger Q = \mathbb 1_m\}$.

    A point is an orthonormal $m$-frame; the chart is the pulse's unitary acting
    on a fixed frame, $\Phi(\phi) = U(\phi)E$, so the product-of-exponentials
    machinery carries over unchanged and the Jacobian and HVP are the group's
    composed with one matrix product.

    **Not a Lie group.** $\mathrm{St}_m(\mathbb C^N) = \mathrm U(N)/\mathrm U(N-m)$
    is a homogeneous space: no identity, no left translation, no trivialisation
    identifying tangent spaces at different points. Everything here is genuinely
    point-dependent, the metric included.

    Attributes:
        dim: The ambient Hilbert-space dimension $N$.
        frame: The number of frame columns $m$; a point is ``(N, m)``.
        base_point: The frame $E$ the chart acts on, ``(N, m)``. Defaults to
            $(\mathbb 1_m, 0)^\intercal$, the note's embedding — the first $m$
            computational basis states. Inherited from `Manifold`, where it is
            $\Phi(0)$ for every space, and normalised to that default in
            ``__post_init__`` so it is always a concrete frame.
        projective: Whether a global phase is physical. ``True`` (the default)
            scores $\lvert\mathrm{Tr}\rvert/m$ and phase-aligns the target inside
            `log`; ``False`` scores $\mathrm{Re}\,\mathrm{Tr}/m$ and leaves it. See
            `log` for why the two must agree.
        log_tol: Convergence tolerance $\kappa$ of the iterative logarithm.
        log_max_iter: Iteration cap of the iterative logarithm.
    """

    dim: int
    frame: int
    # Keyword-only: re-annotating the base's ``projective`` ClassVar as a field
    # inherits its *position* in the base's annotations, which is ahead of
    # ``dim``; kw_only lifts it out of the positional ordering entirely.
    projective: bool = field(default=True, kw_only=True)
    log_tol: float = field(default=1e-13, kw_only=True)
    log_max_iter: int = field(default=100, kw_only=True)

    name: ClassVar[str] = "St(N,m)"

    def __post_init__(self) -> None:
        if not 1 <= self.frame <= self.dim:
            raise ValueError(
                f"{self.name} needs 1 <= frame <= dim, got frame={self.frame}, "
                f"dim={self.dim}."
            )
        # Normalise `base_point` to a concrete frame here rather than resolving
        # it on every read: `Manifold` defaults it to None — "the propagator is
        # the point" — which is meaningful on a group but not on a frame
        # manifold, where the chart's `(N, N)` propagator has to land on an
        # `(N, m)` frame. ``object.__setattr__`` is how a frozen dataclass
        # derives a field, and it runs once, host-side, at construction.
        if self.base_point is None:
            base = jnp.eye(self.dim, self.frame, dtype=jnp.complex128)
        else:
            base = jnp.asarray(self.base_point, dtype=jnp.complex128)
            if base.shape != self.ambient_shape:
                raise ValueError(
                    f"{self.name} with dim={self.dim}, frame={self.frame} expects "
                    f"a {self.ambient_shape} base_point, got {tuple(base.shape)}."
                )
            self.validate_point(base, "base_point")
        object.__setattr__(self, "base_point", base)

    # --- the interface ------------------------------------------------------

    @property
    def ambient_shape(self) -> tuple[int, ...]:
        return (self.dim, self.frame)

    @property
    def manifold_dim(self) -> int:
        r"""$2Nm - m^2$, less one more when the global phase is quotiented out.

        The same $-1$ that separates $\mathfrak{su}(d)$ from $\mathfrak u(d)$.
        """
        return 2 * self.dim * self.frame - self.frame**2 - int(self.projective)

    def validate_point(self, point: np.ndarray, what: str = "point") -> None:
        r"""Check membership: $Q^\dagger Q = \mathbb 1_m$.

        Host-side and numpy; see `geope.geometry.manifold.Manifold.validate_point`.
        """
        point = np.asarray(point)
        deviation = float(np.max(np.abs(np.conj(point).T @ point - np.eye(self.frame))))
        if deviation > _FRAME_ATOL:
            raise ValueError(
                f"{self.name} needs an orthonormal {what}: max|Q^dag Q - 1| is "
                f"{deviation:.3g}, above the {_FRAME_ATOL:g} tolerance."
            )

    def to_tangent(self, point: Array, ambient: Array) -> Array:
        r"""Project onto $T_Q = \{\Delta : Q^\dagger\Delta + \Delta^\dagger Q = 0\}$.

        $Z \mapsto Z - Q\,\mathrm{sym}(Q^\dagger Z)$: it removes only the
        *Hermitian* part of $Q^\dagger Z$, leaving the skew-Hermitian part, which
        is the frame rotation. Contrast
        `geope.geometry.stiefel.sphere.StateSphere`, which removes the whole
        complex component and so lands in $\mathbb{CP}^{n-1}$ rather than the
        sphere.
        """
        overlap = jnp.einsum("ji,...jk->...ik", jnp.conj(point), ambient)
        return ambient - jnp.einsum("ij,...jk->...ik", point, _sym(overlap))

    def _weight(self, point: Array, x: Array, coefficient: float) -> Array:
        r"""$(\mathbb 1 - c\,QQ^\dagger)x$, the shared shape of $W_Q$ and $W_Q^{1/2}$."""
        return x - coefficient * jnp.einsum(
            "ij,...jk->...ik", point, jnp.einsum("ji,...jk->...ik", jnp.conj(point), x)
        )

    def inner(self, point: Array, x: Array, y: Array) -> Array:
        r"""The canonical metric $\mathrm{Re}\,\mathrm{Tr}(x^\dagger W_Q y)$.

        With $W_Q = \mathbb 1 - \tfrac12 QQ^\dagger$: the in-frame rotations carry
        half the weight of the leakage directions, which is what distinguishes
        this from the embedded Frobenius metric and gives it different geodesics.
        """
        return jnp.real(
            jnp.sum(jnp.conj(x) * self._weight(point, y, 0.5), axis=(-2, -1))
        )

    def coefficients(self, point: Array, tangent: Array) -> Array:
        r"""Real coordinates in which the canonical metric *is* the dot product.

        $\Delta \mapsto [\mathrm{Re}\,W_Q^{1/2}\Delta,\ \mathrm{Im}\,W_Q^{1/2}\Delta]$
        flattened, with $W_Q^{1/2} = \mathbb 1 - (1 - \tfrac1{\sqrt2})QQ^\dagger$.
        Then $\sum_k c_k(u)c_k(v) = \mathrm{Re}\,\mathrm{Tr}(u^\dagger W_Q v) =
        \langle u,v\rangle_Q$ exactly, so the `Manifold.coefficients` constant is
        $c = 1$.

        This one hook is the whole canonical-metric story: because `gammas` and
        `omegas` both pass through it, the geodesic least-squares problem is
        automatically posed in the canonical norm, with no change to the
        optimiser. It is the $W_Q^{1/2}$ reweighting of the operands written once,
        in the only place it belongs.

        The image is a proper subspace — $T_Q$ has real dimension $2Nm - m^2$
        inside $2Nm$ coordinates — which the contract explicitly allows: the solve
        needs faithfulness, not surjectivity.
        """
        weighted = self._weight(point, tangent, _HALF_SQRT)
        flat = weighted.reshape(weighted.shape[:-2] + (-1,))
        return jnp.concatenate([jnp.real(flat), jnp.imag(flat)], axis=-1)

    def _align(self, x: Array, y: Array) -> Array:
        r"""Rotate ``y``'s global phase so that $\mathrm{Tr}(x^\dagger y) \ge 0$.

        The horizontal lift of the $\mathrm U(1)$ action, and a no-op when
        ``projective`` is ``False``.
        """
        if not self.projective:
            return y
        overlap = jnp.trace(jnp.conj(x).T @ y)
        magnitude = jnp.abs(overlap)
        # Both branches of a `where` are evaluated, so keep the division safe.
        safe = jnp.where(magnitude > _TINY, magnitude, 1.0)
        phase = jnp.where(magnitude > _TINY, overlap / safe, 1.0 + 0.0j)
        return y * jnp.conj(phase)

    def log(self, x: Array, y: Array, key: Array | None = None) -> Array:
        r"""$\mathrm{Log}_Q(Q_\star)$ by the Zimmermann–Hüper iteration.

        **Why the phase alignment matters.** The canonical logarithm is *not*
        phase-invariant — $\mathrm{Log}_Q(e^{i\theta}Q_\star) \ne
        \mathrm{Log}_Q(Q_\star)$, and its norm grows with $\theta$ — while the
        projective fidelity $\lvert\mathrm{Tr}\rvert/m$ is. Left alone, the
        distance objective the Armijo family minimises and the fidelity
        convergence is tested on would disagree: the optimiser would be forced to
        fix a physically meaningless global phase. With ``projective=True`` the
        target is first rotated onto its horizontal lift, which makes the two
        agree and reproduces `StateSphere` exactly at $m = 1$.

        **What round-trips, in projective mode.** ``exp(log(Q, Q_star))`` recovers
        $Q_\star$ *up to a global phase* — to machine precision, and at fidelity
        exactly 1 — but not the tangent that was exponentiated to reach it. The
        alignment picks the overlap-real representative of the $\mathrm U(1)$
        orbit, which coincides with the submersion's horizontal lift exactly at
        $m = 1$ and only to first order beyond it: $\mathrm{Tr}(Q^\dagger\gamma(t))$
        acquires a small imaginary part along the geodesic even when
        $\mathrm{Tr}(Q^\dagger\Delta) = 0$. The gauge is smooth, phase-invariant
        and consistent, which is all the algorithm needs; ``projective=False``
        round-trips the tangent itself exactly.

        Beyond the injectivity radius this returns the *minimal* geodesic, which
        need not be the one an `exp` was taken along — the usual cut-locus
        caveat, empirically biting around $\lVert\Delta\rVert \gtrsim 2$.

        Args:
            x: The base frame ``(N, m)``.
            y: The end frame ``(N, m)``.
            key: Unused; accepted for interface parity.
        """
        return _stiefel_log(x, self._align(x, y), self.log_tol, self.log_max_iter)[0]

    def exp(self, point: Array, tangent: Array) -> Array:
        r"""$\mathrm{Exp}_Q(\Delta) = (Q\,|\,Q_\perp)\exp(S)(\mathbb 1_m; 0)^\intercal$.

        With $A = Q^\dagger\Delta$, $Q_\perp R = (\mathbb 1 - QQ^\dagger)\Delta$ by
        QR, and $S = \bigl(\begin{smallmatrix} A & -R^\dagger\\ R &
        0\end{smallmatrix}\bigr) \in \mathfrak u(2m)$ — a $2m\times2m$ exponential
        rather than the $N\times N$ one the naive form would need.

        **Not a `Manifold` hook**: the pipeline never needs to move along a
        geodesic, only to read its initial tangent. This exists so that `log` can
        be tested by round trip, which is the only convincing way to validate an
        iterative solver.
        """
        m = self.frame
        dtype = jnp.result_type(point, tangent, jnp.complex128)
        point, tangent = point.astype(dtype), tangent.astype(dtype)

        a = jnp.conj(point).T @ tangent
        q_perp, r = jnp.linalg.qr(tangent - point @ a)
        s = jnp.block([[a, -jnp.conj(r).T], [r, jnp.zeros((m, m), dtype)]])
        embedding = jnp.concatenate(
            [jnp.eye(m, dtype=dtype), jnp.zeros((m, m), dtype)], axis=0
        )
        basis = jnp.concatenate([point, q_perp], axis=1)
        return basis @ jax.scipy.linalg.expm(s) @ embedding

    def tangent_acceleration(self, point: Array, v: Array, w: Array) -> Array:
        r"""$\dot\Omega(0) = W - V\,\mathrm{sym}(Q^\dagger V) - Q\,\mathrm{sym}(V^\dagger V + Q^\dagger W)$.

        The product-rule derivative of $\Omega(t) = \dot X - X\,\mathrm{sym}
        (X^\dagger\dot X)$ along the chart's curve — the projection's own bending,
        which on a group would instead be $V^\dagger V + U^\dagger W$.
        """
        qhv = jnp.conj(point).T @ v
        return (
            w - v @ _sym(qhv) - point @ _sym(jnp.conj(v).T @ v + jnp.conj(point).T @ w)
        )

    def hessian_quadratic_form(
        self, point: Array, a: Array, omega: Array
    ) -> tuple[Array, Array]:
        r"""The block-Jacobi form $\langle Z, E_{12}^{-1}E_{11}Z\rangle$ — phase-blind only.

        There is no eigenangle formula: a general Stiefel manifold is normal
        homogeneous but not *symmetric*, so neither the group's
        $\frac{\mathrm{ad}}{2}\coth\frac{\mathrm{ad}}{2}$ nor the sphere's
        $\theta\cot\theta$ applies. What does exist is exact — the Jacobi
        equation has constant coefficients in a homogeneous moving frame, so
        $\mathcal K^{\mathrm{St}}_S$ is read off the blocks of one operator
        exponential. See `geope.jax.stiefel_hessian_quadratic_form`.

        **Note the negation.** The context hands over ``a = -Log_Q(Q_star)``,
        pointing *away* from the target, and unlike $\mathfrak{su}(d)$ this form
        is genuinely **not even** in its argument — $-\Delta$ addresses the
        geodesic reflection of the target, a different Hessian. Only at $m = N$
        does evenness return, because there the manifold is the group.

        **Unavailable when ``projective``.** Phase alignment inside `log` makes
        the objective the squared distance on the $\mathrm U(1)$ *quotient*
        $\mathrm{St}_m(\mathbb C^N)/\mathrm U(1)$, whose Hessian carries an
        O'Neill term this construction does not have; measured against a finite
        difference of `distance2`, the total-space form is off by up to a few
        percent there. Rather than let `ApproximateQuadraticArmijo` quietly stop
        being exact, this fails — the same way a missing `TangentBundle.hvp` does
        under ``param_transform``. The other four line searches run in both
        modes, and `geope.line_searches.QuadraticArmijo` builds its curvature
        from ``ctx.q``, which uses the radial surrogate $\lVert\Omega\rVert^2$
        and never calls this.
        """
        if self.projective:
            raise NotImplementedError(
                f"{self.name} with projective=True measures distance on the U(1) "
                "quotient, whose Riemannian Hessian this does not implement, so "
                "`ctx.q_exact` and `ctx.rho` are unavailable here. Pass "
                "projective=False for the exact form, or use GoldenSection, Adam, "
                "Armijo or QuadraticArmijo — all four run in either mode; only "
                "ApproximateQuadraticArmijo reads `q_exact`."
            )
        return stiefel_hessian_quadratic_form(point, -a, omega)

    def fidelity(self, x: Array, y: Array) -> Array:
        r"""$\lvert\mathrm{Tr}(y^\dagger x)\rvert/m$, or $\mathrm{Re}\,\mathrm{Tr}/m$.

        The subspace fidelity: it scores only how the frames overlap, and is blind
        to whatever the pulse does outside the $m$ columns — which is the
        redundancy this manifold exists to exploit.
        """
        overlap = jnp.trace(jnp.conj(y).T @ x)
        scored = jnp.abs(overlap) if self.projective else jnp.real(overlap)
        return scored / self.frame

    def infidelity(self, x: Array, y: Array) -> Array:
        r"""$1 - F(x, y)$."""
        return 1.0 - self.fidelity(x, y)
