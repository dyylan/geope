r"""Why ``gammas_and_omegas`` left-trivialises before projecting.

Run it::

    python examples/left_trivialisation.py

The bug
-------
``geope.lie.pauli_projector.project_omegas`` computes
:math:`\mathrm{Re}\,\mathrm{Tr}(P_k M)/d` against a **Hermitian** basis. Split
:math:`M = H + iK` into Hermitian parts; both :math:`\mathrm{Tr}(P_kH)` and
:math:`\mathrm{Tr}(P_kK)` are *real*, so the ``Re`` keeps the first and throws
the second away entirely. **The projection sees only the Hermitian part of its
argument.**

Both quantities the geodesic step feeds it are *ambient*, of the form
:math:`U\cdot(\text{traceless Hermitian})`:

* ``geodesic_hamiltonian`` returns :math:`U g` with
  :math:`g = -i\log(U^\dagger U_T)` traceless Hermitian;
* the Jacobian columns are :math:`\partial_{g,k}U`, and it is
  :math:`iU^\dagger\partial_{g,k}U` that is Hermitian (differentiate
  :math:`U^\dagger U = \mathbb 1`).

:math:`UH` is not Hermitian, so projecting it directly discards most of it --
and *how much* it discards depends on :math:`U`. The downstream least squares
(``linear_comb_projected_coeffs_multigate``) therefore minimised in a
:math:`U`-dependent seminorm rather than in :math:`\langle\cdot,\cdot\rangle_F`,
so the search direction it returned was not the Frobenius-orthogonal projection
of the geodesic tangent that the geodesic step is *defined* to be.

The fix
-------
Multiply by :math:`U^\dagger` first, in ``get_gammas_and_omegas_fn``
(and in the separately testable halves ``get_gammas_fn`` / ``get_omegas_fn``)::

    -  gammaU = geo_fn(unitary, key=key)
    -  omegas = project_omegas_fn(1.0j * omegaUs)
    +  u_dag  = unitary.conj().T
    +  gammaU = u_dag @ geo_fn(unitary, key=key)
    +  omegas = project_omegas_fn(1.0j * (u_dag @ omegaUs))

That recovers :math:`g` and :math:`iU^\dagger\partial_{g,k}U`, which *are*
traceless Hermitian, so the projection is an isometry onto their coefficients
and the solve is exactly the Frobenius-orthogonal projection again.

Why the two operations cannot be commuted
-----------------------------------------
Left translation :math:`X\mapsto U^\dagger X` **is** a Frobenius isometry --
that is not where the loss is, and it is why the fix looks like a no-op.
:math:`\Pi` is an isometry **only on traceless Hermitian matrices**.
:math:`L_{U^\dagger}` is what lands you in that subspace, so
:math:`\Pi\circ L_{U^\dagger}` is an isometry while :math:`\Pi` alone, applied
to :math:`UH`, is not. The order matters because the two maps have different
domains of validity, not because either one is lossy on its own.

Why it matters
--------------
The lemma :math:`\Psi = P\Omega` -- the achieved velocity is the orthogonal
projection of the geodesic tangent -- is what makes the geodesic step *be* the
geodesic step. It is also the hypothesis the second-order line search leans on:
``QuadraticArmijo`` substitutes :math:`\|\Omega\|_F^2` for the intrinsic
curvature term :math:`\langle\Omega,\mathcal K_A\Omega\rangle_F`, which is valid
exactly when :math:`\Omega \parallel A`. Break the lemma and the curvature it
seeds its step from is describing a different direction than the one taken.
"""

from __future__ import annotations

import functools
import itertools as it
import sys

import numpy as np

CHECKS: dict[str, bool] = {}


def check(name: str, ok: bool) -> None:
    """Record a named PASS/FAIL so the script can double as a smoke test."""
    CHECKS[name] = bool(ok)


def banner(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# ---------------------------------------------------------------------------
# A standalone copy of the projection, so stages 1-2 need no geope import.
# This is `project_omegas` verbatim: Re Tr(P_k M) / d.
# ---------------------------------------------------------------------------


def pauli_basis(n: int) -> np.ndarray:
    """The (4**n - 1) traceless n-qubit Pauli strings, identity dropped."""
    single = [
        np.eye(2, dtype=complex),
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.diag([1.0, -1.0]).astype(complex),
    ]
    return np.array(
        [
            functools.reduce(np.kron, [single[i] for i in comb])
            for comb in it.product(range(4), repeat=n)
        ][1:]
    )


def projector(basis: np.ndarray):
    """`(Pi, rebuild)` for a Pauli basis: coefficients out, matrix back in."""
    d = basis.shape[-1]
    pi = lambda M: np.real(np.einsum("ijk,kj->i", basis, M)) / d
    rebuild = lambda c: np.einsum("i,iab->ab", c.astype(complex), basis)
    return pi, rebuild


def herm(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.conj().T)


def random_su(d: int, rng: np.random.Generator) -> np.ndarray:
    """A Haar-ish SU(d) element, via the exponential of a traceless Hermitian."""
    A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    A = herm(A)
    A -= np.trace(A) / d * np.eye(d)
    w, V = np.linalg.eigh(A)
    return V @ np.diag(np.exp(1j * w)) @ V.conj().T


# ---------------------------------------------------------------------------
# 1. The projection only sees the Hermitian part
# ---------------------------------------------------------------------------


def stage_1() -> None:
    banner("1. The projection keeps only the Hermitian part of its argument")
    rng = np.random.default_rng(0)
    B = pauli_basis(1)  # {X, Y, Z}, d = 2
    pi, rebuild = projector(B)

    H = rebuild(rng.normal(size=len(B)))  # traceless Hermitian, by construction
    U = random_su(2, rng)

    lossless = np.allclose(rebuild(pi(H)), H)
    print(f"  rebuild(Pi(H))       == H         {lossless}   <- Pi is an isometry here")
    print(
        f"  rebuild(Pi(U @ H))   == U @ H     {np.allclose(rebuild(pi(U @ H)), U @ H)}"
    )

    # Name exactly what survives: the projection of UH is the projection of its
    # Hermitian part, and nothing else.
    same_as_herm = np.allclose(pi(U @ H), pi(herm(U @ H)))
    print(f"  Pi(U @ H)            == Pi(herm(U @ H))     {same_as_herm}")

    # How much is lost depends on U, and vanishes at U = 1. That is the whole
    # reason the bug is invisible at the start of a run.
    print()
    print("  U = exp(i*theta*Z):   theta      ||U H||_F   kept    discarded")
    Z = B[2]
    for theta in (0.0, np.pi / 8, np.pi / 4, np.pi / 2):
        Ut = np.diag(np.exp(1j * theta * np.diag(Z).real))
        kept = np.linalg.norm(rebuild(pi(Ut @ H)))
        full = np.linalg.norm(Ut @ H)
        print(
            f"                      {theta:8.4f}   {full:9.4f}   {kept:6.4f}"
            f"   {1 - kept / full:8.1%}"
        )
    print("  -> zero loss at theta = 0, and it grows with the rotation.")

    check("Pi is an isometry on traceless Hermitian", lossless)
    check("Pi(U H) discards the anti-Hermitian part", same_as_herm)


# ---------------------------------------------------------------------------
# 2. So Pi o L_U is a U-dependent distortion -- with a kernel
# ---------------------------------------------------------------------------


def induced_map(U: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """The map the ambient projection induces on coefficient vectors.

    Column j is ``Pi(U @ P_j)``, so ``M_U @ c`` are the coefficients the
    projection returns for the matrix ``U @ rebuild(c)``. The least squares
    downstream sees only these, so it is really minimising ``||M_U (Jx - A)||``.
    """
    pi, _ = projector(basis)
    return np.stack([pi(U @ P) for P in basis], axis=1)


def stage_2() -> None:
    banner("2. So the ambient projection is a U-dependent distortion, with a kernel")
    rng = np.random.default_rng(1)

    B1 = pauli_basis(1)
    for label, U in (
        ("U = identity        ", np.eye(2, dtype=complex)),
        ("U = random SU(2)    ", random_su(2, rng)),
    ):
        M = induced_map(U, B1)
        dev = np.linalg.norm(M.T @ M - np.eye(len(B1)))
        print(f"  {label}  ||M_U^T M_U - I|| = {dev:8.4f}")
    identity_is_clean = (
        np.linalg.norm(
            induced_map(np.eye(2, dtype=complex), B1).T
            @ induced_map(np.eye(2, dtype=complex), B1)
            - np.eye(len(B1))
        )
        < 1e-12
    )
    print("  -> lengths AND angles are not preserved, so the solve is minimising")
    print("     in a U-dependent seminorm rather than in the Frobenius inner product.")

    # The sharpest statement: the map is not merely distorted, it has a kernel.
    # For d = 4 the global phase i*1 has det = 1, so it is a legitimate SU(4)
    # element -- exactly the group the two-qubit optimisations move in.
    print()
    B2 = pauli_basis(2)
    pi2, _ = projector(B2)
    U = 1j * np.eye(4, dtype=complex)
    H2 = np.einsum("i,iab->ab", rng.normal(size=len(B2)).astype(complex), B2)
    annihilated = np.linalg.norm(pi2(U @ H2)) < 1e-12
    print(
        f"  at U = i*1 in SU(4)   det(U) = {np.linalg.det(U).real:.0f}  (so U is in SU(4))"
    )
    print(f"    ||Pi(H)||     = {np.linalg.norm(pi2(H2)):9.4f}")
    print(f"    ||Pi(U @ H)|| = {np.linalg.norm(pi2(U @ H2)):9.3e}   <- annihilated")
    print("  -> gammas and omegas would both be exactly zero and the solve")
    print("     would degenerate to 0 = 0. This is a kernel, not a small error.")

    check("Pi o L_U is the identity map at U = 1", identity_is_clean)
    check("Pi o L_U annihilates everything at U = i*1 in SU(4)", annihilated)


# ---------------------------------------------------------------------------
# 3. Therefore the search direction is wrong -- on the real library path
# ---------------------------------------------------------------------------

CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def _ambient_gammas_and_omegas(p, free, key):
    """The pre-fix body, verbatim: project the ambient matrices, no U-dagger.

    Kept inline so the comparison below is against the code that actually
    shipped rather than a strawman.
    """
    import jax.numpy as jnp

    unitary = p.compute_U_fn(free)
    gammaU = p.geo_fn(unitary, key=key)
    gammas = p.project_omegas_fn(jnp.expand_dims(gammaU, axis=0)).squeeze(axis=0) / (
        gammaU.shape[0]
    )
    dUs_t = jnp.transpose(jnp.array(p.jac_fn(free)), [2, 3, 0, 1])
    omegas = jnp.array([p.project_omegas_fn(1.0j * omegaUs) for omegaUs in dUs_t])
    return gammas, omegas[:, p.proj_indices_projdrift_basis, :]


def stage_3():
    banner("3. So the least-squares solve returns the wrong search direction")

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from geope import Parameters, construct_full_pauli_basis
    from geope.geope import linear_comb_projected_coeffs_multigate

    # piecewise_steps = 2 gives 8 controllable dof against dim su(4) = 15, so the
    # Jacobian is rank-deficient and the least-squares residual actually matters.
    p = Parameters(
        basis=construct_full_pauli_basis(2),
        control={1: ["x", "z"], 2: ["x", "z"]},
        drift={(1, 2): ["zz"]},
        drift_values={(1, 2): {"zz": 1.0}},
        target=CNOT,
        piecewise_steps=2,
        seed=0,
    )
    key = jax.random.key(3)
    pidx = np.asarray(p.proj_indices_projdrift_basis)
    K_pd, G, d = p.proj_drift_basis.lie_algebra_dim, p.piecewise_steps, 4
    print(f"  dof = {G * int(pidx.sum())}, dim su(4) = {p.basis.lie_algebra_dim}")

    def cosine(X, Y):
        n = np.linalg.norm(X) * np.linalg.norm(Y)
        return float(np.real(np.trace(X.conj().T @ Y)) / n) if n > 0 else 1.0

    def at(free):
        """Frobenius projection, shipped direction and pre-fix direction at `free`."""
        U = np.asarray(p.compute_U_fn(free))
        A = U.conj().T @ np.asarray(p.geo_fn(U, key=key))  # left-trivialised tangent
        dU = np.transpose(np.asarray(p.jac_fn(free)), [2, 3, 0, 1])  # (G, K_pd, d, d)
        J = 1j * np.einsum("ab,gkbc->gkac", U.conj().T, dU)[:, pidx]  # Hermitian

        # The definition: the Frobenius-orthogonal projection of A onto span(J),
        # by an independent least squares in matrix space.
        Bm = J.reshape(-1, d * d).T
        Br = np.concatenate([Bm.real, Bm.imag], axis=0)
        ar = np.concatenate([A.real.ravel(), A.imag.ravel()])
        r = Br @ np.linalg.lstsq(Br, ar, rcond=None)[0]
        PA = (r[: d * d] + 1j * r[d * d :]).reshape(d, d)

        def achieved(gammas, omegas):
            """Psi = sum_gk x_gk J_gk, the tangent the library's solve asks for.

            Only the *direction* matters: ``gammas`` carries an extra 1/d and
            ``Geope`` renormalises before stepping, so the cosine is the right
            comparison.
            """
            sol = np.asarray(
                linear_comb_projected_coeffs_multigate(omegas, gammas, None)
            )
            return np.einsum("gk,gkab->ab", sol, J)

        conv = lambda t: [np.asarray(v) for v in t]
        return dict(
            U=U,
            A=A,
            PA=PA,
            herm_dev=max(
                np.abs(A - A.conj().T).max(),
                max(np.abs(c - c.conj().T).max() for c in J.reshape(-1, d, d)),
            ),
            psi_fixed=achieved(*conv(p.gammas_and_omegas(free, key))),
            psi_ambient=achieved(*conv(_ambient_gammas_and_omegas(p, free, key))),
        )

    # Evaluated at two iterates. Near the identity the ambient projection is
    # *almost* lossless, so both paths agree -- which is exactly why this bug
    # survived: it is invisible at the start of a run and grows as U moves away.
    rng = np.random.default_rng(7)
    iterates = {
        "start of a run (U ~ 1)": jnp.asarray(p.parameters)[
            :, p.proj_drift_indices
        ].astype(jnp.complex128),
        "a mid-run iterate": jnp.asarray(rng.normal(size=(G, K_pd)) * 0.8).astype(
            jnp.complex128
        ),
    }

    print()
    print(
        f"  {'iterate':26s} {'||U - 1||':>10s} {'cos shipped':>14s} {'cos pre-fix':>14s}"
    )
    out = {}
    for name, free in iterates.items():
        r = at(free)
        out[name] = r
        print(
            f"  {name:26s} {np.linalg.norm(r['U'] - np.eye(d)):10.4f} "
            f"{cosine(r['PA'], r['psi_fixed']):14.10f} "
            f"{cosine(r['PA'], r['psi_ambient']):14.10f}"
        )
    print("  (cos(Psi, P A) is 1 exactly when the solve IS the projection)")

    far = out["a mid-run iterate"]
    print()
    print(f"  left-trivialised A and J are Hermitian to {far['herm_dev']:.1e}")
    print("  -> the fix is invisible near U = 1, and once U has moved away the")
    print("     pre-fix solve keeps only a third of the correct direction.")

    check(
        "shipped solve is the Frobenius projection",
        all(abs(cosine(r["PA"], r["psi_fixed"]) - 1) < 1e-9 for r in out.values()),
    )
    check(
        "pre-fix solve is NOT the Frobenius projection away from U = 1",
        abs(cosine(far["PA"], far["psi_ambient"]) - 1) > 1e-3,
    )
    return far["PA"], far["A"], far["psi_fixed"], far["psi_ambient"]


# ---------------------------------------------------------------------------
# 4. Why it matters: three residual measures agree only with the fix
# ---------------------------------------------------------------------------


def stage_4(PA, A, psi_fixed, psi_ambient) -> None:
    banner("4. Why it matters: the residual measures agree only with the fix")

    def rel_residual(psi):
        """||Xi|| / ||Omega||, with Xi the part of A the direction cannot reach."""
        s = np.real(np.trace(A.conj().T @ psi))
        n2 = np.real(np.trace(psi.conj().T @ psi))
        if n2 <= 0:
            return float("nan")
        return float(
            np.sqrt(max(1.0 - s**2 / (np.real(np.trace(A.conj().T @ A)) * n2), 0.0))
        )

    frob = float(np.linalg.norm(A - PA) / np.linalg.norm(A))
    print(f"  Frobenius optimum  ||A - P A|| / ||A||     {frob:10.7f}")
    print(
        f"  achieved, shipped                          {rel_residual(psi_fixed):10.7f}"
    )
    print(
        f"  achieved, pre-fix                          {rel_residual(psi_ambient):10.7f}"
    )
    print()
    print("  With the fix the achieved mismatch IS the best achievable one, which is")
    print("  the content of the lemma Psi = P Omega. Without it the two disagree, and")
    print("  every second-order quantity resting on that lemma -- QuadraticArmijo's")
    print("  curvature surrogate, the descent bound -- is describing a direction")
    print("  other than the one the optimiser is about to take.")

    check(
        "shipped achieves the Frobenius optimum",
        abs(rel_residual(psi_fixed) - frob) < 1e-8,
    )
    check(
        "pre-fix does not achieve it",
        abs(rel_residual(psi_ambient) - frob) > 1e-4,
    )


def main() -> int:
    print("Why gammas_and_omegas left-trivialises before projecting.")
    print("(the full argument is in this file's module docstring)")
    stage_1()
    stage_2()
    stage_4(*stage_3())

    banner("Summary")
    for name, ok in CHECKS.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    failed = [n for n, ok in CHECKS.items() if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
