from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import jax
from jax import Array
import itertools as it
from typing import Callable

from ..geometry.lie.basis import Basis, traces


@jax.jit
def trace_dot_jit(x: Array, y: Array) -> Array:
    """Compute the trace of the matrix product $\\mathrm{Tr}(xy)$.

    JIT-compiled for use inside scan loops.

    Args:
        x: First matrix ``Array``.
        y: Second matrix ``Array``.

    Returns:
        A scalar trace ``Array``.
    """
    return jnp.trace(x @ y)


def check_xy_comb(comb: tuple[int, ...]) -> bool:
    """Check whether a Pauli index combination is valid for XY-type interactions.

    Allows single-body terms and two-body terms with identical non-identity
    Pauli indices (XX, YY), but not ZZ or mixed two-body terms.

    Args:
        comb: Tuple of integers (0=I, 1=X, 2=Y, 3=Z).

    Returns:
        ``True`` if the combination is allowed.
    """
    if len(np.nonzero(comb)[0]) == 1:
        return True
    elif len(np.nonzero(comb)[0]) > 2:
        return False
    else:
        for i, a in enumerate(comb):
            for j, b in enumerate(comb):
                if (i != j) and (a != b) and (a > 0) and (b > 0):
                    return False
                elif (a == 3) and (b == 3):
                    return False
    return True


def check_Heisenberg_comb(comb: tuple[int, ...]) -> bool:
    """Check whether a Pauli index combination is valid for Heisenberg interactions.

    Allows single-body terms and two-body terms with identical
    non-identity Pauli indices (XX, YY, ZZ).

    Args:
        comb: Tuple of integers (0=I, 1=X, 2=Y, 3=Z).

    Returns:
        ``True`` if the combination is allowed.
    """
    if len(np.nonzero(comb)[0]) == 1:
        return True
    elif len(np.nonzero(comb)[0]) > 2:
        return False
    else:
        for i, a in enumerate(comb):
            for j, b in enumerate(comb):
                if (i != j) and (a != b) and (a > 0) and (b > 0):
                    return False
    return True


def restriction_function(restriction: list[str]) -> Callable[[tuple[int, ...]], bool]:
    """Create a filter function from a list of allowed interaction strings.

    Each string in `restriction` encodes an allowed Pauli combination
    using characters ``'x'``, ``'y'``, ``'z'``.

    Args:
        restriction: List of strings, e.g. ``['xx', 'yy', 'zz']``.

    Returns:
        A ``Callable[[tuple[int, ...]], bool]`` that accepts a Pauli
        index tuple and returns ``True`` if it matches any allowed pattern.
    """
    mapping = {"x": 1, "y": 2, "z": 3}
    restriction_int = [
        sorted([mapping[char] for char in res if char in mapping])
        for res in restriction
    ]

    def check(comb):
        sorted_comb = sorted([c for c in comb if c != 0])
        return sorted_comb in restriction_int

    return check


def restriction_order_function(
    n: int, restriction: dict[int | tuple[int, ...], list[str]]
) -> Callable[[tuple[int, ...]], bool]:
    """Create an ordered restriction filter from a dictionary.

    Args:
        n: Number of qubits.
        restriction: Dictionary mapping qubit indices (or tuples) to
            lists of interaction label strings.

    Returns:
        A ``Callable[[tuple[int, ...]], bool]`` that accepts a Pauli
        index tuple and returns ``True`` if it matches the restriction.
    """
    mapping = {"x": 1, "y": 2, "z": 3}
    restriction_int = []
    for interaction in restriction.keys():
        for label in restriction[interaction]:
            r = [0] * n
            if type(interaction) is int:
                r[interaction - 1] = mapping[label[0]]
            else:
                for i, k in enumerate(interaction):
                    r[k - 1] = mapping[label[i]]
            restriction_int.append(r)

    def check(comb):
        return list(comb) in restriction_int

    return check


def control_to_indices(
    labels: list[str], control: dict, strict: bool = False
) -> list[int]:
    """Map Pauli labels to the indices selected by a control-format dict.

    For each label, build the qubit-index key (a single integer for
    1-body terms, a tuple for multi-body) and the lower-case interaction
    string, then keep the index only if ``control[key]`` lists that
    interaction. Preserves the order of ``labels``.

    Args:
        labels: Sequence of Pauli-string labels, e.g. ``["XII", "ZZI"]``.
        control: Dict mapping qubit index (or tuple of indices) to a
            list of interaction labels, e.g. ``{1: ['x', 'y'], (1, 2): ['xx']}``.
        strict: If ``True``, raise ``ValueError`` when any ``(key, op)``
            entry in ``control`` matches no label in ``labels`` (e.g. a
            typo, a wrong qubit index, or an interaction absent from the
            basis). Defaults to ``False`` (silently ignore such entries).

    Returns:
        The list of indices into ``labels`` that match the control dict.

    Raises:
        ValueError: If ``strict`` is ``True`` and one or more ``(key, op)``
            entries are not present among ``labels``.
    """
    keep = []
    matched = set()
    for idx, label in enumerate(labels):
        non_id = [(pos, c.lower()) for pos, c in enumerate(label) if c != "I"]
        if len(non_id) == 0:
            continue
        sites = [pos + 1 for pos, _ in non_id]
        ops = "".join(c for _, c in non_id)
        key = tuple(sites) if len(sites) > 1 else sites[0]
        allowed = control.get(key)
        if allowed is not None and ops in allowed:
            keep.append(idx)
            matched.add((key, ops))
    if strict:
        requested = {(key, op) for key, ops in control.items() for op in ops}
        missing = requested - matched
        if missing:
            pretty = ", ".join(
                f"{op!r} on qubit(s) {key}" for key, op in sorted(missing, key=str)
            )
            raise ValueError(
                f"Interaction(s) not present in the basis: {pretty}. "
                f"Check the qubit index/tuple, the operator label, and its "
                f"ordering against the available labels."
            )
    return keep


def filter_basis_by_control(basis: Basis, control: dict) -> Basis:
    """Filter a Basis keeping only operators that match a control dict.

    For each basis element, inspect its label, build the qubit-index key
    (a single integer for 1-body terms, a tuple for multi-body) and the
    lower-case interaction label, then keep the element only if
    ``control[key]`` lists the interaction.

    Args:
        basis: The full ``Basis`` to filter.
        control: Dict mapping qubit index (or tuple of indices) to a
            list of interaction labels, e.g. ``{1: ['x', 'y'], (1, 2): ['xx']}``.

    Returns:
        A new ``Basis`` containing only the matching operators. The
        returned basis preserves ``basis._n_qubits_override`` if set.
    """
    keep = control_to_indices(list(basis.labels), control)
    b = basis.basis[keep]
    l = [basis.labels[i] for i in keep]
    n_qubits = (
        basis._n_qubits_override if basis._n_qubits_override is not None else None
    )
    return Basis(b, labels=l, n_qubits=n_qubits)


def make_per_element_transform(transforms: list[Callable | None]) -> Callable:
    """Build a ``param_transform`` from per-element callables.

    Each entry of ``transforms`` maps a single experimental parameter
    to a single basis coefficient. Use ``None`` to mean identity.

    Args:
        transforms: List of callables (or ``None``), one per basis element.

    Returns:
        A callable mapping a ``phi`` vector to a coefficients vector,
        suitable for ``Parameters(param_transform=...)``.

    Example:
        ``transforms = [lambda x: jnp.exp(1j*x), jnp.cos, None]``.
    """

    def param_transform(phi):
        return jnp.array(
            [f(phi[k]) if f is not None else phi[k] for k, f in enumerate(transforms)]
        )

    return param_transform


def construct_restricted_pauli_basis(
    n: int, restriction: list[str] | dict[int | tuple[int, ...], list[str]]
) -> Basis:
    """Construct a Pauli basis restricted by allowed interactions.

    Args:
        n: Number of qubits.
        restriction: Either a list of allowed interaction strings
            or a dictionary mapping qubit indices to interaction
            labels.

    Returns:
        A `Basis` instance containing only the allowed Pauli strings.
    """
    I = np.eye(2).astype(complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    b = []
    l = []
    if type(restriction) is list:
        restriction = restriction_function(restriction)
    elif type(restriction) is dict:
        restriction = restriction_order_function(n, restriction)
    for comb in list(it.product([0, 1, 2, 3], repeat=n))[1:]:
        p = 1.0
        s = ""
        if restriction(comb):
            for c in comb:
                if c == 0:
                    p = np.kron(p, I)
                    s += "I"
                elif c == 1:
                    p = np.kron(p, X)
                    s += "X"
                elif c == 2:
                    p = np.kron(p, Y)
                    s += "Y"
                elif c == 3:
                    p = np.kron(p, Z)
                    s += "Z"
            b.append(p)
            l.append(s)

    return Basis(np.stack(b), labels=l)


def construct_Heisenberg_pauli_basis(n: int) -> Basis:
    """Construct the Pauli basis for a Heisenberg-type Hamiltonian.

    Includes all single-body Pauli terms and two-body terms of the
    form XX, YY, ZZ on any pair of qubits.

    Args:
        n: Number of qubits.

    Returns:
        A `Basis` instance.
    """
    I = np.eye(2).astype(complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    b = []
    l = []
    for comb in list(it.product([0, 1, 2, 3], repeat=n))[1:]:
        p = 1.0
        s = ""
        if check_Heisenberg_comb(comb):
            for c in comb:
                if c == 0:
                    p = np.kron(p, I)
                    s += "I"
                elif c == 1:
                    p = np.kron(p, X)
                    s += "X"
                elif c == 2:
                    p = np.kron(p, Y)
                    s += "Y"
                elif c == 3:
                    p = np.kron(p, Z)
                    s += "Z"
            b.append(p)
            l.append(s)

    return Basis(np.stack(b), labels=l)


def construct_two_body_pauli_basis(n: int) -> Basis:
    """Construct the full two-body Pauli basis.

    Includes all Pauli strings acting on at most two qubits.

    Args:
        n: Number of qubits.

    Returns:
        A `Basis` instance.
    """
    I = np.eye(2).astype(complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    b = []
    l = []
    for comb in list(it.product([0, 1, 2, 3], repeat=n))[1:]:
        p = 1.0
        s = ""
        if len(np.nonzero(comb)[0]) <= 2:
            for c in comb:
                if c == 0:
                    p = np.kron(p, I)
                    s += "I"
                elif c == 1:
                    p = np.kron(p, X)
                    s += "X"
                elif c == 2:
                    p = np.kron(p, Y)
                    s += "Y"
                elif c == 3:
                    p = np.kron(p, Z)
                    s += "Z"
            b.append(p)
            l.append(s)

    return Basis(np.stack(b), labels=l)


def construct_full_pauli_basis(n: int) -> Basis:
    """Construct the full $n$-qubit Pauli basis (excluding identity).

    Contains all $4^n - 1$ non-identity Pauli strings.

    Args:
        n: Number of qubits.

    Returns:
        A `Basis` instance.
    """
    I = np.eye(2).astype(complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)

    b = []
    l = []
    for comb in list(it.product([0, 1, 2, 3], repeat=n))[1:]:
        p = 1.0
        s = ""
        for c in comb:
            if c == 0:
                p = np.kron(p, I)
                s += "I"
            elif c == 1:
                p = np.kron(p, X)
                s += "X"
            elif c == 2:
                p = np.kron(p, Y)
                s += "Y"
            elif c == 3:
                p = np.kron(p, Z)
                s += "Z"
        b.append(p)
        l.append(s)

    return Basis(np.stack(b), labels=l)


def creation_annihilation_operators(
    boson_truncation: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build truncated bosonic creation and annihilation operators.

    Args:
        boson_truncation: Maximum occupation number.

    Returns:
        A tuple ``(a_0, a_minus, a_plus)`` where ``a_0`` is the identity,
        ``a_minus`` is the lowering operator, and ``a_plus`` is the
        raising operator, each of dimension ``boson_truncation + 1``.
    """
    dim = boson_truncation + 1
    coeff = np.sqrt(np.arange(1, dim))
    a_0 = np.eye(dim)
    a_minus = np.diag(coeff, k=1)
    a_plus = np.diag(coeff, k=-1)
    return a_0, a_minus, a_plus


def construct_full_spin_boson_basis(
    n_spins: int, n_bosons: int, boson_truncation: int = 3
) -> Basis:
    """Construct the full spin-boson Pauli-like basis.

    Combines all $n$-qubit Pauli strings with bosonic position ($q$)
    and momentum ($p$) operators on each bosonic mode.

    Args:
        n_spins: Number of spin (qubit) degrees of freedom.
        n_bosons: Number of bosonic modes.
        boson_truncation: Fock-space truncation level. Defaults to 3.

    Returns:
        A `Basis` instance.
    """
    I = np.eye(2).astype(complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)

    a_0, a_minus, a_plus = creation_annihilation_operators(boson_truncation)
    a_norm = np.sqrt(boson_truncation)

    b = []
    l = []
    for comb in list(it.product([0, 1, 2, 3], repeat=n_spins))[1:]:
        p = 1.0
        s = ""
        for c in comb:
            if c == 0:
                p = np.kron(p, I)
                s += "I"
            elif c == 1:
                p = np.kron(p, X)
                s += "X"
            elif c == 2:
                p = np.kron(p, Y)
                s += "Y"
            elif c == 3:
                p = np.kron(p, Z)
                s += "Z"
        for bos_comb in list(it.product([0, 1, 2], repeat=n_bosons)):
            pb = np.copy(p)
            sb = "".join(s)
            for bos_c in bos_comb:
                if bos_c == 0:
                    pb = np.kron(pb, a_0)
                    sb += "i"
                elif bos_c == 1:
                    pb = np.kron(pb, (a_minus + a_plus) / a_norm)
                    sb += "q"
                elif bos_c == 2:
                    pb = np.kron(pb, 1.0j * (a_plus - a_minus) / a_norm)
                    sb += "p"
            b.append(pb)
            l.append(sb)

    return Basis(np.stack(b), labels=l)


def construct_restricted_spin_boson_basis(
    n_spins: int,
    n_bosons: int,
    restriction: list[str] | dict[int | tuple[int, ...], list[str]],
    boson_truncation: int = 3,
) -> Basis:
    """Construct a restricted spin-boson basis.

    Like `construct_full_spin_boson_basis` but only includes Pauli
    strings matching the given restriction.

    Args:
        n_spins: Number of spin (qubit) degrees of freedom.
        n_bosons: Number of bosonic modes.
        restriction: Either a list of allowed interaction strings or
            a dictionary mapping qubit indices to interaction labels.
        boson_truncation: Fock-space truncation level. Defaults to 3.

    Returns:
        A `Basis` instance.
    """
    I = np.eye(2).astype(complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)

    a_0, a_minus, a_plus = creation_annihilation_operators(boson_truncation)
    a_norm = np.sqrt(boson_truncation)

    b = []
    l = []
    if type(restriction) is list:
        restriction = restriction_function(restriction)
    elif type(restriction) is dict:
        restriction = restriction_order_function(n_spins, restriction)
    for comb in list(it.product([0, 1, 2, 3], repeat=n_spins))[1:]:
        p = 1.0
        s = ""
        if restriction(comb):
            for c in comb:
                if c == 0:
                    p = np.kron(p, I)
                    s += "I"
                elif c == 1:
                    p = np.kron(p, X)
                    s += "X"
                elif c == 2:
                    p = np.kron(p, Y)
                    s += "Y"
                elif c == 3:
                    p = np.kron(p, Z)
                    s += "Z"
            for bos_comb in list(it.product([0, 1, 2], repeat=n_bosons)):
                pb = np.copy(p)
                sb = "".join(s)
                for bos_c in bos_comb:
                    if bos_c == 0:
                        pb = np.kron(pb, a_0)
                        sb += "i"
                    elif bos_c == 1:
                        pb = np.kron(pb, (a_minus + a_plus) / a_norm)
                        sb += "q"
                    elif bos_c == 2:
                        pb = np.kron(pb, 1.0j * (a_plus - a_minus) / a_norm)
                        sb += "p"
                b.append(pb)
                l.append(sb)

    return Basis(np.stack(b), labels=l)


def prepare_random_parameters(
    proj_indices: np.ndarray,
    expander: np.ndarray | None = None,
    spread: float = 1.0,
    key: jax.Array = jax.random.key(0),
) -> np.ndarray:
    """Generate a random parameter vector for the projected subspace.

    Samples uniform random values in $[-\\text{spread}, \\text{spread}]$
    and optionally expands them through a constraint matrix.

    Args:
        proj_indices: Boolean ``np.ndarray`` mask indicating projected
            parameter positions.
        expander: Optional constraint expansion ``np.ndarray``.
        spread: Half-width of the uniform sampling range. Defaults to 1.0.
        key: JAX random key. Defaults to ``jax.random.key(0)``.

    Returns:
        A parameter ``np.ndarray`` of the same length as ``proj_indices``
        with random values at projected positions and zeros elsewhere.
    """
    num_indep_params = proj_indices.sum() if expander is None else expander.shape[1]
    randoms = np.array(
        jax.random.uniform(
            key, shape=(num_indep_params,), minval=-spread, maxval=spread
        )
    )
    if expander is not None:
        randoms = expander @ randoms
    parameters = np.zeros_like(proj_indices, dtype=randoms.dtype)
    parameters[proj_indices] = randoms
    return parameters


def multicontrol_unitary(local_unitary: np.ndarray, num_controls: int) -> np.ndarray:
    """Embed a single-qubit unitary as a multi-controlled gate.

    Places `local_unitary` in the bottom-right $2 \\times 2$ block
    of a $2^{n+1} \\times 2^{n+1}$ identity matrix, where $n$ is
    `num_controls`.

    Args:
        local_unitary: A $2 \\times 2$ unitary matrix.
        num_controls: Number of control qubits.

    Returns:
        The full multi-controlled unitary matrix.
    """
    dim = 2 ** (num_controls + 1)
    full_unitary = np.eye(dim, dtype=np.asarray(local_unitary).dtype)
    indices = [dim - 2, dim - 1]
    full_unitary[np.ix_(indices, indices)] = local_unitary
    return full_unitary


def qft_unitary(num_qubits: int) -> np.ndarray:
    """Construct the Quantum Fourier Transform unitary.

    Args:
        num_qubits: Number of qubits.

    Returns:
        A $2^n \\times 2^n$ QFT unitary matrix.
    """
    w = np.exp(1.0j * 2 * np.pi / 2**num_qubits)
    qft_unitary = (1 / np.sqrt(2**num_qubits)) * np.array(
        [[w ** (i * j) for i in range(2**num_qubits)] for j in range(2**num_qubits)]
    )
    return qft_unitary


def merge_constraints(
    constraints: list[np.ndarray], rtol: float = 1e-9, atol: float = 1e-12
) -> list[list[float]]:
    """Merge overlapping linear equality constraints.

    Iteratively merges rows of the constraint matrix that share
    non-zero entries, verifying consistency of the overlap.

    Args:
        constraints: List of 1-D arrays (all same length) representing
            linear constraints.
        rtol: Relative tolerance for consistency check. Defaults to 1e-9.
        atol: Absolute tolerance for consistency check. Defaults to 1e-12.

    Returns:
        A list of merged constraint vectors with no overlapping
        non-zero entries.

    Raises:
        ValueError: If overlapping constraints are inconsistent.
    """
    cons = np.asarray(constraints, dtype=float)  # shape (m, n)
    i = 0

    while i < len(cons):
        j = i + 1
        merged_any = False

        while j < len(cons):
            # indices where both constraints are active
            overlap = (cons[i] != 0) & (cons[j] != 0)

            if overlap.any():
                # compute scale to align row j to row i
                scale = cons[i, overlap][0] / cons[j, overlap][0]

                # check consistency on overlapping indices
                if not np.allclose(
                    cons[i, overlap], cons[j, overlap] * scale, rtol=rtol, atol=atol
                ):
                    raise ValueError(f"Inconsistent constraints at rows {i} and {j}")

                # merge: prefer non-zero from row i, otherwise scaled row j
                cons[i] = np.where(cons[i] != 0, cons[i], cons[j] * scale)

                # remove row j (it is now merged into i)
                cons = np.delete(cons, j, axis=0)
                merged_any = True
            else:
                j += 1

        if not merged_any:
            i += 1

    return cons.tolist()
