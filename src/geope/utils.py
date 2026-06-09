from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import jax
from jax import Array
import itertools as it
from typing import Callable

from . import lie


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


def traces(b_1: np.ndarray, b_2: np.ndarray) -> Array:
    """Compute the trace inner-product Gram matrix between two basis sets.

    Returns a matrix $G_{ij} = \\mathrm{Tr}(B^{(1)}_i B^{(2)}_j)$ for
    all pairs of basis elements.

    Args:
        b_1: First basis tensor ``np.ndarray`` of shape ``(K1, d, d)``.
        b_2: Second basis tensor ``np.ndarray`` of shape ``(K2, d, d)``.

    Returns:
        A complex ``Array`` of shape ``(K1, K2)``.
    """
    indices = []
    len_1 = b_1.shape[0]
    len_2 = b_2.shape[0]
    for i in range(len_1):
        for j in range(len_2):
            indices.append([i, j])
    indices = np.stack(indices)
    jself = jnp.array(b_1)
    jother = jnp.array(b_2)
    carry = jnp.empty((len_1, len_2), dtype=complex)

    def scan_body(c, idx):
        idx, jdx = idx
        c = c.at[idx, jdx].set(trace_dot_jit(jself[idx], jother[jdx]))
        return c, None

    carry, _ = jax.lax.scan(scan_body, init=carry, xs=indices)
    return carry


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


def check_2_local_comb(comb: tuple[int, ...]) -> bool:
    """Check whether a Pauli index combination is at most 2-local.

    Allows any term acting on at most two qubits.

    Args:
        comb: Tuple of integers (0=I, 1=X, 2=Y, 3=Z).

    Returns:
        ``True`` if the combination involves at most two non-identity
        Pauli operators.
    """
    if len(np.nonzero(comb)[0]) == 1:
        return True
    elif len(np.nonzero(comb)[0]) > 2:
        return False
    else:
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
    mapping = {'x': 1, 'y': 2, 'z': 3}
    restriction_int = [sorted([mapping[char] for char in res if char in mapping]) for res in restriction]
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
    mapping = {'x': 1, 'y': 2, 'z': 3}
    restriction_int = []
    for interaction in restriction.keys():
        for label in restriction[interaction]:
            r = [0] * n
            if type(interaction) is int:
                r[interaction-1] = mapping[label[0]]
            else:
                for i,k in enumerate(interaction):
                    r[k-1] = mapping[label[i]]
            restriction_int.append(r)    
    def check(comb):
        return list(comb) in restriction_int
    return check 


def control_to_indices(labels: list[str], control: dict,
                       strict: bool = False) -> list[int]:
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
        non_id = [(pos, c.lower()) for pos, c in enumerate(label) if c != 'I']
        if len(non_id) == 0:
            continue
        sites = [pos + 1 for pos, _ in non_id]
        ops = ''.join(c for _, c in non_id)
        key = tuple(sites) if len(sites) > 1 else sites[0]
        allowed = control.get(key)
        if allowed is not None and ops in allowed:
            keep.append(idx)
            matched.add((key, ops))
    if strict:
        requested = {(key, op) for key, ops in control.items() for op in ops}
        missing = requested - matched
        if missing:
            pretty = ", ".join(f"{op!r} on qubit(s) {key}"
                               for key, op in sorted(missing, key=str))
            raise ValueError(
                f"Interaction(s) not present in the basis: {pretty}. "
                f"Check the qubit index/tuple, the operator label, and its "
                f"ordering against the available labels.")
    return keep


def filter_basis_by_control(basis: lie.Basis, control: dict) -> lie.Basis:
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
    n_qubits = basis._n_qubits_override if basis._n_qubits_override is not None else None
    return lie.Basis(b, labels=l, n_qubits=n_qubits)


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
        return jnp.array([
            f(phi[k]) if f is not None else phi[k]
            for k, f in enumerate(transforms)
        ])
    return param_transform


def construct_restricted_pauli_basis(
    n: int, restriction: list[str] | dict[int | tuple[int, ...], list[str]]
) -> lie.Basis:
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
        p = 1.
        s = ''
        if restriction(comb):
            for c in comb:
                if c == 0:
                    p = np.kron(p, I)
                    s += 'I' 
                elif c == 1:
                    p = np.kron(p, X)
                    s += 'X' 
                elif c == 2:
                    p = np.kron(p, Y)
                    s += 'Y' 
                elif c == 3:
                    p = np.kron(p, Z)
                    s += 'Z' 
            b.append(p)
            l.append(s)

    return lie.Basis(np.stack(b), labels=l)


def construct_Heisenberg_pauli_basis(n: int) -> lie.Basis:
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
        p = 1.
        s = ''
        if check_Heisenberg_comb(comb):
            for c in comb:
                if c == 0:
                    p = np.kron(p, I)
                    s += 'I' 
                elif c == 1:
                    p = np.kron(p, X)
                    s += 'X' 
                elif c == 2:
                    p = np.kron(p, Y)
                    s += 'Y' 
                elif c == 3:
                    p = np.kron(p, Z)
                    s += 'Z' 
            b.append(p)
            l.append(s)

    return lie.Basis(np.stack(b), labels=l)


def construct_two_body_pauli_basis(n: int) -> lie.Basis:
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
        p = 1.
        s = ''
        if len(np.nonzero(comb)[0]) <= 2:
            for c in comb:
                if c == 0:
                    p = np.kron(p, I)
                    s += 'I' 
                elif c == 1:
                    p = np.kron(p, X)
                    s += 'X' 
                elif c == 2:
                    p = np.kron(p, Y)
                    s += 'Y' 
                elif c == 3:
                    p = np.kron(p, Z)
                    s += 'Z' 
            b.append(p)
            l.append(s)

    return lie.Basis(np.stack(b), labels=l)


def construct_full_pauli_basis(n: int) -> lie.Basis:
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
        p = 1.
        s = ''
        for c in comb:
            if c == 0:
                p = np.kron(p, I)
                s += 'I' 
            elif c == 1:
                p = np.kron(p, X)
                s += 'X' 
            elif c == 2:
                p = np.kron(p, Y)
                s += 'Y' 
            elif c == 3:
                p = np.kron(p, Z)
                s += 'Z' 
        b.append(p)
        l.append(s)

    return lie.Basis(np.stack(b), labels=l)


def creation_annihilation_operators(boson_truncation: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
) -> lie.Basis:
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
        p = 1.
        s = ''
        for c in comb:
            if c == 0:
                p = np.kron(p, I)
                s += 'I' 
            elif c == 1:
                p = np.kron(p, X)
                s += 'X' 
            elif c == 2:
                p = np.kron(p, Y)
                s += 'Y' 
            elif c == 3:
                p = np.kron(p, Z)
                s += 'Z'
        for bos_comb in list(it.product([0,1,2], repeat=n_bosons)):
            pb = np.copy(p)
            sb = ''.join(s)
            for bos_c in bos_comb:
                if bos_c == 0:
                    pb = np.kron(pb, a_0)
                    sb += 'i'
                elif bos_c == 1:
                    pb = np.kron(pb, (a_minus + a_plus)/a_norm)
                    sb += 'q'
                elif bos_c == 2:
                    pb = np.kron(pb, 1.j * (a_plus - a_minus)/a_norm)
                    sb += 'p'
            b.append(pb)
            l.append(sb)

    return lie.Basis(np.stack(b), labels=l)


def construct_restricted_spin_boson_basis(
    n_spins: int,
    n_bosons: int,
    restriction: list[str] | dict[int | tuple[int, ...], list[str]],
    boson_truncation: int = 3,
) -> lie.Basis:
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
        p = 1.
        s = ''
        if restriction(comb):
            for c in comb:
                if c == 0:
                    p = np.kron(p, I)
                    s += 'I' 
                elif c == 1:
                    p = np.kron(p, X)
                    s += 'X' 
                elif c == 2:
                    p = np.kron(p, Y)
                    s += 'Y' 
                elif c == 3:
                    p = np.kron(p, Z)
                    s += 'Z' 
            for bos_comb in list(it.product([0,1,2], repeat=n_bosons)):
                pb = np.copy(p)
                sb = ''.join(s)
                for bos_c in bos_comb:
                    if bos_c == 0:
                        pb = np.kron(pb, a_0)
                        sb += 'i'
                    elif bos_c == 1:
                        pb = np.kron(pb, (a_minus + a_plus)/a_norm)
                        sb += 'q'
                    elif bos_c == 2:
                        pb = np.kron(pb, 1.j * (a_plus - a_minus)/a_norm)
                        sb += 'p'
                b.append(pb)
                l.append(sb)

    return lie.Basis(np.stack(b), labels=l)


def prepare_random_parameters(
    proj_indices: np.ndarray,
    expander: np.ndarray | None = None,
    spread: float = 1.0,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a random parameter vector for the projected subspace.

    Samples uniform random values in $[-\\text{spread}, \\text{spread}]$
    and optionally expands them through a constraint matrix.

    Args:
        proj_indices: Boolean ``np.ndarray`` mask indicating projected
            parameter positions.
        expander: Optional constraint expansion ``np.ndarray``.
        spread: Half-width of the uniform sampling range. Defaults to 1.0.
        seed: Random seed for reproducibility. Defaults to ``None``.
            Ignored when ``rng`` is provided.
        rng: Optional NumPy ``Generator``. When given, values are drawn
            from it and the global ``np.random`` state is left untouched.

    Returns:
        A parameter ``np.ndarray`` of the same length as ``proj_indices``
        with random values at projected positions and zeros elsewhere.
    """
    num_indep_params = proj_indices.sum() if expander is None else expander.shape[1]
    if rng is None:
        np.random.seed(seed)
        randoms = (2 * np.random.rand(num_indep_params) - 1) * spread
    else:
        randoms = (2 * rng.random(num_indep_params) - 1) * spread
    if expander is not None:
        randoms = expander @ randoms
    parameters = np.zeros_like(proj_indices, dtype=randoms.dtype)
    parameters[proj_indices] = randoms
    return parameters


def construct_commuting_ansatz_matrix(params: list, sols: dict) -> np.ndarray:
    """Construct the commuting-ansatz substitution matrix.

    Builds a matrix that encodes how free parameters map to the
    full parameter vector through the symbolic solutions.

    Args:
        params: List of symbolic parameter names (or ``0`` / falsy
            for absent parameters).
        sols: Dictionary mapping dependent parameter names to
            symbolic expressions.

    Returns:
        A square numpy array of shape ``(len(params), len(params))``.
    """
    mat = np.zeros((len(params), len(params)))
    for j, h in enumerate(params):
        if h:
            h_sub = {m: 0 for m in params if m}
            h_sub[h] = 1
            for i, s in enumerate(params):
                if i == j:
                    mat[i, j] = 1
                if s in sols:
                    mat[i, j] = sols[s].subs(h_sub)
    return mat


def remove_solution_free_parameters(params: list, sols: dict) -> list[int]:
    """Identify which parameters are free (not determined by solutions).

    Args:
        params: List of symbolic parameter names.
        sols: Dictionary of solved dependent parameters.

    Returns:
        A list of 0s and 1s; ``1`` indicates a free parameter.
    """
    indices = [0 if h in sols else 1 if h else 0 for h in params]
    return indices


def multikron(matrices: list[np.ndarray]) -> np.ndarray:
    """Compute the Kronecker product of a list of matrices.

    Args:
        matrices: List of 2-D arrays.

    Returns:
        The iterated Kronecker product.
    """
    product = matrices[0]
    for mat in matrices[1:]:
        product = np.kron(product, mat)
    return product


def multimatmul(matrices: list[np.ndarray]) -> np.ndarray:
    """Compute the matrix multiplication of a list of matrices.

    Args:
        matrices: List of 2-D arrays.

    Returns:
        The iterated matrix multiplication.
    """
    matmul = matrices[0]
    for mat in matrices[1:]:
        matmul = np.matmul(matmul, mat)
    return matmul


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
    dim = 2**(num_controls+1)
    full_unitary = np.eye(dim)
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
    w = np.exp(1.j * 2 * np.pi / 2 ** num_qubits)
    qft_unitary = (1 / np.sqrt(2 ** num_qubits)) * np.array([[w ** (i * j) for i in range(2 ** num_qubits)] for j in range(2 ** num_qubits)])
    return qft_unitary

def golden_section_search_np(
    f: Callable[[float], float], a: float, b: float, tol: float = 1e-5
) -> tuple[float, float]:
    """Golden-section search using NumPy.

    Finds the minimum of a unimodal function `f` on the interval
    $[a, b]$ to within tolerance `tol`.

    Args:
        f: Scalar-valued unimodal callable.
        a: Left endpoint of the search interval.
        b: Right endpoint of the search interval.
        tol: Convergence tolerance. Defaults to 1e-5.

    Returns:
        A tuple ``(x_min, f_min)`` of the approximate minimiser
        and its function value.

    Example:
        ```python
        f = lambda x: (x - 2) ** 2
        x_min, f_min = golden_section_search_np(f, 1, 5)
        ```

    References:
        [Golden-section search](https://en.wikipedia.org/wiki/Golden-section_search)
    """

    invphi = (np.sqrt(5) - 1) / 2  # 1 / phi
    invphi2 = (3 - np.sqrt(5)) / 2  # 1 / phi^2

    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= tol:
        return (a, b)

    # Required steps to achieve tolerance
    n = int(np.ceil(np.log(tol / h) / np.log(invphi)))

    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)
    yd = f(d)

    for k in range(n - 1):
        if yc < yd:
            # Minimum is in [a, d]: discard the right portion.
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c)
        else:
            # Minimum is in [c, b]: discard the left portion.
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d)
    if yc < yd:
        return c, yc
    else:
        return d, yd


def golden_section_search(
    f: Callable[[Array], Array],
    a_init: float | Array,
    b_init: float | Array,
    tol: float = 1e-5,
) -> tuple[Array, Array]:
    """JIT-compatible golden-section search using JAX.

    Finds the minimum of a unimodal function `f` on the interval
    $[a, b]$ using ``jax.lax.while_loop``, making it compatible
    with JIT compilation.

    Args:
        f: Scalar-valued unimodal callable.
        a_init: Left endpoint of the search interval.
        b_init: Right endpoint of the search interval.
        tol: Convergence tolerance. Defaults to 1e-5.

    Returns:
        A tuple ``(x_min, f_min)`` of the approximate minimiser
        and its function value.

    Example:
        ```python
        f = lambda x: (x - 2) ** 2
        x_min, f_min = golden_section_search(f, 1.0, 5.0)
        ```

    References:
        [Golden-section search](https://en.wikipedia.org/wiki/Golden-section_search)
    """
    phi = (jnp.sqrt(5.0) - 1.0) / 2.0   
    resphi = 1.0 - phi 
    max_iter = jnp.array((jnp.ceil(jnp.log(tol / (b_init - a_init)) / jnp.log(phi))),int) 

    a = a_init
    b = b_init

    x1 = a + resphi * (b - a)
    x2 = a + phi   * (b - a)
    f1 = f(x1)
    f2 = f(x2)

    state0 = (a, b, x1, x2, f1, f2, jnp.array(0, dtype=jnp.int32))

    def cond_fun(state):
        a, b, x1, x2, f1, f2, i = state
        interval_check = (b - a) > tol
        iter_check = i < max_iter
        return jnp.logical_and(interval_check, iter_check)

    def body_fun(state):
        a, b, x1, x2, f1, f2, i = state

        def left_branch(s):
            # Minimum is in [a, x2]: discard the right portion (b <- x2).
            a, b, x1, x2, f1, f2, i = s
            b_new = x2
            x2_new = x1
            f2_new = f1
            x1_new = a + resphi * (b_new - a)
            f1_new = f(x1_new)
            return (a, b_new, x1_new, x2_new, f1_new, f2_new, i + 1)

        def right_branch(s):
            # Minimum is in [x1, b]: discard the left portion (a <- x1).
            a, b, x1, x2, f1, f2, i = s
            a_new = x1
            x1_new = x2
            f1_new = f2
            x2_new = a_new + phi * (b - a_new)
            f2_new = f(x2_new)
            return (a_new, b, x1_new, x2_new, f1_new, f2_new, i + 1)

        return jax.lax.cond(f1 < f2, left_branch, right_branch, state)

    a, b, x1, x2, f1, f2, i = jax.lax.while_loop(cond_fun, body_fun, state0)

    t_best = jnp.where(f1 < f2, x1, x2)
    f_best = jnp.where(f1 < f2, f1, f2)
    return t_best, f_best


def adam_line_search(
    f: Callable[[Array], Array],
    a_init: float | Array,
    b_init: float | Array,
    lr: float = 0.05,
    num_steps: int = 30,
    finite_difference: bool = True,
    fd_step: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    t_init: float | Array = 0.0,
) -> tuple[Array, Array]:
    """JIT-compatible 1-D Adam line search using JAX.

    Minimises a scalar function `f` on the interval $[a, b]$ by running
    a fixed number of Adam steps on the scalar variable ``t``, clipping
    ``t`` back into the interval after every step. Uses
    ``jax.lax.fori_loop`` (fixed step count), making it compatible with
    JIT compilation.

    The gradient ``df/dt`` is obtained either by a finite-difference
    secant from successive evaluations (``finite_difference=True``;
    derivative-free, one ``f`` evaluation per step) or by
    ``jax.value_and_grad`` (``finite_difference=False``; exact, but
    differentiates through ``f``). ``f`` must map a real scalar to a
    real scalar.

    Adam is not monotone, so the best iterate visited is tracked and
    returned — the result is never worse than ``f(t_init)``.

    Args:
        f: Scalar-valued callable (real -> real).
        a_init: Left endpoint of the search interval.
        b_init: Right endpoint of the search interval.
        lr: Adam learning rate. Defaults to 0.05.
        num_steps: Number of Adam iterations. Defaults to 30.
        finite_difference: If ``True`` (default), estimate the gradient
            with a finite-difference secant; otherwise use
            ``jax.value_and_grad``.
        fd_step: Probe size for the finite-difference bootstrap.
            Defaults to 1e-3.
        beta1: First-moment decay. Defaults to 0.9.
        beta2: Second-moment decay. Defaults to 0.999.
        eps: Numerical-stability term. Defaults to 1e-8.
        t_init: Starting point for ``t``. Defaults to 0.0.

    Returns:
        A tuple ``(t_best, f_best)`` of the best minimiser found and its
        function value, matching the ``(x_min, f_min)`` contract of
        :func:`golden_section_search`.

    Example:
        ```python
        f = lambda x: (x - 2.0) ** 2
        x_min, f_min = adam_line_search(f, 0.0, 5.0, lr=0.1, num_steps=200)
        ```

    References:
        [Adam](https://arxiv.org/abs/1412.6980)
    """
    lo = jnp.minimum(a_init, b_init)
    hi = jnp.maximum(a_init, b_init)
    f64 = lambda x: jnp.asarray(x, dtype=jnp.float64)

    def adam_update(i, t, m, v, g):
        # Shared Adam moment update + bias correction + interval clip.
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        step = f64(i) + 1.0
        m_hat = m / (1.0 - beta1 ** step)
        v_hat = v / (1.0 - beta2 ** step)
        t_new = jnp.clip(t - lr * m_hat / (jnp.sqrt(v_hat) + eps), lo, hi)
        return t_new, m, v

    t0 = jnp.clip(f64(t_init), lo, hi)
    f0 = f64(f(t0))

    if finite_difference:
        # Bootstrap: one inward probe to seed (t_prev, f_prev).
        direction = jnp.sign((lo + hi) / 2.0 - t0)
        direction = jnp.where(direction == 0, -1.0, direction)
        t_start = jnp.clip(t0 + direction * fd_step, lo, hi)
        # state: (t, m, v, t_prev, f_prev, t_best, f_best)
        state0 = (t_start, f64(0.0), f64(0.0), t0, f0, t0, f0)

        def body_fun(i, state):
            t, m, v, t_prev, f_prev, t_best, f_best = state
            ft = f64(f(t))
            improved = ft < f_best
            t_best = jnp.where(improved, t, t_best)
            f_best = jnp.where(improved, ft, f_best)
            dt_ = t - t_prev
            dt_safe = jnp.where(dt_ == 0, fd_step, dt_)   # guard exact-zero
            g = (ft - f_prev) / dt_safe                   # secant slope
            t_new, m, v = adam_update(i, t, m, v, g)
            return (t_new, m, v, t, ft, t_best, f_best)

        t, m, v, t_prev, f_prev, t_best, f_best = jax.lax.fori_loop(
            0, num_steps, body_fun, state0)
    else:
        # When ``f`` maps the real ``t`` through complex intermediates (e.g.
        # unitaries), JAX may emit a benign ComplexWarning while forming the
        # real cotangent of ``t``; the gradient is correct (verified against
        # finite differences).
        value_and_grad = jax.value_and_grad(f)
        # state: (t, m, v, t_best, f_best)
        state0 = (t0, f64(0.0), f64(0.0), t0, f0)

        def body_fun(i, state):
            t, m, v, t_best, f_best = state
            ft, g = value_and_grad(t)
            ft = f64(ft)
            improved = ft < f_best
            t_best = jnp.where(improved, t, t_best)
            f_best = jnp.where(improved, ft, f_best)
            t_new, m, v = adam_update(i, t, m, v, g)
            return (t_new, m, v, t_best, f_best)

        t, m, v, t_best, f_best = jax.lax.fori_loop(
            0, num_steps, body_fun, state0)

    # Also consider the final iterate (evaluated once after the loop).
    f_last = f64(f(t))
    take_last = f_last < f_best
    t_best = jnp.where(take_last, t, t_best)
    f_best = jnp.where(take_last, f_last, f_best)
    return t_best, f_best


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
                if not np.allclose(cons[i, overlap],
                                   cons[j, overlap] * scale,
                                   rtol=rtol, atol=atol):
                    raise ValueError(
                        f"Inconsistent constraints at rows {i} and {j}"
                    )

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