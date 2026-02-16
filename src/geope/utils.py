import numpy as np
import jax.numpy as jnp
import jax
import itertools as it

from . import lie


@jax.jit
def trace_dot_jit(x, y):
    return jnp.trace(x @ y)


def traces(b_1, b_2):
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


def check_xy_comb(comb):
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


def check_Heisenberg_comb(comb):
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


def check_2_local_comb(comb):
    if len(np.nonzero(comb)[0]) == 1:
        return True
    elif len(np.nonzero(comb)[0]) > 2:
        return False
    else:
        return True


def restriction_function(restriction):
    mapping = {'x': 1, 'y': 2, 'z': 3}
    restriction_int = [sorted([mapping[char] for char in res if char in mapping]) for res in restriction]
    def check(comb):
        sorted_comb = sorted([c for c in comb if c != 0])
        return sorted_comb in restriction_int
    return check 


def restriction_order_function(n, restriction):
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


def construct_restricted_pauli_basis(n: int, restriction):
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


def construct_Heisenberg_pauli_basis(n: int):
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


def construct_two_body_pauli_basis(n: int):
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


def construct_full_pauli_basis(n: int):
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


def creation_annihilation_operators(boson_truncation: int):
    dim = boson_truncation + 1
    coeff = np.sqrt(np.arange(1, dim))
    a_0 = np.eye(dim)
    a_minus = np.diag(coeff, k=1)
    a_plus = np.diag(coeff, k=-1)
    return a_0, a_minus, a_plus


def construct_full_spin_boson_basis(n_spins: int, n_bosons: int, boson_truncation: int=3):
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


def construct_restricted_spin_boson_basis(n_spins: int, n_bosons: int, restriction, boson_truncation: int=3):
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


def prepare_random_parameters(proj_indices, expander=None, spread=1.0, seed=None):
    np.random.seed(seed)
    num_indep_params = proj_indices.sum() if expander is None else expander.shape[1]
    randoms = (2 * np.random.rand(num_indep_params) - 1) * spread 
    if expander is not None:
        randoms = expander @ randoms
    parameters = np.zeros_like(proj_indices, dtype=randoms.dtype)
    parameters[proj_indices] = randoms
    return parameters


def construct_commuting_ansatz_matrix(params, sols):
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


def remove_solution_free_parameters(params, sols):
    indices = [0 if h in sols else 1 if h else 0 for h in params]
    return indices


def multikron(matrices):
    product = matrices[0]
    for mat in matrices[1:]:
        product = np.kron(product, mat)
    return product


def multicontrol_unitary(local_unitary, num_controls):
    dim = 2**(num_controls+1)
    full_unitary = np.eye(dim)
    indices = [dim - 2, dim - 1]
    full_unitary[np.ix_(indices, indices)] = local_unitary
    return full_unitary

def qft_unitary(num_qubits):
    w = np.exp(1.j * 2 * np.pi / 2 ** num_qubits)
    qft_unitary = (1 / np.sqrt(2 ** num_qubits)) * np.array([[w ** (i * j) for i in range(2 ** num_qubits)] for j in range(2 ** num_qubits)])
    return qft_unitary

def golden_section_search_np(f, a, b, tol=1e-5):
    """Golden-section search.

    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.

    Example:
    f = lambda x: (x-2)**2
    a = 1
    b = 5
    tol = 1e-5
    (c,d) = gss(f, a, b, tol)
    print(c, d)
    1.9999959837979107 2.0000050911830893
    source: https://en.wikipedia.org/wiki/Golden-section_search
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
        if yc > yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c)
        else:
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


def golden_section_search(f, a_init, b_init, tol=1e-5):
    """Golden-section search.

    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.

    Example:
    f = lambda x: (x-2)**2
    a = 1
    b = 5
    tol = 1e-5
    (c,d) = gss(f, a, b, tol)
    print(c, d)
    1.9999959837979107 2.0000050911830893
    source: https://en.wikipedia.org/wiki/Golden-section_search
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
            a, b, x1, x2, f1, f2, i = s
            b_new = x2
            x2_new = x1
            f2_new = f1
            x1_new = a + resphi * (b_new - a)
            f1_new = f(x1_new)
            return (a, b_new, x1_new, x2_new, f1_new, f2_new, i + 1)

        def right_branch(s):
            a, b, x1, x2, f1, f2, i = s
            a_new = x1
            x1_new = x2
            f1_new = f2
            x2_new = a_new + phi * (b - a_new)
            f2_new = f(x2_new)
            return (a_new, b, x1_new, x2_new, f1_new, f2_new, i + 1)

        return jax.lax.cond(f1 > f2, left_branch, right_branch, state)

    a, b, x1, x2, f1, f2, i = jax.lax.while_loop(cond_fun, body_fun, state0)

    t_best = jnp.where(f1 > f2, x1, x2)
    f_best = jnp.where(f1 > f2, f1, f2)
    return t_best, f_best


def merge_constraints(constraints, rtol=1e-9, atol=1e-12):
    """
    constraints: list of 1D sequences, all same length
    returns: list of merged constraints (lists) with no overlaps
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