"""Find the k-hot penalty A(Σ_{i∈g} x_i − k)² from the QUBO matrices (quad, lin) alone.

The penalty adds the same coupling 2A to every pair in group g, and A(1 − 2k) to the linear term of each variable.
Procedure:
 0. Sort positive couplings in descending order, and do not treat values below the largest gap as penalty candidates (the penalty is assumed to be chosen larger than the cost).
 1. Bucket positive couplings by value (values within relative tolerance REL are treated as the same value). For the graph made only of pairs with value v,
    take maximal cliques (size 2 or more) as candidates for A = v/2.
    In case the cost is added to pairs with the same value and the values vary (e.g. portfolio), sort couplings in descending order and
    do the same for pairs above the largest gap.
 2. For each candidate, view the linear terms as "the sum of the candidate penalties + the rest" and solve for k by least squares.
    Keep only those within KTOL of an integer with 1 ≤ k ≤ |g| − 1.
 3. Choose groups to harden: greedily take non-overlapping groups starting from the largest (overlapping groups are left as penalties).
NOTE: k cannot in principle be determined from the matrices alone: since A(Σx−k)² = A(Σx−k')² + 2A(k'−k)Σx + const,
   the "uniform part over the whole group" of the linear cost terms cannot be separated from k. Here we assume "the linear cost terms are sufficiently smaller than A" and round.
NOTE: What is found merely "looks like a penalty"; whether it is truly a constraint is decided by the user (if the formula is given, this guess is unnecessary).
"""
import itertools

import networkx as nx
import numpy as np

REL = 1e-6
KTOL = 0.05


def _cliques(pairs):
    G = nx.Graph()
    G.add_edges_from(pairs)
    return [frozenset(c) for c in nx.find_cliques(G) if len(c) >= 2]


def candidates(quad):
    pos = {e: v for e, v in quad.items() if v > 0}
    if not pos:
        return []
    out = {}
    # The penalty is usually the largest coupling (A is chosen larger than the cost). Values below the largest gap in the descending order of positive couplings are not candidates
    sv = np.sort(np.array(list(pos.values())))
    cut = sv[int(np.argmax(sv[1:] / np.maximum(sv[:-1], 1e-12))) + 1] if len(sv) >= 2 else sv[0]
    # 1a. By value
    vals = sorted(set(pos.values()))
    classes = []
    for v in vals:
        if classes and abs(v - classes[-1][0]) <= REL * max(abs(v), 1e-12):
            classes[-1][1].append(v)
        else:
            classes.append((v, [v]))
    for v0, _ in classes:
        if v0 < cut * (1 - REL):
            continue
        pairs = [e for e, v in pos.items() if abs(v - v0) <= REL * max(abs(v0), 1e-12)]
        for c in _cliques(pairs):
            if len(c) * (len(c) - 1) // 2 == sum(1 for a, b in itertools.combinations(sorted(c), 2) if (a, b) in pairs or (b, a) in pairs):
                out.setdefault(c, v0 / 2)
    # 1b. Above the largest gap (when values vary)
    if len(sv) >= 2:
        pairs = [e for e, v in pos.items() if v >= cut]
        for c in _cliques(pairs):
            ws = [pos.get((min(a, b), max(a, b)), 0.0) for a, b in itertools.combinations(c, 2)]
            out.setdefault(c, float(np.median(ws)) / 2)
    return [(c, A) for c, A in out.items()]


def fit_k(cands, lin, n):
    """Solve the linear terms lin_i ≈ Σ_g A_g (1 − 2 k_g) [i∈g] + the rest for k_g by least squares."""
    if not cands:
        return []
    M = np.zeros((n, len(cands)))
    for j, (c, A) in enumerate(cands):
        for i in c:
            M[i, j] = -2 * A
    y = np.array([lin.get(i, 0.0) for i in range(n)]) - M.sum(1) * 0  # lin = Σ A(1−2k) = Σ A − Σ 2A k
    y = y - np.array([sum(A for c, A in cands if i in c) for i in range(n)])
    rows = [i for i in range(n) if M[i].any()]
    k, *_ = np.linalg.lstsq(M[rows], y[rows], rcond=None)
    out = []
    for (c, A), kj in zip(cands, k):
        kr = round(kj)
        if abs(kj - kr) <= KTOL and 1 <= kr <= len(c) - 1:
            out.append((c, int(kr), A, float(kj)))
    return out


def detect(quad, lin, n):
    return fit_k(candidates(quad), lin, n)


def choose_disjoint(found):
    """Greedily select non-overlapping groups starting from the largest (for equal sizes, the one with the larger A)."""
    used, out = set(), []
    for c, k, A, kf in sorted(found, key=lambda t: (-len(t[0]), -t[2])):
        if not (used & c):
            out.append((c, k, A, kf))
            used |= c
    return out
