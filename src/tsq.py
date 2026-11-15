"""Minimal implementation of Two-Step QAOA (run with blueqat SDK GitHub version 2.2.0).

Bit order: blueqat statevector uses index = sum_i x_i 2^i (q0 is the least significant).
The QUBO is held as f(x) = x^T Q x (Q symmetric, diagonal gives linear coefficients).
"""
import itertools
import math

import numpy as np
import torch
from blueqat import Circuit

BACKEND = 'statevector'


# ---------------------------------------------------------------- problem
def bits_table(n):
    """Row = basis index, column = value x_i of qubit i."""
    idx = np.arange(2 ** n)
    return ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)


def qubo_values(Q, X):
    return np.einsum('bi,ij,bj->b', X, Q, X)


def to_ising(Q):
    """With x = (1 - z)/2, x^T Q x = const + sum h_i z_i + sum_{i<j} J_ij z_i z_j."""
    Q = np.asarray(Q, float)
    n = len(Q)
    S = (Q + Q.T) / 2
    h = np.zeros(n)
    J = {}
    const = 0.0
    for i in range(n):
        const += S[i, i] / 2
        h[i] -= S[i, i] / 2
    for i in range(n):
        for j in range(i + 1, n):
            w = 2 * S[i, j]          # coefficient of x_i x_j
            const += w / 4
            h[i] -= w / 4
            h[j] -= w / 4
            J[(i, j)] = J.get((i, j), 0.0) + w / 4
    return const, h, J


def khot_penalty(n, groups, ks):
    """QUBO matrix of sum_g (sum_{i in g} x_i - k_g)^2 (constant dropped)."""
    P = np.zeros((n, n))
    for g, k in zip(groups, ks):
        for i in g:
            P[i, i] += 1 - 2 * k
            for j in g:
                if j != i:
                    P[i, j] += 1
    return P


def feasible_mask(X, groups, ks):
    m = np.ones(len(X), bool)
    for g, k in zip(groups, ks):
        m &= X[:, list(g)].sum(1) == k
    return m


# ---------------------------------------------------------------- circuit components
def add_cost(c, h, J, gamma):
    """exp(-i gamma H_ising)。rz(t)=exp(-i t Z/2), rzz(t)=exp(-i t ZZ/2)。"""
    for i, hi in enumerate(h):
        if hi != 0:
            c.rz(2 * gamma * hi)[i]
    for (i, j), w in J.items():
        if w != 0:
            c.rzz(2 * gamma * w)[i, j]
    return c


def add_xmixer(c, qubits, beta):
    for i in qubits:
        c.rx(2 * beta)[i]
    return c


def xy_pairs(group, kind):
    g = list(group)
    if len(g) < 2:
        return []
    if kind == 'complete':
        return list(itertools.combinations(g, 2))
    if kind == 'ring':
        ring = [(g[a], g[(a + 1) % len(g)]) for a in range(len(g))]
        if len(g) == 2:
            ring = ring[:1]
        # even/odd order (parity ring)
        return ring[0::2] + ring[1::2]
    raise ValueError(kind)


def check_disjoint(groups):
    seen = set()
    for g in groups:
        if seen & set(g):
            raise ValueError('groups overlap. The per-group XY mixer does not preserve the weight of overlapping groups')
        seen |= set(g)


def add_xymixer(c, groups, beta, kind='ring', compiled=False):
    """Product of exp(-i beta (XX+YY)/2) applied per group in the order of xy_pairs (not an exact exponential of the sum of XY terms.
    Adjacent pairs do not commute, so this is an ansatz as an 'ordered product').
    With rxx(t)=exp(-i t XX/2), ryy(t)=exp(-i t YY/2), XX and YY commute.
    Each factor preserves the weight of group g. Only for non-overlapping groups is the k-hot property of every group exactly preserved.
    If compiled=True, build each pair from 2 CNOTs (xy_pair_2cx; equivalence to the same unitary is checked in test_tsq)."""
    check_disjoint(groups)
    for g in groups:
        for i, j in xy_pairs(g, kind):
            if compiled:
                xy_pair_2cx(c, beta, i, j)
            else:
                c.rxx(beta)[i, j]
                c.ryy(beta)[i, j]
    return c


def xy_pair_2cx(c, beta, i, j):
    """exp(-i beta (XX+YY)/2) with 2 CNOTs. Build CNOT·(Rx⊗Rz)·CNOT = exp(-i(XX+ZZ)beta/2),
    then apply Rx(π/2) to both to turn ZZ into YY."""
    h = math.pi / 2
    c.rx(h)[i].rx(h)[j].cx[i, j].rx(beta)[i].rz(beta)[j].cx[i, j].rx(-h)[i].rx(-h)[j]
    return c


def dicke_vector(n, groups, ks):
    """Product of Dicke states D(|g|, k) for each group (qubits not in any group are |+>)."""
    X = bits_table(n)
    amp = np.ones(2 ** n)
    covered = set()
    for g, k in zip(groups, ks):
        ok = X[:, list(g)].sum(1) == k
        amp *= ok / math.sqrt(math.comb(len(g), k))
        covered |= set(g)
    for i in range(n):
        if i not in covered:
            amp *= 1 / math.sqrt(2)
    return torch.tensor(amp, dtype=torch.complex128)


# ---------------------------------------------------------------- approach
def run_state(n, build, initial=None):
    c = Circuit(n)
    build(c)
    kw = {'initial': initial} if initial is not None else {}
    return c.statevector(backend=BACKEND, **kw)


def state_penalty_qaoa(n, ising_total, params, p):
    """Penalty QAOA: |+>^n, X mixer, H = C + lam P. params = [g1..gp, b1..bp]."""
    _, h, J = ising_total

    def build(c):
        c.h[:]
        for l in range(p):
            add_cost(c, h, J, params[l])
            add_xmixer(c, range(n), params[p + l])
    return run_state(n, build)


def state_xy_dicke(n, groups, ks, ising_cost, params, p, kind='ring'):
    """Constraint-preserving QAOA (Hadfield type): from an exact Dicke product state, XY mixer, H = C."""
    _, h, J = ising_cost
    covered = set(i for g in groups for i in g)
    free = [i for i in range(n) if i not in covered]

    def build(c):
        for l in range(p):
            add_cost(c, h, J, params[l])
            add_xymixer(c, groups, params[p + l], kind)
            if free:
                add_xmixer(c, free, params[p + l])
    return run_state(n, build, initial=dicke_vector(n, groups, ks))


def state_two_step(n, groups, ks, ising_pen, ising_cost, params, p1, p2, kind='ring'):
    """Two-Step QAOA (v2 equations (19)(20)).
    Stage 1: from |+>^n with X mixer, H_C1 = sum (sum x - k)^2 for p1 layers.
    Stage 2: from the stage 1 output with XY mixer, H_C2 = C for p2 layers.
    params = [g1(p1), b1(p1), g2(p2), b2(p2)]."""
    _, hp, Jp = ising_pen
    _, hc, Jc = ising_cost
    covered = set(i for g in groups for i in g)
    free = [i for i in range(n) if i not in covered]

    def build(c):
        c.h[:]
        for l in range(p1):
            add_cost(c, hp, Jp, params[l])
            add_xmixer(c, range(n), params[p1 + l])
        o = 2 * p1
        for l in range(p2):
            add_cost(c, hc, Jc, params[o + l])
            add_xymixer(c, groups, params[o + p2 + l], kind)
            if free:
                add_xmixer(c, free, params[o + p2 + l])
    return run_state(n, build)


# ---------------------------------------------------------------- metrics
def metrics(probs, fvals, feas):
    """1 sample = 1 bitstring from 1 measurement.
    P_feas: probability of satisfying constraints / P_opt: probability of the constrained optimum (no post-selection, unconditional value) /
    r_feas: energy normalized among feasible samples (Cmax - <C>)/(Cmax - Cmin), 1 is the optimum and 0 is the worst feasible solution.
            NOTE: This is not a multiplicative approximation ratio. It also does not include the cost of post-selection (1/P_feas trials)."""
    p = np.asarray(probs, float)
    fv = fvals[feas]
    cmin, cmax = fv.min(), fv.max()
    opt = feas & (np.abs(fvals - cmin) <= 1e-9)   # optimal = within 1e-9 of the constrained minimum (not judged by relative error)
    pf = p[feas].sum()
    r = np.nan
    if pf > 1e-12 and cmax > cmin:
        r = (cmax - (p[feas] * fv).sum() / pf) / (cmax - cmin)
    return dict(P_feas=float(pf), P_opt=float(p[opt].sum()), r_feas=float(r))


def optimize(loss_fn, n_params, restarts=8, iters=200, seed=0, lr=0.05, init_scale=0.5):
    """Minimize with Adam over several initial values. Return the best (loss, params)."""
    g = torch.Generator().manual_seed(seed)
    best = (math.inf, None)
    for _ in range(restarts):
        x = (torch.rand(n_params, generator=g, dtype=torch.float64) * 2 - 1) * init_scale
        x.requires_grad_(True)
        opt = torch.optim.Adam([x], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            L = loss_fn(x)
            L.backward()
            opt.step()
        with torch.no_grad():
            L = float(loss_fn(x))
        if L < best[0]:
            best = (L, x.detach().clone())
    return best


# ---------------------------------------------------------------- fast equivalent (for sweeping)
# Compute the same circuit as blueqat directly via diagonal phase multiplication, 1-qubit rotations, and XY two-level rotations.
# Run the optimization here, then pass the obtained parameters through the blueqat version to confirm they match (verify_against_blueqat).
class Fast:
    """NOTE: xymix handles only non-overlapping groups (same as add_xymixer)."""
    def __init__(self, n):
        self.n = n
        self.X = bits_table(n)
        self.idx = np.arange(2 ** n)

    def diag(self, h, J, const=0.0):
        z = 1 - 2 * self.X
        e = const + z @ np.asarray(h) + sum(w * z[:, i] * z[:, j] for (i, j), w in J.items())
        return torch.tensor(e, dtype=torch.float64)

    def plus(self):
        return torch.full((2 ** self.n,), 2 ** (-self.n / 2), dtype=torch.complex128)

    @staticmethod
    def cost(s, d, gamma):
        return s * torch.exp(-1j * gamma * d)

    def xmix(self, s, qubits, beta):
        n = self.n
        t = s.reshape((2,) * n)
        c, sn = torch.cos(beta), -1j * torch.sin(beta)
        for q in qubits:
            ax = n - 1 - q             # axis 0 of reshape is q_{n-1}
            a0, a1 = t.select(ax, 0), t.select(ax, 1)
            t = torch.stack([c * a0 + sn * a1, sn * a0 + c * a1], dim=ax)
        return t.reshape(-1)

    def xymix(self, s, groups, beta, kind='ring'):
        """exp(-i beta (XX+YY)/2) is a rotation between |01>,|10>: [[cos, -i sin],[-i sin, cos]]."""
        check_disjoint(groups)
        c, sn = torch.cos(beta), -1j * torch.sin(beta)
        for g in groups:
            for i, j in xy_pairs(g, kind):
                xi, xj = self.X[:, i], self.X[:, j]
                a = torch.as_tensor(np.where((xi == 1) & (xj == 0))[0])   # |..1_i 0_j..>
                b = a + (1 << j) - (1 << i)                                # pairwise |..0_i 1_j..>
                sa, sb = s[a], s[b]
                s = s.index_copy(0, a, c * sa + sn * sb).index_copy(0, b, sn * sa + c * sb)
        return s


def fast_penalty_qaoa(F, d_total, params, p):
    s = F.plus()
    for l in range(p):
        s = F.xmix(F.cost(s, d_total, params[l]), range(F.n), params[p + l])
    return s


def fast_xy(F, groups, d_cost, params, p, s0, kind='ring'):
    covered = set(i for g in groups for i in g)
    free = [i for i in range(F.n) if i not in covered]
    s = s0
    for l in range(p):
        s = F.xymix(F.cost(s, d_cost, params[l]), groups, params[p + l], kind)
        if free:
            s = F.xmix(s, free, params[p + l])
    return s


def fast_two_step(F, groups, d_pen, d_cost, params, p1, p2, kind='ring'):
    s = fast_penalty_qaoa(F, d_pen, params[:2 * p1], p1)
    return fast_xy(F, groups, d_cost, params[2 * p1:], p2, s, kind)


def portfolio_instance(seed, n):
    """Experiment 2 problem: 0.5 x^T S x - mu^T x, S = A A^T / n, mu ~ U(0,1)."""
    r = np.random.default_rng(1000 + seed)
    A = r.normal(size=(n, n))
    S = A @ A.T / n
    mu = r.uniform(0, 1, n)
    return 0.5 * S - np.diag(mu)
