"""Two-Step QAOA の最小実装（blueqat SDK の GitHub 版 2.2.0 で回す）。

ビットの向き: blueqat の statevector は index = sum_i x_i 2^i（q0 が最下位）。
QUBO は f(x) = x^T Q x（Q は対称、対角が一次の係数）で持つ。
"""
import itertools
import math

import numpy as np
import torch
from blueqat import Circuit

BACKEND = 'statevector'


# ---------------------------------------------------------------- 問題
def bits_table(n):
    """行 = 基底 index、列 = 量子ビット i の値 x_i。"""
    idx = np.arange(2 ** n)
    return ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)


def qubo_values(Q, X):
    return np.einsum('bi,ij,bj->b', X, Q, X)


def to_ising(Q):
    """x = (1 - z)/2 として x^T Q x = const + sum h_i z_i + sum_{i<j} J_ij z_i z_j。"""
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
            w = 2 * S[i, j]          # x_i x_j の係数
            const += w / 4
            h[i] -= w / 4
            h[j] -= w / 4
            J[(i, j)] = J.get((i, j), 0.0) + w / 4
    return const, h, J


def khot_penalty(n, groups, ks):
    """sum_g (sum_{i in g} x_i - k_g)^2 の QUBO 行列（定数は捨てる）。"""
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


# ---------------------------------------------------------------- 回路の部品
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
        # 偶・奇の順（parity ring）
        return ring[0::2] + ring[1::2]
    raise ValueError(kind)


def check_disjoint(groups):
    seen = set()
    for g in groups:
        if seen & set(g):
            raise ValueError('組が重なっている。組ごとの XY ミキサーは、重なる組の重みを保たない')
        seen |= set(g)


def add_xymixer(c, groups, beta, kind='ring', compiled=False):
    """各組で exp(-i beta (XX+YY)/2) を、xy_pairs の順に掛けた積（XY 項の和の厳密な指数ではない。
    隣り合う対は可換でないので、これは「順序つきの積」という ansatz）。
    rxx(t)=exp(-i t XX/2), ryy(t)=exp(-i t YY/2) で、XX と YY は可換。
    各因子は組 g の重みを保つ。重なりの無い組に限り、どの組の k-hot も厳密に保たれる。
    compiled=True なら 1 対を CNOT 2 個で組む（xy_pair_2cx。同じユニタリであることは test_tsq で確認）。"""
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
    """exp(-i beta (XX+YY)/2) を CNOT 2 個で。CNOT·(Rx⊗Rz)·CNOT = exp(-i(XX+ZZ)beta/2) を作り、
    両方に Rx(π/2) を掛けて ZZ を YY に回す。"""
    h = math.pi / 2
    c.rx(h)[i].rx(h)[j].cx[i, j].rx(beta)[i].rz(beta)[j].cx[i, j].rx(-h)[i].rx(-h)[j]
    return c


def dicke_vector(n, groups, ks):
    """各組が Dicke 状態 D(|g|, k) の積（組に入らない量子ビットは |+>）。"""
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


# ---------------------------------------------------------------- 方式
def run_state(n, build, initial=None):
    c = Circuit(n)
    build(c)
    kw = {'initial': initial} if initial is not None else {}
    return c.statevector(backend=BACKEND, **kw)


def state_penalty_qaoa(n, ising_total, params, p):
    """素の QAOA: |+>^n、X ミキサー、H = C + lam P。params = [g1..gp, b1..bp]。"""
    _, h, J = ising_total

    def build(c):
        c.h[:]
        for l in range(p):
            add_cost(c, h, J, params[l])
            add_xmixer(c, range(n), params[p + l])
    return run_state(n, build)


def state_xy_dicke(n, groups, ks, ising_cost, params, p, kind='ring'):
    """制約保存 QAOA（Hadfield 型）: 厳密な Dicke 積状態から、XY ミキサー、H = C。"""
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
    """Two-Step QAOA（v2 の式 (19)(20)）。
    段1: |+>^n から X ミキサー、H_C1 = sum (sum x - k)^2 を p1 層。
    段2: 段1 の出力から XY ミキサー、H_C2 = C を p2 層。
    params = [g1(p1), b1(p1), g2(p2), b2(p2)]。"""
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


# ---------------------------------------------------------------- 物差し
def metrics(probs, fvals, feas):
    """1 標本 = 1 回の測定で得たビット列 1 本。
    P_feas: 制約を満たす確率 / P_opt: 制約つき最適解の確率（事後選択しない、無条件の値）/
    r_feas: 満たした標本の中で正規化したエネルギー (Cmax - <C>)/(Cmax - Cmin)、1 が最適・0 が最悪の可行解。
            ⚠️ 掛け算の近似比ではない。事後選択の手間（1/P_feas 回）も入っていない。"""
    p = np.asarray(probs, float)
    fv = fvals[feas]
    cmin, cmax = fv.min(), fv.max()
    opt = feas & (np.abs(fvals - cmin) <= 1e-9)   # 最適 = 制約つき最小値と 1e-9 以内（相対誤差では判定しない）
    pf = p[feas].sum()
    r = np.nan
    if pf > 1e-12 and cmax > cmin:
        r = (cmax - (p[feas] * fv).sum() / pf) / (cmax - cmin)
    return dict(P_feas=float(pf), P_opt=float(p[opt].sum()), r_feas=float(r))


def optimize(loss_fn, n_params, restarts=8, iters=200, seed=0, lr=0.05, init_scale=0.5):
    """Adam で何度か初期値を変えて最小化。最良の (loss, params) を返す。"""
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


# ---------------------------------------------------------------- 速い同等品（掃引用）
# blueqat と同じ回路を、対角の位相掛け・1 量子ビット回転・XY の 2 準位回転で直接計算する。
# 最適化はこちらで回し、得たパラメータを blueqat 版に通して一致を確かめる（verify_against_blueqat）。
class Fast:
    """⚠️ xymix は重なりの無い組だけを扱う（add_xymixer と同じ）。"""
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
            ax = n - 1 - q             # reshape の軸 0 が q_{n-1}
            a0, a1 = t.select(ax, 0), t.select(ax, 1)
            t = torch.stack([c * a0 + sn * a1, sn * a0 + c * a1], dim=ax)
        return t.reshape(-1)

    def xymix(self, s, groups, beta, kind='ring'):
        """exp(-i beta (XX+YY)/2) は |01>,|10> の間の回転 [[cos, -i sin],[-i sin, cos]]。"""
        check_disjoint(groups)
        c, sn = torch.cos(beta), -1j * torch.sin(beta)
        for g in groups:
            for i, j in xy_pairs(g, kind):
                xi, xj = self.X[:, i], self.X[:, j]
                a = torch.as_tensor(np.where((xi == 1) & (xj == 0))[0])   # |..1_i 0_j..>
                b = a + (1 << j) - (1 << i)                                # 対の |..0_i 1_j..>
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
    """実験2 の問題: 0.5 x^T S x - mu^T x、S = A A^T / n、mu ~ U(0,1)。"""
    r = np.random.default_rng(1000 + seed)
    A = r.normal(size=(n, n))
    S = A @ A.T / n
    mu = r.uniform(0, 1, n)
    return 0.5 * S - np.diag(mu)
