"""QUBO の行列（quad・lin）だけから、k-hot の罰則 A(Σ_{i∈g} x_i − k)² を見つける。

罰則は、組 g の中の対すべてに同じ結合 2A を、各変数の一次の項に A(1 − 2k) を足す。
手順:
 0. 正の結合を大きい順に並べ、最大の段差より下の値は罰則の候補にしない（罰則はコストより大きく選ばれる前提）。
 1. 正の結合を値ごとに束ねる（相対誤差 REL 以内は同じ値とみなす）。値 v の対だけで作ったグラフの
    極大クリーク（大きさ 2 以上）を、A = v/2 の候補にする。
    コストが同じ対に足されて値がばらつく場合（ポートフォリオなど）に備え、結合を大きい順に並べて
    最大の段差より上の対でも同じことをする。
 2. 候補ごとに、一次の項を「候補の罰則の和 ＋ 残り」と見て k を最小二乗で解く。
    整数から KTOL 以内で、1 ≤ k ≤ |g| − 1 のものだけを残す。
 3. 固める組を選ぶ: 重ならない組を、大きいものから貪欲に取る（重なる組は罰則のまま残す）。
⚠️ k は行列だけからは原理的に決まらない: A(Σx−k)² = A(Σx−k')² ＋ 2A(k'−k)Σx ＋ 定数 なので、
   コストの一次の項の「組全体に一様な部分」と k を分けられない。ここでは「コストの一次の項は A より十分小さい」と仮定して丸める。
⚠️ 見つけたものは「罰則の形をしている」だけで、本当に制約かどうかは利用者が決める（式を渡せば、この推測は要らない）。
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
    # 罰則はふつう一番大きい結合（A はコストより大きく選ぶ）。正の結合を大きい順に並べた最大の段差より下は候補にしない
    sv = np.sort(np.array(list(pos.values())))
    cut = sv[int(np.argmax(sv[1:] / np.maximum(sv[:-1], 1e-12))) + 1] if len(sv) >= 2 else sv[0]
    # 1a. 値ごと
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
    # 1b. 最大の段差より上（値がばらつく場合）
    if len(sv) >= 2:
        pairs = [e for e, v in pos.items() if v >= cut]
        for c in _cliques(pairs):
            ws = [pos.get((min(a, b), max(a, b)), 0.0) for a, b in itertools.combinations(c, 2)]
            out.setdefault(c, float(np.median(ws)) / 2)
    return [(c, A) for c, A in out.items()]


def fit_k(cands, lin, n):
    """一次の項 lin_i ≈ Σ_g A_g (1 − 2 k_g) [i∈g] ＋ 残り を、k_g について最小二乗で解く。"""
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
    """重ならない組を、大きいものから貪欲に選ぶ（同じ大きさなら A の大きいもの）。"""
    used, out = set(), []
    for c, k, A, kf in sorted(found, key=lambda t: (-len(t[0]), -t[2])):
        if not (used & c):
            out.append((c, k, A, kf))
            used |= c
    return out
