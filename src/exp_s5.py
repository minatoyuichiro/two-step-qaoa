"""S5（と S4 の強化版）: 定式化ライブラリ の問題で、重ならない組を固める。学習は S1b と同じ INTERP 法（pipe.train_interp）、p ≤ 5。
彩色のように組が全変数を覆う問題では、全部を固められる（残る罰則なし）。以下は S4 の説明をそのまま引き継ぐ。

S4: 組が重なる問題（割当・TSP など）で、重ならない組だけを固め、残りは罰則のまま残す（一部だけ固める）。

1 つ = 定式化ライブラリ の qrandom が作った問題 1 つ（名前・大きさ・種）× 方式 × 段2 の層の数 p。
問題も罰則の重み A の段階づけ（A_start から倍、上限 A_safe）も、定式化ライブラリ の qsolve の問題クラスそのもの。
正解の制約 = 組み立ての途中で add_onehot / add_sum_eq（係数なし・K が整数）が呼ばれた組（S3 と同じ記録の仕方）。
  cost = QUBO 全体 − 記録した罰則すべて（= 純粋なコスト）。可行 = 記録した制約すべてを満たす。最適 = 可行の中で cost が最小。
方式:
  penalty_ladder : 素の QAOA（|+>・X ミキサー・QUBO 全体）。定式化ライブラリ と同じ段階づけで A を上げる。
  partial_ladder : 重ならない組（定式化ライブラリ が固めに選ぶ groups と同じ）を XY で固める。段1 は組ごとに表の角度（学習 0 回）、
                   段2 のコストは「QUBO 全体 − 固めた組の罰則」（残りの罰則はそのまま）。A は同じく段階づけ。
                   段1 は前もって作った表（results/s2）から引く。表に無い大きさなら止まる。
  uniform_feasible : 可行解を一様に引く（古典の基準）。
  ※ 学習ではハミルトニアンを最大絶対値で割る。段階づけでは、判定に使った抽出をそのまま答えにし、全部数える。
                   学習は固めた組について事後選択した期待値で行い、ショットは 1/P(固めた組がすべて満たされる) 倍で数える。
手間の数え方は S1 と同じ（評価 1 回 = 1000 ショット、勾配はパラメータシフト）。
使い方: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/path/to/formulation_library:. python3 exp_s5.py out.json name:size [name:size ...]
"""
import json
import math
import sys
import time

import numpy as np
import torch

import qopt_lite as qo
import qrandom
import qsolve

import pipe
import sym
import tsq

calls = []
_onehot, _sumeq = qo.add_onehot, qo.add_sum_eq


def rec_onehot(quad, linear, idxs, A):
    idxs = list(idxs)
    calls.append((tuple(idxs), 1, float(A)))
    return _onehot(quad, linear, idxs, A)


def rec_sumeq(quad, linear, idxs, K, A, coef=None):
    idxs = list(idxs)
    if coef is None and float(K).is_integer():
        calls.append((tuple(idxs), int(K), float(A)))
    return _sumeq(quad, linear, idxs, K, A, coef=coef)


qo.add_onehot, qo.add_sum_eq = rec_onehot, rec_sumeq

OUT = sys.argv[1]
JOBS = [(a.split(':')[0], int(a.split(':')[1])) for a in sys.argv[2:]]
SEEDS = tuple(int(s) for s in __import__("os").environ.get("SEEDS", "1,2,3,4,5").split(","))   # 例ごとに並べて回すとき SEEDS=3 などで絞る
PMAX = 5
S_EVAL, S_FINAL = 1000, 1000
RESTARTS, ITERS, LR = 4, 150, 0.05
MAXQ = 16


def build(name, data, A=None):
    calls.clear()
    P = qsolve.PROBLEMS[name](data) if A is None else qsolve.PROBLEMS[name](data, A=A)
    return P, list(calls)


def qubo_matrix(P):
    n = P.n
    Q = np.zeros((n, n))
    for i, v in P.linear.items():
        Q[i, i] += v
    for (i, j), v in P.quad.items():
        Q[i, j] += v / 2
        Q[j, i] += v / 2
    return Q


def penalty_matrix(n, cons):
    M = np.zeros((n, n))
    const = 0.0
    for g, k, A in cons:
        M += A * tsq.khot_penalty(n, [g], [k])
        const += A * k * k
    return M, const


def train(loss, n_params, seed):
    g = torch.Generator().manual_seed(seed)
    best = (math.inf, None)
    for _ in range(RESTARTS):
        x = ((torch.rand(n_params, generator=g, dtype=torch.float64) * 2 - 1) * 0.5).requires_grad_(True)
        opt = torch.optim.Adam([x], lr=LR)
        for _ in range(ITERS):
            opt.zero_grad()
            L = loss(x)
            if float(L.detach()) < best[0]:
                best = (float(L.detach()), x.detach().clone())
            L.backward()
            opt.step()
    return best[1], RESTARTS * ITERS * (1 + 2 * n_params)


_s1cache = {}
TABLES = sorted(__import__('glob').glob(__import__('os').path.join(__import__('os').path.dirname(__file__) or '.', '../results/s2/table_*.json')))


def stage1_weights(m, k):
    """組の大きさ m・k の段1 を、前もって作って固定した表（results/s2/table_*.json）から引く（P_k ≥ 0.99 の最小 p1。
    無ければ表の最良）。表に無ければ止まる（その場で作らない）。重み w ごとの振幅を返す。"""
    if (m, k) not in _s1cache:
        rows = [r for f in TABLES for r in json.load(open(f)) if r['n'] == m and r['k'] == k]
        if not rows:
            raise SystemExit(f'段1 の表に ({m}, {k}) が無い。先に exp_s2_table.py で作ること')
        ok = sorted([r for r in rows if r['P_k'] >= 0.99], key=lambda r: r['p1'])
        r = ok[0] if ok else max(rows, key=lambda r: r['P_k'])
        S = sym.Sym(m)
        amp = S.stage1(torch.tensor(r['params'], dtype=torch.float64), k, r['p1']).detach()
        _s1cache[(m, k)] = (r['p1'], r['P_k'], amp)
    return _s1cache[(m, k)]


def product_state(F, groups, ks):
    """固めた組ごとに段1 の対称状態、残りの量子ビットは |+>。全体はその積。"""
    n = F.n
    amp = torch.ones(2 ** n, dtype=torch.complex128)
    covered = set()
    for g, k in zip(groups, ks):
        p1, pk, a = stage1_weights(len(g), k)
        w = F.X[:, list(g)].sum(1)
        amp = amp * a[torch.tensor(w)] / torch.tensor([math.sqrt(math.comb(len(g), int(x))) for x in w], dtype=torch.complex128)
        covered |= set(g)
    free = n - len(covered)
    return amp * 2 ** (-free / 2)


def make_data(name, size, seed):
    """coloring_dense は、定式化ライブラリ の彩色クラスに渡す密なグラフ（辺の確率 0.7、3 色）。定式化ライブラリ の乱数のグラフは疎で、
    正しい塗り分けが多すぎて物差しにならないため。それ以外は 定式化ライブラリ の qrandom をそのまま使う。"""
    if name == 'coloring_dense':
        import random
        r = random.Random(seed)
        edges = [[i, j] for i in range(size) for j in range(i + 1, size) if r.random() < 0.7]
        return {'n': size, 'edges': edges, 'colors': 3}
    return qrandom.generate(name, size, seed)


rows = []
for name, size in JOBS:
    qname = 'coloring' if name == 'coloring_dense' else name
    for seed in SEEDS:
        data = make_data(name, size, seed)
        P0, cons0 = build(qname, data)
        n = P0.n
        if n > MAXQ or P0.n != getattr(P0, 'n_core', P0.n):
            print(f'skip {name}:{size} n={n}', flush=True)
            break
        F = tsq.Fast(n)
        cons_t = [(g, k) for g, k, A in cons0 if len(g) >= 2 and 1 <= k <= len(g) - 1]
        feas = tsq.feasible_mask(F.X, [g for g, k in cons_t], [k for g, k in cons_t])
        Qf = qubo_matrix(P0)
        M0, c0 = penalty_matrix(n, cons0)
        cost = tsq.qubo_values(Qf - M0, F.X) + P0.const - c0
        P2, cons2 = build(qname, data, A=P0.A_start * 3.7)
        M2, c2 = penalty_matrix(n, cons2)
        cost2 = tsq.qubo_values(qubo_matrix(P2) - M2, F.X) + P2.const - c2
        assert np.abs(cost - cost2).max() < 1e-6 * max(1.0, np.abs(cost).max()), (name, np.abs(cost - cost2).max())
        if feas.sum() == 0:
            continue
        cmin = cost[feas].min()
        optm = feas & (np.abs(cost - cmin) <= 1e-9)
        hard = [tuple(g) for g in getattr(P0, 'groups', []) if len(g) >= 2]
        hard_k = [1] * len(hard)
        hard_t = torch.tensor(tsq.feasible_mask(F.X, hard, hard_k))
        s_hard0 = product_state(F, hard, hard_k)
        P_hard = float((s_hard0.abs() ** 2)[hard_t].sum())
        ladder = []
        a = P0.A_start
        while a < P0.A_safe:
            ladder.append(a); a *= 2
        ladder.append(P0.A_safe)
        covers_all = len(set(i for g in hard for i in g)) == n and len(cons_t) == len(hard)
        base = dict(name=name, size=size, seed=seed, n=n, n_constraints=len(cons_t), n_hard=len(hard), covers_all=covers_all,
                    P_hard_stage1=P_hard, n_feasible=int(feas.sum()), n_optimal=int(optm.sum()), A_start=P0.A_start, A_safe=P0.A_safe)

        def draw(s, tag):
            pr = (s.abs() ** 2).detach().numpy(); pr = pr / pr.sum()
            return np.random.default_rng(seed * 100003 + tag).choice(len(pr), size=S_FINAL, p=pr)

        def finish(method, s, shots, p, smp, **kw):
            pr = (s.abs() ** 2).detach().numpy(); pr = pr / pr.sum()
            ok = smp[feas[smp]]
            gap = float(cost[ok].min() - cmin) if len(ok) else None
            po, pf = float(pr[optm].sum()), float(pr[feas].sum())
            row = {**base, 'method': method, 'p': p, **kw, 'P_feas': pf, 'P_opt': po, 'P_opt_given_feas': po / max(pf, 1e-300),
                   'shots_total': shots, 'answer_feasible': len(ok) > 0, 'answer_gap': gap,
                   'P_batch_hits_opt': 1 - (1 - po) ** S_FINAL,
                   'shots_to_opt_95': (math.ceil(math.log(0.05) / math.log(1 - po)) if 0 < po < 1 else None),
                   'measured': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            rows.append(row)
            print(json.dumps({a: row[a] for a in ('name', 'seed', 'method', 'p', 'P_feas', 'P_opt', 'shots_total')}), flush=True)
            json.dump(rows, open(OUT, 'w'), indent=1)

        unif = torch.tensor(np.where(feas, 1.0, 0.0) / math.sqrt(feas.sum()), dtype=torch.complex128)
        finish('uniform_feasible', unif, S_FINAL, 0, draw(unif, 1))

        def states(method, A, tag):
            PA, consA = build(qname, data, A=A)
            QA = qubo_matrix(PA)
            if method == 'penalty_ladder':
                d = torch.tensor(tsq.qubo_values(QA, F.X) + PA.const)
                d = d / d.abs().max()
                res = pipe.train_interp(lambda x, p: (tsq.fast_penalty_qaoa(F, d, x, p).abs() ** 2) @ d, PMAX, seed * 10 + tag)
                return {p: (tsq.fast_penalty_qaoa(F, d, x, p).detach(), ev * S_EVAL) for p, (x, ev) in res.items()}
            Mh, ch = penalty_matrix(n, [(g, k, A_) for g, k, A_ in consA if tuple(g) in set(hard)])
            d = torch.tensor(tsq.qubo_values(QA - Mh, F.X) + PA.const - ch)
            d = d / d.abs().max()

            def L(x, p):
                pr = tsq.fast_xy(F, hard, d, x, p, s_hard0).abs() ** 2
                return (pr * hard_t) @ d / (pr * hard_t).sum()
            res = pipe.train_interp(L, PMAX, seed * 10 + tag)
            return {p: (tsq.fast_xy(F, hard, d, x, p, s_hard0).detach(), ev * S_EVAL / P_hard) for p, (x, ev) in res.items()}

        for method, tag in (('penalty_ladder', 1), ('partial_ladder', 2)):
            rungs, stop = [], {}
            for A in ladder:
                st = states(method, A, tag)
                rungs.append((A, st))
                for p in range(1, PMAX + 1):
                    if p in stop:
                        continue
                    smp = draw(st[p][0], 1000 * tag + 10 * len(rungs) + p)
                    if feas[smp].any() or A == ladder[-1]:
                        stop[p] = (st[p][0], sum(r[1][p][1] + S_FINAL for r in rungs), smp, A, len(rungs))
                if len(stop) == PMAX:
                    break
            for p, (s, shots, smp, Au, nr) in sorted(stop.items()):
                finish(method if not covers_all or method == 'penalty_ladder' else 'two_step_all', s, shots, p, smp, A_used=Au, rungs=nr)
