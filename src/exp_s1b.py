"""S1b: S1 の強化版。全方式の学習を INTERP 法（pipe.train_interp）に揃え、p ≤ 5、例を増やす。

1 つ = ポートフォリオの例 1 つ（n 銘柄から k 銘柄、seed で決まる）× 方式 × 段2 の層の数 p。
手間（ショット）の数え方は S1 と同じ目安: 評価 1 回 = S_EVAL ショット、勾配はパラメータシフト 2 回/角度とみなす。
深さ p の手間には、深さ 1〜p−1 の学習も含める（INTERP は前の深さから始めるので）。
事後選択で学習する方式は、学習のショットを 1/P_k 倍で数える。どの方式も、学習ではハミルトニアンを最大絶対値で割る。
方式:
  uniform_feasible     可行解を一様に引く（古典の基準）
  penalty_ladder       素の QAOA。罰則 A は 定式化ライブラリ の Portfolio と同じ A_start から、制約を破ったら倍（上限 A_safe）。
                       深さ p ごとに「その深さの抽出で可行が出た最初の段」で止まり、そこまでの段の手間を全部足す。
                       判定に使った抽出をそのまま答えにする。
  penalty_ladder_low   同じ段階づけを A_start/16 から
  penalty_grid_deploy  A = σ·{0.03, 0.1, 0.3, 1, 3} を全部学習し、各 1000 ショットの可行で最良の目的が一番良い点を選ぶ（σ だけ神託）
  penalty_grid         同じ 5 点から、厳密な P_opt で選ぶ（神託）
  two_step             段1 は表から（学習 0 回）→ 段2 は XY リング、事後選択の <C> で学習
  bp_xy                偏った積状態（CNOT 0 個）→ XY → 事後選択
  xy_dicke             厳密な Dicke から XY（参照）
使い方: python3 exp_s1b.py n k seeds out.json table.json[,table2.json...]
"""
import json
import math
import sys
import time

import numpy as np
import torch

import pipe
import sym
import tsq

N, K = int(sys.argv[1]), int(sys.argv[2])
SEEDS = [int(s) for s in sys.argv[3].split(',')]
OUT = sys.argv[4]
TABLES = sys.argv[5].split(',')
PMAX = 5
S_EVAL, S_FINAL = 1000, 1000
GRID = (0.03, 0.1, 0.3, 1.0, 3.0)
P_TARGET = 0.99
import os
KIND = os.environ.get('XY', 'ring')          # XY ミキサーの対: ring（既定）か complete（全結合）
ONLY_XY = os.environ.get('ONLY_XY') == '1'     # 1 なら制約を保つ方式だけ回す（S1c: 全結合 XY の追加実験）


def stage1_from_table(n, k):
    rows = [r for f in TABLES for r in json.load(open(f)) if r['n'] == n and r['k'] == k]
    if not rows:
        raise SystemExit(f'段1 の表に ({n}, {k}) が無い')
    ok = sorted([r for r in rows if r['P_k'] >= P_TARGET], key=lambda r: r['p1'])
    r = ok[0] if ok else max(rows, key=lambda r: r['P_k'])
    return r['p1'], torch.tensor(r['params'], dtype=torch.float64), r['P_k']


def portfolio_A(Q, n):
    """定式化ライブラリ の Portfolio と同じ A_start・A_safe。"""
    lin = {i: Q[i, i] for i in range(n)}
    quad = {(i, j): Q[i, j] + Q[j, i] for i in range(n) for j in range(i + 1, n)}
    R = sum(abs(v) for v in lin.values()) + sum(abs(v) for v in quad.values())
    step = max(abs(lin[i]) + sum(abs(quad[(min(i, j), max(i, j))]) for j in range(n) if j != i) for i in range(n))
    return 2.0 * step, R + 1.0


rows = []
n, k = N, K
F = tsq.Fast(n)
groups, ks = [tuple(range(n))], [k]
feas = tsq.feasible_mask(F.X, groups, ks)
feas_t = torch.tensor(feas)
D = tsq.dicke_vector(n, groups, ks)
ip = tsq.to_ising(tsq.khot_penalty(n, groups, ks))
dp = F.diag(ip[1], ip[2], ip[0]) + k * k
p1, th1, _ = stage1_from_table(n, k)
s1 = tsq.fast_penalty_qaoa(F, dp, th1, p1).detach()
pk1 = float((s1.abs() ** 2)[feas_t].sum())
q = k / n
pk_bp = sym.biased_product_Pk(n, k)
bp = torch.tensor(np.prod(np.where(F.X == 1, math.sqrt(q), math.sqrt(1 - q)), axis=1), dtype=torch.complex128)

for seed in SEEDS:
    Q = tsq.portfolio_instance(seed, n)
    fvals = tsq.qubo_values(Q, F.X)
    fv = fvals[feas]
    cmin, sigma = fv.min(), fv.max() - fv.min()
    optmask = feas & (np.abs(fvals - cmin) <= 1e-9)
    ic = tsq.to_ising(Q)
    dc = F.diag(ic[1], ic[2], ic[0])
    dcn = dc / dc.abs().max()
    A_start, A_safe = portfolio_A(Q, n)
    base = dict(n=n, k=k, seed=seed, sigma=float(sigma), binding=bool(fv.min() > fvals.min() + 1e-9),
                A_start_over_sigma=A_start / sigma, n_optimal=int(optmask.sum()))

    def draw(s, tag):
        pr = (s.abs() ** 2).detach().numpy(); pr = pr / pr.sum()
        return np.random.default_rng(seed * 100003 + tag).choice(len(pr), size=S_FINAL, p=pr)

    def finish(method, s, shots, p, smp, **kw):
        pr = (s.abs() ** 2).detach().numpy(); pr = pr / pr.sum()
        m = tsq.metrics(pr, fvals, feas)
        ok = smp[feas[smp]]
        gap = float(fvals[ok].min() - cmin) if len(ok) else None
        row = {**base, 'method': method, 'p': p, **kw, **m, 'shots_total': shots,
               'answer_feasible': len(ok) > 0, 'answer_gap': gap, 'answer_optimal': gap is not None and gap <= 1e-9,
               'P_batch_hits_opt': 1 - (1 - m['P_opt']) ** S_FINAL,
               'shots_to_opt_95': (math.ceil(math.log(0.05) / math.log(1 - m['P_opt'])) if 0 < m['P_opt'] < 1 else None),
               'measured': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        rows.append(row)
        print(json.dumps({a: row[a] for a in ('seed', 'method', 'p', 'P_feas', 'P_opt', 'shots_total')}), flush=True)
        json.dump(rows, open(OUT, 'w'), indent=1)

    if not ONLY_XY:
        finish('uniform_feasible', D, S_FINAL, 0, draw(D, 1))

    def penalty_states(A, tag):
        dt = dc + A * dp
        dn = dt / dt.abs().max()
        res = pipe.train_interp(lambda x, p: (tsq.fast_penalty_qaoa(F, dn, x, p).abs() ** 2) @ dn, PMAX, seed * 10 + tag)
        return {p: (tsq.fast_penalty_qaoa(F, dn, x, p).detach(), ev * S_EVAL) for p, (x, ev) in res.items()}

    # 段階づけ（深さごとに止まる段を決める。段は必要になったときだけ学習する）
    for method, A0, tag in (() if ONLY_XY else (('penalty_ladder', A_start, 1), ('penalty_ladder_low', A_start / 16, 2))):
        rungs, A = [], A0
        stop = {}
        while len(stop) < PMAX:
            st = penalty_states(A, tag)
            rungs.append((A, st))
            for p in range(1, PMAX + 1):
                if p in stop:
                    continue
                smp = draw(st[p][0], 1000 * tag + 10 * len(rungs) + p)
                if feas[smp].any() or A >= A_safe:
                    shots = sum(r[1][p][1] + S_FINAL for r in rungs)     # 止まった段までの学習と抽出を全部
                    stop[p] = (st[p][0], shots, smp, A, len(rungs))
            if A >= A_safe:
                break
            A = min(2 * A, A_safe)
        for p, (s, shots, smp, Au, nr) in sorted(stop.items()):
            finish(method, s, shots, p, smp, A_used_over_sigma=Au / sigma, rungs=nr)

    # λ の格子
    cand = {p: [] for p in range(1, PMAX + 1)}
    for c in (() if ONLY_XY else GRID):
        st = penalty_states(c * sigma, 3)
        for p in range(1, PMAX + 1):
            smp = draw(st[p][0], 3000 + int(c * 100) * 10 + p)
            okc = fvals[smp[feas[smp]]]
            cand[p].append(dict(c=c, s=st[p][0], smp=smp, ev=st[p][1],
                                po=float((st[p][0].abs() ** 2)[torch.tensor(optmask)].sum()),
                                key=(okc.min() if len(okc) else math.inf, -len(okc))))
    for p in (() if ONLY_XY else range(1, PMAX + 1)):
        shots = sum(d['ev'] for d in cand[p]) + S_FINAL * len(GRID)
        b = max(cand[p], key=lambda d: d['po'])
        finish('penalty_grid', b['s'], shots, p, b['smp'], lam_over_sigma=b['c'])
        b = min(cand[p], key=lambda d: d['key'])
        finish('penalty_grid_deploy', b['s'], shots, p, b['smp'], lam_over_sigma=b['c'])

    # 制約を保つ方式（事後選択の <C> で学習）
    for method, s0, pk, tag in (('two_step', s1, pk1, 4), ('bp_xy', bp, pk_bp, 5), ('xy_dicke', D, 1.0, 6))[: (1 if ONLY_XY else 3)]:
        def L(x, p, s0=s0):
            pr = tsq.fast_xy(F, groups, dcn, x, p, s0, KIND).abs() ** 2
            return (pr * feas_t) @ dcn / (pr * feas_t).sum()
        res = pipe.train_interp(L, PMAX, seed * 10 + tag)
        for p, (x, ev) in res.items():
            s = tsq.fast_xy(F, groups, dcn, x, p, s0, KIND).detach()
            finish(method + ('' if KIND == 'ring' else '_' + KIND), s, ev * S_EVAL / pk + S_FINAL, p, draw(s, 4000 + 100 * tag + p), P_k_stage1=pk,
                   **({'p1': p1} if method == 'two_step' else {}), xy=KIND)
