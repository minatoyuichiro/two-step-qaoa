"""S5 (and the hardened version of S4): harden non-overlapping groups in formulation library instances. Training uses the same INTERP method as S1b (pipe.train_interp), p <= 5.
In problems where the groups cover all variables, such as coloring, everything can be hardened (no remaining penalty). The description of S4 is carried over below.

S4: In problems where groups overlap (assignment, TSP, etc.), harden only the non-overlapping groups and leave the rest as penalties (partial hardening).

One = one instance (name, size, seed) created by qrandom of formulation library x method x number of stage 2 layers p.
The problem and the ladder of penalty weights A (multiply from A_start, upper bound A_safe) are the formulation library qsolve problem class itself.
Correct constraints = the groups for which add_onehot / add_sum_eq (no coefficients, integer K) were called during construction (same recording as S3).
  cost = whole QUBO - all recorded penalties (= pure cost). Feasible = satisfies all recorded constraints. Optimal = minimum cost among feasible.
Methods:
  penalty_ladder : penalty QAOA (|+>, X mixer, whole QUBO). Raise A with the same ladder as formulation library.
  partial_ladder : harden non-overlapping groups (same as the groups formulation library chooses to harden) with XY. Stage 1 uses the table angles per group (0 training),
                   stage 2 cost is "whole QUBO - penalties of hardened groups" (remaining penalties unchanged). A is laddered the same way.
                   Stage 1 is looked up from a prebuilt table (results/s2). If the size is not in the table, stop.
  uniform_feasible : draw feasible solutions uniformly (classical baseline).
  NOTE: In training, divide the Hamiltonian by the maximum absolute value. In the ladder, use the extraction used for the decision as the answer and count all.
                   Training uses the post-selection expected value over the hardened groups, and shots are counted multiplied by 1/P(all hardened groups satisfied).
Cost is counted the same as S1 (1 evaluation = 1000 shots, gradients by parameter shift).
Usage: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/path/to/formulation_library:. python3 exp_s5.py out.json name:size [name:size ...]
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
SEEDS = tuple(int(s) for s in __import__("os").environ.get("SEEDS", "1,2,3,4,5").split(","))   # Limit with SEEDS=3 etc. when running over instances one by one
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
    """Look up stage 1 for group sizes m, k from a prebuilt fixed table (results/s2/table_*.json) (minimum p1 with P_k >= 0.99.
    Otherwise the best in the table). If not in the table, stop (do not build on the fly). Return the amplitudes for each weight w."""
    if (m, k) not in _s1cache:
        rows = [r for f in TABLES for r in json.load(open(f)) if r['n'] == m and r['k'] == k]
        if not rows:
            raise SystemExit(f'Stage 1 table has no ({m}, {k}). Build it first with exp_s2_table.py')
        ok = sorted([r for r in rows if r['P_k'] >= 0.99], key=lambda r: r['p1'])
        r = ok[0] if ok else max(rows, key=lambda r: r['P_k'])
        S = sym.Sym(m)
        amp = S.stage1(torch.tensor(r['params'], dtype=torch.float64), k, r['p1']).detach()
        _s1cache[(m, k)] = (r['p1'], r['P_k'], amp)
    return _s1cache[(m, k)]


def product_state(F, groups, ks):
    """Stage 1 symmetric state for each hardened group, remaining qubits in |+>. The whole is their product."""
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
    """coloring_dense is a dense graph (edge probability 0.7, 3 colors) passed to the coloring class of formulation library. The random graphs of formulation library are sparse, and
there are too many proper colorings to serve as a benchmark. Otherwise use qrandom of formulation library as is."""
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
