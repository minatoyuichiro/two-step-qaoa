"""Cross-check at the size of the portfolio experiments (n=10, k=5): the fast state-vector routine against blueqat circuits,
for penalty QAOA, the Dicke-initialised XY ansatz and Two-Step (stage 1 from the table), with ring and complete-graph mixers.
The check is on random stage-2 / QAOA angles (the optimised angles of the experiments were not stored)."""
import json
import numpy as np
import torch
import tsq

n, k = 10, 5
rng = np.random.default_rng(11)
F = tsq.Fast(n)
groups, ks = [tuple(range(n))], [k]
Q = tsq.portfolio_instance(0, n)
P = tsq.khot_penalty(n, groups, ks)
ic, ip = tsq.to_ising(Q), tsq.to_ising(P)
dc, dp = F.diag(ic[1], ic[2], ic[0]), F.diag(ip[1], ip[2], ip[0])
row = min((r for r in json.load(open('../results/s2/table_3.json')) if r['n'] == n and r['k'] == k and r['P_k'] >= 0.99), key=lambda r: r['p1'])
th1 = torch.tensor(row['params'], dtype=torch.float64)
p1 = row['p1']
worst = 0.0
for kind in ('ring', 'complete'):
    th2 = torch.tensor(rng.normal(size=4))
    a = tsq.state_two_step(n, groups, ks, ip, ic, torch.cat([th1, th2]), p1, 2, kind)
    b = tsq.fast_two_step(F, groups, dp, dc, torch.cat([th1, th2]), p1, 2, kind)
    f1 = abs(np.vdot(a.numpy(), b.numpy()))
    a = tsq.state_xy_dicke(n, groups, ks, ic, th2, 2, kind)
    b = tsq.fast_xy(F, groups, dc, th2, 2, tsq.dicke_vector(n, groups, ks), kind)
    f2 = abs(np.vdot(a.numpy(), b.numpy()))
    print(kind, 'two_step fidelity', f1, ' xy_dicke fidelity', f2)
    worst = max(worst, abs(f1 - 1), abs(f2 - 1))
it = tsq.to_ising(Q + 0.8 * P)
th = torch.tensor(rng.normal(size=4))
f3 = abs(np.vdot(tsq.state_penalty_qaoa(n, it, th, 2).numpy(), tsq.fast_penalty_qaoa(F, F.diag(it[1], it[2], it[0]), th, 2).numpy()))
print('penalty fidelity', f3)
assert max(worst, abs(f3 - 1)) < 1e-9
print('blueqat n=10 ok')
