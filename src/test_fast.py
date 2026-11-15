"""Fast の結果が blueqat の statevector と（大域位相を除き）一致するか。"""
import numpy as np, torch, tsq

def fid(a, b):
    f = abs(np.vdot(a.detach().numpy(), b.detach().numpy()))
    assert abs(f - 1) < 1e-10, f
    return f

rng = np.random.default_rng(3)
n = 6; groups = [(0, 1, 2), (3, 4, 5)]; ks = [1, 2]
Q = rng.normal(size=(n, n)); Q = (Q + Q.T) / 2
P = tsq.khot_penalty(n, groups, ks)
F = tsq.Fast(n)
ic, ip = tsq.to_ising(Q), tsq.to_ising(P)
dc, dp = F.diag(*ic[1:]), F.diag(*ip[1:])
for kind in ('ring', 'complete'):
    th = torch.tensor(rng.normal(size=8))
    a = tsq.state_two_step(n, groups, ks, ip, ic, th, 2, 2, kind)
    b = tsq.fast_two_step(F, groups, dp, dc, th, 2, 2, kind)
    print(kind, 'two_step fid', fid(a, b))
    a = tsq.state_xy_dicke(n, groups, ks, ic, th[:4], 2, kind)
    b = tsq.fast_xy(F, groups, dc, th[:4], 2, tsq.dicke_vector(n, groups, ks), kind)
    print(kind, 'xy_dicke fid', fid(a, b))
it = tsq.to_ising(Q + 1.7 * P)
th = torch.tensor(rng.normal(size=6))
print('penalty fid', fid(tsq.state_penalty_qaoa(n, it, th, 3), tsq.fast_penalty_qaoa(F, F.diag(*it[1:]), th, 3)))

# 重なる組は受け付けない
try:
    tsq.Fast(4).xymix(tsq.Fast(4).plus(), [(0, 1), (1, 2)], torch.tensor(0.3))
    raise SystemExit('重なる組が通ってしまった')
except ValueError:
    print('overlap rejected ok')
