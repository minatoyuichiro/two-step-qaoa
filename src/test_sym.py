"""Check whether Sym (the symmetric subspace) matches the 2^n-dimensional computation (tsq.Fast, already verified against blueqat)."""
import numpy as np, torch, tsq, sym, math

rng = np.random.default_rng(5)
for n, k in ((4, 1), (6, 3), (7, 2), (9, 4)):
    F = tsq.Fast(n)
    dp = F.diag(*tsq.to_ising(tsq.khot_penalty(n, [range(n)], [k]))[1:])
    S = sym.Sym(n)
    for p1 in (1, 3):
        th = torch.tensor(rng.normal(size=2 * p1))
        full = tsq.fast_penalty_qaoa(F, dp, th, p1)
        pw = np.array([float((full.abs() ** 2)[torch.tensor(F.X.sum(1) == w)].sum()) for w in range(n + 1)])
        ps = (S.stage1(th, k, p1).abs() ** 2).numpy()
        assert np.abs(pw - ps).max() < 1e-12, (n, k, p1, np.abs(pw - ps).max())
    # P_k of the biased product state
    q = k / n
    th = 2 * math.asin(math.sqrt(q))
    st = tsq.run_state(n, lambda c: c.ry(th)[:]).numpy()
    pk = float((np.abs(st) ** 2)[F.X.sum(1) == k].sum())
    assert abs(pk - sym.biased_product_Pk(n, k)) < 1e-12
print('sym ok')
