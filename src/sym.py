"""段1 を対称部分空間（基底 |D(n,w)>, w=0..n）で厳密に計算する。

段1 は |+>^n・X ミキサー・(sum x - k)^2 だけで、どれも量子ビットの入れ替えに対して対称。
よって状態は n+1 次元から出ない。sum_i X_i = 2 J_x、<w+1|J_x|w> = sqrt((w+1)(n-w))/2。
2^n 次元の計算（tsq.Fast）との一致は test_sym.py で確かめる。
"""
import math

import numpy as np
import torch


class Sym:
    def __init__(self, n):
        self.n = n
        w = np.arange(n + 1)
        off = np.sqrt((w[:-1] + 1) * (n - w[:-1])) / 2
        Jx = np.diag(off, 1) + np.diag(off, -1)
        lam, V = np.linalg.eigh(Jx)
        self.lam = torch.tensor(lam)
        self.V = torch.tensor(V, dtype=torch.complex128)
        self.w = torch.tensor(w, dtype=torch.float64)
        # |+>^n の重み w 成分の振幅 sqrt(C(n,w) / 2^n)
        self.plus = torch.tensor([math.sqrt(math.comb(n, int(x)) / 2 ** n) for x in w], dtype=torch.complex128)

    def xmix(self, s, beta):
        """exp(-i beta sum X) = exp(-i 2 beta J_x)。"""
        ph = torch.exp(-2j * beta * self.lam)
        return self.V @ (ph * (self.V.conj().T @ s))

    def stage1(self, params, k, p1):
        d = (self.w - k) ** 2
        s = self.plus
        for l in range(p1):
            s = self.xmix(s * torch.exp(-1j * params[l] * d), params[p1 + l])
        return s

    def P_k(self, params, k, p1):
        return self.stage1(params, k, p1).abs()[k] ** 2


def biased_product_Pk(n, k):
    """各量子ビットを確率 k/n で 1 にした積状態（CNOT 0 個）の P_k。k 成分は厳密に Dicke。"""
    q = k / n
    return math.comb(n, k) * q ** k * (1 - q) ** (n - k)


def minimize(loss, x0, iters=300):
    """L-BFGS-B（勾配は torch の自動微分）。評価した中で最も良かった点を返す（最後の点ではなく）。"""
    from scipy.optimize import minimize as _min
    best = [math.inf, x0.clone()]

    def f(v):
        x = torch.tensor(v, dtype=torch.float64, requires_grad=True)
        L = loss(x)
        L.backward()
        if float(L.detach()) < best[0]:
            best[0], best[1] = float(L.detach()), x.detach().clone()
        return float(L.detach()), x.grad.numpy().copy()
    _min(f, x0.numpy(), jac=True, method='L-BFGS-B', options={'maxiter': iters})
    return best[0], best[1]


def adam(loss, x0, iters=300, lr=0.05):
    """Adam。途中で最も良かった点を返す。L-BFGS が狭い谷を外すときの補い。"""
    x = x0.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    best = [math.inf, x0.clone()]
    for _ in range(iters):
        opt.zero_grad()
        L = loss(x)
        if float(L.detach()) < best[0]:
            best = [float(L.detach()), x.detach().clone()]
        L.backward()
        opt.step()
    return best[0], best[1]


def interp_table(n, k, pmax, restarts=128, iters=300, seed=0):
    """INTERP 法: p1 の解から、層を 1 つ足した p1+1 の初期値を作る（Zhou ら 2020 の線形補間）。
    それに加えて乱数の初期値も試し、最良を残す。返り値: [(p1, P_k, params), ...]。"""
    S = Sym(n)
    g = torch.Generator().manual_seed(seed)
    out = []
    prev = None
    for p1 in range(1, pmax + 1):
        loss = lambda x: -S.P_k(x, k, p1)
        cands = []
        if prev is not None:
            gm, bt = prev[:p1 - 1].numpy(), prev[p1 - 1:].numpy()
            def interp(v):
                q = len(v)
                return np.array([(i / q) * (v[i - 1] if i - 1 >= 0 else 0) + ((q - i) / q) * (v[i] if i < q else 0)
                                 for i in range(q + 1)])
            cands.append(torch.tensor(np.concatenate([interp(gm), interp(bt)])))
            # 前の解に 0 の層を足したもの（P_k は前と同じ = 下がらないことの保証）
            cands.append(torch.cat([prev[:p1 - 1], torch.zeros(1, dtype=torch.float64),
                                    prev[p1 - 1:], torch.zeros(1, dtype=torch.float64)]))
        for _ in range(restarts):
            # 周期の全範囲から取る: (w-k)^2 は整数なので γ の周期は 2π、β の周期は π
            u = torch.rand(2 * p1, generator=g, dtype=torch.float64) * 2 - 1
            cands.append(torch.cat([u[:p1] * math.pi, u[p1:] * math.pi / 2]))
        best = (math.inf, None)
        for i, x0 in enumerate(cands):
            r = minimize(loss, x0, iters)
            if r[0] < best[0]:
                best = r
            if i < 16:                     # 先頭 16 個（補間・0 足し・乱数）は Adam → L-BFGS でも試す
                a = adam(loss, x0, iters)
                r = minimize(loss, a[1], iters)
                if r[0] < best[0]:
                    best = r
        prev = best[1]
        out.append((p1, -best[0], prev.tolist()))
    return out
