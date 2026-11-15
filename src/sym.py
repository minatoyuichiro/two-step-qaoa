"""Exactly compute stage 1 in the symmetric subspace (basis |D(n,w)>, w=0..n).

Stage 1 consists only of |+>^n, the X mixer, and (sum x - k)^2, all of which are symmetric under qubit permutation.
Thus the state stays within the n+1 dimensions. sum_i X_i = 2 J_x, <w+1|J_x|w> = sqrt((w+1)(n-w))/2.
Agreement with the 2^n-dimensional computation (tsq.Fast) is checked in test_sym.py.
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
        # Amplitude of the weight-w component of |+>^n: sqrt(C(n,w) / 2^n)
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
    """P_k of the product state with each qubit set to 1 with probability k/n (zero CNOTs). The k component is exactly Dicke."""
    q = k / n
    return math.comb(n, k) * q ** k * (1 - q) ** (n - k)


def minimize(loss, x0, iters=300):
    """L-BFGS-B (gradients via torch autograd). Return the best point evaluated (not the last point)."""
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
    """Adam. Return the best point along the way. A fallback for when L-BFGS misses a narrow valley."""
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
    """INTERP method: from the p1 solution, build an initial value for p1+1 with one extra layer (linear interpolation of Zhou et al. 2020).
    In addition, also try random initial values and keep the best. Returns: [(p1, P_k, params), ...]."""
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
            # The previous solution with a zero layer appended (P_k same as before = guarantee of no degradation)
            cands.append(torch.cat([prev[:p1 - 1], torch.zeros(1, dtype=torch.float64),
                                    prev[p1 - 1:], torch.zeros(1, dtype=torch.float64)]))
        for _ in range(restarts):
            # Take from the full period range: (w-k)^2 is an integer, so the period of gamma is 2pi and that of beta is pi
            u = torch.rand(2 * p1, generator=g, dtype=torch.float64) * 2 - 1
            cands.append(torch.cat([u[:p1] * math.pi, u[p1:] * math.pi / 2]))
        best = (math.inf, None)
        for i, x0 in enumerate(cands):
            r = minimize(loss, x0, iters)
            if r[0] < best[0]:
                best = r
            if i < 16:                     # For the first 16 (interpolation, zero-padding, random), also try Adam -> L-BFGS
                a = adam(loss, x0, iters)
                r = minimize(loss, a[1], iters)
                if r[0] < best[0]:
                    best = r
        prev = best[1]
        out.append((p1, -best[0], prev.tolist()))
    return out
