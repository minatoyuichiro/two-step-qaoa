"""Learning components shared by S1b and S5.

train_interp: train by adding layers one at a time (the INTERP method, Zhou et al. 2020). The initial values at depth p are
  the linear interpolation of the solution at depth p-1 plus a zero layer (which keeps the same values as the previous depth), plus RAND random initial values.
  For each candidate, run Adam for ITERS steps and keep the best point along the way.
  The cost of the result at depth p = the cumulative number of evaluations used for training at depths 1..p (since obtaining the answer at depth p requires training the previous depths as well).
The caller must divide the Hamiltonian by its maximum absolute value before passing it (use the same treatment for all methods).
"""
import math

import numpy as np
import torch

ITERS, LR, RAND = 150, 0.05, 2


def _adam(loss, x0, iters):
    x = x0.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=LR)
    best = (math.inf, x0.clone())
    for _ in range(iters):
        opt.zero_grad()
        L = loss(x)
        v = float(L.detach())
        if v < best[0]:
            best = (v, x.detach().clone())
        L.backward()
        opt.step()
    return best


def _interp(v):
    q = len(v)
    return np.array([(i / q) * (v[i - 1] if i >= 1 else 0.0) + ((q - i) / q) * (v[i] if i < q else 0.0) for i in range(q + 1)])


def train_interp(loss_at, pmax, seed, iters=ITERS, rand=RAND):
    """Minimize loss_at(x, p) for p = 1..pmax. Returns: {p: (params, cumulative number of evaluations)}.
    number of evaluations = steps × (1 + 2·number of parameters) (an estimate treating it as parameter shift)."""
    g = torch.Generator().manual_seed(seed)
    out, prev, evals = {}, None, 0
    for p in range(1, pmax + 1):
        cands = []
        if prev is not None:
            gm, bt = prev[:p - 1].numpy(), prev[p - 1:].numpy()
            cands.append(torch.tensor(np.concatenate([_interp(gm), _interp(bt)])))
            z = torch.zeros(1, dtype=torch.float64)
            cands.append(torch.cat([prev[:p - 1], z, prev[p - 1:], z]))
        for _ in range(rand if prev is not None else rand + 2):
            cands.append((torch.rand(2 * p, generator=g, dtype=torch.float64) * 2 - 1) * 0.5)
        best = (math.inf, None)
        for x0 in cands:
            r = _adam(lambda x: loss_at(x, p), x0, iters)
            if r[0] < best[0]:
                best = r
        evals += len(cands) * iters * (1 + 2 * 2 * p)
        prev = best[1]
        out[p] = (prev, evals)
    return out


def bootstrap_median_ci(x, n=2000, seed=0, q=(2.5, 97.5)):
    """95% interval of the median of the ratio per instance (resampling instances by bootstrap)."""
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    meds = [np.median(r.choice(x, size=len(x), replace=True)) for _ in range(n)]
    return float(np.median(x)), float(np.percentile(meds, q[0])), float(np.percentile(meds, q[1]))
