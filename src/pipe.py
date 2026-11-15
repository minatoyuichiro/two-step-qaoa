"""S1b・S5 で共通に使う学習の部品（Codex r2 の指摘 #8・#10 を受けて）。

train_interp: 層を 1 つずつ足して学習する（INTERP 法、Zhou ら 2020）。深さ p の初期値は、
  深さ p−1 の解の線形補間と、0 の層を足したもの（こちらは前の深さと同じ値を保つ）、それに乱数の初期値 RAND 個。
  候補ごとに Adam を ITERS 歩、途中で最良の点を残す。
  深さ p の結果の手間 = 深さ 1〜p の学習に使った評価回数の累計（深さ p の答えを得るには、前の深さも学習するので）。
ハミルトニアンは、呼ぶ側で最大絶対値で割ってから渡すこと（全方式で同じ扱いにする）。
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
    """loss_at(x, p) を p = 1..pmax について最小化。返り値: {p: (params, 評価回数の累計)}。
    評価回数 = 歩数 × (1 + 2·パラメータ数)（パラメータシフトとみなした目安）。"""
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
    """例ごとの比の中央値の 95% 区間（例をブートストラップで引き直す）。"""
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    meds = [np.median(r.choice(x, size=len(x), replace=True)) for _ in range(n)]
    return float(np.median(x)), float(np.percentile(meds, q[0])), float(np.percentile(meds, q[1]))
