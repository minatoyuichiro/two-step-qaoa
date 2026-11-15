"""Build the exact Dicke-state circuit of Bärtschi–Eidenbenz (2019) in blueqat and count CNOTs.

Qubit indices use the paper's 1..n shifted to 0..n-1. The initial state is |0^{n-k} 1^k> (the last k are 1).
Count CRy as 2 CNOTs and CCRy as 4 CNOTs (alternating Ry(θ/4) and CX).
"""
import math
import sys

import numpy as np
import torch
from blueqat import Circuit

import tsq


def cry(c, th, ctl, tgt):
    c.ry(th / 2)[tgt].cx[ctl, tgt].ry(-th / 2)[tgt].cx[ctl, tgt]


def ccry(c, th, c1, c2, tgt):
    q = th / 4
    c.ry(q)[tgt].cx[c2, tgt].ry(-q)[tgt].cx[c1, tgt].ry(q)[tgt].cx[c2, tgt].ry(-q)[tgt].cx[c1, tgt]


def scs(c, l, k):
    """SCS_{l,k} (the paper's 1-based index l)."""
    a, b = l - 2, l - 1          # the paper's l-1, l
    c.cx[a, b]
    cry(c, 2 * math.acos(math.sqrt(1 / l)), b, a)
    c.cx[a, b]
    for i in range(2, k + 1):
        t = l - i - 1            # the paper's l-i
        c.cx[t, b]
        ccry(c, 2 * math.acos(math.sqrt(i / l)), b, t + 1, t)
        c.cx[t, b]


def dicke_circuit(n, k):
    c = Circuit(n)
    for q in range(n - k, n):
        c.x[q]
    for l in range(n, k, -1):
        scs(c, l, k)
    for l in range(k, 1, -1):
        scs(c, l, l - 1)
    return c


def n_cx(c):
    return sum(1 for g in c.ops if g.lowername == 'cx')


if __name__ == '__main__':
    print('n k  F(Dicke)  CNOT(Bärtschi)  CNOT(stage 1, per layer)')
    for n in (4, 6, 8, 10):
        for k in sorted({1, 2, n // 2}):
            c = dicke_circuit(n, k)
            s = c.statevector(backend='statevector')
            D = tsq.dicke_vector(n, [tuple(range(n))], [k])
            f = float(abs(torch.vdot(D, s)) ** 2)
            print(n, k, f'{f:.12f}', n_cx(c), n * (n - 1))
