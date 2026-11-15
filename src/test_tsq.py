"""Cross-check the components of tsq.py against numpy dense matrices (expm)."""
import numpy as np
import torch
from scipy.linalg import expm

import tsq

I2 = np.eye(2); X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1, -1])


def op(n, d):
    """d = {qubit: 2x2}. index = sum x_i 2^i, so order the Kronecker product starting from q_{n-1}."""
    m = np.array([[1]])
    for q in reversed(range(n)):
        m = np.kron(m, d.get(q, I2))
    return m


def test_ising_matches_qubo():
    rng = np.random.default_rng(1)
    n = 5
    Q = rng.normal(size=(n, n)); Q = (Q + Q.T) / 2
    const, h, J = tsq.to_ising(Q)
    Xb = tsq.bits_table(n)
    f = tsq.qubo_values(Q, Xb)
    z = 1 - 2 * Xb
    e = const + z @ h + sum(w * z[:, i] * z[:, j] for (i, j), w in J.items())
    assert np.allclose(f, e)


def test_penalty_qubo():
    n = 6; groups = [(0, 1, 2), (3, 4, 5)]; ks = [1, 2]
    P = tsq.khot_penalty(n, groups, ks)
    Xb = tsq.bits_table(n)
    ref = sum((Xb[:, list(g)].sum(1) - k) ** 2 for g, k in zip(groups, ks))
    const = sum(k * k for k in ks)
    assert np.allclose(tsq.qubo_values(P, Xb) + const, ref)


def test_layers_match_expm():
    rng = np.random.default_rng(2)
    n = 4
    Q = rng.normal(size=(n, n)); Q = (Q + Q.T) / 2
    const, h, J = tsq.to_ising(Q)
    gam, bet = 0.37, -0.81
    Hc = sum(h[i] * op(n, {i: Z}) for i in range(n)) + sum(w * op(n, {i: Z, j: Z}) for (i, j), w in J.items())
    Hx = sum(op(n, {i: X}) for i in range(n))
    plus = np.ones(2 ** n) / 2 ** (n / 2)
    ref = expm(-1j * bet * Hx) @ expm(-1j * gam * Hc) @ plus
    st = tsq.state_penalty_qaoa(n, (const, h, J), torch.tensor([gam, bet]), 1).numpy()
    assert abs(abs(np.vdot(ref, st)) - 1) < 1e-10


def test_xy_pair_and_weight():
    n = 4
    st = tsq.run_state(n, lambda c: (c.x[0], tsq.add_xymixer(c, [(0, 1)], 0.6, 'ring')))
    st = st.numpy()
    Hxy = (op(n, {0: X, 1: X}) + op(n, {0: Y, 1: Y})) / 2
    e0 = np.zeros(2 ** n); e0[1] = 1
    ref = expm(-1j * 0.6 * Hxy) @ e0
    assert abs(abs(np.vdot(ref, st)) - 1) < 1e-10
    # Weight preservation: multiplying XY into Dicke does not leak outside weight 2
    groups = [(0, 1, 2, 3)]
    for kind in ('ring', 'complete'):
        s = tsq.run_state(n, lambda c: tsq.add_xymixer(c, groups, 0.9, kind),
                          initial=tsq.dicke_vector(n, groups, [2])).numpy()
        w = tsq.bits_table(n).sum(1)
        assert np.abs(s[w != 2]).max() < 1e-12


def test_xy_pair_2cx():
    """The XY of 2 CNOTs equals rxx·ryy as a unitary (up to global phase)."""
    for t in (0.3, -1.1, 2.4):
        cols_a, cols_b = [], []
        for b in range(4):
            e = torch.zeros(4, dtype=torch.complex128); e[b] = 1
            cols_a.append(tsq.run_state(2, lambda c: c.rxx(t)[0, 1].ryy(t)[0, 1], initial=e).numpy())
            cols_b.append(tsq.run_state(2, lambda c: tsq.xy_pair_2cx(c, t, 0, 1), initial=e).numpy())
        A, B = np.array(cols_a).T, np.array(cols_b).T
        assert abs(abs(np.trace(A.conj().T @ B)) / 4 - 1) < 1e-12


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k)
