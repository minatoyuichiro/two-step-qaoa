# Two-Step QAOA — code and data for arXiv:2408.05383 (v3)

Code, raw results and LaTeX source for

> Y. Minato, *Two-Step QAOA: Enhancing Quantum Optimization by Decomposing K-hot Constraints in QUBO Formulations*, arXiv:2408.05383 (v3).

Two-Step QAOA hardens k-hot constraints `sum_{i in g} x_i = k` of a penalty-form QUBO:
stage 1 is QAOA on the penalty `(sum x - k)^2` alone (X mixer, from `|+>`), stage 2 is QAOA on the cost alone with an XY mixer,
followed by post-selection on the hardened constraints.

## Layout

| path | content |
|---|---|
| `src/tsq.py` | circuits in the [blueqat SDK](https://github.com/minatoyuichiro/blueqatSDK) (penalty QAOA, Dicke-initialised XY ansatz, Two-Step) and an equivalent fast state-vector routine used for the sweeps |
| `src/sym.py` | stage 1 in the (n+1)-dimensional symmetric subspace; stage-1 angle tables |
| `src/pipe.py` | layer-by-layer (INTERP) training, bootstrap interval |
| `src/detect.py` | heuristic detection of k-hot penalties in a QUBO matrix |
| `src/dicke_circuit.py` | Bärtschi–Eidenbenz Dicke-state circuit (for reference) |
| `src/test_*.py` | checks against exact matrix exponentials, blueqat vs. fast routine, symmetric subspace vs. full space |
| `src/exp_s2_table.py` | stage-1 tables (paper Table 1, Fig. 1) |
| `src/exp_s1b.py` | portfolio experiments (paper Fig. 2, Tables 2–3) |
| `src/exp_s3_detect.py`, `src/exp_s5.py` | detection and multi-group experiments (paper Tables 4–5). These need an in-house library of QUBO formulations that is **not** part of this repository; their raw results are included |
| `src/make_paper_assets.py` | builds the paper's figures and tables from `results/` |
| `results/` | raw results (JSON) behind every number in the paper; `results/s2/` holds the stage-1 angle tables |
| `paper/` | LaTeX source and PDF of v3 |

Comments and docstrings in the code are in Japanese.

## Running

Requirements: Python 3.10+, `torch` (CPU is enough), `numpy`, `scipy`, `networkx`, `matplotlib`,
and the blueqat SDK from GitHub (branch `release-2.2.0`; not the PyPI release).

```
git clone -b release-2.2.0 https://github.com/minatoyuichiro/blueqatSDK
cd src
export PYTHONPATH=/path/to/blueqatSDK:.
python3 test_tsq.py && python3 test_fast.py && python3 test_sym.py
python3 exp_s2_table.py ../results/s2/my_table.json 8:4 12:6          # stage-1 table for (n,k) = (8,4), (12,6)
python3 exp_s1b.py 10 5 0,1,2 out.json ../results/s2/table_3.json     # portfolio, n=10, k=5, seeds 0-2
XY=complete ONLY_XY=1 python3 exp_s1b.py 10 5 0,1,2 out_c.json ../results/s2/table_3.json   # complete-graph XY mixer
python3 make_paper_assets.py                                          # figures and tables -> ../paper/figs, ../paper/tables
```

All simulations are noiseless state-vector simulations (up to 16 qubits; stage-1 tables up to 32 qubits per group).
Shot counts in the paper are an optimiser-budget proxy, not finite-shot optimisation.

## License

Apache License 2.0 (see `LICENSE`).
