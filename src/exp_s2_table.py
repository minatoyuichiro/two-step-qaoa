"""S2: Table of stage 1 angles. One row = one (n, k, p1) group.
Stage 1 does not include cost, so this table can be reused regardless of the problem contents (intended to be bundled with the library).
Usage: python3 exp_s2_table.py <out.json> n:k [n:k ...]"""
import json, math, sys, time
import sym

out_path, pairs = sys.argv[1], [tuple(map(int, a.split(':'))) for a in sys.argv[2:]]
PMAX = 6
rows = []
for n, k in pairs:
    t0 = time.time()
    for p1, pk, params in sym.interp_table(n, k, PMAX):
        row = dict(n=n, k=k, p1=p1, P_k=pk, params=params, P_k_plus=math.comb(n, k) / 2 ** n,
                   P_k_biased_product=sym.biased_product_Pk(n, k), cnot_per_layer=n * (n - 1),
                   seconds=round(time.time() - t0, 1), measured=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        rows.append(row)
        print(json.dumps({a: row[a] for a in ('n', 'k', 'p1', 'P_k', 'seconds')}), flush=True)
        json.dump(rows, open(out_path, 'w'), indent=1)
