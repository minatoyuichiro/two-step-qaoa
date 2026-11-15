"""S2: 段1 の角度の表。1 行 = (n, k, p1) の組 1 つ。
段1 はコストを含まないので、この表は問題の中身によらず使い回せる（ライブラリに同梱する想定）。
使い方: python3 exp_s2_table.py <out.json> n:k [n:k ...]"""
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
