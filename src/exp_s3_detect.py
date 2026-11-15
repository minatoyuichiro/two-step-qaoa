"""S3: 定式化ライブラリ の定式化から作った QUBO だけを渡して、k-hot の罰則を当てられるか。

1 つ = (定式化の名前, 大きさ, 種) で qrandom が作った問題 1 つ。
正解 = 問題を組む途中で qopt_lite.add_onehot / add_sum_eq が「係数なし（全部 1）・K が整数」で呼ばれた組と K。
       （不等式の余り変数つき・重みつきの和は k-hot ではないので正解に入れない）
検出は QUBO（quad・linear）だけを見る（detect.py）。組と k が両方一致したものだけを当たりと数える。
⚠️ この実験は、公開していない社内の定式化ライブラリ（qsolve・qrandom・qopt_lite）が要る。結果は results/ に入れてある。
使い方: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/path/to/formulation_library:. python3 exp_s3_detect.py out.json
"""
import json
import sys
import time

import qopt_lite as qo
import qrandom
import qsolve

import detect

calls = []
_onehot, _sumeq = qo.add_onehot, qo.add_sum_eq


def rec_onehot(quad, linear, idxs, A):
    idxs = list(idxs)
    calls.append((frozenset(idxs), 1, float(A), len(idxs)))
    return _onehot(quad, linear, idxs, A)


def rec_sumeq(quad, linear, idxs, K, A, coef=None):
    idxs = list(idxs)
    # 柔らかい目的（シフトの「公平さ」w·(勤務日数 − t)² など）は制約ではないので正解に入れない。呼んだ行に fair が出るかで見分ける
    ctx = ''.join(__import__('inspect').getframeinfo(sys._getframe(1)).code_context or [])
    if coef is None and float(K).is_integer() and 'fair' not in ctx:
        calls.append((frozenset(idxs), int(K), float(A), len(idxs)))
    return _sumeq(quad, linear, idxs, K, A, coef=coef)


qo.add_onehot, qo.add_sum_eq = rec_onehot, rec_sumeq
OUT = sys.argv[1]
SIZES = {}          # 既定の大きさと、その 1.5 倍（上限 MAX_N）
rows = []
for name in qsolve.names():
    d = qrandom.DEFAULT[name]
    sizes = [d] if isinstance(d, str) else sorted({d, min(int(d * 1.5), qrandom.MAX_N)})
    for size in sizes:
        for seed in (1, 2, 3):
            calls.clear()
            try:
                data = qrandom.generate(name, size, seed)
                P = qsolve.PROBLEMS[name](data)
            except Exception as e:
                rows.append(dict(name=name, size=size, seed=seed, error=f'{type(e).__name__}: {str(e)[:120]}'))
                continue
            # 変数の数が 1 以下の組（K=0/1 の自明なもの）と、K が 0 や |g| のもの（全部 0 / 全部 1）は k-hot として数えない
            truth = {(g, k) for g, k, A, m in calls if m >= 2 and 1 <= k <= m - 1}
            t0 = time.time()
            found = detect.detect(P.quad, P.linear, P.n)
            dt = time.time() - t0
            got = {(c, k) for c, k, A, kf in found}
            tp = len(truth & got)
            cand = {c for c, A in detect.candidates(P.quad)}
            sup_t, sup_g = {g for g, k in truth}, {c for c, k in got}
            sup_tp = len(sup_t & cand)               # 候補の段階で組が当たった数
            fin_sup_tp = len(sup_t & sup_g)           # 最終判定（k も整数に合ったもの）で組が当たった数
            chosen = detect.choose_disjoint(found)
            qgroups = {frozenset(g) for g in getattr(P, 'groups', []) if len(g) >= 2}
            row = dict(name=name, size=size, seed=seed, n=P.n, n_truth=len(truth), n_found=len(got), tp=tp,
                       fp=len(got - truth), fn=len(truth - got), support_tp=sup_tp, n_candidates=len(cand), final_support_tp=fin_sup_tp,
                       overlapping_truth=any(a[0] & b[0] for a, b in __import__('itertools').combinations(truth, 2)),
                       n_chosen=len(chosen), chosen_vars=sum(len(c) for c, *_ in chosen),
                       library_groups=len(qgroups), library_groups_found=len(qgroups & {c for c, *_ in found}),
                       seconds=round(dt, 3),
                       missed=[(sorted(g), k) for g, k in list(truth - got)[:3]],
                       extra=[(sorted(c), k) for c, k in list(got - truth)[:3]])
            rows.append(row)
            print(json.dumps({a: row[a] for a in ('name', 'size', 'seed', 'n', 'n_truth', 'tp', 'fp', 'fn', 'support_tp', 'n_candidates', 'n_chosen')}), flush=True)
json.dump(rows, open(OUT, 'w'), indent=1)
