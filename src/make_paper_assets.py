"""Build the paper figures and tables from results/ -> paper/figs/*.pdf, paper/tables/*.tex
Colors use the dataviz default sequence (fixed order, validated). Light surfaces only (for print). Series should also be distinguishable by legend + marker shape."""
import collections
import glob
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pipe

R = '../results'
FIG, TAB = '../paper/figs', '../paper/tables'
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
MK = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.6,
                     'legend.frameon': False, 'font.family': 'serif'})


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += json.load(open(f))
    return rows


# ---------------------------------------------------------------- S2: stage 1 table
s2 = load(f'{R}/s2/table_*.json')
by = {}
for r in s2:
    by.setdefault((r['n'], r['k']), {})[r['p1']] = r
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
NS = [4, 8, 12, 16, 20, 24, 32]          # Up to 7 series (do not cycle colors). Same n uses the same color on left and right
for ax, kind, title in ((axs[0], 'half', r'half filling, $k=n/2$'), (axs[1], 'one', r'one-hot, $k=1$')):
    for i, n in enumerate(NS):
        k = n // 2 if kind == 'half' else 1
        if (n, k) not in by:
            continue
        d = by[(n, k)]
        ps = sorted(d)
        ax.plot(ps, [max(1 - d[p]['P_k'], 1e-6) for p in ps], marker=MK[i], ms=4, color=C[i], label=f'n={n}')
    ax.set_yscale('log')
    ax.set_ylim(8e-7, 1.2)
    ax.axhline(0.01, color=INK2, lw=0.8, ls='--')
    ax.text(1.0, 0.0125, r'$P_k=0.99$', color=INK2, fontsize=7)
    ax.set_xlabel(r'stage-1 depth $p_1$')
    ax.set_title(title, fontsize=8, color=INK)
    ax.set_xticks(range(1, 7))
axs[0].set_ylabel(r'leakage $1-P_k$ (floor $10^{-6}$)')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=7, fontsize=7, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(f'{FIG}/stage1_table.pdf')
plt.close(fig)

lines = [r'\begin{table}[t]\centering\small', r'\caption{Stage-1 table: best found $P_k$ per group size $n$, weight $k$ and depth $p_1$ (symmetric-subspace optimization; lower bounds on the achievable value). '
         r"$|+\rangle$ and ``product'' are the $P_k$ of $|+\rangle^{\otimes n}$ and of the biased product state. Last column: smallest $p_1$ with $P_k\ge0.99$ (--: not reached by $p_1=6$).}",
         r'\label{tab:stage1}', r'\begin{tabular}{rrrrrrrrrrr}\toprule', r'$n$ & $k$ & $|+\rangle$ & product & $p_1{=}1$ & 2 & 3 & 4 & 5 & 6 & $p_1^{0.99}$\\\midrule']
for (n, k), d in sorted(by.items()):
    if n < 4:
        continue
    r0 = next(iter(d.values()))
    need = min([p for p, r in d.items() if r['P_k'] >= 0.99], default='--')
    lines.append(f"{n} & {k} & {r0['P_k_plus']:.3f} & {r0['P_k_biased_product']:.3f} & " + ' & '.join(f"{d[p]['P_k']:.4f}" for p in range(1, 7)) + f' & {need}\\\\')
lines += [r'\bottomrule\end{tabular}\end{table}']
open(f'{TAB}/stage1.tex', 'w').write('\n'.join(lines) + '\n')

# ---------------------------------------------------------------- S1b: portfolio
s1 = load(f'{R}/s1b/*.json')
NAMES = collections.OrderedDict([('two_step_complete', 'Two-Step (complete XY)'), ('two_step', 'Two-Step (ring XY)'), ('penalty_ladder', 'penalty, ladder from $A_{\\rm start}$'),
                                 ('penalty_ladder_low', 'penalty, ladder from $A_{\\rm start}/16$'),
                                 ('penalty_grid_deploy', 'penalty, grid (sample-selected)'),
                                 ('penalty_grid', 'penalty, grid (oracle-selected)'), ('bp_xy', 'biased product + XY')])
PLAIN = {'two_step_complete': 'Two-Step (complete XY)', 'two_step': 'Two-Step (ring XY)', 'penalty_ladder': r'penalty, ladder from $A_{start}$', 'penalty_ladder_low': r'penalty, ladder from $A_{start}/16$',
         'penalty_grid_deploy': 'penalty, grid (sample-selected)', 'penalty_grid': 'penalty, grid (oracle)', 'bp_xy': 'biased product + XY'}
sizes = sorted({(r['n'], r['k']) for r in s1})
if sizes:
    fig, axs = plt.subplots(1, len(sizes), figsize=(2.3 * len(sizes) + 0.4, 2.5), sharey=True, squeeze=False)
    for ax, (n, k) in zip(axs[0], sizes):
        Rk = [r for r in s1 if r['n'] == n and r['k'] == k]
        for i, m in enumerate(NAMES):
            ps = sorted({r['p'] for r in Rk if r['method'] == m})
            ys = [np.mean([r['P_opt'] for r in Rk if r['method'] == m and r['p'] == p]) for p in ps]
            ax.plot(ps, ys, marker=MK[i], ms=4, color=C[i], label=PLAIN[m], ls=':' if m == 'penalty_grid' else '-', lw=2.2 if m.startswith('two_step') else 1.4)
        u = np.mean([r['P_opt'] for r in Rk if r['method'] == 'uniform_feasible'])
        ax.axhline(u, color=INK2, lw=0.9, ls='--')
        ax.text(5.0, u * 0.78, 'uniform feasible', color=INK2, fontsize=6.5, ha='right', va='top')
        ax.set_yscale('log')
        ax.set_xlabel('stage-2 / QAOA depth $p$')
        ax.set_title(f'n={n}, k={k} ({len({r["seed"] for r in Rk})} instances)', fontsize=8)
        ax.set_xticks(range(1, 6))
    axs[0][0].set_ylabel(r'$P_{\rm opt}$ (mean over instances)')
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(f'{FIG}/portfolio.pdf')
    plt.close(fig)

    for ref, tag, lab in (('two_step', 'portfolio', 'ring'), ('two_step_complete', 'portfolio_complete', 'complete-graph')):
        lines = [r'\begin{table}[t]\centering\small',
                 r'\caption{Portfolio problems, Two-Step with the ' + lab + r' XY mixer: paired ratio $P_{\rm opt}^{\rm Two\text{-}Step}/P_{\rm opt}^{\rm baseline}$ per instance at the same depth $p$; median with 95\% bootstrap interval of the median, and the number of instances in which Two-Step is better. '
                 r'Budgets: ladders as used (all stopped at their first rung), grids five weights, post-selected methods with overhead $1/P_k$. The biased-product baseline uses the ring XY mixer.}',
                 r'\label{tab:' + tag + '}', r'\resizebox{\linewidth}{!}{\begin{tabular}{llccc}\toprule', r'$(n,k)$ & baseline & $p=1$ & $p=3$ & $p=5$\\\midrule']
        for (n, k) in sizes:
            Rk = [r for r in s1 if r['n'] == n and r['k'] == k]
            seeds = sorted({r['seed'] for r in Rk})
            others = ['uniform_feasible'] + [m for m in NAMES if not m.startswith('two_step')] + (['two_step'] if ref == 'two_step_complete' else [])
            for m in others:
                cells = []
                for p in (1, 3, 5):
                    xs = []
                    for sd in seeds:
                        a = [r for r in Rk if r['method'] == ref and r['p'] == p and r['seed'] == sd]
                        b = [r for r in Rk if r['method'] == m and r['seed'] == sd and (r['p'] == p or m == 'uniform_feasible')]
                        if a and b and b[0]['P_opt'] > 0:
                            xs.append(a[0]['P_opt'] / b[0]['P_opt'])
                    if xs:
                        med, lo, hi = pipe.bootstrap_median_ci(xs)
                        cells.append(f'{med:.2f} [{lo:.2f}, {hi:.2f}] ({sum(x > 1 for x in xs)}/{len(xs)})')
                    else:
                        cells.append('--')
                labm = 'uniform feasible' if m == 'uniform_feasible' else NAMES[m]
                lines.append(f'({n},{k}) & {labm} & ' + ' & '.join(cells) + r'\\')
            lines.append(r'\midrule')
        lines[-1] = r'\bottomrule\end{tabular}}\end{table}'
        open(f'{TAB}/{tag}.tex', 'w').write('\n'.join(lines) + '\n')

# ---------------------------------------------------------------- S3: detection
s3 = [r for r in json.load(open(f'{R}/s3_detect.json')) if 'error' not in r]
tot = collections.Counter()
per = collections.OrderedDict()
for r in s3:
    for kk in ('n_truth', 'support_tp', 'n_candidates', 'n_found', 'tp', 'fp'):
        tot[kk] += r[kk]
        per.setdefault(r['name'], collections.Counter())[kk] += r[kk]
    per[r['name']]['inst'] += 1
lines = [r'\begin{table}[t]\centering\small',
         rf"\caption{{Detecting k-hot penalties from QUBO matrices of {len(s3)} instances of {len(per)} formulations. Truth: the constraints passed to the penalty builders. "
         rf"Final detections are counted correct only if both the support and $k$ match. Totals: {tot['n_truth']} constraints, {tot['n_found']} detections, {tot['tp']} correct "
         rf"(precision {tot['tp'] / tot['n_found']:.2f}, recall {tot['tp'] / tot['n_truth']:.2f}); candidate supports before the $k$ fit: {tot['n_candidates']}, of which {tot['support_tp']} match a true support.}}",
         r'\label{tab:detect}', r'\begin{tabular}{lrrrr}\toprule', r'formulation & constraints & detections & correct & wrong\\\midrule']
for name, a in per.items():
    if a['n_truth'] == 0 and a['n_found'] == 0:
        continue
    lines.append(f"{name} & {a['n_truth']} & {a['n_found']} & {a['tp']} & {a['fp']}\\\\")
lines += [r'\bottomrule\end{tabular}\end{table}']
open(f'{TAB}/detect.tex', 'w').write('\n'.join(lines) + '\n')

# ---------------------------------------------------------------- S5 / S4b: problems with multiple groups
s5 = load(f'{R}/s5/*.json')
if s5:
    s5 = list({(r['name'], r['size'], r['seed'], r['method'], r['p']): r for r in s5}.values())
    LAB = {'coloring_dense': 'graph 3-coloring (dense)', 'assignment': 'assignment', 'qap': 'QAP', 'tsp': 'TSP'}
    lines = [r'\begin{table}[t]\centering\small',
             r"\caption{Problems with several one-hot groups (mean over instances, depth $p=5$, ring XY mixer). ``Hard'' rows: the one-hot groups chosen by the formulation code are hardened with XY mixers; "
             r'for coloring these are all constraints, for the permutation problems only the rows (columns remain penalties). '
             r"$P_{\rm opt|feas}$ is the optimum probability among feasible samples. Penalty weights follow the formulation code's ladder.}",
             r'\label{tab:multi}', r'\begin{tabular}{lrrlccc}\toprule', r'problem & qubits & inst. & method & $P_{\rm feas}$ & $P_{\rm opt}$ & $P_{\rm opt|feas}$\\\midrule']
    for name, size in sorted({(r['name'], r['size']) for r in s5}):
        Rk = [r for r in s5 if r['name'] == name and r['size'] == size]
        nq = Rk[0]['n']
        for m, lab in (('uniform_feasible', 'uniform feasible'), ('penalty_ladder', 'penalty QAOA'), ('two_step_all', 'Two-Step (all hard)'), ('partial_ladder', 'Two-Step (rows hard)')):
            S = [r for r in Rk if r['method'] == m and r['p'] in (0, 5)]
            if not S:
                continue
            lines.append(f"{LAB.get(name, name)} {size} & {nq} & {len(S)} & {lab} & {np.mean([r['P_feas'] for r in S]):.3f} & {np.mean([r['P_opt'] for r in S]):.4f} & {np.mean([r['P_opt_given_feas'] for r in S]):.4f}\\\\")
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule\end{tabular}\end{table}'
    open(f'{TAB}/multi.tex', 'w').write('\n'.join(lines) + '\n')
print('assets written')
