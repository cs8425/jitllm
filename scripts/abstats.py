#!/usr/bin/env python3
"""abstats.py: median of per-round ratios with IQR for a decisionbench-ab.sh run.

    abstats.py OUT.tsv [WINDOW_SECONDS]

Ratios are B over A for words (more is B faster) and A over B for the p50 (more is
B faster); a run whose IQR/median exceeds 0.10 is rejected (AGENTS.md RULE 2).
"""
import statistics, sys

def q(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)

rows = [list(map(float, l.split())) for l in open(sys.argv[1]) if l.strip()]
win = float(sys.argv[2]) if len(sys.argv) > 2 else 32.0
for name, ratio, scale in (("p50 speedup (B p50 / A p50)", lambda r: r[3] / r[1], 1), ("words speedup (A words / B words)", lambda r: r[2] / r[4], 1)):
    rs = [ratio(r) for r in rows]
    med = statistics.median(rs)
    iqr = q(rs, .75) - q(rs, .25)
    print(f"{name}: median {med:.4f} IQR {iqr:.4f} IQR/median {iqr/med:.4f} n={len(rs)} {'REJECT' if iqr/med > .10 else 'ok'}")
print("A p50 median %.3f ms, B p50 median %.3f ms" % (statistics.median(r[1] for r in rows) / 1e3, statistics.median(r[3] for r in rows) / 1e3))
k = 32.0 / win
print("A words in 32 s median %.0f, B %.0f (counted in %gs windows, scaled by %.3g)" % (statistics.median(r[2] for r in rows) * k, statistics.median(r[4] for r in rows) * k, win, k))
