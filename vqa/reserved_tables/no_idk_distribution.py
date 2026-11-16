"""DEV ONLY. Remove the IDK / NK votes; how are the remaining (real) votes distributed per class?
Per question with m >= 2 remaining votes:
  s = top share = nmax / m;  c = format-normalised concentration (s - 1/L)/(1 - 1/L) (L = #options; free-form: c = s)
  H = normalised entropy of the remaining votes = H / log(min(L, m))  (0 = all agree, 1 = as spread as possible)
  unanimous = all remaining votes agree;  distributed = c < 0.5
KBQA twice: empty results kept as one answer cluster, and removed like IDK.
Also stratified by m (how many non-IDK votes there are), since few votes look concentrated by chance less often.
"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_hypothesis as H, check_hypothesis_v2 as V2

def per_q(r, mod, L, drop_empty):
    N, n0, m, clusters, is_corr = mod.extract_clusters(r, ignore_idk=True)[:5]
    if drop_empty:
        clusters = [c for c in clusters if not all(str(r["pred_answer"][i]).strip() == "" for i in c)]
    sizes = np.array(sorted((len(c) for c in clusters), reverse=True), float)
    mm = sizes.sum()
    if mm < 2:
        return dict(m=int(mm))
    s = sizes[0] / mm
    c = s if L is None else (s - 1 / L) / (1 - 1 / L)
    p = sizes / mm
    Hn = float(-(p * np.log(p)).sum() / math.log(min(L or mm, mm))) if min(L or mm, mm) > 1 else 0.0
    return dict(m=int(mm), s=s, c=c, H=Hn, K=len(sizes), unan=len(sizes) == 1,
                top_ok=any(is_corr[i] for i in max(clusters, key=len)))

def summarise(qs):
    v = [q for q in qs if q["m"] >= 2]
    if not v:
        return None
    a = lambda k: np.array([q[k] for q in v], float)
    return dict(n=len(qs), n_used=len(v), m01=len(qs) - len(v), unan=100 * a("unan").mean(),
                dist=100 * (a("c") < 0.5).mean(), med_c=float(np.median(a("c"))), mean_H=float(a("H").mean()),
                K=float(a("K").mean()), top_ok=100 * a("top_ok").mean(),
                strata={b: (100 * np.mean([q["unan"] for q in v if lo <= q["m"] <= hi]) if any(lo <= q["m"] <= hi for q in v) else None,
                            sum(lo <= q["m"] <= hi for q in v))
                        for b, (lo, hi) in {"2-9": (2, 9), "10-19": (10, 19), "20": (20, 20)}.items()})

def auroc(a, u):
    a, u = np.asarray(a, float), np.asarray(u, float)
    return float((u[:, None] > a[None, :]).mean() + 0.5 * (u[:, None] == a[None, :]).mean()) if len(a) and len(u) else float("nan")

Lmap = V2.n_options()
runs = []
for ds in H.VQA:
    for st in ("aware", "unaware"):
        runs.append((f"{ds}", st, H.load_vqa(ds, st), H.EF, ds, False))
for ds, mod in H.PART:
    rows = H.load_partner(ds)
    if mod is H.EK:
        runs.append((f"{ds} (empty kept)", "aware", rows, mod, ds, False))
        runs.append((f"{ds} (empty removed)", "aware", rows, mod, ds, True))
    else:
        runs.append((ds, "aware", rows, mod, ds, False))

md = ["# Distribution of the non-IDK votes (dev only)", "",
      "Questions with fewer than 2 non-IDK votes are left out (count in 'm<2'). distributed = normalised concentration c < 0.5.",
      "AUROC: low concentration -> unanswerable (0.5 = no signal).", "",
      "| dataset | setting | class | n | m<2 | unanimous % | distributed % | median c | mean norm. entropy | mean K | top answer correct % | AUROC (low c) |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
strat = ["", "## Unanimous % by number of non-IDK votes m (n in brackets)", "",
         "| dataset | setting | class | m 2-9 | m 10-19 | m = 20 |", "|---|---|---|---|---|---|"]
for name, st, rows, mod, ds, de in runs:
    L = lambda r: None if ds in V2.FREE else Lmap[r["id"]]
    qa = [per_q(r, mod, L(r), de) for r in rows if not H.is_unans(r)]
    qu = [per_q(r, mod, L(r), de) for r in rows if H.is_unans(r)]
    A, U = summarise(qa), summarise(qu)
    au = auroc([-q["c"] for q in qa if q["m"] >= 2], [-q["c"] for q in qu if q["m"] >= 2])
    for cls, S in (("answerable", A), ("unanswerable", U)):
        if S is None:
            md.append(f"| {name} | {st} | {cls} | – | all | – | – | – | – | – | – | – |"); continue
        md.append(f"| {name} | {st} | {cls} | {S['n']} | {S['m01']} | {S['unan']:.0f} | {S['dist']:.0f} | {S['med_c']:.2f} | "
                  f"{S['mean_H']:.2f} | {S['K']:.1f} | {S['top_ok']:.0f} | {au:.2f} |")
        strat.append(f"| {name} | {st} | {cls} | " + " | ".join(
            (f"{v:.0f} ({n})" if v is not None else f"– ({n})") for v, n in S["strata"].values()) + " |")
out = os.path.expanduser("~/vqa-datasets/results/dev/hypothesis/no_idk_distribution.md")
open(out, "w").write("\n".join(md + strat) + "\n")
print("\n".join(md + strat))
