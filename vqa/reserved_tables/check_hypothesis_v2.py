"""Format-aware hypothesis check on DEV (v1 = check_hypothesis.py, kept).

Hypothesis: answerable -> samples concentrate on the right answer; unanswerable -> samples disagree, OR pile onto IDK.

"Disagree" must respect the answer space: with Yes/No the samples can only split 2 ways. For each question
  L  = number of possible real answers: 2 (UVQA, SpatialUncertain yes/no), 3-4 (SpatialUncertain MCQ, without the
       "Cannot determine" option, which counts as IDK), unbounded for free-form (UNK-VQA, HaloQuest, math, KBQA)
  s  = top real-answer share = nmax / m          (m = non-IDK samples)
  c  = normalised concentration = (s - 1/L) / (1 - 1/L)   (c = s for free-form): 0 = as split as the format
       allows, 1 = unanimous
Patterns:  IDK-heavy  n0 >= 10
           Split      n0 < 10 and c < C            (default C = 0.5: Yes/No below 15:5, 4-way below 62%, free-form below 50%)
           Conc-correct / Conc-wrong   n0 < 10, c >= C, top answer correct / wrong
"Many different answers" (K, singletons) is reported for free-form datasets only.
--empty-idk: count an empty KBQA answer (empty query result) as IDK.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_hypothesis as H

FREE = {"unkvqa", "haloquest", "ReliableMath", "brokenmath_type_answer_AU", "grailqa", "graphqa"}
PATS = ["Conc-correct", "Conc-wrong", "Split", "IDK-heavy"]


def n_options():
    sys.path.insert(0, os.path.expanduser("~/vqa-datasets"))
    import vqa_data as V
    L = {}
    for it in V.load("uvqa"):
        L[it["id"]] = 2
    for it in V.load("spatialuncertain"):
        L[it["id"]] = len(it["options"]) - (it["abstain_idx"] is not None)
    return L


def item_stats(r, mod, L, empty_idk):
    if empty_idk:
        r = dict(r)
        lab = list(r["pred_label"])
        for i, a in enumerate(r["pred_answer"]):
            if str(a).strip() == "":
                lab[i] = "ABSTAIN"
        r["pred_label"] = lab
    s = H.stats(r, mod)
    share = s["nmax"] / s["m"] if s["m"] else float("nan")
    c = share if L is None else ((share - 1 / L) / (1 - 1 / L) if s["m"] else float("nan"))
    s.update(share=share, c=c, L=L)
    return s


def pattern(s, M, C):
    if s["n0"] >= M:
        return "IDK-heavy"
    if s["c"] < C:
        return "Split"
    return "Conc-correct" if s["top_ok"] else "Conc-wrong"


def run(name, st, rows, mod, Lmap, ds, M, C, empty_idk=False):
    out = {"A": [], "U": []}
    for r in rows:
        L = None if ds in FREE else Lmap[r["id"]]
        s = item_stats(r, mod, L, empty_idk)
        if s["N"] == 0:
            continue
        s["pattern"] = pattern(s, M, C)
        out["U" if H.is_unans(r) else "A"].append(s)
    res = {"name": name, "setting": st, "free": ds in FREE}
    for c in "AU":
        ss = out[c]
        res[c] = dict(n=len(ss), **{p: H.wilson(sum(x["pattern"] == p for x in ss), len(ss)) for p in PATS})
    ok = lambda c, x: x["pattern"] == "Conc-correct" if c == "A" else x["pattern"] in ("Split", "IDK-heavy")
    res["holds"] = {c: H.wilson(sum(ok(c, x) for x in out[c]), len(out[c])) for c in "AU"}
    na = [x for x in out["A"] if x["n0"] < M and x["m"]]
    nu = [x for x in out["U"] if x["n0"] < M and x["m"]]
    res["auroc_c"] = H.auroc([-x["c"] for x in na], [-x["c"] for x in nu])     # low concentration -> unanswerable
    res["auroc_n0"] = H.auroc([x["n0"] for x in out["A"]], [x["n0"] for x in out["U"]])
    res["mean_c"] = (float(np.mean([x["c"] for x in na])) if na else float("nan"),
                     float(np.mean([x["c"] for x in nu])) if nu else float("nan"))
    if ds in FREE:
        res["K"] = (float(np.mean([x["K"] for x in na])) if na else float("nan"),
                    float(np.mean([x["K"] for x in nu])) if nu else float("nan"))
        res["auroc_K"] = H.auroc([x["K"] for x in na], [x["K"] for x in nu])
    # split answerable questions: is the correct answer in the split at all?
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.expanduser("~/vqa-datasets/results/dev/hypothesis"))
    a = ap.parse_args()
    Lmap = n_options()
    runs = []
    for ds in H.VQA:
        for st in ("aware", "unaware"):
            runs.append(run(f"VQA {ds}", st, H.load_vqa(ds, st), H.EF, Lmap, ds, a.M, a.C))
    for ds, mod in H.PART:
        rows = H.load_partner(ds)
        runs.append(run(ds, "aware", rows, mod, Lmap, ds, a.M, a.C))
        if mod is H.EK:
            runs.append(run(ds + " (empty=IDK)", "aware", rows, mod, Lmap, ds, a.M, a.C, empty_idk=True))
    f = H.fmt
    md = [f"# Format-aware hypothesis check on dev (IDK-heavy: n0 >= {a.M}; Split: normalised concentration c < {a.C})", "",
          "| dataset | setting | class | n | Conc-correct | Conc-wrong | Split | IDK-heavy | **holds** |", "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        for c in "AU":
            x = r[c]
            md.append(f"| {r['name']} | {r['setting']} | {'answerable' if c == 'A' else 'unanswerable'} | {x['n']} | "
                      + " | ".join(f(x[p]) for p in PATS) + f" | **{f(r['holds'][c])}** |")
    md += ["", "## Disagreement as a signal (questions with n0 < M; AUROC unanswerable vs answerable, 0.5 = none)", "",
           "| dataset | setting | mean c A / U | AUROC low c | AUROC n0 | K A / U (free-form) | AUROC K |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        k = f"{r['K'][0]:.1f} / {r['K'][1]:.1f}" if "K" in r else "n/a (Yes/No or MCQ)"
        ak = f"{r['auroc_K']:.2f}" if "auroc_K" in r else "–"
        md.append(f"| {r['name']} | {r['setting']} | {r['mean_c'][0]:.2f} / {r['mean_c'][1]:.2f} | {r['auroc_c']:.2f} | {r['auroc_n0']:.2f} | {k} | {ak} |")
    os.makedirs(a.out, exist_ok=True)
    open(f"{a.out}/hypothesis_v2_M{a.M}_C{a.C}.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
