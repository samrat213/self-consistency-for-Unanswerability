"""Final dev-set distribution audit (all datasets with sample files): how the 20 samples are distributed per class.

Per question, the 20 samples are split into
  n0 = IDK / NK samples (partner's is_sample_abstention), e = empty-result samples (KBQA only: a query that returns []),
  m  = the rest (real, non-empty answers) in K clusters, top share s = nmax / m,
  c  = normalised concentration (s - 1/L) / (1 - 1/L), L = number of possible real answers (2 Yes/No, 3-4 MCQ,
       unbounded free-form -> c = s).
Outcome (first match, M = 10 of 20):
  IDK-heavy   n0 >= M          Empty-heavy  e >= M (KBQA)
  Split       c < 0.5 (or no single outcome reaches M)
  Conc-correct / Conc-wrong    one real answer dominates (c >= 0.5 and nmax >= M), correct / wrong
Classes: answerable; unanswerable; KBQA unanswerable split into schema-level (gold NK) and data-level (gold query, empty
answer), following FUn (JAIR_FUn_V6, Sec. 3): correct outputs are answer / NK / valid query with empty result.
"""
import collections, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_hypothesis as H
import check_hypothesis_v2 as V2

M = 10
OUTS = ["Conc-correct", "Conc-wrong", "Split", "IDK-heavy", "Empty-heavy"]


def kbqa_class(r):
    if str(r.get("gt_label")) == "ANSWERABLE":
        return "answerable"
    return "unans: schema-level" if str(r.get("golden_s_expr")) == "NK" else "unans: data-level"


def outcome(r, mod, L, kbqa):
    N, n0, m, clusters, is_corr = mod.extract_clusters(r, ignore_idk=True)[:5]
    empty = set()
    if kbqa:
        empty = {i for c in clusters for i in c if str(r["pred_answer"][i]).strip() == ""}
        clusters = [c for c in clusters if not set(c) & empty]
    e = len(empty)
    real = sum(len(c) for c in clusters)
    sizes = sorted((len(c) for c in clusters), reverse=True)
    nmax = sizes[0] if sizes else 0
    s = nmax / real if real else 0.0
    c = s if L is None else ((s - 1 / L) / (1 - 1 / L) if real else 0.0)
    top_ok = bool(clusters) and any(is_corr[i] for i in max(clusters, key=len))
    if n0 >= M:
        o = "IDK-heavy"
    elif e >= M:
        o = "Empty-heavy"
    elif nmax >= M and c >= 0.5:
        o = "Conc-correct" if top_ok else "Conc-wrong"
    else:
        o = "Split"
    return dict(o=o, n0=n0, e=e, K=len(sizes), c=c, unanimous=(nmax == N))


def table(rows, mod, Lmap, ds, kbqa):
    by = collections.defaultdict(list)
    for r in rows:
        L = None if ds in V2.FREE else Lmap[r["id"]]
        cls = kbqa_class(r) if kbqa else ("unanswerable" if H.is_unans(r) else "answerable")
        by[cls].append(outcome(r, mod, L, kbqa))
    return by


def main():
    Lmap = V2.n_options()
    runs = []
    for ds in H.VQA:
        for st in ("aware", "unaware"):
            runs.append((f"VQA {ds}", st, table(H.load_vqa(ds, st), H.EF, Lmap, ds, False), False))
    for ds, mod in H.PART:
        kb = mod is H.EK
        runs.append((ds, "aware", table(H.load_partner(ds), mod, Lmap, ds, kb), kb))
    md = ["# Final dev distribution audit (M = 10 of 20; Split = normalised concentration < 0.5 or nothing reaches 10)", "",
          "% of dev questions per class (n). Empty-heavy only exists for KBQA.", "",
          "| dataset | setting | class | n | Conc-correct | Conc-wrong | Split | IDK-heavy | Empty-heavy | unanimous wrong |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    js = {}
    for name, st, by, kb in runs:
        for cls in sorted(by, key=lambda x: (x != "answerable", x)):
            xs = by[cls]
            n = len(xs)
            cnt = collections.Counter(x["o"] for x in xs)
            uw = sum(x["o"] == "Conc-wrong" and x["unanimous"] for x in xs)
            md.append(f"| {name} | {st} | {cls} | {n} | " + " | ".join(
                (f"{100*cnt[o]/n:.0f}" if (o != "Empty-heavy" or kb) else "–") for o in OUTS) + f" | {uw} |")
            js[f"{name}|{st}|{cls}"] = dict(n=n, **{o: cnt[o] for o in OUTS}, unanimous_wrong=uw)
    out = os.path.expanduser("~/vqa-datasets/results/dev/hypothesis")
    open(f"{out}/final_audit.md", "w").write("\n".join(md) + "\n")
    json.dump(js, open(f"{out}/final_audit.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
