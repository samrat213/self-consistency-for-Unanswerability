"""Check the hypothesis on DEV splits only:
  answerable   -> the 20 samples concentrate on the right answer
  unanswerable -> many different answers, OR they concentrate on the IDK token

Per question (clusters and IDK detection exactly as the partner's experiments_fast.py / experiments_fast_KBQA.py):
  n0 = IDK samples, m = 20 - n0 real answers in K clusters, nmax = largest real-answer cluster.
Pattern (M = majority threshold, default 10 of 20):
  IDK-heavy      n0 >= M
  Conc-correct   n0 < M, nmax >= M, largest cluster correct
  Conc-wrong     n0 < M, nmax >= M, largest cluster wrong   (for unanswerable: a confident hallucination)
  Scattered      n0 < M, nmax < M  (no single answer or IDK reaches M)
Prediction: answerable -> Conc-correct; unanswerable -> IDK-heavy or Scattered.

  python check_hypothesis.py [--M 10] [--out results/dev/hypothesis]
"""
import argparse, json, math, os, sys
import numpy as np

REPO = os.path.expanduser("~/scu-sync")
PARTNER = os.path.expanduser("~/uploads/partner_1116")
sys.path.insert(0, REPO)
import experiments_fast as EF
import experiments_fast_KBQA as EK

VQA = ["uvqa", "unkvqa", "spatialuncertain", "haloquest"]
PART = [("ReliableMath", EF), ("brokenmath_type_answer_AU", EF), ("grailqa", EK), ("graphqa", EK)]
PATS = ["Conc-correct", "Conc-wrong", "Scattered", "IDK-heavy"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, c - h), min(1.0, c + h)


def auroc(a, u):
    """P(score_U > score_A) + 0.5 ties."""
    a, u = np.asarray(a, float), np.asarray(u, float)
    if not len(a) or not len(u):
        return float("nan")
    gt = (u[:, None] > a[None, :]).mean()
    eq = (u[:, None] == a[None, :]).mean()
    return float(gt + 0.5 * eq)


def stats(item, mod):
    r = mod.extract_clusters(item, ignore_idk=True)
    N, n0, m, clusters, is_corr = r[0], r[1], r[2], r[3], r[4]
    sizes = sorted((len(c) for c in clusters), reverse=True)
    nmax = sizes[0] if sizes else 0
    top_ok = bool(clusters) and any(is_corr[i] for i in max(clusters, key=len))
    return dict(N=N, n0=n0, m=m, K=len(sizes), nmax=nmax, single=sum(s == 1 for s in sizes),
                top_ok=top_ok, sizes=sizes)


def pattern(s, M):
    if s["n0"] >= M:
        return "IDK-heavy"
    if s["nmax"] >= M:
        return "Conc-correct" if s["top_ok"] else "Conc-wrong"
    return "Scattered"


def load_vqa(ds, setting):
    d = json.load(open(f"{REPO}/vqa/data/{ds}_{setting}.json"))
    dev = set(d["split"]["dev"])
    return [r for r in d["results"] if r["id"] in dev]


def load_partner(ds):
    val = {x["question"].strip() for x in json.load(open(f"{PARTNER}/splits/val/{ds}.json"))}
    res = json.load(open(f"{PARTNER}/raw/{ds}_abstaintion_samples.json"))["results"]
    return [r for r in res if str(r.get("question", "")).strip() in val]


def is_unans(r):
    return r.get("should_abstain") in (True, "True", "true")


def analyse(rows, mod, M):
    out = {"A": [], "U": []}
    for r in rows:
        s = stats(r, mod)
        if s["N"] == 0:
            continue
        s["pattern"] = pattern(s, M)
        out["U" if is_unans(r) else "A"].append(s)
    res = {}
    for cls, ss in out.items():
        n = len(ss)
        res[cls] = dict(n=n, **{p: wilson(sum(s["pattern"] == p for s in ss), n) for p in PATS})
        # n0 shape among this class (for the two-mode idea)
        n0 = np.array([s["n0"] for s in ss])
        res[cls]["n0_bins"] = {"0": int((n0 == 0).sum()), "1-9": int(((n0 >= 1) & (n0 < M)).sum()),
                               f"{M}-19": int(((n0 >= M) & (n0 < 20)).sum()), "20": int((n0 == 20).sum())}
    A, U = out["A"], out["U"]
    pred_ok = lambda cls, s: (s["pattern"] == "Conc-correct") if cls == "A" else (s["pattern"] in ("IDK-heavy", "Scattered"))
    res["holds"] = {c: wilson(sum(pred_ok(c, s) for s in out[c]), len(out[c])) for c in "AU"}
    # separation of the raw signals (higher = more unanswerable-looking)
    sig = {"n0 (IDK count)": lambda s: s["n0"], "-nmax (largest real cluster)": lambda s: -s["nmax"],
           "K (real clusters)": lambda s: s["K"],
           "K among non-IDK-heavy": None}
    res["auroc"] = {}
    for name, f in sig.items():
        if f is None:
            a = [s["K"] for s in A if s["n0"] < M]
            u = [s["K"] for s in U if s["n0"] < M]
        else:
            a, u = [f(s) for s in A], [f(s) for s in U]
        res["auroc"][name] = auroc(a, u)
    # "many different answers": singletons and K when the unanswerable q is not IDK-heavy
    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")
    res["scatter"] = {c: dict(K=mean([s["K"] for s in out[c] if s["n0"] < M]),
                              singletons=mean([s["single"] for s in out[c] if s["n0"] < M]),
                              nmax_over_m=mean([s["nmax"] / s["m"] for s in out[c] if s["n0"] < M and s["m"]]))
                      for c in "AU"}
    # among unanswerable: is IDK-heavy vs scattered a split into two groups (bimodal n0)?
    res["U_conc_wrong_unanimous"] = sum(s["pattern"] == "Conc-wrong" and s["nmax"] == 20 for s in U)
    return res


def fmt(p):
    return f"{100*p[0]:.0f} ({100*p[1]:.0f}–{100*p[2]:.0f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--out", default=os.path.expanduser("~/vqa-datasets/results/dev/hypothesis"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = []
    for ds in VQA:
        for st in ("aware", "unaware"):
            runs.append((f"VQA {ds}", st, load_vqa(ds, st), EF))
    for ds, mod in PART:
        runs.append((ds, "aware", load_partner(ds), mod))
    allres = {}
    md = [f"# Hypothesis check on dev (M = {a.M} of 20)", "",
          "% of questions (95% Wilson CI). Prediction: answerable → Conc-correct; unanswerable → Scattered or IDK-heavy.", "",
          "| dataset | setting | class | n | Conc-correct | Conc-wrong | Scattered | IDK-heavy | **holds** |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, st, rows, mod in runs:
        r = analyse(rows, mod, a.M)
        allres[f"{name}|{st}"] = r
        for c in "AU":
            x = r[c]
            md.append(f"| {name} | {st} | {'answerable' if c == 'A' else 'unanswerable'} | {x['n']} | "
                      + " | ".join(fmt(x[p]) for p in PATS) + f" | **{fmt(r['holds'][c])}** |")
    md += ["", "## Separation (AUROC, unanswerable vs answerable; 0.5 = no signal)", "",
           "| dataset | setting | n0 | −nmax | K | K (not IDK-heavy) | U unanimous wrong |", "|---|---|---|---|---|---|---|"]
    for k, r in allres.items():
        name, st = k.split("|")
        au = r["auroc"]
        md.append(f"| {name} | {st} | " + " | ".join(f"{v:.2f}" for v in au.values()) + f" | {r['U_conc_wrong_unanimous']} |")
    md += ["", "## Scatter when not IDK-heavy (means) and n0 distribution", "",
           "| dataset | setting | K A / U | singletons A / U | nmax/m A / U | n0 bins A | n0 bins U |", "|---|---|---|---|---|---|---|"]
    for k, r in allres.items():
        name, st = k.split("|")
        s = r["scatter"]
        md.append(f"| {name} | {st} | {s['A']['K']:.1f} / {s['U']['K']:.1f} | {s['A']['singletons']:.1f} / {s['U']['singletons']:.1f} | "
                  f"{s['A']['nmax_over_m']:.2f} / {s['U']['nmax_over_m']:.2f} | {r['A']['n0_bins']} | {r['U']['n0_bins']} |")
    open(f"{a.out}/hypothesis_M{a.M}.md", "w").write("\n".join(md) + "\n")
    json.dump(allres, open(f"{a.out}/hypothesis_M{a.M}.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
