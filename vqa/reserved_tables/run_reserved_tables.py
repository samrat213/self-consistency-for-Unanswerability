"""Fit the reserved-tables models on dev, tune tau on dev (Overall F1), report test once.

Datasets: VQA uvqa/unkvqa/spatialuncertain/haloquest (aware + unaware) and the partner's ReliableMath,
brokenmath_type_answer_AU, grailqa, graphqa (aware). KBQA convention for EVERY method here: an empty query result is
an abstention, never an answer (differs from the dashboard, which answers with it).
Also: 5-fold CV inside dev (params + tau refit on 4 folds, scored on the 5th) for every method; RTcv picks J by it.

  python run_reserved_tables.py [--out ~/vqa-datasets/results/reserved_tables]
"""
import argparse, json, os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_hypothesis as H
import check_hypothesis_v2 as V2
import reserved_tables as RT
from sklearn.linear_model import LogisticRegression

SPECS = {"RT0 Hurdle-CRP (empty = answer cluster)": RT.Spec(urn=False, empty_table=False, J=1),
         "RT1 + option-aware urn": RT.Spec(urn=True, empty_table=False, J=1),
         "RT2 + empty table": RT.Spec(urn=True, empty_table=True, J=1),
         "RT3 + 2-component mixture": RT.Spec(urn=True, empty_table=True, J=2)}


def load(name):
    ds, st = name
    Lmap = V2.n_options() if ds in ("uvqa", "spatialuncertain") else {}
    if ds in H.VQA:
        d = json.load(open(f"{H.REPO}/vqa/data/{ds}_{st}.json"))
        dev_ids, test_ids = set(d["split"]["dev"]), set(d["split"]["test"])
        rows = d["results"]
        dev = [r for r in rows if r["id"] in dev_ids]
        test = [r for r in rows if r["id"] in test_ids]
        mod, kb = H.EF, False
    else:
        mod = dict(H.PART)[ds]
        kb = mod is H.EK
        val = {x["question"].strip() for x in json.load(open(f"{H.PARTNER}/splits/val/{ds}.json"))}
        tst = {x["question"].strip() for x in json.load(open(f"{H.PARTNER}/splits/test/{ds}.json"))}
        rows = json.load(open(f"{H.PARTNER}/raw/{ds}_abstaintion_samples.json"))["results"]
        dev = [r for r in rows if str(r.get("question", "")).strip() in val]
        # partner's split_output_data: a question text in both val and test files counts as val only
        test = [r for r in rows if str(r.get("question", "")).strip() in tst and str(r.get("question", "")).strip() not in val]
    f = lambda rs: ([RT.featurise(r, mod, None if ds in V2.FREE else Lmap[r["id"]], kb) for r in rs],
                    np.array([H.is_unans(r) for r in rs]))
    (xd, yd), (xt, yt) = f(dev), f(test)
    keep = lambda xs, ys: ([x for x in xs if x["N"] > 0], ys[[x["N"] > 0 for x in xs]])
    return keep(xd, yd), keep(xt, yt)


# ---------- baselines as score functions (higher = more unanswerable-looking) ----------
def sc_abstain(xs):
    """majority over outcome groups: IDK, empty, each real cluster; abstain if IDK or empty wins."""
    return np.array([max(x["n0"], x["e"]) >= (x["sizes"][0] if x["sizes"] else 0) for x in xs])


def feats(xs):
    def c(x):
        if not x["r"]:
            return 0.0
        s = x["sizes"][0] / x["r"]
        return s if not x["L"] else (s - 1 / x["L"]) / (1 - 1 / x["L"])
    return np.array([[x["n0"] / x["N"], x["e"] / x["N"], c(x), len(x["sizes"]) / x["N"]] for x in xs])


def fit_scorer(method, xd, yd):
    if method in SPECS:
        m = RT.ReservedTables(SPECS[method]).fit(xd, yd)
        return m.score
    if method == "CRP_k (outcome groups)":
        return lambda xs: np.array([x["n_groups"] for x in xs], float)
    if method == "LogReg (n0, e, c, K)":
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(feats(xd), yd)
        return lambda xs: lr.decision_function(feats(xs))
    raise KeyError(method)


def evaluate(method, xd, yd, xt, yt):
    if method == "SC (majority, IDK/empty = abstain)":
        return RT.f1_from(RT.cells(xt, yt, sc_abstain(xt))), None
    sc = fit_scorer(method, xd, yd)
    sd = sc(xd)
    tau = RT.tune_tau(sd, xd, yd)
    return RT.f1_from(RT.cells(xt, yt, sc(xt) > tau)), tau


def cv(method, xd, yd, k=5, seed=0):
    """grouped by nothing (questions independent within these splits), class-stratified folds."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(xd))
    fold = np.zeros(len(xd), int)
    for cls in (False, True):
        ii = idx[yd == cls]
        rng.shuffle(ii)
        fold[ii] = np.arange(len(ii)) % k
    out = []
    for f in range(k):
        tr, te = fold != f, fold == f
        xtr = [x for x, t in zip(xd, tr) if t]
        xte = [x for x, t in zip(xd, te) if t]
        try:
            r, _ = evaluate(method, xtr, yd[tr], xte, yd[te])
            out.append(r["f1"])
        except Exception:
            out.append(np.nan)
    return float(np.nanmean(out))


METHODS = ["SC (majority, IDK/empty = abstain)", "CRP_k (outcome groups)", "LogReg (n0, e, c, K)"] + list(SPECS)


def run_one(name):
    t0 = time.time()
    (xd, yd), (xt, yt) = load(name)
    res = {"n_dev": len(xd), "n_test": len(xt), "kbqa": xd[0]["kbqa"], "methods": {}}
    for m in METHODS:
        if m.startswith("RT2") and not xd[0]["kbqa"]:
            pass   # identical to RT1 off KBQA; still run so the ladder is complete
        test, tau = evaluate(m, xd, yd, xt, yt)
        res["methods"][m] = dict(test=test, tau=tau, cv=cv(m, xd, yd))
    # RTcv: J = 1 or 2 by dev CV
    j = "RT3 + 2-component mixture" if res["methods"]["RT3 + 2-component mixture"]["cv"] > \
        res["methods"]["RT2 + empty table"]["cv"] else "RT2 + empty table"
    res["methods"]["RTcv (J by dev CV)"] = dict(res["methods"][j], chose=j.split()[0])
    res["secs"] = round(time.time() - t0, 1)
    return name, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/vqa-datasets/results/reserved_tables"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = [(ds, st) for ds in H.VQA for st in ("aware", "unaware")] + [(ds, "aware") for ds, _ in H.PART]
    allres = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for name, res in ex.map(run_one, names):
            allres[f"{name[0]}|{name[1]}"] = res
            print(f"{name[0]:26s} {name[1]:7s} done in {res['secs']}s", flush=True)
    json.dump(allres, open(f"{a.out}/results.json", "w"), indent=1, default=float)
    cols = METHODS + ["RTcv (J by dev CV)"]
    md = ["# Reserved-tables model vs baselines (test Overall F1 ×100; dev 5-fold CV mean in brackets)", "",
          "Fitted + tau tuned on dev only; test touched once. KBQA: empty result = abstention for all methods.", "",
          "| dataset | setting | " + " | ".join(c.split(" (")[0] if not c.startswith("RT") else c.split()[0] for c in cols) + " |",
          "|---|---|" + "---|" * len(cols)]
    for k, r in allres.items():
        ds, st = k.split("|")
        cells = []
        best = max(r["methods"][c]["test"]["f1"] for c in cols)
        for c in cols:
            v = r["methods"][c]
            s = f"{100*v['test']['f1']:.1f}"
            s = f"**{s}**" if v["test"]["f1"] >= best - 1e-12 else s
            cv_ = v.get("cv")
            cells.append(s + (f" [{100*cv_:.1f}]" if cv_ is not None and not np.isnan(cv_) else "") +
                         (f" ({v['chose']})" if "chose" in v else ""))
        md.append(f"| {ds} | {st} | " + " | ".join(cells) + " |")
    md += ["", "Columns: " + "; ".join(cols), ""]
    md += ["## A-acc / U-acc / answered (test %) for CRP_k, RT0, RT2, RTcv", "",
           "| dataset | setting | CRP_k | RT0 | RT2 | RTcv |", "|---|---|---|---|---|---|"]
    for k, r in allres.items():
        ds, st = k.split("|")
        g = lambda c: "{:.0f}/{:.0f}/{:.0f}".format(*(100 * r["methods"][c]["test"][q] for q in ("A_acc", "U_acc", "answered")))
        md.append(f"| {ds} | {st} | {g('CRP_k (outcome groups)')} | {g('RT0 Hurdle-CRP (empty = answer cluster)')} | "
                  f"{g('RT2 + empty table')} | {g('RTcv (J by dev CV)')} |")
    open(f"{a.out}/results.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
