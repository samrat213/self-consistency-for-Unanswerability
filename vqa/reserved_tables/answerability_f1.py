"""Overall F1 vs answerability F1 (a wrong answer to an answerable question counts as a correct 'answer' decision).
Each method's tau is tuned on dev for the metric it is scored on; dev 5-fold CV (selection evidence) + test once.
Methods are fixed in advance (no selection on test). KBQA: an empty result is never given as an answer (as in
run_reserved_tables.py), for every method.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_hypothesis as H, run_reserved_tables as R, reserved_tables as RT
from sklearn.linear_model import LogisticRegression
from concurrent.futures import ProcessPoolExecutor

def f1_metric(c, kind):
    ac, aw, au, ua, uu = c
    if kind == "answerability":
        ac, aw = ac + aw, 0
    return RT.f1_from(np.array([ac, aw, au, ua, uu]))["f1"]

def tune(scores, xs, ys, kind):
    v = np.unique(scores)
    taus = np.concatenate([[v[0] - 1], (v[:-1] + v[1:]) / 2, [v[-1] + 1]])
    res = []
    for t in taus:
        c = RT.cells(xs, ys, scores > t)
        res.append(-1 if c[0] + c[1] + c[3] == 0 else f1_metric(c, kind))
    res = np.array(res); good = taus[res >= res.max() - 1e-12]
    return float(good[len(good) // 2])

HURDLE = RT.Spec(urn=False, empty_table=False, J=1)
def scorer(method, xd, yd):
    if method == "CRP_k": return lambda xs: np.array([x["n_groups"] for x in xs], float)
    if method == "scUn + T": return lambda xs: np.array([-(x["sizes"][0] if x["sizes"] else 0) / x["N"] for x in xs])
    if method == "Hurdle_CRP": m = RT.ReservedTables(HURDLE).fit(xd, yd); return m.score
    if method == "LogReg_Features":
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(R.feats(xd), yd); return lambda xs: lr.decision_function(R.feats(xs))

def evaluate(method, xd, yd, xt, yt, kind):
    if method == "Majority vote":
        return f1_metric(RT.cells(xt, yt, R.sc_abstain(xt)), kind)
    sc = scorer(method, xd, yd)
    tau = tune(sc(xd), xd, yd, kind)
    return f1_metric(RT.cells(xt, yt, sc(xt) > tau), kind)

def cv(method, xd, yd, kind, k=5, seed=0):
    rng = np.random.default_rng(seed); fold = np.zeros(len(xd), int)
    for c in (False, True):
        ii = np.where(yd == c)[0]; rng.shuffle(ii); fold[ii] = np.arange(len(ii)) % k
    out = []
    for f in range(k):
        tr, te = fold != f, fold == f
        out.append(evaluate(method, [x for x, t in zip(xd, tr) if t], yd[tr], [x for x, t in zip(xd, te) if t], yd[te], kind))
    return float(np.mean(out))

METHODS = ["Majority vote", "CRP_k", "scUn + T", "Hurdle_CRP", "LogReg_Features"]

def one(name):
    (xd, yd), (xt, yt) = R.load(name)
    r = {}
    for kind in ("overall", "answerability"):
        for m in METHODS:
            r[(kind, m)] = (cv(m, xd, yd, kind), evaluate(m, xd, yd, xt, yt, kind))
    return name, r

if __name__ == "__main__":
    names = [(ds, st) for ds in H.VQA for st in ("aware", "unaware")] + [(ds, "aware") for ds, _ in H.PART]
    with ProcessPoolExecutor(8) as ex:
        res = dict(ex.map(one, names))
    md = []
    for kind, title in (("overall", "Overall F1 (tau tuned for Overall F1)"), ("answerability", "Answerability F1 (tau tuned for answerability F1)")):
        md += [f"## {title}: test ×100 [dev CV]", "", "| dataset | setting | " + " | ".join(METHODS) + " | best on test |", "|---|---|" + "---|" * (len(METHODS) + 1)]
        for (ds, st), r in res.items():
            vals = {m: r[(kind, m)] for m in METHODS}
            best = max(vals, key=lambda m: vals[m][1])
            md.append(f"| {ds} | {st} | " + " | ".join(f"{100*t:.1f} [{100*c:.1f}]" for c, t in vals.values()) + f" | {best} |")
        md.append("")
    out = os.path.expanduser("~/vqa-datasets/results/reserved_tables/answerability_f1.md")
    open(out, "w").write("\n".join(md)); print("\n".join(md))
