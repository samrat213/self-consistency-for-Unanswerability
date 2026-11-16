"""Write the feature-based logistic regression ("LogReg_Features") into the partner's dashboard/results.json.

Features per question (same as run_reserved_tables.feats): IDK share n0/N, empty share e/N (KBQA; 0 elsewhere),
normalised concentration c of the largest real cluster ((s - 1/L)/(1 - 1/L) for L options, s for free-form), K/N.
Logistic regression (sklearn, C = 1) fitted on dev; score = decision function (log-odds of unanswerable); tau tuned on
dev for Overall F1. Abstain if score > tau, else answer the largest real cluster. Same entries, splits and naming as
export_rt_dashboard.py (KBQA rows: "LogReg_Features (empty = abstain)").

  python export_logreg_dashboard.py --repo ~/scu-sync
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_hypothesis as H
import run_reserved_tables as R
import reserved_tables as RT
import export_rt_dashboard as X
from sklearn.linear_model import LogisticRegression

FEATS = ["n0/N", "e/N", "concentration c", "K/N"]


def build(name):
    (xd, yd), (xt, yt) = R.load(name)
    dev_items, test_items, mod = X.raw_rows(name)
    nz = lambda its: [it for it in its if mod.extract_clusters(it, ignore_idk=True)[0] > 0]
    dev_items, test_items = nz(dev_items), nz(test_items)
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(R.feats(xd), yd)
    sd, st = lr.decision_function(R.feats(xd)), lr.decision_function(R.feats(xt))
    tau = RT.tune_tau(sd, xd, yd)
    e = X.entry(test_items, xt, yt, st, tau, mod)
    e["val_set"] = X.entry(dev_items, xd, yd, sd, tau, mod)
    e["learned_threshold"] = round(float(tau), 4)
    e["threshold_label"] = f"τ = {tau:.2f}"
    e["learned_params"] = {"tau": float(tau), "intercept": round(float(lr.intercept_[0]), 4),
                           "coef": {f: round(float(c), 4) for f, c in zip(FEATS, lr.coef_[0])}}
    e["method_note"] = ("LogReg_Features: logistic regression on (IDK share, empty share, normalised concentration, "
                        "K/N), fitted on dev; tau tuned on dev for Overall F1.")
    return name, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.expanduser("~/scu-sync"))
    a = ap.parse_args()
    names = [(ds, st) for ds in H.VQA for st in ("aware", "unaware")] + [(ds, "aware") for ds, _ in H.PART]
    path = f"{a.repo}/dashboard/results.json"
    dash = json.load(open(path))
    for name in names:
        name, e = build(name)
        ds, st = name
        key = X.DS_KEY.get(ds, ds)
        aw = "_abstaintion" if st == "aware" else ""
        meth = "LogReg_Features (empty = abstain)" if ds in ("grailqa", "graphqa") else "LogReg_Features"
        dash["result_qwen38"].setdefault(key, {}).setdefault(aw, {})[meth] = e
        print(f"{key:28s} {st:8s} {meth:36s} test F1 {100*e['metrics']['overall_f1']:.0f}  dev F1 {100*e['val_set']['metrics']['overall_f1']:.0f}  tau {e['learned_threshold']}", flush=True)
    json.dump(dash, open(path, "w"), indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
