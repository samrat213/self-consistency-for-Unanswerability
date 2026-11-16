"""Rows whose threshold is tuned on dev for ANSWERABILITY F1 (a wrong answer to an answerable question counts as a
correct 'answer' decision): Hurdle_CRP, CRP_k, scUn + T, LogReg_Features, for the 4 VQA datasets (aware + unaware)
and the partner's ReliableMath, brokenmath_type_answer_AU, grailqa, graphqa (aware). Same item format, splits and
decision rule as export_logreg_dashboard.py (KBQA: an empty result is never given as an answer; rows suffixed
"(empty = abstain)"). Scores: answerability_f1.py (Hurdle_CRP = Beta-Binomial IDK + CRP, CRP_k = number of outcome
groups incl. IDK/empty, scUn + T = top real-answer share of 20).

  python export_ansf1_dashboard.py --repo ~/scu-sync
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_hypothesis as H, run_reserved_tables as R, export_rt_dashboard as X, answerability_f1 as AF

METHODS = {"Hurdle_CRP": "Hurdle_CRP", "CRP_k": "CRP_k", "scUn + T": "scUn + T", "LogReg_Features": "LogReg_Features"}

def build(name, method):
    (xd, yd), (xt, yt) = R.load(name)
    dev_items, test_items, mod = X.raw_rows(name)
    nz = lambda its: [it for it in its if mod.extract_clusters(it, ignore_idk=True)[0] > 0]
    dev_items, test_items = nz(dev_items), nz(test_items)
    sc = AF.scorer(method, xd, yd)
    sd, st = sc(xd), sc(xt)
    tau = AF.tune(sd, xd, yd, "answerability")
    e = X.entry(test_items, xt, yt, st, tau, mod)
    e["val_set"] = X.entry(dev_items, xd, yd, sd, tau, mod)
    e["learned_threshold"] = round(float(tau), 4)
    e["threshold_label"] = f"τ = {tau:.2f}"
    e["tuned_on"] = "Answerability F1"
    e["method_note"] = f"{method}: threshold tuned on dev for answerability F1 (wrong answer to an answerable question = correct decision)."
    return e

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", default=os.path.expanduser("~/scu-sync")); a = ap.parse_args()
    names = [(ds, st) for ds in H.VQA for st in ("aware", "unaware")] + [(ds, "aware") for ds, _ in H.PART]
    path = f"{a.repo}/dashboard/results.json"; dash = json.load(open(path))
    for name in names:
        ds, st = name; key = X.DS_KEY.get(ds, ds); aw = "_abstaintion" if st == "aware" else ""
        for m in METHODS:
            e = build(name, m)
            meth = f"{m} [tuned: answerability F1]" + (" (empty = abstain)" if ds in ("grailqa", "graphqa") else "")
            dash["result_qwen38"].setdefault(key, {}).setdefault(aw, {})[meth] = e
            print(f"{key:26s} {st:8s} {meth:52s} test overall F1 {100*e['metrics']['overall_f1']:3.0f}", flush=True)
    json.dump(dash, open(path, "w"), indent=2, allow_nan=False)

if __name__ == "__main__":
    main()
