"""Write the final reserved-tables model ("Reserved_Tables") into the partner's dashboard/results.json.

Model (chosen by dev CV only): Beta-Binomial on the IDK count + CRP on the real answers; KBQA adds a separate table for
empty query results (Dirichlet-Multinomial over NK / empty / real). One component per class, fitted by maximum
likelihood on dev; tau tuned on dev for Overall F1. Abstain if log p(x|U) - log p(x|A) > tau, else answer the largest
real (non-IDK, non-empty) cluster.

Rows: result_qwen38 / <dataset> / "_abstaintion" | "" / "Reserved_Tables" for the 4 VQA datasets (aware + unaware) and
the partner's ReliableMath, brokenmath_type_answer_AU, grailqa, graphqa (aware; his files are result_qwen38 — greedy and
self-consistency counts match the dashboard's result_qwen38 rows). KBQA rows are named
"Reserved_Tables (empty = abstain)" because an empty result counts as an abstention, unlike the other KBQA rows.
Entry fields follow experiments_fast.process_run (matrix, error_analysis, histograms, metrics + AURC, val_set).

  python export_rt_dashboard.py --repo ~/scu-sync
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_hypothesis as H
import run_reserved_tables as R
import reserved_tables as RT

SPEC = RT.Spec(urn=False, empty_table=True, J=1)
DS_KEY = {"uvqa": "VQA_UVQA", "unkvqa": "VQA_UNKVQA", "spatialuncertain": "VQA_SpatialUncertain",
          "haloquest": "VQA_HaloQuest"}


def raw_rows(name):
    """the item dicts in the same order/filter as R.load (needed for partner-style metadata)."""
    ds, st = name
    if ds in H.VQA:
        d = json.load(open(f"{H.REPO}/vqa/data/{ds}_{st}.json"))
        dv, ts = set(d["split"]["dev"]), set(d["split"]["test"])
        return [r for r in d["results"] if r["id"] in dv], [r for r in d["results"] if r["id"] in ts], H.EF
    mod = dict(H.PART)[ds]
    val = {x["question"].strip() for x in json.load(open(f"{H.PARTNER}/splits/val/{ds}.json"))}
    tst = {x["question"].strip() for x in json.load(open(f"{H.PARTNER}/splits/test/{ds}.json"))}
    rows = json.load(open(f"{H.PARTNER}/raw/{ds}_abstaintion_samples.json"))["results"]
    q = lambda r: str(r.get("question", "")).strip()
    return [r for r in rows if q(r) in val], [r for r in rows if q(r) in tst and q(r) not in val], mod


def entry(items, xs, ys, scores, tau, mod):
    mod.precompute_dataset_caches(items)
    res = {"ans_count": 0, "unans_count": 0, "ans_correct_in_samples": 0, "unans_correct_in_samples": 0,
           "matrix_gt_pred_counts": {"ANSWERABLE": {"Pred_ANSWERABLE_correct": 0, "Pred_ANSWERABLE_wrong": 0, "Pred_Unanswerable": 0},
                                     "ABSTAIN": {"Pred_ANSWERABLE": 0, "Pred_Unanswerable": 0}},
           "error_analysis": {"FN_missed_answerable": {"count": 0, "correct_present_count": 0, "correct_was_most_freq_count": 0, "sum_correct_freq": 0},
                              "FP_hallucinated_unanswerable": {"count": 0, "correct_present_count": 0, "sum_winning_freq": 0, "sum_total_clusters": 0},
                              "TN_successful_abstention": {"count": 0, "sum_total_clusters": 0},
                              "answerable_max_freq_histogram": {str(i): 0 for i in range(21)},
                              "unanswerable_max_freq_histogram": {str(i): 0 for i in range(21)}}}
    ea, mx = res["error_analysis"], res["matrix_gt_pred_counts"]
    xs_items = list(zip(items, xs))
    assert len(items) == len(xs) == len(scores)
    for (it, x), y, s in zip(xs_items, ys, scores):
        max_freq, total_clusters, correct_present, correct_freq, corr_most = mod.get_item_metadata(it, ignore_abstention_tokens=True)
        hk = str(min(max_freq, 20))
        abstain = bool(s > tau) or x["r"] == 0
        if y:
            res["unans_count"] += 1
            res["unans_correct_in_samples"] += bool(correct_present)
            ea["unanswerable_max_freq_histogram"][hk] += 1
            if abstain:
                mx["ABSTAIN"]["Pred_Unanswerable"] += 1
                ea["TN_successful_abstention"]["count"] += 1
                ea["TN_successful_abstention"]["sum_total_clusters"] += total_clusters
            else:
                mx["ABSTAIN"]["Pred_ANSWERABLE"] += 1
                f = ea["FP_hallucinated_unanswerable"]
                f["count"] += 1
                f["correct_present_count"] += bool(correct_present)
                f["sum_winning_freq"] += max_freq
                f["sum_total_clusters"] += total_clusters
        else:
            res["ans_count"] += 1
            res["ans_correct_in_samples"] += bool(correct_present)
            ea["answerable_max_freq_histogram"][hk] += 1
            if abstain:
                mx["ANSWERABLE"]["Pred_Unanswerable"] += 1
                f = ea["FN_missed_answerable"]
                f["count"] += 1
                if correct_present:
                    f["correct_present_count"] += 1
                    f["sum_correct_freq"] += correct_freq
                    f["correct_was_most_freq_count"] += bool(corr_most)
            else:
                mx["ANSWERABLE"]["Pred_ANSWERABLE_correct" if x["top_ok"] else "Pred_ANSWERABLE_wrong"] += 1
    fn, fp, tn = ea["FN_missed_answerable"], ea["FP_hallucinated_unanswerable"], ea["TN_successful_abstention"]
    fn["avg_correct_freq_when_present"] = round(fn.pop("sum_correct_freq") / fn["correct_present_count"], 2) if fn["correct_present_count"] else 0.0
    fp["avg_winning_freq"] = round(fp.pop("sum_winning_freq") / fp["count"], 2) if fp["count"] else 0.0
    fp["avg_total_clusters"] = round(fp.pop("sum_total_clusters") / fp["count"], 2) if fp["count"] else 0.0
    tn["avg_total_clusters"] = round(tn.pop("sum_total_clusters") / tn["count"], 2) if tn["count"] else 0.0
    res = H.EF.calculate_metrics(res)
    can = [x["r"] > 0 for x in xs]
    wrong = [bool(y) or not x["top_ok"] for x, y in zip(xs, ys)]
    res["metrics"].update(H.EF.risk_coverage_curve(-np.asarray(scores), can, wrong, total_n=len(xs)))
    return res


def build(name):
    (xd, yd), (xt, yt) = R.load(name)
    dev_items, test_items, mod = raw_rows(name)
    nz = lambda its: [it for it in its if mod.extract_clusters(it, ignore_idk=True)[0] > 0]   # same filter as R.load
    dev_items, test_items = nz(dev_items), nz(test_items)
    model = RT.ReservedTables(SPEC).fit(xd, yd)
    sd, st = model.score(xd), model.score(xt)
    tau = RT.tune_tau(sd, xd, yd)
    e = entry(test_items, xt, yt, st, tau, mod)
    e["val_set"] = entry(dev_items, xd, yd, sd, tau, mod)
    e["learned_threshold"] = round(float(tau), 4)
    e["threshold_label"] = f"τ = {tau:.2f}"
    pr = lambda m: {"weights": [round(float(np.exp(w)), 4) for w in m[0]],
                    "params": [[round(float(v), 4) for v in np.exp(p)] for p in m[1]]}
    e["learned_params"] = {"tau": float(tau), "A": pr(model.mA), "U": pr(model.mU),
                           "param_order": "KBQA: alpha_NK, alpha_empty, alpha_real, theta; else: a, b, theta"}
    e["method_note"] = ("Reserved_Tables: Beta-Binomial IDK count + CRP over real answers (KBQA: Dirichlet-Multinomial "
                        "over NK / empty / real + CRP), one component per class, fitted on dev, tau tuned on dev for Overall F1.")
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
        key = DS_KEY.get(ds, ds)
        aw = "_abstaintion" if st == "aware" else ""
        meth = "Reserved_Tables (empty = abstain)" if ds in ("grailqa", "graphqa") else "Reserved_Tables"
        dash["result_qwen38"].setdefault(key, {}).setdefault(aw, {})[meth] = e
        print(f"{key:28s} {st:8s} {meth:36s} test F1 {100*e['metrics']['overall_f1']:.0f}  dev F1 {100*e['val_set']['metrics']['overall_f1']:.0f}  tau {e['learned_threshold']}", flush=True)
    json.dump(dash, open(path, "w"), indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
