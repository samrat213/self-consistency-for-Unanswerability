"""Run the partner's experiments_fast.py (all 22 methods with their keep/drop-IDK ablations) on the VQA votes in
vqa/data/, unchanged, and write the rows to dashboard/results.json under
result_qwen38/VQA_<DS>/{"_abstaintion", ""}. Existing VQA rows with other names (our earlier rule rows) are kept.

  python vqa/run_fast_vqa.py                      # tunes VQA rows on Overall F1, writes the dashboard
  python vqa/run_fast_vqa.py --metric overall_accuracy --out vqa/x.json   # other metric, side file only

The dev/test split is the fixed one stored in each data file (split by item id, since VQA question texts repeat).
"""
import argparse, json, os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
import experiments_fast as E

KEYS = {"uvqa": "VQA_UVQA", "unkvqa": "VQA_UNKVQA", "spatialuncertain": "VQA_SpatialUncertain",
        "haloquest": "VQA_HaloQuest"}


def run(metric):
    if metric:
        E.TARGET_METRIC = metric
    work = tempfile.mkdtemp(prefix="fastvqa_")
    E.BASE_DIRECTORY = work
    os.makedirs(f"{work}/result_qwen38", exist_ok=True)
    out = {}
    for ds, key in KEYS.items():
        for setting, aw in (("aware", "_abstaintion"), ("unaware", "")):
            d = json.load(open(f"{HERE}/data/{ds}_{setting}.json"))
            rows = [dict(r, question=r["id"]) for r in d["results"]]
            json.dump({"results": rows}, open(f"{work}/result_qwen38/{ds}{aw}_samples.json", "w"))
            g = json.load(open(f"{HERE}/data/{ds}_{setting}_greedy.json"))      # temperature-0 answers
            json.dump({"results": [dict(r, question=r["id"]) for r in g["results"]]},
                      open(f"{work}/result_qwen38/{ds}{aw}_greedy.json", "w"))
            dev, test = set(d["split"]["dev"]), set(d["split"]["test"])
            _, _, _, res = E.process_combination("result_qwen38", ds, aw, dev, test, E.REQUESTED_EXPERIMENTS)
            out.setdefault(key, {})[aw] = {k: res[k] for k in sorted(res)}
            best = max(res, key=lambda k: res[k]["metrics"]["overall_f1"] if "Oracle" not in k else -1)
            print(f"{ds:16s} {setting:7s} {len(res)} rows; best non-oracle F1 {best} "
                  f"{res[best]['metrics']['overall_f1']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="overall_f1", help="dev tuning metric for the VQA rows (overrides experiments_fast.TARGET_METRIC)")
    ap.add_argument("--out", default=None, help="write here instead of merging into the dashboard")
    a = ap.parse_args()
    print("TARGET_METRIC =", a.metric or E.TARGET_METRIC)
    out = run(a.metric)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2, allow_nan=False)
        return
    path = f"{REPO}/dashboard/results.json"
    dash = json.load(open(path))
    for key, by_aw in out.items():
        for aw, rows in by_aw.items():
            dash["result_qwen38"].setdefault(key, {}).setdefault(aw, {}).update(rows)
    json.dump(dash, open(path, "w"), indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
