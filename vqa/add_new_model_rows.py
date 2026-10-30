"""Run the generative-restaurant models added to experiments_CRP.py (Binomial / constrained-BB gates, SingleRest,
PYP without IDK table) on the VQA votes in vqa/data/ and write them to dashboard/results.json under
result_qwen38/VQA_<DS>/{"_abstaintion", ""}. The four earlier TwoRest models are recomputed with the current code
too, so every model row comes from the same code as the math rows; the change against the old row is printed.
Rule rows (Greedy ... CRP_theta) and the Oracle rows are left untouched.

  python vqa/add_new_model_rows.py
"""
import json, os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
import experiments_CRP as E

KEYS = {"uvqa": "VQA_UVQA", "unkvqa": "VQA_UNKVQA", "spatialuncertain": "VQA_SpatialUncertain",
        "haloquest": "VQA_HaloQuest"}
NEW = ["TwoRest_Binomial_Hurdle_CRP", "TwoRest_Binomial_ZI_PYP", "TwoRest_ConstrainedBB_ZI_PYP",
       "SingleRest_Binomial_ZI_PYP_Surprisal", "SingleRest_Binomial_ZI_PYP_Predictive",
       "SingleRest_ZI_PYP_Surprisal", "SingleRest_ZI_PYP_Predictive", "SingleRest_PYP_NoIDK", "TwoRest_PYP_NoIDK"]
OLD = ["TwoRest_CRP", "TwoRest_Reserved_CRP", "TwoRest_Hurdle_CRP", "TwoRest_ZI_PYP"]


def main():
    work = tempfile.mkdtemp(prefix="newrows_")
    E.BASE_DIRECTORY = work
    E.RUN_ALL_16_ABLATION_COMBOS = False
    os.makedirs(f"{work}/result_qwen38", exist_ok=True)
    path = f"{REPO}/dashboard/results.json"
    dash = json.load(open(path))
    for ds, key in KEYS.items():
        for setting, aw in (("aware", "_abstaintion"), ("unaware", "")):
            d = json.load(open(f"{HERE}/data/{ds}_{setting}.json"))
            json.dump({"results": [dict(r, question=r["id"]) for r in d["results"]]},
                      open(f"{work}/result_qwen38/{ds}{aw}_samples.json", "w"))
            dev, test = set(d["split"]["dev"]), set(d["split"]["test"])
            tgt = dash["result_qwen38"][key][aw]
            _, _, _, res = E.process_combination("result_qwen38", ds, aw, dev, test, NEW + OLD)
            for k, v in res.items():
                f1 = round(100 * v["metrics"]["overall_f1"], 1)
                old = f"was {100 * tgt[k]['metrics']['overall_f1']:.0f}" if k in tgt else "new"
                tgt[k] = v
                print(f"{ds:16s} {setting:7s} {k:45s} {f1:5.1f}  {old}", flush=True)
    json.dump(dash, open(path, "w"), indent=2, allow_nan=False)      # same format as the committed file


if __name__ == "__main__":
    main()
