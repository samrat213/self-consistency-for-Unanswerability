"""Export the 20-sample VQA votes (aware setting) into the partner's sample format for vqa/ in his repo.

Writes <repo>/vqa/data/<ds>_aware.json = {"split": {"dev": [ids], "test": [ids]}, "results": [item, ...]} where
item = {id, should_abstain, pred_label[20], pred_answer[20], is_correct[20]} as read by experiments_CRP.extract_clusters:
pred_label is "ABSTAIN" for an Unanswerable vote, else "ANSWERABLE"; pred_answer is the canonical answer (option
id / Yes / No / UNK-VQA judge cluster id / UNANS); is_correct follows analyze.load_votes' ok map.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as AZ, vqa_data as V

REPO = os.path.expanduser("~/self-consistency-for-Unanswerability")


def main():
    os.makedirs(f"{REPO}/vqa/data", exist_ok=True)
    for ds in ("uvqa", "unkvqa", "spatialuncertain"):
        items = {it["id"]: it for it in V.load(ds)}
        split = json.load(open(f"{V.ROOT}/subsets/{ds}_split.json"))
        qs = AZ.load_votes(ds, "aware", items)
        res = []
        for i in split["dev"] + split["test"]:
            if i not in qs:
                continue
            q = qs[i]
            res.append(dict(id=i, should_abstain=not q["item"]["answerable"],
                            pred_label=["ABSTAIN" if v in (V.UNANS, V.INVALID) else "ANSWERABLE" for v in q["votes"]],
                            pred_answer=list(q["votes"]),
                            is_correct=[bool(q["ok"].get(v, False)) for v in q["votes"]]))
        keep = {r["id"] for r in res}
        out = dict(split={k: [i for i in split[k] if i in keep] for k in ("dev", "test")}, results=res)
        json.dump(out, open(f"{REPO}/vqa/data/{ds}_aware.json", "w"))
        print(ds, len(out["split"]["dev"]), "dev /", len(out["split"]["test"]), "test")


if __name__ == "__main__":
    main()
