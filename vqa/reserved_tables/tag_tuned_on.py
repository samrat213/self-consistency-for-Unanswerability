"""Add a 'tuned_on' field to every dashboard entry (what its threshold was tuned on, dev split).
Verified 2026-11-16 by re-running the partner's learn_best_threshold on his raw dev data (result_qwen38):
math datasets reproduce 6/6 stored thresholds with overall_accuracy (experiments_fast.py), KBQA 6/6 with overall_f1
(experiments_fast_KBQA.py). Other models are assumed to come from the same scripts. Our VQA rows and LogReg_Features
rows: Overall F1. Rows without a tuned threshold: 'none (fixed rule)'. Existing 'tuned_on' fields are kept.
"""
import json, re, sys
path = sys.argv[1]
d = json.load(open(path))
FIXED = re.compile(r"^(01_Greedy|02_self_Consistency|03_Majority|22_Oracle|Greedy$|self Consistency$|Majority$|Adaptive_Consistency_beta_0\.95$)")
MATH = {"ReliableMath", "brokenmath_type_answer_AU", "brokenmath_type_proof_AU", "UA_commonsense", "UA_math"}
KBQA = {"grailqa", "graphqa"}
counts = {}
for m in d:
    for ds in d[m]:
        for aw in d[m][ds]:
            for k, e in d[m][ds][aw].items():
                if not isinstance(e, dict) or "metrics" not in e or "tuned_on" in e:
                    continue
                if FIXED.match(k):
                    t = "none (fixed rule)"
                elif ds.startswith("VQA_") or k.startswith("LogReg_Features"):
                    t = "Overall F1"
                elif ds in MATH:
                    t = "Overall accuracy"
                elif ds in KBQA:
                    t = "Overall F1"
                else:
                    t = None
                if t:
                    e["tuned_on"] = t
                    counts[t] = counts.get(t, 0) + 1
                else:
                    counts["UNTAGGED " + ds] = counts.get("UNTAGGED " + ds, 0) + 1
json.dump(d, open(path, "w"), indent=2, allow_nan=False)
print(counts)
