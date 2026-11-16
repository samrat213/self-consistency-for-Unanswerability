"""Final reserved-tables model, chosen on dev CV only (see run_reserved_tables.py): Beta-Binomial IDK + CRP,
plus a separate empty-result table for KBQA (no option-aware urn, J = 1). Off KBQA this is exactly RT0 / Hurdle_CRP.
Reports dev 5-fold CV, then test once. Baseline rows are fixed methods (not selected on test).
"""
import json, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_reserved_tables as R, reserved_tables as RT, check_hypothesis as H

R.SPECS["RTfinal"] = RT.Spec(urn=False, empty_table=True, J=1)
NAME = "RTfinal"


def one(name):
    (xd, yd), (xt, yt) = R.load(name)
    cvs = R.cv(NAME, xd, yd)
    test, tau = R.evaluate(NAME, xd, yd, xt, yt)
    return name, dict(cv=cvs, test=test, tau=tau, n_dev=len(xd), n_test=len(xt))


if __name__ == "__main__":
    names = [(ds, st) for ds in H.VQA for st in ("aware", "unaware")] + [(ds, "aware") for ds, _ in H.PART]
    out = {}
    with ProcessPoolExecutor(8) as ex:
        for name, r in ex.map(one, names):
            out[f"{name[0]}|{name[1]}"] = r
            print(f"{name[0]:26s} {name[1]:7s} devCV {100*r['cv']:.1f}  test {100*r['test']['f1']:.1f}", flush=True)
    d = os.path.expanduser("~/vqa-datasets/results/reserved_tables")
    json.dump(out, open(f"{d}/final.json", "w"), indent=1, default=float)
