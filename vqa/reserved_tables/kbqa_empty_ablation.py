"""DEV-ONLY ablation of how KBQA empty results enter the likelihood (decision identical: empty never answered).
(a) empty = an ordinary answer cluster in the CRP (partner's Hurdle_CRP view)   Spec(empty_table=False)
(b) empty merged into IDK: n0 += e, CRP over real answers only                    Spec(empty_table=False) on merged x
(c) separate empty table: Dirichlet-Multinomial over NK / empty / real            Spec(empty_table=True)
5-fold CV inside dev, repeated over 20 seeds; also the decision effect alone: (a') empty answered (partner's decision).
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_reserved_tables as R, reserved_tables as RT

def merged(xs):
    return [dict(x, n0=x["n0"] + x["e"], e=0, sizes_e=x["sizes"]) for x in xs]

VARIANTS = {"(a) empty = answer cluster": (RT.Spec(urn=False, empty_table=False), lambda xs: xs),
            "(b) empty merged into IDK": (RT.Spec(urn=False, empty_table=False), merged),
            "(c) separate empty table": (RT.Spec(urn=False, empty_table=True), lambda xs: xs)}

def cv(xd, yd, spec, tf, seed, k=5):
    rng = np.random.default_rng(seed); fold = np.zeros(len(xd), int)
    for c in (False, True):
        ii = np.where(yd == c)[0]; rng.shuffle(ii); fold[ii] = np.arange(len(ii)) % k
    f1 = []
    for f in range(k):
        tr, te = fold != f, fold == f
        xtr = tf([x for x, t in zip(xd, tr) if t]); xte = tf([x for x, t in zip(xd, te) if t])
        m = RT.ReservedTables(spec).fit(xtr, yd[tr])
        tau = RT.tune_tau(m.score(xtr), xtr, yd[tr])
        f1.append(RT.f1_from(RT.cells(xte, yd[te], m.score(xte) > tau))["f1"])
    return np.mean(f1)

for ds in ("grailqa", "graphqa"):
    (xd, yd), _ = R.load((ds, "aware"))
    print(f"{ds}: dev {len(xd)} questions ({int(yd.sum())} unanswerable)")
    res = {}
    for name, (spec, tf) in VARIANTS.items():
        s = np.array([cv(xd, yd, spec, tf, seed) for seed in range(20)])
        res[name] = s
        print(f"  {name:30s} dev CV F1 {100*s.mean():5.1f} ± {100*s.std():.1f} (sd over 20 seeds)")
    d = res["(c) separate empty table"] - res["(b) empty merged into IDK"]
    d2 = res["(c) separate empty table"] - res["(a) empty = answer cluster"]
    print(f"  (c)-(b): {100*d.mean():+.1f}, (c) better in {int((d>0).sum())}/20 seeds;  (c)-(a): {100*d2.mean():+.1f}, better in {int((d2>0).sum())}/20")
