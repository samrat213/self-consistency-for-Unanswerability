"""Which tuning target gives better final accuracy: Overall F1 or answerability F1?
For each method, tau is tuned on dev with each target; judged by balanced accuracy (aw = wrong) and by
answerability balanced accuracy (aw = correct 'answer' decision). Dev: 5-fold CV x 5 seeds (selection evidence); test once.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import answerability_f1 as AF, run_reserved_tables as R, reserved_tables as RT, check_hypothesis as H
from concurrent.futures import ProcessPoolExecutor

def bal(c, answ=False):
    ac, aw, au, ua, uu = c
    ra = (ac + aw) / (ac + aw + au) if answ else ac / (ac + aw + au)
    return 100 * (ra + uu / (ua + uu)) / 2

def run(method, xd, yd, xt, yt, kind):
    if method == "Majority vote": return RT.cells(xt, yt, R.sc_abstain(xt))
    sc = AF.scorer(method, xd, yd); tau = AF.tune(sc(xd), xd, yd, kind)
    return RT.cells(xt, yt, sc(xt) > tau)

def cvc(method, xd, yd, kind, seeds=5, k=5):
    tot = np.zeros(5)
    for seed in range(seeds):
        rng = np.random.default_rng(seed); fold = np.zeros(len(xd), int)
        for c in (False, True):
            ii = np.where(yd == c)[0]; rng.shuffle(ii); fold[ii] = np.arange(len(ii)) % k
        for f in range(k):
            tr, te = fold != f, fold == f
            tot += run(method, [x for x, t in zip(xd, tr) if t], yd[tr], [x for x, t in zip(xd, te) if t], yd[te], kind)
    return tot   # pooled counts over folds and seeds

METH = ["CRP_k", "scUn + T", "Hurdle_CRP", "LogReg_Features"]
def one(n):
    (xd, yd), (xt, yt) = R.load(n); out = {}
    for m in METH:
        for kind in ("overall", "answerability"):
            out[(m, kind)] = (cvc(m, xd, yd, kind), run(m, xd, yd, xt, yt, kind))
    return n, out

if __name__ == "__main__":
    names = [(ds, st) for ds in H.VQA for st in ("aware", "unaware")] + [(ds, "aware") for ds, _ in H.PART]
    with ProcessPoolExecutor(8) as ex: res = list(ex.map(one, names))
    for answ, label in ((False, "balanced accuracy (wrong answer = error)"), (True, "answerability balanced accuracy (wrong answer = correct decision)")):
        print(f"\n## {label}: tuned Overall F1 -> tuned answerability F1   (dev CV | test)")
        wins_dev = wins_test = ties_dev = n = 0; dd = []; dt = []
        for (ds, st), out in res:
            cells = []
            for m in METH:
                (cd_o, ct_o), (cd_a, ct_a) = out[(m, "overall")], out[(m, "answerability")]
                d1, d2 = bal(cd_a, answ) - bal(cd_o, answ), bal(ct_a, answ) - bal(ct_o, answ)
                dd.append(d1); dt.append(d2); n += 1
                cells.append(f"{d1:+5.1f}|{d2:+5.1f}")
            print(f"{ds[:16]:16s} {st:8s} " + "  ".join(f"{m[:10]:>10s} {c}" for m, c in zip(METH, cells)))
        dd, dt = np.array(dd), np.array(dt)
        print(f"answerability-F1 tuning minus Overall-F1 tuning: dev mean {dd.mean():+.2f} (better {int((dd>0.05).sum())}, worse {int((dd<-0.05).sum())}, same {int((abs(dd)<=0.05).sum())} of {n}); "
              f"test mean {dt.mean():+.2f} (better {int((dt>0.05).sum())}, worse {int((dt<-0.05).sum())})")
