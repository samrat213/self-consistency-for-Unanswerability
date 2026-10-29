"""CRP-family experiments on the VQA unanswerability data (Qwen3.8-27B, answerability-AWARE prompts, 20 samples/q).

Data: vqa/data/<ds>_aware.json (export of the 20-sample votes in the experiments_CRP.py item format + fixed
dev/test split, 10% dev). Everything is tuned on dev and reported on test unless a section says "dev only".

  E1  experiments_CRP.py pipeline as-is (process_combination): baselines, TwoRest_* models, Oracle_Intermediate
  E2  sequential stopping: SPRT on the TwoRest_Hurdle_CRP log-likelihood ratio (and its IDK-only half) vs fixed-n
      rules and Adaptive Consistency -> Overall F1 vs average samples used
  E3  does the Pitman-Yor discount have anything to use? evenness sum_k log n_k at fixed K (dev only)
  E4  hard vs unanswerable: which signals separate answerable-but-failed questions from unanswerable ones (dev only)
  E5  hierarchical CRP from the Colab notebook (alpha ~ Normal per class) fitted to the real samples (dev only)

Run:  python vqa/experiments_vqa.py   (numpy + scipy; matplotlib optional for plots)
Out:  vqa/results_vqa.json, vqa/RESULTS_raw.md, vqa/plots/*.png
"""
import collections, json, math, os, sys, tempfile

import numpy as np
from scipy.special import betainc, betaln, gammaln
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import experiments_CRP as E  # noqa: E402

DATASETS = ["uvqa", "unkvqa", "spatialuncertain"]
N = 20
N_PERM = 10          # sample orders averaged over in E2 (samples are i.i.d.; order is arbitrary)
BUDGETS = [1.5, 2, 3, 4, 5, 7, 10, 15, 20]
FIXED_N = [1, 2, 3, 4, 5, 7, 10, 15, 20]
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- helpers
def load(ds):
    d = json.load(open(f"{HERE}/data/{ds}_aware.json"))
    by = {r["id"]: r for r in d["results"]}
    return [by[i] for i in d["split"]["dev"]], [by[i] for i in d["split"]["test"]]


def f1_counts(c):
    """Overall F1 as in experiments_CRP.calculate_metrics, unrounded. c = dict(a_ok, a_wrong, a_abst, u_ans, u_abst)."""
    nA, nU = c["a_ok"] + c["a_wrong"] + c["a_abst"], c["u_ans"] + c["u_abst"]
    answered, abst = c["a_ok"] + c["a_wrong"] + c["u_ans"], c["a_abst"] + c["u_abst"]
    f = lambda p, r: 2 * p * r / (p + r) if p + r else 0.0
    fa = f(c["a_ok"] / answered if answered else 0, c["a_ok"] / nA if nA else 0)
    fu = f(c["u_abst"] / abst if abst else 0, c["u_abst"] / nU if nU else 0)
    return dict(f1=100 * (fa + fu) / 2, ans_f1=100 * fa, unans_f1=100 * fu,
                A_acc=100 * c["a_ok"] / nA if nA else 0, U_acc=100 * c["u_abst"] / nU if nU else 0,
                answered=100 * answered / (nA + nU))


def score(unans, abstain, correct):
    """Vectorised over questions (any leading axes are averaged as separate runs). Returns f1_counts dict."""
    unans, abstain, correct = np.broadcast_arrays(unans, abstain, correct)
    c = dict(a_ok=np.sum(~unans & ~abstain & correct), a_wrong=np.sum(~unans & ~abstain & ~correct),
             a_abst=np.sum(~unans & abstain), u_ans=np.sum(unans & ~abstain), u_abst=np.sum(unans & abstain))
    return f1_counts({k: float(v) for k, v in c.items()})


def auroc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v; ranks[m] = ranks[m].mean()
    n1 = y.sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (~y).sum()))


def permuted(item, perm):
    out = dict(item)
    for k in ("pred_label", "pred_answer", "is_correct"):
        out[k] = [item[k][j] for j in perm]
    return out


def prefix(item, t):
    out = dict(item)
    for k in ("pred_label", "pred_answer", "is_correct"):
        out[k] = item[k][:t]
    return out


def m2(x):
    return f"{x:.1f}"


# ---------------------------------------------------------------- E1: partner's pipeline as-is
E1_METHODS = ["self Consistency", "Majority", "self Consistency + T", "scUn", "CRP_k", "CRP_theta",
              "Adaptive_Consistency_beta", "TwoRest_CRP", "TwoRest_Reserved_CRP", "TwoRest_Hurdle_CRP",
              "TwoRest_ZI_PYP", "Oracle_Intermediate"]


def e1(ds, dev, test, work):
    """Writes the items as <work>/result_qwen38/<ds>_abstaintion_samples.json (question = id) and calls
    experiments_CRP.process_combination exactly as its __main__ does, one requested experiment at a time."""
    E.BASE_DIRECTORY = work
    E.RUN_ALL_16_ABLATION_COMBOS = True
    os.makedirs(f"{work}/result_qwen38", exist_ok=True)
    items = [dict(r, question=r["id"]) for r in dev + test]
    json.dump({"results": items}, open(f"{work}/result_qwen38/{ds}_abstaintion_samples.json", "w"))
    val_q, test_q = {r["id"] for r in dev}, {r["id"] for r in test}
    out = {}
    for m in E1_METHODS:
        _, _, _, _, res = E.process_combination("result_qwen38", ds, "_abstaintion", val_q, test_q, [m])
        for v in res.values():                      # 16-way grid cells -> flat keys "<method>__V_..__T_.."
            if isinstance(v, dict) and "ablation_grid" in v:
                out.update(v.pop("ablation_grid"))
        out.update(res)
    return out


def e1_row(r):
    M = r["metrics"]; mx = r["matrix_gt_pred_counts"]
    answered = mx["ANSWERABLE"]["Pred_ANSWERABLE_correct"] + mx["ANSWERABLE"]["Pred_ANSWERABLE_wrong"] + mx["ABSTAIN"]["Pred_ANSWERABLE"]
    tot = r["ans_count"] + r["unans_count"]
    return dict(f1=100 * M["overall_f1"], A_acc=100 * M["answerable"]["accuracy"], U_acc=100 * M["unanswerable"]["accuracy"],
                answered=100 * answered / tot, dev_f1=100 * r["val_set"]["metrics"]["overall_f1"] if "val_set" in r else None)


# ---------------------------------------------------------------- per-prefix statistics for E2
def trajectories(items, perms, params):
    """For each perm p, question q, prefix length t=1..N: hurdle LLR, IDK-only LLR, K (IDK kept as a cluster),
    #Unanswerable votes, top-cluster-incl-IDK is IDK / correct, top substantive answer exists / correct, AC prob."""
    P, Q = len(perms), len(items)
    shp = (P, Q, N)
    T = {k: np.zeros(shp) for k in ("llr", "llr_idk", "K", "n0", "ac")}
    for k in ("top_idk", "top_ok", "sub_exists", "sub_ok"):
        T[k] = np.zeros(shp, bool)
    aU, bU, aA, bA = params["a_U"], params["b_U"], params["a_A"], params["b_A"]
    for p, perm in enumerate(perms):
        for q, it in enumerate(items):
            itp = permuted(it, perm)
            for t in range(1, N + 1):
                pre = prefix(itp, t)
                T["llr"][p, q, t - 1] = E.compute_two_restaurant_llr(pre, params)
                Nn, n0, m, cl_keep, ok, lab = E.extract_clusters(pre, ignore_idk=False)
                T["llr_idk"][p, q, t - 1] = ((gammaln(n0 + aU) + gammaln(m + bU) - gammaln(t + aU + bU)) -
                                             (gammaln(n0 + aA) + gammaln(m + bA) - gammaln(t + aA + bA)) -
                                             (betaln(aU, bU) - betaln(aA, bA)))
                sizes = sorted((len(c) for c in cl_keep), reverse=True)
                T["K"][p, q, t - 1] = len(cl_keep)
                T["n0"][p, q, t - 1] = n0
                big = max(cl_keep, key=len)[0]
                T["top_idk"][p, q, t - 1] = lab[big] in E.UNANS_LABELS
                T["top_ok"][p, q, t - 1] = bool(ok[big])
                a, b = sizes[0], sizes[1] if len(sizes) > 1 else 0
                T["ac"][p, q, t - 1] = 1.0 - betainc(a + 1.0, b + 1.0, 0.5)
                _, _, _, cl_sub, _, _ = E.extract_clusters(pre, ignore_idk=True)
                if cl_sub:
                    T["sub_exists"][p, q, t - 1] = True
                    T["sub_ok"][p, q, t - 1] = bool(ok[max(cl_sub, key=len)[0]])
    return T


def seq_decide(stat, A, B, end_tau, t_min=1):
    """SPRT: stop at first t >= t_min with stat >= A (abstain) or stat <= B (answer); undecided at N -> stat > end_tau
    abstains. Returns (abstain[P,Q], stop_t[P,Q] 1-based)."""
    P, Q, _ = stat.shape
    hit = (stat >= A) | (stat <= B)
    hit[:, :, :t_min - 1] = False
    any_hit = hit.any(-1)
    first = np.where(any_hit, hit.argmax(-1), N - 1)
    s = np.take_along_axis(stat, first[..., None], -1)[..., 0]
    abst = np.where(any_hit, s >= A, s > end_tau)
    return abst, first + 1


def outcome_answer_sub(T, abst, stop):
    """Answer = top substantive cluster in the prefix; no substantive answer -> abstain."""
    idx = (stop - 1)[..., None]
    exists = np.take_along_axis(T["sub_exists"], idx, -1)[..., 0]
    ok = np.take_along_axis(T["sub_ok"], idx, -1)[..., 0]
    return abst | ~exists, ok


def outcome_answer_top(T, abst, stop):
    """Answer = top cluster incl. IDK (IDK on top -> abstain), as the partner's CRP_k / AC do with IDK kept."""
    idx = (stop - 1)[..., None]
    top_idk = np.take_along_axis(T["top_idk"], idx, -1)[..., 0]
    ok = np.take_along_axis(T["top_ok"], idx, -1)[..., 0]
    return abst | top_idk, ok


def frontier(cands):
    """cands: list of (name/params, dev_res, dev_avg_t, test_res, test_avg_t). For each budget pick max dev F1 with
    dev avg samples <= budget (ties -> fewer samples); never-answer candidates are excluded."""
    rows = []
    for b in BUDGETS:
        ok = [c for c in cands if c[2] <= b + 1e-9 and c[1]["answered"] > 0]
        if not ok:
            rows.append(None); continue
        best = max(ok, key=lambda c: (round(c[1]["f1"], 6), -c[2]))
        rows.append(dict(budget=b, params=best[0], dev_f1=best[1]["f1"], dev_avg_t=best[2],
                         **{k: best[3][k] for k in ("f1", "A_acc", "U_acc", "answered")}, avg_t=best[4]))
    return rows


def e2(dev, test, hurdle_params):
    perms = [np.arange(N)] + [rng.permutation(N) for _ in range(N_PERM - 1)]
    Td, Tt = trajectories(dev, perms, hurdle_params), trajectories(test, perms, hurdle_params)
    yd = np.array([r["should_abstain"] for r in dev])[None, :]
    yt = np.array([r["should_abstain"] for r in test])[None, :]
    tau = hurdle_params["tau"]
    out = {}

    def run_seq(stat_key, end_tau, answer_fn):
        qs = np.quantile(Td[stat_key], np.linspace(0, 1, 31))
        As = np.unique(np.r_[qs, np.inf]); Bs = np.unique(np.r_[qs, -np.inf])
        cands = []
        for A in As:
            for B in Bs:
                if B >= A:
                    continue
                res = []
                for T, y in ((Td, yd), (Tt, yt)):
                    abst, stop = seq_decide(T[stat_key], A, B, end_tau)
                    abst, ok = answer_fn(T, abst, stop)
                    res += [score(y, abst, ok), float(stop.mean())]
                cands.append(((float(A), float(B)), *res))
        return frontier(cands), cands

    # end-of-budget fallback for the IDK-only SPRT: tune its own tau on dev at t=N
    def tune_end(stat_key):
        best = (-1, 0.0)
        for tau_ in np.unique(np.quantile(Td[stat_key][..., -1], np.linspace(0, 1, 41))):
            abst, ok = outcome_answer_sub(Td, Td[stat_key][..., -1] > tau_, np.full(Td["K"].shape[:2], N))
            r = score(yd, abst, ok)
            if r["answered"] > 0 and r["f1"] > best[0]:
                best = (r["f1"], float(tau_))
        return best[1]

    out["SPRT hurdle"], cands_h = run_seq("llr", tau, outcome_answer_sub)
    out["SPRT IDK-only"], _ = run_seq("llr_idk", tune_end("llr_idk"), outcome_answer_sub)

    # Adaptive Consistency (stop when Beta posterior P(top > second) >= thr; never stops -> abstain)
    cands = []
    for thr in np.r_[np.linspace(0.5, 0.95, 10), 0.97, 0.99, 0.995, 0.999]:
        res = []
        for T, y in ((Td, yd), (Tt, yt)):
            abst, stop = seq_decide(T["ac"], thr, -np.inf, end_tau=np.inf)  # stat >= thr -> "stop" ...
            stopped = (T["ac"] >= thr).any(-1)
            abst, ok = outcome_answer_top(T, ~stopped, stop)                 # ... and answer when it stopped
            res += [score(y, abst, ok), float(stop.mean())]
        cands.append(((float(thr),), *res))
    out["Adaptive Consistency"] = frontier(cands)

    # fixed-n rules on the first n samples, thresholds tuned on dev at that n
    def fixed(n, rule):
        i = n - 1
        best = None
        if rule == "Unans votes":
            grid = [(u, lambda T, u=u: T["n0"][..., i] >= u) for u in range(1, n + 1)] + [(None, lambda T: np.zeros(T["n0"].shape[:2], bool))]
            ans = outcome_answer_sub
        elif rule == "CRP_k":
            grid = [(k, lambda T, k=k: T["K"][..., i] > k) for k in range(1, n + 1)]
            ans = outcome_answer_top
        else:  # hurdle LLR at n
            grid = [(float(tq), lambda T, tq=tq: T["llr"][..., i] > tq) for tq in np.unique(np.quantile(Td["llr"][..., i], np.linspace(0, 1, 41)))]
            ans = outcome_answer_sub
        for thr, f in grid:
            rs = []
            for T, y in ((Td, yd), (Tt, yt)):
                abst, ok = ans(T, f(T), np.full(T["K"].shape[:2], n))
                rs.append(score(y, abst, ok))
            if rs[0]["answered"] > 0 and (best is None or rs[0]["f1"] > best[1]["f1"] + 1e-9):
                best = (thr, *rs)
        return dict(n=n, params=best[0], dev_f1=best[1]["f1"], **{k: best[2][k] for k in ("f1", "A_acc", "U_acc", "answered")})

    for rule in ("Unans votes", "CRP_k", "Hurdle LLR"):
        out[f"fixed-n {rule}"] = [fixed(n, rule) for n in FIXED_N]
    # sanity: SPRT with A=+inf, B=-inf must equal TwoRest_Hurdle_CRP at 20 samples (orig order = perm 0)
    a0, s0 = seq_decide(Tt["llr"][:1], np.inf, -np.inf, tau)
    a0, ok0 = outcome_answer_sub({k: v[:1] for k, v in Tt.items()}, a0, s0)
    out["_check_hurdle20_perm0"] = score(yt, a0, ok0)
    return out


# ---------------------------------------------------------------- E3-E5 (dev only)
def full_stats(items, ignore_idk=False):
    rows = []
    for it in items:
        Nn, n0, m, cl, ok, lab = E.extract_clusters(it, ignore_idk=ignore_idk)
        sizes = sorted((len(c) for c in cl), reverse=True)
        top = max(cl, key=len)[0] if cl else None
        rows.append(dict(sizes=sizes, K=len(sizes), n0=n0, unans=it["should_abstain"],
                         share=sizes[0] / N if sizes else 0, c2=sizes[1] if len(sizes) > 1 else 0,
                         evenness=float(np.sum(np.log(sizes))) if sizes else 0.0,
                         sc_ok=bool(ok[top]) if top is not None else False,
                         sc_idk=(lab[top] in E.UNANS_LABELS) if top is not None else True))
    return rows


def e3(dev):
    out = {}
    for mode, ign in (("IDK kept as a cluster", False), ("IDK dropped", True)):
        rows = [r for r in full_stats(dev, ign) if r["K"] > 0]
        res = []
        for lab, sel in (("K=2", lambda r: r["K"] == 2), ("K=3", lambda r: r["K"] == 3), ("K>=4", lambda r: r["K"] >= 4)):
            g = [r for r in rows if sel(r)]
            y = [r["unans"] for r in g]
            res.append(dict(group=lab, n=len(g), n_unans=int(sum(y)),
                            auroc_evenness=auroc(y, [r["evenness"] for r in g]),
                            auroc_n_unans=auroc(y, [r["n0"] for r in g])))
        out[mode] = res
    return out


def e4(dev, hurdle_params):
    rows = full_stats(dev)
    for r, it in zip(rows, dev):
        r["llr"] = E.compute_two_restaurant_llr(it, hurdle_params)
    grp = {"easy": [r for r in rows if not r["unans"] and r["sc_ok"]],
           "hard": [r for r in rows if not r["unans"] and not r["sc_ok"]],
           "unans": [r for r in rows if r["unans"]]}
    sig = {"#Unanswerable votes": lambda r: r["n0"], "K (IDK kept)": lambda r: r["K"],
           "1 - top share": lambda r: 1 - r["share"], "evenness sum log n_k": lambda r: r["evenness"],
           "Hurdle LLR (fit on dev)": lambda r: r["llr"]}
    res = {"counts": {k: len(v) for k, v in grp.items()},
           "hard_sc_abstained": sum(r["sc_idk"] for r in grp["hard"]), "signals": {}}
    for name, f in sig.items():
        pair = lambda a, b: auroc([1] * len(grp[a]) + [0] * len(grp[b]), [f(r) for r in grp[a] + grp[b]])
        res["signals"][name] = {"unans vs hard": pair("unans", "hard"), "unans vs easy": pair("unans", "easy"),
                                "hard vs easy": pair("hard", "easy")}
    return res


GRID = np.linspace(0.05, 12.0, 400)


def _lw(sizes, mu, sig):
    n, K = sum(sizes), len(sizes)
    return K * np.log(GRID) - (gammaln(GRID + n) - gammaln(GRID)) + sum(gammaln(c) for c in sizes) + norm.logpdf(GRID, mu, sig)


def _lse(x):
    m = x.max(); return m + np.log(np.exp(x - m).sum())


def e5(dev):
    rows = [r for r in full_stats(dev) if r["K"] > 0]
    y = np.array([r["unans"] for r in rows])
    mle = np.array([GRID[np.argmax(_lw(r["sizes"], 0, 1e9))] for r in rows])
    init = {True: (5.0, 2.0), False: (2.0, 1.0)}                      # notebook's init priors
    pm = {True: [], False: []}
    for r in rows:
        lw = _lw(r["sizes"], *init[r["unans"]]); w = np.exp(lw - lw.max())
        pm[r["unans"]].append(float(w @ GRID / w.sum()))
    fitN = {k: (float(np.mean(v)), float(np.std(v) + 1e-4)) for k, v in pm.items()}
    pi0 = float(y.mean())
    post = []
    for r in rows:
        l0 = math.log(pi0) + _lse(_lw(r["sizes"], *fitN[True])); l1 = math.log(1 - pi0) + _lse(_lw(r["sizes"], *fitN[False]))
        post.append(1 / (1 + math.exp(l1 - l0)))
    return dict(alpha_mle_A=(float(mle[~y].mean()), float(mle[~y].std())), alpha_mle_U=(float(mle[y].mean()), float(mle[y].std())),
                fitted_prior_A=fitN[False], fitted_prior_U=fitN[True],
                auroc_posterior=auroc(y, post), auroc_K=auroc(y, [r["K"] for r in rows]))


# ---------------------------------------------------------------- main
def main():
    results = {}
    work = tempfile.mkdtemp(prefix="vqa_crp_")
    for ds in DATASETS:
        dev, test = load(ds)
        print(f"== {ds}: dev {len(dev)}, test {len(test)}", flush=True)
        r1 = e1(ds, dev, test, work)
        hurdle = r1["TwoRest_Hurdle_CRP"]["learned_params"]
        print("  E1 done; E2 ...", flush=True)
        r2 = e2(dev, test, hurdle)
        results[ds] = dict(E1={k: e1_row(v) for k, v in r1.items() if isinstance(v, dict) and "metrics" in v},
                           E1_params={k: v.get("learned_params", v.get("learned_threshold")) for k, v in r1.items()
                                      if isinstance(v, dict) and ("learned_params" in v or "learned_threshold" in v) and "__" not in k},
                           E2=r2, E3=e3(dev), E4=e4(dev, hurdle), E5=e5(dev))
        print("  done", flush=True)
    json.dump(results, open(f"{HERE}/results_vqa.json", "w"), indent=1, default=float)
    print(f"wrote {HERE}/results_vqa.json")


if __name__ == "__main__":
    main()
