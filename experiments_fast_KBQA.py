import os
import json
from tqdm import tqdm
import numpy as np
import scipy.special
import scipy.optimize
import concurrent.futures

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIRECTORY = "/home/scai/phd/aiz248314/KBQA/scUn/UA-Bench/"
DATA_DIRECTORY = os.path.join(BASE_DIRECTORY, "data")
MODELS = [
    "result_qwen3", "result_qwen36", "result_qwen38",
    "result_gemini36", "result_gemini38", "result_qwen35", "result_qwen35_base"
]
OUTPUT_FILE = "/home/scai/phd/aiz248314/KBQA/scUn/UA-Bench/results.json"

# Options:
#   Standard UA-Bench : 'overall_f1', 'balanced_accuracy', 'overall_accuracy', 'ans_f1'
#   KBQA-Specific     : 'kbqa_overall_f1', 'kbqa_overall_em', 'kbqa_macro_au_f1', 'kbqa_macro_au_em'
TARGET_METRIC = "overall_f1"

datasets = [
    "grailqa",
    # "webqsp",
    "graphqa",
]

ORDERED_EXPERIMENTS = [
    (1,  'Greedy'),
    (2,  'self_Consistency'),
    (3,  'Majority'),
    (4,  'self_Consistency_T'),
    (5,  'scUn'),
    (6,  'CRP_k'),
    (7,  'CRP_theta'),
    (8,  'Adaptive_Consistency_beta'),
    (9,  'SingleRest_PYP_NoIDK'),
    (10, 'SingleRest_Binomial_ZI_PYP_Surprisal'),
    (11, 'SingleRest_Binomial_ZI_PYP_Predictive'),
    (12, 'SingleRest_ZI_PYP_Surprisal'),
    (13, 'SingleRest_ZI_PYP_Predictive'),
    (14, 'TwoRest_CRP'),
    (15, 'TwoRest_PYP_NoIDK'),
    (16, 'TwoRest_Reserved_CRP'),
    (17, 'TwoRest_Binomial_Hurdle_CRP'),
    (18, 'TwoRest_Hurdle_CRP'),
    (19, 'TwoRest_Binomial_ZI_PYP'),
    (20, 'TwoRest_ConstrainedBB_ZI_PYP'),
    (21, 'TwoRest_ZI_PYP'),
    (22, 'Oracle_Intermediate'),
]
REQUESTED_EXPERIMENTS = [name for _, name in ORDERED_EXPERIMENTS]

THREE_ABLATION_COMBOS = [
    ("1_keepIDK_outOf20", False, 'N'),
    ("2_dropIDK_outOf20", True,  'N'),
    ("3_dropIDK_outOfM",  True,  'm'),
]

awareness = ['_abstaintion', '']

UNANS_PHRASES = ["i don't know", "insufficient", "cannot be answered", "not enough information", "nk", "nlf", "no logical form"]
UNANS_LABELS = {"ABSTAIN", "DATA_UNCERTAIN", "MODEL_UNCERTAIN", "UNANSWERABLE"}


# ==========================================
# 1. FAST CACHED CLUSTERING & KBQA EXTRACTION
# ==========================================
def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def is_sample_abstention(lbl, ans, lf=""):
    lbl_str = str(lbl).strip() if lbl is not None else ""
    ans_str = str(ans).strip().lower() if ans is not None else ""
    lf_str = str(lf).strip().lower() if lf is not None else ""
    if lbl_str in UNANS_LABELS:
        return True
    if lf_str in {"nk", "abstain", "nlf", "no logical form"}:
        return True
    if any(p in ans_str for p in UNANS_PHRASES):
        return True
    return False


def _extract_item_arrays(item):
    """Safely extracts per-sample arrays: is_correct, pred_label, pred_answer, em, f1, top_predictions, answer."""
    raw_corr = item.get("is_correct", [])
    if not isinstance(raw_corr, list):
        raw_corr = [raw_corr]
    is_correct_list = [
        bool(x[0]) if isinstance(x, list) and len(x) > 0 else bool(x)
        for x in raw_corr
    ]
    N_full = len(is_correct_list)

    raw_lbl = item.get("pred_label", [])
    if not isinstance(raw_lbl, list):
        raw_lbl = [raw_lbl]
    pred_label_list = [
        str(x[0]).strip() if isinstance(x, list) and len(x) > 0 else ("" if isinstance(x, list) or x is None else str(x).strip())
        for x in raw_lbl
    ]

    pred_answer_list = item.get("pred_answer", [])
    if not isinstance(pred_answer_list, list):
        pred_answer_list = [pred_answer_list]

    raw_em = item.get("em", [])
    if not isinstance(raw_em, list) or len(raw_em) < N_full:
        em_list = [1.0 if c else 0.0 for c in is_correct_list]
    else:
        em_list = [float(x[0]) if isinstance(x, list) and len(x) > 0 else float(x or 0.0) for x in raw_em]

    raw_f1 = item.get("f1", [])
    if not isinstance(raw_f1, list) or len(raw_f1) < N_full:
        f1_list = [1.0 if c else 0.0 for c in is_correct_list]
    else:
        f1_list = [float(x[0]) if isinstance(x, list) and len(x) > 0 else float(x or 0.0) for x in raw_f1]

    top_preds = item.get("top_predictions", [""] * N_full)
    if not isinstance(top_preds, list):
        top_preds = [top_preds]

    exec_answers = item.get("answer", [[] for _ in range(N_full)])
    if not isinstance(exec_answers, list):
        exec_answers = [exec_answers]

    return is_correct_list, pred_label_list, pred_answer_list, em_list, f1_list, top_preds, exec_answers


def _compute_clusters_raw(item, ignore_idk=False, up_to_k=None):
    is_correct_list, pred_label_list, pred_answer_list, em_list, f1_list, top_preds, exec_answers = _extract_item_arrays(item)

    compare_label = item.get("compare_label", {})
    if not isinstance(compare_label, dict):
        compare_label = {}

    N_full = len(is_correct_list)
    if N_full == 0:
        return 0, 0, 0, [], is_correct_list, pred_label_list, em_list, f1_list

    N_total = min(up_to_k, N_full) if up_to_k is not None else N_full

    idk_indices = []
    substantive_indices = []
    for i in range(N_total):
        lbl = pred_label_list[i] if i < len(pred_label_list) else ""
        ans = pred_answer_list[i] if i < len(pred_answer_list) else ""
        lf = top_preds[i] if i < len(top_preds) else ""
        if is_sample_abstention(lbl, ans, lf):
            idk_indices.append(i)
        else:
            substantive_indices.append(i)

    n_0 = len(idk_indices)
    m = len(substantive_indices)

    active_pool = substantive_indices if ignore_idk else list(range(N_total))
    active_set = set(active_pool)

    clusters = []
    visited = set()
    for i in active_pool:
        if i not in visited:
            matches = compare_label.get(str(i), [])
            cluster_indices = [j for j, val in enumerate(matches) if val == 1 and j in active_set]

            if not cluster_indices:
                lexical_cluster = []
                if pred_answer_list and i < len(pred_answer_list):
                    ans_i = str(pred_answer_list[i]).strip().lower()
                    for j in active_pool:
                        if j < len(pred_answer_list):
                            if ans_i == str(pred_answer_list[j]).strip().lower():
                                lexical_cluster.append(j)
                cluster_indices = lexical_cluster if lexical_cluster else [i]

            clusters.append(cluster_indices)
            visited.update(cluster_indices)

    return N_total, n_0, m, clusters, is_correct_list, pred_label_list, em_list, f1_list


_MLE_THETA_CACHE = {}
def get_mle_theta(denom_size, K):
    key = (int(denom_size), int(K))
    if key in _MLE_THETA_CACHE:
        return _MLE_THETA_CACHE[key]
    if denom_size <= 0 or K <= 1:
        val = 0.0
    elif K >= denom_size:
        val = 100.0
    else:
        low, high = 1e-5, 1e4
        for _ in range(40):
            mid = (low + high) / 2.0
            if sum(mid / (mid + i) for i in range(denom_size)) > K:
                high = mid
            else:
                low = mid
        val = mid
    _MLE_THETA_CACHE[key] = val
    return val


def precompute_dataset_caches(data):
    """Caches clusters, KBQA EM/F1 arrays, and sequential Beta trajectories once per dataset split."""
    for item in data:
        if "_fast_cache" in item:
            continue
        cache = {}
        _, _, _, _, _, top_preds, exec_answers = _extract_item_arrays(item)

        empty_exec_count = 0
        for lf, ans in zip(top_preds, exec_answers):
            lf_clean = str(lf).strip().lower()
            if lf_clean not in {"", "nk", "abstain", "nlf", "no logical form", "error"}:
                if not ans or ans == [] or ans == ["0"] or ans == "null":
                    empty_exec_count += 1

        for ign in (False, True):
            N, n_0, m, clusters, is_corr, pred_lbl, em_list, f1_list = _compute_clusters_raw(
                item, ignore_idk=ign, up_to_k=None
            )
            total_clusters = len(clusters)
            cluster_sizes = [len(c) for c in clusters]
            max_freq = max(cluster_sizes) if cluster_sizes else 0

            correct_freq = sum(1 for x in is_corr if x is True)
            correct_present = correct_freq > 0
            oracle_best_f1 = max(f1_list) if f1_list else 0.0
            oracle_best_em = max(em_list) if em_list else 0.0

            is_correct_most_freq = False
            if correct_present and clusters:
                correct_cluster_sizes = [len(c) for c in clusters if any(is_corr[idx] for idx in c)]
                if correct_cluster_sizes:
                    is_correct_most_freq = (max(correct_cluster_sizes) == max_freq)

            multi_sparql_in_top_cluster = False
            if clusters:
                largest_c = max(clusters, key=len)
                unique_lfs = {
                    " ".join(str(top_preds[idx]).strip().lower().split())
                    for idx in largest_c if idx < len(top_preds)
                }
                if len(unique_lfs) > 1:
                    multi_sparql_in_top_cluster = True

            seq_steps = []
            for k in range(1, N + 1):
                _, n_0_k, _, cl_k, _, _, _, _ = _compute_clusters_raw(item, ignore_idk=ign, up_to_k=k)
                if not cl_k:
                    seq_steps.append(None)
                    continue
                c_sizes = sorted([len(c) for c in cl_k], reverse=True)
                a = float(c_sizes[0])
                b_sec = float(c_sizes[1]) if len(c_sizes) > 1 else 0.0
                b_N = max(b_sec, float(n_0_k)) if ign else b_sec
                prob_m = float(1.0 - scipy.special.betainc(a + 1.0, b_sec + 1.0, 0.5))
                prob_N = float(1.0 - scipy.special.betainc(a + 1.0, b_N + 1.0, 0.5))
                best_i_k = max(cl_k, key=len)[0]
                seq_steps.append((prob_N, prob_m, pred_lbl[best_i_k], is_corr[best_i_k], best_i_k))

            cache[ign] = {
                'extract': (N, n_0, m, clusters, is_corr, pred_lbl, em_list, f1_list),
                'meta': (max_freq, total_clusters, correct_present, correct_freq, is_correct_most_freq),
                'kbqa_meta': {
                    'empty_exec_count': empty_exec_count,
                    'oracle_best_f1': oracle_best_f1,
                    'oracle_best_em': oracle_best_em,
                    'multi_sparql_in_top_cluster': multi_sparql_in_top_cluster,
                },
                'seq': seq_steps
            }
        item["_fast_cache"] = cache


def extract_clusters(item, ignore_idk=False, up_to_k=None):
    if up_to_k is None and "_fast_cache" in item:
        return item["_fast_cache"][bool(ignore_idk)]['extract'][:6]
    return _compute_clusters_raw(item, ignore_idk=ignore_idk, up_to_k=up_to_k)[:6]


def extract_clusters_with_kbqa(item, ignore_idk=False, up_to_k=None):
    if up_to_k is None and "_fast_cache" in item:
        return item["_fast_cache"][bool(ignore_idk)]['extract']
    return _compute_clusters_raw(item, ignore_idk=ignore_idk, up_to_k=up_to_k)


def get_item_metadata(item, ignore_abstention_tokens=True):
    if "_fast_cache" in item:
        return item["_fast_cache"][bool(ignore_abstention_tokens)]['meta']
    N, n_0, m, clusters, is_correct_list, _ = extract_clusters(
        item, ignore_idk=ignore_abstention_tokens
    )
    if N == 0:
        return 0, 0, False, 0, False
    total_clusters = len(clusters)
    cluster_sizes = [len(c) for c in clusters]
    max_freq = max(cluster_sizes) if cluster_sizes else 0
    correct_freq = sum(1 for x in is_correct_list if x is True)
    correct_present = correct_freq > 0
    is_correct_most_freq = False
    if correct_present and clusters:
        correct_cluster_sizes = [len(c) for c in clusters if any(is_correct_list[idx] for idx in c)]
        if correct_cluster_sizes:
            is_correct_most_freq = (max(correct_cluster_sizes) == max_freq)
    return max_freq, total_clusters, correct_present, correct_freq, is_correct_most_freq


# ==========================================
# 2. CONSENSUS METHODS (RETURNING best_i)
# ==========================================
def get_generalized_sc_consensus(
    item, method='majority', threshold=0.5, ignore_idk=False, denom_mode='N'
):
    N, n_0, m, clusters, is_correct_list, pred_label_list = extract_clusters(
        item, ignore_idk=ignore_idk
    )
    if N == 0 or not clusters:
        return "DATA_UNCERTAIN", None, None

    largest_cluster = max(clusters, key=len)
    max_votes = len(largest_cluster)
    best_i = largest_cluster[0]

    if method == 'most_freq':
        return pred_label_list[best_i], is_correct_list[best_i], best_i

    denom = m if denom_mode == 'm' else N
    if denom <= 0:
        return "DATA_UNCERTAIN", None, None

    if (max_votes / float(denom)) >= threshold:
        return pred_label_list[best_i], is_correct_list[best_i], best_i
    else:
        return "DATA_UNCERTAIN", None, None


def get_generalized_crp_consensus(
    item, method='crp_k', threshold=None, ignore_idk=False, denom_mode='N'
):
    N, n_0, m, clusters, is_correct_list, pred_label_list = extract_clusters(
        item, ignore_idk=ignore_idk
    )
    if N == 0 or not clusters:
        return "DATA_UNCERTAIN", None, None

    K = len(clusters)
    denom = m if denom_mode == 'm' else N
    if denom <= 0:
        return "DATA_UNCERTAIN", None, None

    is_uncertain = False
    if method == 'crp_k' and threshold is not None:
        effective_k = K * (float(N) / float(denom)) if denom_mode == 'm' else K
        if effective_k > threshold:
            is_uncertain = True
    elif method == 'crp_theta' and threshold is not None:
        theta = get_mle_theta(denom, K)
        if theta > threshold:
            is_uncertain = True

    if is_uncertain:
        return "DATA_UNCERTAIN", None, None

    largest_cluster = max(clusters, key=len)
    best_i = largest_cluster[0]
    return pred_label_list[best_i], is_correct_list[best_i], best_i


def get_Adaptive_Consistency_beta_consensus(
    item, threshold=0.95, ignore_idk=False, denom_mode='N'
):
    if "_fast_cache" in item:
        seq_steps = item["_fast_cache"][bool(ignore_idk)]['seq']
        use_m = (denom_mode == 'm')
        for step in seq_steps:
            if step is None:
                continue
            prob = step[1] if use_m else step[0]
            if prob >= threshold:
                return step[2], step[3], step[4]
        return "DATA_UNCERTAIN", None, None

    is_correct_list = item.get("is_correct", [])
    if not is_correct_list:
        return "DATA_UNCERTAIN", None, None
    N_full = len(is_correct_list)

    for k in range(1, N_full + 1):
        N_k, n_0_k, m_k, clusters, is_corr, pred_lbl = extract_clusters(
            item, ignore_idk=ignore_idk, up_to_k=k
        )
        if not clusters:
            continue
        cluster_sizes = sorted([len(c) for c in clusters], reverse=True)
        a = float(cluster_sizes[0])
        if denom_mode == 'm':
            b = float(cluster_sizes[1]) if len(cluster_sizes) > 1 else 0.0
        else:
            b_second = float(cluster_sizes[1]) if len(cluster_sizes) > 1 else 0.0
            b = max(b_second, float(n_0_k)) if ignore_idk else b_second
        prob = float(1.0 - scipy.special.betainc(a + 1.0, b + 1.0, 0.5))
        if prob >= threshold:
            largest_cluster = max(clusters, key=len)
            best_i = largest_cluster[0]
            return pred_lbl[best_i], is_corr[best_i], best_i

    return "DATA_UNCERTAIN", None, None


def get_oracle_intermediate_consensus(item, ignore_idk=False):
    N, n_0, m, clusters, is_correct_list, pred_label_list = extract_clusters(
        item, ignore_idk=ignore_idk
    )
    if N == 0 or not clusters:
        return "DATA_UNCERTAIN", None, None

    K = len(clusters)
    denom = m if ignore_idk else N

    largest_cluster = max(clusters, key=len)
    best_i = largest_cluster[0]

    if K == 1 and denom == N:
        return pred_label_list[best_i], is_correct_list[best_i], best_i
    if K >= N:
        return "DATA_UNCERTAIN", None, None

    should_abstain = item.get("should_abstain") is True
    if should_abstain:
        return "DATA_UNCERTAIN", None, None
    else:
        return pred_label_list[best_i], is_correct_list[best_i], best_i


# ==========================================
# 3. VECTORIZED PARAMETER FITTING & FAST THRESHOLD SWEEP
# ==========================================
def extract_seating_stats(data, ignore_idk_in_substantive=True):
    stats = []
    for item in data:
        N, n_0, m, clusters, _, _ = extract_clusters(item, ignore_idk=ignore_idk_in_substantive)
        sizes = [len(c) for c in clusters]
        m_eff = sum(sizes)
        K = len(sizes)
        gt_unans = (item.get("should_abstain") is True)
        stats.append((N, n_0, m_eff, K, sizes, gt_unans))
    return stats


def fit_binomial_pi_mle(n0_list, N_list):
    total_n0 = float(sum(n0_list))
    total_N = float(sum(N_list))
    if total_N <= 0:
        return 0.5
    pi = (total_n0 + 0.5) / (total_N + 1.0)
    return float(np.clip(pi, 1e-4, 1.0 - 1e-4))


def fit_constrained_beta_binomial_mle(n0_list, N_list):
    if not n0_list:
        return 0.5
    n0_arr = np.array(n0_list, dtype=float)
    N_arr = np.array(N_list, dtype=float)

    def neg_ll(a):
        b = 1.0 - a
        ll = np.sum(
            scipy.special.betaln(n0_arr + a, (N_arr - n0_arr) + b)
            - scipy.special.betaln(a, b)
        )
        return -float(ll)

    res = scipy.optimize.minimize_scalar(neg_ll, bounds=(1e-4, 1.0 - 1e-4), method='bounded')
    return float(np.clip(res.x, 1e-4, 1.0 - 1e-4))


def fit_beta_binomial_mle(n0_list, N_list):
    if not n0_list:
        return 0.5, 0.5
    n0_arr = np.array(n0_list, dtype=float)
    N_arr = np.array(N_list, dtype=float)

    def neg_ll(params):
        a, b = params
        if a <= 1e-5 or b <= 1e-5:
            return 1e12
        ll = np.sum(
            scipy.special.betaln(n0_arr + a, (N_arr - n0_arr) + b)
            - scipy.special.betaln(a, b)
        )
        return -ll

    res = scipy.optimize.minimize(
        neg_ll, x0=[1.0, 1.0], bounds=[(1e-4, 500.0), (1e-4, 500.0)], method='L-BFGS-B'
    )
    return float(res.x[0]), float(res.x[1])


def fit_crp_alpha_mle(m_list, K_list):
    valid = [(m, K) for m, K in zip(m_list, K_list) if m >= 1 and K >= 1]
    if not valid:
        return 0.5
    m_arr = np.array([x[0] for x in valid], dtype=float)
    K_arr = np.array([x[1] for x in valid], dtype=float)

    def neg_ll(params):
        alpha = params[0]
        if alpha <= 1e-5:
            return 1e12
        ll = np.sum(
            K_arr * np.log(alpha)
            + scipy.special.gammaln(alpha)
            - scipy.special.gammaln(m_arr + alpha)
        )
        return -ll

    res = scipy.optimize.minimize(
        neg_ll, x0=[1.0], bounds=[(1e-4, 500.0)], method='L-BFGS-B'
    )
    return float(res.x[0])


def fit_reserved_crp_mle(N_list, n0_list, K_list):
    if not N_list:
        return 1.0, 1.0
    N_arr = np.array(N_list, dtype=float)
    n0_arr = np.array(n0_list, dtype=float)
    K_arr = np.array(K_list, dtype=float)

    def neg_ll(params):
        alpha, beta = params
        if alpha <= 1e-5 or beta <= 1e-5:
            return 1e12
        ll = np.sum(
            scipy.special.gammaln(alpha + beta)
            - scipy.special.gammaln(alpha + beta + N_arr)
            + scipy.special.gammaln(beta + n0_arr)
            - scipy.special.gammaln(beta)
            + K_arr * np.log(alpha)
        )
        return -ll

    res = scipy.optimize.minimize(
        neg_ll, x0=[1.0, 1.0], bounds=[(1e-4, 500.0), (1e-4, 500.0)], method='L-BFGS-B'
    )
    return float(res.x[0]), float(res.x[1])


def fit_pyp_mle(m_list, K_list, sizes_list):
    valid = [
        (m, K, sizes)
        for m, K, sizes in zip(m_list, K_list, sizes_list)
        if m >= 2 and K >= 1
    ]
    if not valid:
        return 1.0, 0.1

    m_arr = np.array([x[0] for x in valid], dtype=float)
    K_arr = np.array([x[1] for x in valid], dtype=int)
    max_K = int(np.max(K_arr))
    j_vals = np.arange(1, max_K, dtype=float)
    j_counts = np.array([np.sum(K_arr > j) for j in range(1, max_K)], dtype=float) if max_K > 1 else np.array([])

    all_gt1 = [s for _, _, sizes in valid for s in sizes if s > 1]
    if all_gt1:
        uniq_s, s_counts = np.unique(np.array(all_gt1, dtype=float), return_counts=True)
    else:
        uniq_s, s_counts = np.array([]), np.array([])

    n_valid = float(len(valid))

    def neg_ll(params):
        alpha, d = params
        if d < 1e-5 or d >= 1.0 - 1e-5 or alpha <= -d + 1e-5:
            return 1e12
        total_ll = 0.0
        if len(j_vals) > 0:
            total_ll += np.sum(j_counts * np.log(alpha + j_vals * d))
        total_ll -= (np.sum(scipy.special.gammaln(alpha + m_arr)) - n_valid * scipy.special.gammaln(alpha + 1.0))
        if len(uniq_s) > 0:
            total_ll += np.sum(s_counts * (scipy.special.gammaln(uniq_s - d) - scipy.special.gammaln(1.0 - d)))
        return -float(total_ll)

    res = scipy.optimize.minimize(
        neg_ll, x0=[1.0, 0.15], bounds=[(1e-4, 500.0), (1e-4, 0.999)], method='L-BFGS-B'
    )
    return float(res.x[0]), float(res.x[1])


def pyp_log_likelihood_single_partition(m_eff, K, sizes, alpha, d):
    if m_eff <= 0 or K <= 0:
        return 0.0
    ll = 0.0
    if K > 1:
        j_vals = np.arange(1, K, dtype=float)
        ll += np.sum(np.log(alpha + j_vals * d))
    ll -= (scipy.special.gammaln(alpha + m_eff) - scipy.special.gammaln(alpha + 1.0))
    sizes_arr = np.array(sizes, dtype=float)
    gt1 = sizes_arr[sizes_arr > 1]
    if len(gt1) > 0:
        ll += np.sum(scipy.special.gammaln(gt1 - d) - scipy.special.gammaln(1.0 - d))
    return float(ll)


def compute_generative_score(item, params):
    method = params['method']
    test_ignore_idk = params.get('test_ignore_idk', True)

    N, n_0, m, clusters, _, _ = extract_clusters(item, ignore_idk=test_ignore_idk)
    if N == 0:
        return 1e6

    sizes = [len(c) for c in clusters]
    m_eff = sum(sizes)
    K = len(sizes)

    if method == 'TwoRest_Binomial_Hurdle_CRP':
        piU, alU = params['pi_U'], params['alpha_U']
        piA, alA = params['pi_A'], params['alpha_A']
        ll_idk_diff = n_0 * np.log(piU / piA) + m * np.log((1.0 - piU) / (1.0 - piA))
        ll_crp_U = (K * np.log(alU) + scipy.special.gammaln(alU) - scipy.special.gammaln(m + alU)) if m >= 1 else 0.0
        ll_crp_A = (K * np.log(alA) + scipy.special.gammaln(alA) - scipy.special.gammaln(m + alA)) if m >= 1 else 0.0
        return float(ll_idk_diff + (ll_crp_U - ll_crp_A))

    elif method == 'TwoRest_Binomial_ZI_PYP':
        piU, alU, dU = params['pi_U'], params['alpha_U'], params['d_U']
        piA, alA, dA = params['pi_A'], params['alpha_A'], params['d_A']
        llr = n_0 * np.log(piU / piA) + m * np.log((1.0 - piU) / (1.0 - piA))
        if m >= 1 and K >= 1:
            llr += (
                pyp_log_likelihood_single_partition(m, K, sizes, alU, dU)
                - pyp_log_likelihood_single_partition(m, K, sizes, alA, dA)
            )
        return float(llr)

    elif method == 'TwoRest_ConstrainedBB_ZI_PYP':
        aU, alU, dU = params['a_U'], params['alpha_U'], params['d_U']
        aA, alA, dA = params['a_A'], params['alpha_A'], params['d_A']
        bU, bA = 1.0 - aU, 1.0 - aA
        llr = (
            (scipy.special.betaln(n_0 + aU, m + bU) - scipy.special.betaln(aU, bU))
            - (scipy.special.betaln(n_0 + aA, m + bA) - scipy.special.betaln(aA, bA))
        )
        if m >= 1 and K >= 1:
            llr += (
                pyp_log_likelihood_single_partition(m, K, sizes, alU, dU)
                - pyp_log_likelihood_single_partition(m, K, sizes, alA, dA)
            )
        return float(llr)

    elif method == 'SingleRest_Binomial_ZI_PYP_Surprisal':
        pi0, al0, d0 = params['pi_0'], params['alpha_0'], params['d_0']
        ll_idk = n_0 * np.log(pi0) + m * np.log(1.0 - pi0)
        ll_pyp = pyp_log_likelihood_single_partition(m, K, sizes, al0, d0) if (m >= 1 and K >= 1) else 0.0
        return float(-(ll_idk + ll_pyp))

    elif method == 'SingleRest_Binomial_ZI_PYP_Predictive':
        pi0, al0, d0 = params['pi_0'], params['alpha_0'], params['d_0']
        p_idk_next = (n_0 + pi0) / float(N + 1.0)
        p_new_table = ((al0 + K * d0) / float(m + al0)) if m > 0 else 1.0
        max_s = max(sizes) if sizes else 0
        p_stay_dom = ((max_s - d0) / float(m + al0)) if (m > 0 and max_s > 0) else 0.0
        return float(p_idk_next + (1.0 - p_idk_next) * (p_new_table - p_stay_dom))

    elif method == 'TwoRest_CRP':
        if m_eff == 0 or K == 0:
            return 1e6
        aU, aA = params['alpha_U'], params['alpha_A']
        ll_U = K * np.log(aU) + scipy.special.gammaln(aU) - scipy.special.gammaln(m_eff + aU)
        ll_A = K * np.log(aA) + scipy.special.gammaln(aA) - scipy.special.gammaln(m_eff + aA)
        return float(ll_U - ll_A)

    elif method == 'TwoRest_Reserved_CRP':
        aU, bU = params['alpha_U'], params['beta_U']
        aA, bA = params['alpha_A'], params['beta_A']
        ll_U = (
            scipy.special.gammaln(aU + bU) - scipy.special.gammaln(aU + bU + N)
            + scipy.special.gammaln(bU + n_0) - scipy.special.gammaln(bU)
            + K * np.log(aU)
        )
        ll_A = (
            scipy.special.gammaln(aA + bA) - scipy.special.gammaln(aA + bA + N)
            + scipy.special.gammaln(bA + n_0) - scipy.special.gammaln(bA)
            + K * np.log(aA)
        )
        return float(ll_U - ll_A)

    elif method == 'TwoRest_Hurdle_CRP':
        aU, bU, alU = params['a_U'], params['b_U'], params['alpha_U']
        aA, bA, alA = params['a_A'], params['b_A'], params['alpha_A']
        ll_idk_U = scipy.special.betaln(n_0 + aU, m + bU) - scipy.special.betaln(aU, bU)
        ll_idk_A = scipy.special.betaln(n_0 + aA, m + bA) - scipy.special.betaln(aA, bA)
        ll_crp_U = (K * np.log(alU) + scipy.special.gammaln(alU) - scipy.special.gammaln(m + alU)) if m >= 1 else 0.0
        ll_crp_A = (K * np.log(alA) + scipy.special.gammaln(alA) - scipy.special.gammaln(m + alA)) if m >= 1 else 0.0
        return float((ll_idk_U + ll_crp_U) - (ll_idk_A + ll_crp_A))

    elif method == 'TwoRest_ZI_PYP':
        aU, bU, alU, dU = params['a_U'], params['b_U'], params['alpha_U'], params['d_U']
        aA, bA, alA, dA = params['a_A'], params['b_A'], params['alpha_A'], params['d_A']
        llr = (
            (scipy.special.betaln(n_0 + aU, m + bU) - scipy.special.betaln(aU, bU))
            - (scipy.special.betaln(n_0 + aA, m + bA) - scipy.special.betaln(aA, bA))
        )
        if m >= 1 and K >= 1:
            llr += (
                pyp_log_likelihood_single_partition(m, K, sizes, alU, dU)
                - pyp_log_likelihood_single_partition(m, K, sizes, alA, dA)
            )
        return float(llr)

    elif method == 'SingleRest_ZI_PYP_Surprisal':
        a0, b0, al0, d0 = params['a_0'], params['b_0'], params['alpha_0'], params['d_0']
        ll_idk = scipy.special.betaln(n_0 + a0, m + b0) - scipy.special.betaln(a0, b0)
        ll_pyp = pyp_log_likelihood_single_partition(m, K, sizes, al0, d0) if (m >= 1 and K >= 1) else 0.0
        return float(-(ll_idk + ll_pyp))

    elif method == 'SingleRest_ZI_PYP_Predictive':
        a0, b0, al0, d0 = params['a_0'], params['b_0'], params['alpha_0'], params['d_0']
        p_idk_next = (n_0 + a0) / float(N + a0 + b0)
        p_new_table = ((al0 + K * d0) / float(m + al0)) if m > 0 else 1.0
        max_s = max(sizes) if sizes else 0
        p_stay_dom = ((max_s - d0) / float(m + al0)) if (m > 0 and max_s > 0) else 0.0
        return float(p_idk_next + (1.0 - p_idk_next) * (p_new_table - p_stay_dom))

    elif method == 'TwoRest_PYP_NoIDK':
        if m_eff == 0 or K == 0:
            return 1e6
        alU, dU = params['alpha_U'], params['d_U']
        alA, dA = params['alpha_A'], params['d_A']
        ll_U = pyp_log_likelihood_single_partition(m_eff, K, sizes, alU, dU)
        ll_A = pyp_log_likelihood_single_partition(m_eff, K, sizes, alA, dA)
        return float(ll_U - ll_A)

    elif method == 'SingleRest_PYP_NoIDK':
        if m_eff == 0 or K == 0:
            return 1e6
        al0, d0 = params['alpha_0'], params['d_0']
        ll_pyp = pyp_log_likelihood_single_partition(m_eff, K, sizes, al0, d0)
        max_s = max(sizes)
        predictive_ratio = (al0 + K * d0) / max(max_s - d0, 1e-4)
        return float(-ll_pyp + predictive_ratio)

    return 0.0


def fast_vectorized_tau_search(val_data, val_scores, candidate_taus, test_ignore_idk=True):
    n_items = len(val_data)
    if n_items == 0 or not candidate_taus:
        return 0.0

    gt_abstain = np.zeros(n_items, dtype=bool)
    base_pred_ans = np.zeros(n_items, dtype=bool)
    base_pred_unans = np.zeros(n_items, dtype=bool)
    base_corr = np.zeros(n_items, dtype=bool)
    base_em = np.zeros(n_items, dtype=float)
    base_f1 = np.zeros(n_items, dtype=float)

    for idx, item in enumerate(val_data):
        gt_abstain[idx] = (item.get("should_abstain") is True)
        N, n_0, m, clusters, is_corr, pred_lbl, em_list, f1_list = extract_clusters_with_kbqa(
            item, ignore_idk=test_ignore_idk
        )
        if not clusters:
            base_pred_unans[idx] = True
        else:
            best_i = max(clusters, key=len)[0]
            lbl = pred_lbl[best_i]
            base_pred_ans[idx] = (lbl == "ANSWERABLE")
            base_pred_unans[idx] = (lbl in UNANS_LABELS)
            base_corr[idx] = bool(is_corr[best_i])
            base_em[idx] = float(em_list[best_i]) if best_i < len(em_list) else float(is_corr[best_i])
            base_f1[idx] = float(f1_list[best_i]) if best_i < len(f1_list) else float(is_corr[best_i])

    scores_col = np.asarray(val_scores, dtype=float)[:, None]
    taus_row = np.asarray(candidate_taus, dtype=float)[None, :]

    abstain_override = (scores_col > taus_row)

    is_pred_ans = (~abstain_override) & base_pred_ans[:, None]
    is_pred_unans = abstain_override | base_pred_unans[:, None]
    is_corr_mat = (~abstain_override) & base_corr[:, None]

    gt_ans_col = (~gt_abstain)[:, None]
    gt_abs_col = gt_abstain[:, None]

    tp_a = np.sum(gt_ans_col & is_pred_ans & is_corr_mat, axis=0, dtype=float)
    fp_a = np.sum((gt_ans_col & is_pred_ans & (~is_corr_mat)) | (gt_abs_col & is_pred_ans), axis=0, dtype=float)
    tp_u = np.sum(gt_abs_col & is_pred_unans, axis=0, dtype=float)
    fp_u = np.sum(gt_ans_col & is_pred_unans, axis=0, dtype=float)

    # Parentheses around (gt_ans_col & is_pred_ans) prevent float/bool bitwise_and TypeError
    ans_f1_sum = np.sum((gt_ans_col & is_pred_ans) * base_f1[:, None], axis=0, dtype=float)
    ans_em_sum = np.sum((gt_ans_col & is_pred_ans) * base_em[:, None], axis=0, dtype=float)
    unans_sys_sum = tp_u

    ans_count = float(np.sum(~gt_abstain))
    unans_count = float(np.sum(gt_abstain))
    total_count = ans_count + unans_count

    best_tau = float(candidate_taus[0])
    best_score = -1.0

    for t_idx, tau in enumerate(candidate_taus):
        tpa, fpa = tp_a[t_idx], fp_a[t_idx]
        tpu, fpu = tp_u[t_idx], fp_u[t_idx]

        ans_acc = (tpa / ans_count) if ans_count > 0 else 0.0
        ans_prec = (tpa / (tpa + fpa)) if (tpa + fpa) > 0 else 0.0
        ans_f1 = (2.0 * ans_prec * ans_acc / (ans_prec + ans_acc)) if (ans_prec + ans_acc) > 0 else 0.0

        unans_acc = (tpu / unans_count) if unans_count > 0 else 0.0
        unans_prec = (tpu / (tpu + fpu)) if (tpu + fpu) > 0 else 0.0
        unans_f1 = (2.0 * unans_prec * unans_acc / (unans_prec + unans_acc)) if (unans_prec + unans_acc) > 0 else 0.0

        if TARGET_METRIC == 'overall_f1':
            score = round((ans_f1 + unans_f1) / 2.0, 2)
        elif TARGET_METRIC == 'balanced_accuracy':
            score = round((ans_acc + unans_acc) / 2.0, 2)
        elif TARGET_METRIC == 'overall_accuracy':
            score = round((tpa + tpu) / total_count, 2) if total_count > 0 else 0.0
        elif TARGET_METRIC == 'ans_f1':
            score = round(ans_f1, 2)
        elif TARGET_METRIC == 'kbqa_overall_f1':
            score = round((ans_f1_sum[t_idx] + unans_sys_sum[t_idx]) / total_count, 4) if total_count > 0 else 0.0
        elif TARGET_METRIC == 'kbqa_overall_em':
            score = round((ans_em_sum[t_idx] + unans_sys_sum[t_idx]) / total_count, 4) if total_count > 0 else 0.0
        elif TARGET_METRIC == 'kbqa_macro_au_f1':
            k_a_f1 = (ans_f1_sum[t_idx] / ans_count) if ans_count > 0 else 0.0
            k_u_f1 = (unans_sys_sum[t_idx] / unans_count) if unans_count > 0 else 0.0
            score = round((k_a_f1 + k_u_f1) / 2.0, 4)
        elif TARGET_METRIC == 'kbqa_macro_au_em':
            k_a_em = (ans_em_sum[t_idx] / ans_count) if ans_count > 0 else 0.0
            k_u_em = (unans_sys_sum[t_idx] / unans_count) if unans_count > 0 else 0.0
            score = round((k_a_em + k_u_em) / 2.0, 4)
        else:
            score = round((tpa + tpu) / total_count, 2) if total_count > 0 else 0.0

        if score > best_score:
            best_score = score
            best_tau = float(tau)

    return best_tau


IDK_TABLE_GENERATIVE_METHODS = {
    'TwoRest_Binomial_Hurdle_CRP',
    'TwoRest_Binomial_ZI_PYP',
    'TwoRest_ConstrainedBB_ZI_PYP',
    'SingleRest_Binomial_ZI_PYP_Surprisal',
    'SingleRest_Binomial_ZI_PYP_Predictive',
    'TwoRest_Reserved_CRP',
    'TwoRest_Hurdle_CRP',
    'TwoRest_ZI_PYP',
    'SingleRest_ZI_PYP_Surprisal',
    'SingleRest_ZI_PYP_Predictive',
}

NO_IDK_TABLE_GENERATIVE_METHODS = {
    'SingleRest_PYP_NoIDK',
    'TwoRest_CRP',
    'TwoRest_PYP_NoIDK',
}

GENERATIVE_METHODS = IDK_TABLE_GENERATIVE_METHODS | NO_IDK_TABLE_GENERATIVE_METHODS


def train_generative_restaurant_model(
    val_data, method='TwoRest_Binomial_ZI_PYP', val_ignore_idk=True, test_ignore_idk=True
):
    if method in IDK_TABLE_GENERATIVE_METHODS:
        val_ignore_idk = True
        test_ignore_idk = True

    stats = extract_seating_stats(val_data, ignore_idk_in_substantive=val_ignore_idk)
    ans_stats = [s for s in stats if not s[5]]
    unans_stats = [s for s in stats if s[5]]
    if not ans_stats:
        ans_stats = stats
    if not unans_stats:
        unans_stats = stats

    params = {
        'method': method,
        'val_ignore_idk': val_ignore_idk,
        'test_ignore_idk': test_ignore_idk
    }

    if method == 'TwoRest_Binomial_Hurdle_CRP':
        piA = fit_binomial_pi_mle([s[1] for s in ans_stats], [s[0] for s in ans_stats])
        piU = fit_binomial_pi_mle([s[1] for s in unans_stats], [s[0] for s in unans_stats])
        alA = fit_crp_alpha_mle([s[2] for s in ans_stats], [s[3] for s in ans_stats])
        alU = fit_crp_alpha_mle([s[2] for s in unans_stats], [s[3] for s in unans_stats])
        params.update({
            'pi_A': round(piA, 4), 'alpha_A': round(alA, 4),
            'pi_U': round(piU, 4), 'alpha_U': round(alU, 4)
        })

    elif method == 'TwoRest_Binomial_ZI_PYP':
        piA = fit_binomial_pi_mle([s[1] for s in ans_stats], [s[0] for s in ans_stats])
        piU = fit_binomial_pi_mle([s[1] for s in unans_stats], [s[0] for s in unans_stats])
        alA, dA = fit_pyp_mle([s[2] for s in ans_stats], [s[3] for s in ans_stats], [s[4] for s in ans_stats])
        alU, dU = fit_pyp_mle([s[2] for s in unans_stats], [s[3] for s in unans_stats], [s[4] for s in unans_stats])
        params.update({
            'pi_A': round(piA, 4), 'alpha_A': round(alA, 4), 'd_A': round(dA, 4),
            'pi_U': round(piU, 4), 'alpha_U': round(alU, 4), 'd_U': round(dU, 4)
        })

    elif method == 'TwoRest_ConstrainedBB_ZI_PYP':
        aA = fit_constrained_beta_binomial_mle([s[1] for s in ans_stats], [s[0] for s in ans_stats])
        aU = fit_constrained_beta_binomial_mle([s[1] for s in unans_stats], [s[0] for s in unans_stats])
        alA, dA = fit_pyp_mle([s[2] for s in ans_stats], [s[3] for s in ans_stats], [s[4] for s in ans_stats])
        alU, dU = fit_pyp_mle([s[2] for s in unans_stats], [s[3] for s in unans_stats], [s[4] for s in unans_stats])
        params.update({
            'a_A': round(aA, 4), 'b_A_implied': round(1.0 - aA, 4), 'alpha_A': round(alA, 4), 'd_A': round(dA, 4),
            'a_U': round(aU, 4), 'b_U_implied': round(1.0 - aU, 4), 'alpha_U': round(alU, 4), 'd_U': round(dU, 4)
        })

    elif method in ['SingleRest_Binomial_ZI_PYP_Surprisal', 'SingleRest_Binomial_ZI_PYP_Predictive']:
        pi0 = fit_binomial_pi_mle([s[1] for s in ans_stats], [s[0] for s in ans_stats])
        al0, d0 = fit_pyp_mle([s[2] for s in ans_stats], [s[3] for s in ans_stats], [s[4] for s in ans_stats])
        params.update({
            'pi_0': round(pi0, 4), 'alpha_0': round(al0, 4), 'd_0': round(d0, 4)
        })

    elif method == 'TwoRest_CRP':
        params['alpha_A'] = round(fit_crp_alpha_mle([s[2] for s in ans_stats], [s[3] for s in ans_stats]), 4)
        params['alpha_U'] = round(fit_crp_alpha_mle([s[2] for s in unans_stats], [s[3] for s in unans_stats]), 4)

    elif method == 'TwoRest_Reserved_CRP':
        aA, bA = fit_reserved_crp_mle([s[0] for s in ans_stats], [s[1] for s in ans_stats], [s[3] for s in ans_stats])
        aU, bU = fit_reserved_crp_mle([s[0] for s in unans_stats], [s[1] for s in unans_stats], [s[3] for s in unans_stats])
        params.update({'alpha_A': round(aA, 4), 'beta_A': round(bA, 4), 'alpha_U': round(aU, 4), 'beta_U': round(bU, 4)})

    elif method == 'TwoRest_Hurdle_CRP':
        aA, bA = fit_beta_binomial_mle([s[1] for s in ans_stats], [s[0] for s in ans_stats])
        aU, bU = fit_beta_binomial_mle([s[1] for s in unans_stats], [s[0] for s in unans_stats])
        alA = fit_crp_alpha_mle([s[2] for s in ans_stats], [s[3] for s in ans_stats])
        alU = fit_crp_alpha_mle([s[2] for s in unans_stats], [s[3] for s in unans_stats])
        params.update({
            'a_A': round(aA, 4), 'b_A': round(bA, 4), 'alpha_A': round(alA, 4),
            'a_U': round(aU, 4), 'b_U': round(bU, 4), 'alpha_U': round(alU, 4)
        })

    elif method == 'TwoRest_ZI_PYP':
        aA, bA = fit_beta_binomial_mle([s[1] for s in ans_stats], [s[0] for s in ans_stats])
        aU, bU = fit_beta_binomial_mle([s[1] for s in unans_stats], [s[0] for s in unans_stats])
        alA, dA = fit_pyp_mle([s[2] for s in ans_stats], [s[3] for s in ans_stats], [s[4] for s in ans_stats])
        alU, dU = fit_pyp_mle([s[2] for s in unans_stats], [s[3] for s in unans_stats], [s[4] for s in unans_stats])
        params.update({
            'a_A': round(aA, 4), 'b_A': round(bA, 4), 'alpha_A': round(alA, 4), 'd_A': round(dA, 4),
            'a_U': round(aU, 4), 'b_U': round(bU, 4), 'alpha_U': round(alU, 4), 'd_U': round(dU, 4)
        })

    elif method == 'TwoRest_PYP_NoIDK':
        alA, dA = fit_pyp_mle([s[2] for s in ans_stats], [s[3] for s in ans_stats], [s[4] for s in ans_stats])
        alU, dU = fit_pyp_mle([s[2] for s in unans_stats], [s[3] for s in unans_stats], [s[4] for s in unans_stats])
        params.update({
            'alpha_A': round(alA, 4), 'd_A': round(dA, 4),
            'alpha_U': round(alU, 4), 'd_U': round(dU, 4)
        })

    elif method in ['SingleRest_ZI_PYP_Surprisal', 'SingleRest_ZI_PYP_Predictive']:
        a0, b0 = fit_beta_binomial_mle([s[1] for s in ans_stats], [s[0] for s in ans_stats])
        al0, d0 = fit_pyp_mle([s[2] for s in ans_stats], [s[3] for s in ans_stats], [s[4] for s in ans_stats])
        params.update({
            'a_0': round(a0, 4), 'b_0': round(b0, 4),
            'alpha_0': round(al0, 4), 'd_0': round(d0, 4)
        })

    elif method == 'SingleRest_PYP_NoIDK':
        al0, d0 = fit_pyp_mle([s[2] for s in ans_stats], [s[3] for s in ans_stats], [s[4] for s in ans_stats])
        params.update({'alpha_0': round(al0, 4), 'd_0': round(d0, 4)})

    val_eval_params = dict(params, test_ignore_idk=val_ignore_idk)
    val_scores = [compute_generative_score(item, val_eval_params) for item in val_data]
    finite_scores = [s for s in val_scores if np.isfinite(s) and abs(s) < 1e5]
    if not finite_scores:
        finite_scores = [0.0]

    candidate_taus = sorted(list(set(np.percentile(finite_scores, np.linspace(0, 100, 45)))))
    if 0.0 not in candidate_taus:
        candidate_taus.append(0.0)

    best_tau = fast_vectorized_tau_search(
        val_data, val_scores, candidate_taus, test_ignore_idk=val_ignore_idk
    )
    params['tau'] = round(best_tau, 4)
    return params


def get_generative_restaurant_consensus(item, params):
    score = compute_generative_score(item, params)
    tau = params.get('tau', 0.0)
    if score > tau:
        return "DATA_UNCERTAIN", None, None

    test_ignore_idk = params.get('test_ignore_idk', True)
    N, n_0, m, clusters, is_correct_list, pred_label_list = extract_clusters(
        item, ignore_idk=test_ignore_idk
    )
    if not clusters:
        return "DATA_UNCERTAIN", None, None

    largest_cluster = max(clusters, key=len)
    best_i = largest_cluster[0]
    return pred_label_list[best_i], is_correct_list[best_i], best_i


# ==========================================
# 4. EVALUATION, KBQA SUB-METRICS & AURC PIPELINE
# ==========================================
def calculate_metrics(res):
    matrix = res["matrix_gt_pred_counts"]

    tp_a = matrix["ANSWERABLE"]["Pred_ANSWERABLE_correct"]
    fp_a = matrix["ANSWERABLE"]["Pred_ANSWERABLE_wrong"] + matrix["ABSTAIN"]["Pred_ANSWERABLE"]
    ans_count = res['ans_count']
    unans_count = res['unans_count']
    total_count = ans_count + unans_count

    ans_correct_in_samples = res.get('ans_correct_in_samples', 0)
    ans_sample_recall = (ans_correct_in_samples / ans_count) if ans_count > 0 else 0.0
    ans_accuracy = (tp_a / ans_count) if ans_count > 0 else 0.0

    pred_pos_a = tp_a + fp_a
    ans_precision = (tp_a / pred_pos_a) if pred_pos_a > 0 else 0.0
    ans_f1 = 2 * (ans_precision * ans_accuracy) / (ans_precision + ans_accuracy) if (ans_precision + ans_accuracy) > 0 else 0.0
    coverage = (pred_pos_a / total_count) if total_count > 0 else 0.0
    selective_accuracy = ans_precision

    tp_u = matrix["ABSTAIN"]["Pred_Unanswerable"]
    fp_u = matrix["ANSWERABLE"]["Pred_Unanswerable"]

    unans_correct_in_samples = res.get('unans_correct_in_samples', 0)
    unans_sample_recall = (unans_correct_in_samples / unans_count) if unans_count > 0 else 0.0
    unans_accuracy = (tp_u / unans_count) if unans_count > 0 else 0.0

    pred_pos_u = tp_u + fp_u
    unans_precision = (tp_u / pred_pos_u) if pred_pos_u > 0 else 0.0
    unans_f1 = 2 * (unans_precision * unans_accuracy) / (unans_precision + unans_accuracy) if (unans_precision + unans_accuracy) > 0 else 0.0

    total_correct = tp_a + tp_u
    overall_accuracy = (total_correct / total_count) if total_count > 0 else 0.0
    balanced_accuracy = (ans_accuracy + unans_accuracy) / 2.0

    res["metrics"] = {
        "answerable": {
            "precision": round(ans_precision, 2),
            "accuracy": round(ans_accuracy, 2),
            "ans_correct_in_samples": round(ans_sample_recall, 2),
            "f1": round(ans_f1, 2),
            "coverage": round(coverage, 2),
            "selective_accuracy": round(selective_accuracy, 2),
        },
        "unanswerable": {
            "precision": round(unans_precision, 2),
            "accuracy": round(unans_accuracy, 2),
            "unans_correct_in_samples": round(unans_sample_recall, 2),
            "f1": round(unans_f1, 2)
        },
        "overall_accuracy": round(overall_accuracy, 2),
        "overall_f1": round((ans_f1 + unans_f1) / 2.0, 2),
        "balanced_accuracy": round(balanced_accuracy, 2)
    }

    kbqa_raw = res.pop("_kbqa_accum", None)
    if kbqa_raw:
        def _finalize_bucket(b):
            cnt = b["count"]
            if cnt <= 0:
                return {"count": 0, "EM": 0.0, "F1_L": 0.0, "Strict_LF_EM": 0.0, "Strict_Exec_F1": 0.0}
            return {
                "count": cnt,
                "EM": round((b["sys_em_sum"] / cnt) * 100.0, 2),
                "F1_L": round((b["sys_f1_sum"] / cnt) * 100.0, 2),
                "Strict_LF_EM": round((b["strict_em_sum"] / cnt) * 100.0, 2),
                "Strict_Exec_F1": round((b["strict_f1_sum"] / cnt) * 100.0, 2),
            }

        overall_b = _finalize_bucket(kbqa_raw["buckets"]["all"])
        ans_b = _finalize_bucket(kbqa_raw["buckets"]["A"])
        unans_b = _finalize_bucket(kbqa_raw["buckets"]["U"])
        nk_b = _finalize_bucket(kbqa_raw["buckets"]["NK"])
        na_b = _finalize_bucket(kbqa_raw["buckets"]["NA"])

        level_summary = {
            lvl: _finalize_bucket(b)
            for lvl, b in kbqa_raw["levels"].items() if b["count"] > 0
        }
        missing_elem_summary = {
            elem: _finalize_bucket(b)
            for elem, b in kbqa_raw["missing_elements"].items() if b["count"] > 0
        }

        macro_au_f1 = round((ans_b["F1_L"] + unans_b["F1_L"]) / 2.0, 2)
        macro_au_em = round((ans_b["EM"] + unans_b["EM"]) / 2.0, 2)

        answered_a_cnt = kbqa_raw["answered_A_count"]
        selective_a_f1 = round((kbqa_raw["answered_A_f1_sum"] / answered_a_cnt) * 100.0, 2) if answered_a_cnt > 0 else 0.0
        selective_a_em = round((kbqa_raw["answered_A_em_sum"] / answered_a_cnt) * 100.0, 2) if answered_a_cnt > 0 else 0.0

        res["kbqa_metrics"] = {
            "overall": overall_b,
            "A_overall": ans_b,
            "U_overall": unans_b,
            "NK_overall": nk_b,
            "NA_overall": na_b,
            "macro_AU_F1_L": macro_au_f1,
            "macro_AU_EM": macro_au_em,
            "selective_A_F1_L": selective_a_f1,
            "selective_A_EM": selective_a_em,
            "by_level": level_summary,
            "by_missing_kb_element": missing_elem_summary,
            "uniqueness_diagnostics": {
                "ans_oracle_best_F1_L": round((kbqa_raw["ans_oracle_f1_sum"] / ans_count) * 100.0, 2) if ans_count > 0 else 0.0,
                "ans_oracle_best_EM": round((kbqa_raw["ans_oracle_em_sum"] / ans_count) * 100.0, 2) if ans_count > 0 else 0.0,
                "ans_f1_minus_em_gap": round(ans_b["F1_L"] - ans_b["EM"], 2),
                "ans_partial_match_count_0_lt_f1_lt_1": kbqa_raw["partial_f1_count"],
                "ans_right_answer_wrong_lf_count": kbqa_raw["f1_one_em_zero_count"],
                "multi_sparql_same_answer_top_cluster_rate": round(kbqa_raw["multi_sparql_top_cluster_count"] / max(1, total_count), 4),
                "avg_empty_kg_executions_per_question_A": round(kbqa_raw["empty_exec_sum_A"] / max(1, ans_count), 2),
                "avg_empty_kg_executions_per_question_NA": round(kbqa_raw["empty_exec_sum_NA"] / max(1, na_b["count"]), 2) if na_b["count"] > 0 else 0.0,
                "avg_empty_kg_executions_per_question_NK": round(kbqa_raw["empty_exec_sum_NK"] / max(1, nk_b["count"]), 2) if nk_b["count"] > 0 else 0.0,
            }
        }

        res["metrics"]["kbqa_overall_f1"] = round(overall_b["F1_L"] / 100.0, 4)
        res["metrics"]["kbqa_overall_em"] = round(overall_b["EM"] / 100.0, 4)
        res["metrics"]["kbqa_macro_au_f1"] = round(macro_au_f1 / 100.0, 4)
        res["metrics"]["kbqa_macro_au_em"] = round(macro_au_em / 100.0, 4)

    return res


def risk_coverage_curve(conf, can_answer, is_wrong, total_n, max_points=50):
    conf = np.asarray(conf, dtype=float)
    can = np.asarray(can_answer, dtype=bool)
    wrong = np.asarray(is_wrong, dtype=bool)
    idx = np.where(can)[0]
    if len(idx) == 0 or total_n <= 0:
        return {"aurc": None, "max_coverage": 0.0, "risk_coverage_curve": []}

    order = idx[np.argsort(-conf[idx], kind="mergesort")]
    c = conf[order]
    cum_wrong = np.cumsum(wrong[order].astype(float))
    k = np.arange(1, len(order) + 1, dtype=float)

    ends = np.where(np.append(c[1:] != c[:-1], True))[0]
    cov = k[ends] / float(total_n)
    risk = cum_wrong[ends] / k[ends]
    prev_cov = np.concatenate([[0.0], cov[:-1]])
    max_cov = float(cov[-1])
    aurc = float(np.sum((cov - prev_cov) * risk) / max_cov)

    if len(cov) > max_points:
        keep = np.unique(np.linspace(0, len(cov) - 1, max_points).astype(int))
        cov, risk = cov[keep], risk[keep]
    return {
        "aurc": round(aurc, 4),
        "max_coverage": round(max_cov, 4),
        "risk_coverage_curve": [[round(float(a), 4), round(float(b), 4)] for a, b in zip(cov, risk)],
    }


def _item_answer_confidence(item, method, threshold, ignore_idk, denom_mode):
    if method in GENERATIVE_METHODS:
        ignore_idk = threshold.get('test_ignore_idk', True)

    N, n_0, m, clusters, is_corr, pred_lbl = extract_clusters(item, ignore_idk=ignore_idk)
    if N == 0 or not clusters:
        return (-1e9, False, False)

    best = max(clusters, key=len)
    best_i = best[0]
    denom = m if denom_mode == 'm' else N

    if method in ('majority', 'most_freq', 'adapted_sc'):
        if denom <= 0:
            return (-1e9, False, False)
        conf = len(best) / float(denom)
    elif method == 'crp_k':
        if denom <= 0:
            return (-1e9, False, False)
        K = len(clusters)
        conf = -(K * float(N) / denom if denom_mode == 'm' else float(K))
    elif method == 'crp_theta':
        if denom <= 0:
            return (-1e9, False, False)
        conf = -get_mle_theta(denom, len(clusters))
    elif method == 'Adaptive_Consistency_beta':
        if "_fast_cache" not in item:
            return None
        seq = item["_fast_cache"][bool(ignore_idk)]['seq']
        last = next((s for s in reversed(seq) if s is not None), None)
        if last is None:
            return (-1e9, False, False)
        conf = last[1] if denom_mode == 'm' else last[0]
    elif method in GENERATIVE_METHODS:
        conf = -compute_generative_score(item, threshold)
    else:
        return None

    can_answer = (pred_lbl[best_i] == "ANSWERABLE")
    gt_unans = (item.get("should_abstain") is True)
    is_wrong = gt_unans or (not bool(is_corr[best_i]))
    return (conf, can_answer, is_wrong)


def compute_aurc_metrics(data, method, threshold, ignore_idk, denom_mode):
    rows = []
    for item in data:
        r = _item_answer_confidence(item, method, threshold, ignore_idk, denom_mode)
        if r is None:
            return {"aurc": None, "max_coverage": None, "risk_coverage_curve": []}
        rows.append(r)
    if not rows:
        return {"aurc": None, "max_coverage": None, "risk_coverage_curve": []}
    conf, can, wrong = zip(*rows)
    return risk_coverage_curve(conf, can, wrong, total_n=len(rows))


def _new_kbqa_bucket():
    return {"count": 0, "sys_em_sum": 0.0, "sys_f1_sum": 0.0, "strict_em_sum": 0.0, "strict_f1_sum": 0.0}


def _add_to_kbqa_bucket(b, sys_em, sys_f1, strict_em, strict_f1):
    b["count"] += 1
    b["sys_em_sum"] += sys_em
    b["sys_f1_sum"] += sys_f1
    b["strict_em_sum"] += strict_em
    b["strict_f1_sum"] += strict_f1


def process_run(
    data,
    method,
    threshold=None,
    ignore_abstention_tokens=False,
    denom_mode='N',
    disable_tqdm=True
):
    res = {
        'ans_count': 0, 'unans_count': 0,
        'ans_correct_in_samples': 0,
        'unans_correct_in_samples': 0,
        "matrix_gt_pred_counts": {
            "ANSWERABLE": {"Pred_ANSWERABLE_correct": 0, "Pred_ANSWERABLE_wrong": 0, "Pred_Unanswerable": 0},
            "ABSTAIN": {"Pred_ANSWERABLE": 0, "Pred_Unanswerable": 0}
        },
        "error_analysis": {
            "FN_missed_answerable": {"count": 0, "correct_present_count": 0, "correct_was_most_freq_count": 0, "sum_correct_freq": 0},
            "FP_hallucinated_unanswerable": {"count": 0, "sum_winning_freq": 0, "sum_total_clusters": 0, 'correct_present_count': 0},
            "TN_successful_abstention": {"count": 0, "sum_total_clusters": 0},
            "answerable_max_freq_histogram": {str(i): 0 for i in range(21)},
            "unanswerable_max_freq_histogram": {str(i): 0 for i in range(21)}
        },
        "_kbqa_accum": {
            "buckets": {k: _new_kbqa_bucket() for k in ("all", "A", "U", "NK", "NA")},
            "levels": {},
            "missing_elements": {k: _new_kbqa_bucket() for k in ("T", "R", "E", "F", "E_NA", "E_NK")},
            "answered_A_count": 0,
            "answered_A_f1_sum": 0.0,
            "answered_A_em_sum": 0.0,
            "ans_oracle_f1_sum": 0.0,
            "ans_oracle_em_sum": 0.0,
            "partial_f1_count": 0,
            "f1_one_em_zero_count": 0,
            "multi_sparql_top_cluster_count": 0,
            "empty_exec_sum_A": 0,
            "empty_exec_sum_NA": 0,
            "empty_exec_sum_NK": 0,
        }
    }

    kbqa_acc = res["_kbqa_accum"]
    iterator = tqdm(data, disable=disable_tqdm)

    for item in iterator:
        should_abstain = item.get("should_abstain")
        gt = "ABSTAIN" if should_abstain is True else "ANSWERABLE"

        _, _, _, em_list, f1_list, _, _ = _extract_item_arrays(item)
        max_freq, total_clusters, correct_present, correct_freq, is_correct_most_freq = get_item_metadata(
            item, ignore_abstention_tokens=ignore_abstention_tokens
        )

        if "_fast_cache" in item:
            k_meta = item["_fast_cache"][bool(ignore_abstention_tokens)]['kbqa_meta']
            empty_exec_cnt = k_meta['empty_exec_count']
            oracle_f1 = k_meta['oracle_best_f1']
            oracle_em = k_meta['oracle_best_em']
            if k_meta['multi_sparql_in_top_cluster']:
                kbqa_acc["multi_sparql_top_cluster_count"] += 1
        else:
            empty_exec_cnt = 0
            oracle_f1 = max(f1_list) if f1_list else 0.0
            oracle_em = max(em_list) if em_list else 0.0

        if not ignore_abstention_tokens:
            max_freq_ign, _, _, _, _ = get_item_metadata(item, ignore_abstention_tokens=True)
        else:
            max_freq_ign = max_freq

        hist_key = str(min(max_freq_ign, 20))

        golden_sexpr = str(item.get("golden_s_expr") or item.get("s_expression") or "").strip()
        if gt == "ABSTAIN":
            unans_subtype = "NK" if golden_sexpr == "NK" else "NA"
            kbqa_acc[f"empty_exec_sum_{unans_subtype}"] += empty_exec_cnt
            res['unans_count'] += 1
            if correct_present:
                res['unans_correct_in_samples'] += 1
            if hist_key in res["error_analysis"]["unanswerable_max_freq_histogram"]:
                res["error_analysis"]["unanswerable_max_freq_histogram"][hist_key] += 1
        else:
            unans_subtype = "A"
            kbqa_acc["empty_exec_sum_A"] += empty_exec_cnt
            kbqa_acc["ans_oracle_f1_sum"] += oracle_f1
            kbqa_acc["ans_oracle_em_sum"] += oracle_em
            res['ans_count'] += 1
            if correct_present:
                res['ans_correct_in_samples'] += 1
            if hist_key in res["error_analysis"]["answerable_max_freq_histogram"]:
                res["error_analysis"]["answerable_max_freq_histogram"][hist_key] += 1

        if method == 'greedy':
            pred_val = item.get("pred_label", [])
            eff_pred = pred_val[0] if isinstance(pred_val, list) and len(pred_val) > 0 else pred_val
            corr_val = item.get("is_correct", [])
            eff_correct = corr_val[0] if isinstance(corr_val, list) and len(corr_val) > 0 else corr_val
            best_i = 0
        elif method in ['majority', 'most_freq']:
            eff_pred, eff_correct, best_i = get_generalized_sc_consensus(
                item, method=method, threshold=threshold,
                ignore_idk=ignore_abstention_tokens, denom_mode=denom_mode
            )
        elif method == 'adapted_sc':
            eff_pred, eff_correct, best_i = get_generalized_sc_consensus(
                item, method='majority', threshold=threshold,
                ignore_idk=True, denom_mode=denom_mode
            )
        elif method in ['crp_k', 'crp_theta']:
            eff_pred, eff_correct, best_i = get_generalized_crp_consensus(
                item, method=method, threshold=threshold,
                ignore_idk=ignore_abstention_tokens, denom_mode=denom_mode
            )
        elif method == 'Adaptive_Consistency_beta':
            eff_pred, eff_correct, best_i = get_Adaptive_Consistency_beta_consensus(
                item, threshold=threshold,
                ignore_idk=ignore_abstention_tokens, denom_mode=denom_mode
            )
        elif method in GENERATIVE_METHODS:
            eff_pred, eff_correct, best_i = get_generative_restaurant_consensus(item, params=threshold)
        elif method == 'Oracle_Intermediate':
            eff_pred, eff_correct, best_i = get_oracle_intermediate_consensus(
                item, ignore_idk=ignore_abstention_tokens
            )
        else:
            eff_pred, eff_correct, best_i = "DATA_UNCERTAIN", None, None

        if isinstance(eff_pred, list):
            eff_pred = str(eff_pred[0]).strip() if len(eff_pred) > 0 else ""
        elif eff_pred is not None:
            eff_pred = str(eff_pred).strip()
        else:
            eff_pred = ""

        if isinstance(eff_correct, list):
            eff_correct = bool(eff_correct[0]) if len(eff_correct) > 0 else False

        is_pred_unanswerable = eff_pred in UNANS_LABELS
        is_pred_answerable = (eff_pred == "ANSWERABLE")

        sample_em = float(em_list[best_i]) if (best_i is not None and best_i < len(em_list)) else 0.0
        sample_f1 = float(f1_list[best_i]) if (best_i is not None and best_i < len(f1_list)) else 0.0

        if gt == "ANSWERABLE":
            if is_pred_answerable:
                sys_em, sys_f1 = sample_em, sample_f1
                strict_em, strict_f1 = sample_em, sample_f1
                kbqa_acc["answered_A_count"] += 1
                kbqa_acc["answered_A_em_sum"] += sample_em
                kbqa_acc["answered_A_f1_sum"] += sample_f1
                if 0.0 < sample_f1 < 1.0:
                    kbqa_acc["partial_f1_count"] += 1
                if sample_f1 == 1.0 and sample_em == 0.0:
                    kbqa_acc["f1_one_em_zero_count"] += 1
            else:
                sys_em, sys_f1 = 0.0, 0.0
                strict_em, strict_f1 = 0.0, 0.0
        else:
            if is_pred_unanswerable:
                sys_em, sys_f1 = 1.0, 1.0
                if unans_subtype == "NK":
                    strict_em, strict_f1 = 1.0, 1.0
                else:
                    _, _, _, cl_ign, _, _, em_ign, f1_ign = extract_clusters_with_kbqa(item, ignore_idk=True)
                    top_idx = max(cl_ign, key=len)[0] if cl_ign else None
                    strict_em = float(em_ign[top_idx]) if (top_idx is not None and top_idx < len(em_ign)) else 0.0
                    strict_f1 = float(f1_ign[top_idx]) if (top_idx is not None and top_idx < len(f1_ign)) else 0.0
            else:
                sys_em, sys_f1 = (sample_em, sample_em) if unans_subtype == "NA" else (0.0, 0.0)
                strict_em, strict_f1 = sample_em, sample_f1

        _add_to_kbqa_bucket(kbqa_acc["buckets"]["all"], sys_em, sys_f1, strict_em, strict_f1)
        _add_to_kbqa_bucket(kbqa_acc["buckets"]["A" if gt == "ANSWERABLE" else "U"], sys_em, sys_f1, strict_em, strict_f1)
        if gt == "ABSTAIN":
            _add_to_kbqa_bucket(kbqa_acc["buckets"][unans_subtype], sys_em, sys_f1, strict_em, strict_f1)

        lvl_key = f"{'A' if gt == 'ANSWERABLE' else 'U'}_{item.get('level', 'iid')}"
        if lvl_key not in kbqa_acc["levels"]:
            kbqa_acc["levels"][lvl_key] = _new_kbqa_bucket()
        _add_to_kbqa_bucket(kbqa_acc["levels"][lvl_key], sys_em, sys_f1, strict_em, strict_f1)

        if gt == "ABSTAIN":
            missing_elems = str(item.get("Missing_KB_elements", ""))
            for code in ("T", "R", "E", "F"):
                if code in missing_elems:
                    _add_to_kbqa_bucket(kbqa_acc["missing_elements"][code], sys_em, sys_f1, strict_em, strict_f1)
                    if code == "E":
                        _add_to_kbqa_bucket(kbqa_acc["missing_elements"][f"E_{unans_subtype}"], sys_em, sys_f1, strict_em, strict_f1)

        if gt == "ANSWERABLE":
            if is_pred_answerable:
                if eff_correct:
                    res["matrix_gt_pred_counts"][gt]['Pred_ANSWERABLE_correct'] += 1
                else:
                    res["matrix_gt_pred_counts"][gt]['Pred_ANSWERABLE_wrong'] += 1
            elif is_pred_unanswerable:
                res["matrix_gt_pred_counts"][gt]['Pred_Unanswerable'] += 1
                res["error_analysis"]["FN_missed_answerable"]["count"] += 1
                if correct_present:
                    res["error_analysis"]["FN_missed_answerable"]["correct_present_count"] += 1
                    res["error_analysis"]["FN_missed_answerable"]["sum_correct_freq"] += correct_freq
                    if is_correct_most_freq:
                        res["error_analysis"]["FN_missed_answerable"]["correct_was_most_freq_count"] += 1

        elif gt == "ABSTAIN":
            if is_pred_answerable:
                res["matrix_gt_pred_counts"][gt]['Pred_ANSWERABLE'] += 1
                res["error_analysis"]["FP_hallucinated_unanswerable"]["count"] += 1
                res["error_analysis"]["FP_hallucinated_unanswerable"]["correct_present_count"] += 1
                res["error_analysis"]["FP_hallucinated_unanswerable"]["sum_winning_freq"] += max_freq
                res["error_analysis"]["FP_hallucinated_unanswerable"]["sum_total_clusters"] += total_clusters
            elif is_pred_unanswerable:
                res["matrix_gt_pred_counts"][gt]['Pred_Unanswerable'] += 1
                res["error_analysis"]["TN_successful_abstention"]["count"] += 1
                res["error_analysis"]["TN_successful_abstention"]["sum_total_clusters"] += total_clusters

    analysis = res["error_analysis"]
    fn_cp = analysis["FN_missed_answerable"]["correct_present_count"]
    analysis["FN_missed_answerable"]["avg_correct_freq_when_present"] = round(analysis["FN_missed_answerable"]["sum_correct_freq"] / fn_cp, 2) if fn_cp > 0 else 0.0
    del analysis["FN_missed_answerable"]["sum_correct_freq"]

    fp_c = analysis["FP_hallucinated_unanswerable"]["count"]
    analysis["FP_hallucinated_unanswerable"]["avg_winning_freq"] = round(analysis["FP_hallucinated_unanswerable"]["sum_winning_freq"] / fp_c, 2) if fp_c > 0 else 0.0
    analysis["FP_hallucinated_unanswerable"]["avg_total_clusters"] = round(analysis["FP_hallucinated_unanswerable"]["sum_total_clusters"] / fp_c, 2) if fp_c > 0 else 0.0
    del analysis["FP_hallucinated_unanswerable"]["sum_winning_freq"]
    del analysis["FP_hallucinated_unanswerable"]["sum_total_clusters"]

    tn_c = analysis["TN_successful_abstention"]["count"]
    analysis["TN_successful_abstention"]["avg_total_clusters"] = round(analysis["TN_successful_abstention"]["sum_total_clusters"] / tn_c, 2) if tn_c > 0 else 0.0
    del analysis["TN_successful_abstention"]["sum_total_clusters"]

    res = calculate_metrics(res)
    res["metrics"].update(
        compute_aurc_metrics(data, method, threshold, ignore_abstention_tokens, denom_mode)
    )
    return res


# ==========================================
# 5. FAST THRESHOLD & 3-COMBO ABLATION RUNNER
# ==========================================
def extract_target_score(res):
    if TARGET_METRIC == 'overall_accuracy': return res["metrics"]["overall_accuracy"]
    elif TARGET_METRIC == 'balanced_accuracy': return res["metrics"]["balanced_accuracy"]
    elif TARGET_METRIC == 'overall_f1': return res["metrics"]["overall_f1"]
    elif TARGET_METRIC == 'ans_precision': return res["metrics"]["answerable"]["precision"]
    elif TARGET_METRIC == 'ans_accuracy': return res["metrics"]["answerable"]["accuracy"]
    elif TARGET_METRIC == 'ans_f1': return res["metrics"]["answerable"]["f1"]
    elif TARGET_METRIC in ('kbqa_overall_f1', 'kbqa_overall_em', 'kbqa_macro_au_f1', 'kbqa_macro_au_em'):
        return res["metrics"].get(TARGET_METRIC, res["metrics"]["overall_f1"])
    return res["metrics"]["overall_accuracy"]


def learn_best_threshold(val_data, method, val_ignore_idk=True, val_denom='N'):
    if method in ['majority', 'adapted_sc']:
        test_thresholds = [i / 20.0 for i in range(1, 21)]
    elif method == 'crp_k':
        test_thresholds = list(range(1, 21))
    elif method == 'crp_theta':
        test_thresholds = list(np.arange(0, 20, 0.2))
    elif method == 'Adaptive_Consistency_beta':
        test_thresholds = list(np.arange(0.05, 1.0, 0.05))
    else:
        return None

    if method in ['crp_k', 'crp_theta']:
        val_scores = []
        for item in val_data:
            N, n_0, m, clusters, _, _ = extract_clusters(item, ignore_idk=val_ignore_idk)
            denom = m if val_denom == 'm' else N
            if N == 0 or not clusters or denom <= 0:
                val_scores.append(1e6)
            elif method == 'crp_k':
                K = len(clusters)
                val_scores.append(K * (float(N) / float(denom)) if val_denom == 'm' else float(K))
            else:
                val_scores.append(get_mle_theta(denom, len(clusters)))
        return fast_vectorized_tau_search(val_data, val_scores, test_thresholds, test_ignore_idk=val_ignore_idk)

    best_threshold = test_thresholds[0]
    best_score = -1.0
    for t in test_thresholds:
        res = process_run(
            val_data, method=method, threshold=t,
            ignore_abstention_tokens=val_ignore_idk, denom_mode=val_denom
        )
        score = extract_target_score(res)
        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold


def run_3_ablation_combos(val_samples, test_samples, numbered_prefix, internal_method, fixed_threshold=None, skip_keep_idk=False):
    combo_results = {}
    for combo_tag, ignore_idk, denom_mode in THREE_ABLATION_COMBOS:
        if skip_keep_idk and not ignore_idk:
            continue

        opt_t = fixed_threshold if fixed_threshold is not None else learn_best_threshold(
            val_samples, internal_method, val_ignore_idk=ignore_idk, val_denom=denom_mode
        )
        key = f"{numbered_prefix}__{combo_tag}"

        test_res = process_run(
            test_samples, method=internal_method, threshold=opt_t,
            ignore_abstention_tokens=ignore_idk, denom_mode=denom_mode
        )
        val_res = process_run(
            val_samples, method=internal_method, threshold=opt_t,
            ignore_abstention_tokens=ignore_idk, denom_mode=denom_mode
        )
        test_res['val_set'] = val_res
        test_res['learned_threshold'] = round(opt_t, 4) if isinstance(opt_t, float) else opt_t
        test_res['ablation_config'] = {
            'ignore_idk': ignore_idk,
            'denom_mode': denom_mode
        }
        combo_results[key] = test_res

    return combo_results


# ==========================================
# 6. DATA SPLIT & EXECUTION UTILS
# ==========================================
def get_split_identifiers(split, dataset):
    split_dir = os.path.join(DATA_DIRECTORY, split)
    identifiers = set()
    for ext in ['.json', '.jsonl']:
        filepath = os.path.join(split_dir, f"{dataset}{ext}")
        if os.path.exists(filepath):
            if ext == '.json':
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'data' in data:
                            data = data['data']
                        for item in data:
                            q = str(item.get('question') or item.get('qid') or '').strip()
                            if q:
                                identifiers.add(q)
                            if 'qid' in item:
                                identifiers.add(str(item['qid']).strip())
                except Exception:
                    pass
            elif ext == '.jsonl':
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            obj = json.loads(line)
                            q = str(obj.get('question') or obj.get('qid') or '').strip()
                            if q:
                                identifiers.add(q)
                            if 'qid' in obj:
                                identifiers.add(str(obj['qid']).strip())
                except Exception:
                    pass
            break
    return identifiers


def split_output_data(output_data, val_questions, test_questions, val_ratio_fallback=0.2):
    val_data, test_data = [], []
    if val_questions or test_questions:
        for item in output_data:
            q = str(item.get('question', '')).strip()
            qid_str = str(item.get('qid', '')).strip()
            if (q and q in val_questions) or (qid_str and qid_str in val_questions):
                val_data.append(item)
            elif (q and q in test_questions) or (qid_str and qid_str in test_questions):
                test_data.append(item)

    if not val_data and not test_data and output_data:
        rng = np.random.default_rng(100)
        ans_items = [x for x in output_data if not x.get("should_abstain")]
        unans_items = [x for x in output_data if x.get("should_abstain")]
        for group in (ans_items, unans_items):
            indices = np.arange(len(group))
            rng.shuffle(indices)
            n_val = max(1, int(round(len(group) * val_ratio_fallback))) if len(group) > 1 else 0
            for idx_pos, item_idx in enumerate(indices):
                if idx_pos < n_val:
                    val_data.append(group[item_idx])
                else:
                    test_data.append(group[item_idx])

    return val_data, test_data


def load_data(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f).get('results', [])
    except Exception:
        return []


def process_combination(model, dataset, aware, val_q, test_q, req_exps):
    model_dir = os.path.join(BASE_DIRECTORY, model)
    greedy_file = os.path.join(model_dir, f"{dataset}{aware}_greedy.json")
    samples_file = os.path.join(model_dir, f"{dataset}{aware}_samples.json")

    local_result = {}

    greedy_all_data = load_data(greedy_file) if 'Greedy' in req_exps else []
    samples_all_data = load_data(samples_file) if any(e != 'Greedy' for e in req_exps) else []

    if not greedy_all_data and 'Greedy' in req_exps and samples_all_data:
        greedy_all_data = []
        for item in samples_all_data:
            g_item = dict(item)
            for k in ("top_predictions", "answer", "boxed", "pred_label", "pred_answer", "em", "f1", "is_correct"):
                if isinstance(g_item.get(k), list) and len(g_item[k]) > 0:
                    g_item[k] = [g_item[k][0]]
            g_item["compare_label"] = {"0": [1]}
            greedy_all_data.append(g_item)

    greedy_val, greedy_test = split_output_data(greedy_all_data, val_q, test_q) if greedy_all_data else ([], [])
    val_samples, test_samples = split_output_data(samples_all_data, val_q, test_q) if samples_all_data else ([], [])

    if greedy_val and greedy_test:
        precompute_dataset_caches(greedy_val)
        precompute_dataset_caches(greedy_test)
    if val_samples and test_samples:
        precompute_dataset_caches(val_samples)
        precompute_dataset_caches(test_samples)

    for idx, exp_name in ORDERED_EXPERIMENTS:
        if exp_name not in req_exps:
            continue
        prefix = f"{idx:02d}_{exp_name}"

        if exp_name == 'Greedy':
            if greedy_test:
                local_result[prefix] = process_run(greedy_test, method='greedy')
                local_result[prefix]['val_set'] = process_run(greedy_val, method='greedy')
            continue

        if not (val_samples and test_samples):
            continue

        if exp_name == 'self_Consistency':
            for tag, ign_idk in [("1_keepIDK", False), ("2_dropIDK", True)]:
                key = f"{prefix}__{tag}"
                local_result[key] = process_run(test_samples, method='most_freq', ignore_abstention_tokens=ign_idk)
                local_result[key]['val_set'] = process_run(val_samples, method='most_freq', ignore_abstention_tokens=ign_idk)

        elif exp_name in ['Majority', 'self_Consistency_T', 'scUn', 'CRP_k', 'CRP_theta', 'Adaptive_Consistency_beta']:
            internal_map = {
                'Majority': ('majority', 0.5, False),
                'self_Consistency_T': ('majority', None, False),
                'scUn': ('adapted_sc', None, True),
                'CRP_k': ('crp_k', None, False),
                'CRP_theta': ('crp_theta', None, False),
                'Adaptive_Consistency_beta': ('Adaptive_Consistency_beta', None, False),
            }
            internal_m, fixed_t, skip_keep = internal_map[exp_name]
            combos_dict = run_3_ablation_combos(
                val_samples, test_samples, prefix, internal_m,
                fixed_threshold=fixed_t, skip_keep_idk=skip_keep
            )
            local_result.update(combos_dict)

        elif exp_name in NO_IDK_TABLE_GENERATIVE_METHODS:
            for tag, ign_idk in [("1_keepIDK", False), ("2_dropIDK", True)]:
                key = f"{prefix}__{tag}"
                learned_params = train_generative_restaurant_model(
                    val_samples, method=exp_name, val_ignore_idk=ign_idk, test_ignore_idk=ign_idk
                )
                local_result[key] = process_run(
                    test_samples, method=exp_name, threshold=learned_params, ignore_abstention_tokens=ign_idk
                )
                local_result[key]['val_set'] = process_run(
                    val_samples, method=exp_name, threshold=learned_params, ignore_abstention_tokens=ign_idk
                )
                local_result[key]['learned_params'] = learned_params

        elif exp_name in IDK_TABLE_GENERATIVE_METHODS:
            learned_params = train_generative_restaurant_model(val_samples, method=exp_name)
            local_result[prefix] = process_run(
                test_samples, method=exp_name, threshold=learned_params, ignore_abstention_tokens=True
            )
            local_result[prefix]['val_set'] = process_run(
                val_samples, method=exp_name, threshold=learned_params, ignore_abstention_tokens=True
            )
            local_result[prefix]['learned_params'] = learned_params

        elif exp_name == 'Oracle_Intermediate':
            for tag, ign_idk in [("1_keepIDK", False), ("2_dropIDK", True)]:
                key = f"{prefix}__{tag}"
                local_result[key] = process_run(
                    test_samples, method='Oracle_Intermediate', ignore_abstention_tokens=ign_idk
                )
                local_result[key]['val_set'] = process_run(
                    val_samples, method='Oracle_Intermediate', ignore_abstention_tokens=ign_idk
                )

    return model, dataset, aware, local_result


if __name__ == "__main__":
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                master_result = json.load(f)
        except Exception:
            master_result = {model: {} for model in MODELS}
    else:
        master_result = {model: {} for model in MODELS}

    print(f"Running KBQA Evaluation for Models: {MODELS}")
    print(f"Target Optimization Metric: {TARGET_METRIC}")
    dataset_splits = {
        ds: {'val': get_split_identifiers('val', ds), 'test': get_split_identifiers('test', ds)}
        for ds in datasets
    }

    tasks = []
    for model in MODELS:
        if model not in master_result:
            master_result[model] = {}
        for dataset in datasets:
            if dataset not in master_result[model]:
                master_result[model][dataset] = {}
            val_q, test_q = dataset_splits[dataset]['val'], dataset_splits[dataset]['test']
            for aware in awareness:
                if aware not in master_result[model][dataset]:
                    master_result[model][dataset][aware] = {}
                tasks.append((model, dataset, aware, val_q, test_q, REQUESTED_EXPERIMENTS))

    max_workers = min(8, os.cpu_count() or 4)
    print(f"Dispatching {len(tasks)} dataset-level tasks across {max_workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_combination, *task): task for task in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing KBQA Combinations"):
            try:
                mod, ds, aw, res_dict = future.result()
                if res_dict:
                    combined = master_result[mod][ds].get(aw, {})
                    combined.update(res_dict)
                    master_result[mod][ds][aw] = {k: combined[k] for k in sorted(combined.keys())}
            except Exception as exc:
                task_info = futures[future]
                print(f"Task generated an exception for {task_info[:3]}: {exc}")

    print(f"Writing ordered KBQA results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(master_result, f, indent=2)
    print("Done!")