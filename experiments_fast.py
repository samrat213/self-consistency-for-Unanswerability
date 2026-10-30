import os
import json
import itertools
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
MODELS = ["result_qwen38"]
OUTPUT_FILE = '/home/scai/phd/aiz248314/KBQA/scUn/UA-Bench/results.json'

TARGET_METRIC = 'overall_f1'  # Options: 'overall_accuracy', 'balanced_accuracy', 'overall_f1', 'ans_f1'

datasets = [
    'ReliableMath', 
    "brokenmath_type_answer_AU",
    # "brokenmath_type_proof_AU",
]

REQUESTED_EXPERIMENTS = [
    'Greedy', 
    'Majority', 
    'self Consistency', 
    'self Consistency + T', 
    'scUn',
    'CRP_k',
    'CRP_theta',
    'Adaptive_Consistency_beta',
    # --- Low-Parameter 1-Param Gate Models (Binomial pi & Constrained b = 1 - a) ---
    'TwoRest_Binomial_Hurdle_CRP',
    'TwoRest_Binomial_ZI_PYP',
    'TwoRest_ConstrainedBB_ZI_PYP',
    'SingleRest_Binomial_ZI_PYP_Surprisal',
    'SingleRest_Binomial_ZI_PYP_Predictive',
    # --- 2-Param Gate & Reserved-Atom Models ---
    'TwoRest_CRP',
    'TwoRest_Reserved_CRP',
    'TwoRest_Hurdle_CRP',
    'TwoRest_ZI_PYP',
    'SingleRest_ZI_PYP_Surprisal',
    'SingleRest_ZI_PYP_Predictive',
    # --- PYP Without IDK Table (Keep vs Drop IDK in Train & Test) ---
    # 'SingleRest_PYP_NoIDK',
    # 'TwoRest_PYP_NoIDK',
    # --- Theoretical Upper Bound Oracle ---
    'Oracle_Intermediate'
]

RUN_ALL_16_ABLATION_COMBOS = False
awareness = ['_abstaintion', '']

UNANS_PHRASES = ["i don't know", "insufficient", "cannot be answered", "not enough information"]
UNANS_LABELS = {"ABSTAIN", "DATA_UNCERTAIN", "MODEL_UNCERTAIN", "UNANSWERABLE"}


# ==========================================
# 1. FAST CACHED CLUSTERING & EXTRACTION
# ==========================================
def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def is_sample_abstention(lbl, ans):
    lbl_str = str(lbl).strip() if lbl is not None else ""
    ans_str = str(ans).strip().lower() if ans is not None else ""
    if lbl_str in UNANS_LABELS:
        return True
    if any(p in ans_str for p in UNANS_PHRASES):
        return True
    return False


def _compute_clusters_raw(item, ignore_idk=False, up_to_k=None):
    is_correct_list = item.get("is_correct", [])
    if not isinstance(is_correct_list, list):
        is_correct_list = [is_correct_list]

    pred_label_list = item.get("pred_label", [])
    if not isinstance(pred_label_list, list):
        pred_label_list = [pred_label_list]

    pred_answer_list = item.get("pred_answer", [])
    if not isinstance(pred_answer_list, list):
        pred_answer_list = [pred_answer_list]

    compare_label = item.get("compare_label", {})
    if not isinstance(compare_label, dict):
        compare_label = {}

    N_full = len(is_correct_list)
    if N_full == 0:
        return 0, 0, 0, [], is_correct_list, pred_label_list

    N_total = min(up_to_k, N_full) if up_to_k is not None else N_full

    idk_indices = []
    substantive_indices = []
    for i in range(N_total):
        lbl = pred_label_list[i] if i < len(pred_label_list) else ""
        ans = pred_answer_list[i] if i < len(pred_answer_list) else ""
        if is_sample_abstention(lbl, ans):
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

    return N_total, n_0, m, clusters, is_correct_list, pred_label_list


# Pre-compute MLE theta lookup table for N, K in [0..50] -> O(1) instant lookup!
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
    """
    Runs ONCE per dataset split. Caches clusters, metadata, and sequential Beta trajectories
    on every item so process_run and threshold sweeps are 100x+ faster.
    """
    for item in data:
        if "_fast_cache" in item:
            continue
        cache = {}
        for ign in (False, True):
            N, n_0, m, clusters, is_corr, pred_lbl = _compute_clusters_raw(item, ignore_idk=ign, up_to_k=None)
            total_clusters = len(clusters)
            cluster_sizes = [len(c) for c in clusters]
            max_freq = max(cluster_sizes) if cluster_sizes else 0

            correct_freq = sum(1 for x in is_corr if x is True)
            correct_present = correct_freq > 0
            is_correct_most_freq = False
            if correct_present and clusters:
                correct_cluster_sizes = [len(c) for c in clusters if any(is_corr[idx] for idx in c)]
                if correct_cluster_sizes:
                    is_correct_most_freq = (max(correct_cluster_sizes) == max_freq)

            # Precompute sequential steps for Adaptive_Consistency_beta
            seq_steps = []
            for k in range(1, N + 1):
                _, n_0_k, _, cl_k, _, _ = _compute_clusters_raw(item, ignore_idk=ign, up_to_k=k)
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
                seq_steps.append((prob_N, prob_m, pred_lbl[best_i_k], is_corr[best_i_k]))

            cache[ign] = {
                'extract': (N, n_0, m, clusters, is_corr, pred_lbl),
                'meta': (max_freq, total_clusters, correct_present, correct_freq, is_correct_most_freq),
                'seq': seq_steps
            }
        item["_fast_cache"] = cache


def extract_clusters(item, ignore_idk=False, up_to_k=None):
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
# 2. BASELINE CONSENSUS & ORACLE METHODS
# ==========================================
def get_generalized_sc_consensus(
    item, method='majority', threshold=0.5, ignore_idk=False, denom_mode='N'
):
    N, n_0, m, clusters, is_correct_list, pred_label_list = extract_clusters(
        item, ignore_idk=ignore_idk
    )
    if N == 0 or not clusters:
        return "DATA_UNCERTAIN", None

    largest_cluster = max(clusters, key=len)
    max_votes = len(largest_cluster)
    best_i = largest_cluster[0]

    if method == 'most_freq':
        return pred_label_list[best_i], is_correct_list[best_i]

    denom = m if denom_mode == 'm' else N
    if denom <= 0:
        return "DATA_UNCERTAIN", None

    if (max_votes / float(denom)) >= threshold:
        return pred_label_list[best_i], is_correct_list[best_i]
    else:
        return "DATA_UNCERTAIN", None


def get_generalized_crp_consensus(
    item, method='crp_k', threshold=None, ignore_idk=False, denom_mode='N'
):
    N, n_0, m, clusters, is_correct_list, pred_label_list = extract_clusters(
        item, ignore_idk=ignore_idk
    )
    if N == 0 or not clusters:
        return "DATA_UNCERTAIN", None

    K = len(clusters)
    denom = m if denom_mode == 'm' else N
    if denom <= 0:
        return "DATA_UNCERTAIN", None

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
        return "DATA_UNCERTAIN", None

    largest_cluster = max(clusters, key=len)
    best_i = largest_cluster[0]
    return pred_label_list[best_i], is_correct_list[best_i]


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
                return step[2], step[3]
        return "DATA_UNCERTAIN", None

    is_correct_list = item.get("is_correct", [])
    if not is_correct_list:
        return "DATA_UNCERTAIN", None
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
            return pred_lbl[best_i], is_corr[best_i]

    return "DATA_UNCERTAIN", None


def get_oracle_intermediate_consensus(item, ignore_idk=False):
    N, n_0, m, clusters, is_correct_list, pred_label_list = extract_clusters(
        item, ignore_idk=ignore_idk
    )
    if N == 0 or not clusters:
        return "DATA_UNCERTAIN", None

    K = len(clusters)
    denom = m if ignore_idk else N

    largest_cluster = max(clusters, key=len)
    best_i = largest_cluster[0]

    if K == 1 and denom == N:
        return pred_label_list[best_i], is_correct_list[best_i]
    if K >= N:
        return "DATA_UNCERTAIN", None

    should_abstain = item.get("should_abstain") is True
    if should_abstain:
        return "DATA_UNCERTAIN", None
    else:
        return pred_label_list[best_i], is_correct_list[best_i]


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
    """
    Fully vectorized Pitman-Yor MLE using pre-aggregated count arrays (50x faster than Python loop).
    """
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
    # j_counts[idx] = how many items have K > j_vals[idx]
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
    test_ignore_idk = params.get('test_ignore_idk', method not in ['TwoRest_CRP'])

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
        aU, aA = params['alpha_U'], params['alpha_A']
        ll_U = K * np.log(aU) + scipy.special.gammaln(aU) - scipy.special.gammaln(N + aU)
        ll_A = K * np.log(aA) + scipy.special.gammaln(aA) - scipy.special.gammaln(N + aA)
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
    """
    Evaluates ALL candidate_taus simultaneously in < 0.2ms using 2D NumPy broadcasting,
    producing the EXACT same metric (including round(..., 2)) as calling process_run in a loop.
    """
    n_items = len(val_data)
    if n_items == 0 or not candidate_taus:
        return 0.0

    gt_abstain = np.zeros(n_items, dtype=bool)
    base_pred_ans = np.zeros(n_items, dtype=bool)
    base_pred_unans = np.zeros(n_items, dtype=bool)
    base_corr = np.zeros(n_items, dtype=bool)

    for idx, item in enumerate(val_data):
        gt_abstain[idx] = (item.get("should_abstain") is True)
        N, n_0, m, clusters, is_corr, pred_lbl = extract_clusters(item, ignore_idk=test_ignore_idk)
        if not clusters:
            base_pred_unans[idx] = True
        else:
            best_i = max(clusters, key=len)[0]
            lbl = pred_lbl[best_i]
            base_pred_ans[idx] = (lbl == "ANSWERABLE")
            base_pred_unans[idx] = (lbl in UNANS_LABELS)
            base_corr[idx] = bool(is_corr[best_i])

    scores_col = np.asarray(val_scores, dtype=float)[:, None]   # Shape: (M, 1)
    taus_row = np.asarray(candidate_taus, dtype=float)[None, :]  # Shape: (1, T)

    # Whenever score > tau, prediction is overridden to "DATA_UNCERTAIN" (Unanswerable)
    abstain_override = (scores_col > taus_row)  # Shape: (M, T)

    is_pred_ans = (~abstain_override) & base_pred_ans[:, None]
    is_pred_unans = abstain_override | base_pred_unans[:, None]
    is_corr_mat = (~abstain_override) & base_corr[:, None]

    gt_ans_col = (~gt_abstain)[:, None]
    gt_abs_col = gt_abstain[:, None]

    tp_a = np.sum(gt_ans_col & is_pred_ans & is_corr_mat, axis=0, dtype=float)
    fp_a = np.sum((gt_ans_col & is_pred_ans & (~is_corr_mat)) | (gt_abs_col & is_pred_ans), axis=0, dtype=float)
    tp_u = np.sum(gt_abs_col & is_pred_unans, axis=0, dtype=float)
    fp_u = np.sum(gt_ans_col & is_pred_unans, axis=0, dtype=float)

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
        elif TARGET_METRIC == 'ans_precision':
            score = round(ans_prec, 2)
        elif TARGET_METRIC == 'ans_accuracy':
            score = round(ans_acc, 2)
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
    # 'SingleRest_Binomial_ZI_PYP_Predictive',
    'TwoRest_CRP',
    'TwoRest_Reserved_CRP',
    'TwoRest_Hurdle_CRP',
    'TwoRest_ZI_PYP',
    'SingleRest_ZI_PYP_Surprisal',
    # 'SingleRest_ZI_PYP_Predictive',
}

GENERATIVE_METHODS = IDK_TABLE_GENERATIVE_METHODS | {'TwoRest_PYP_NoIDK', 'SingleRest_PYP_NoIDK'}
GENERATIVE_METHODS = IDK_TABLE_GENERATIVE_METHODS | {'TwoRest_PYP_NoIDK', 'SingleRest_PYP_NoIDK'}


def train_generative_restaurant_model(
    val_data, method='TwoRest_Binomial_ZI_PYP', val_ignore_idk=True, test_ignore_idk=True
):
    if method == 'TwoRest_CRP':
        val_ignore_idk = False
        test_ignore_idk = False
    elif method in IDK_TABLE_GENERATIVE_METHODS:
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
        params['alpha_A'] = round(fit_crp_alpha_mle([s[0] for s in ans_stats], [s[3] for s in ans_stats]), 4)
        params['alpha_U'] = round(fit_crp_alpha_mle([s[0] for s in unans_stats], [s[3] for s in unans_stats]), 4)

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

    # Compute validation scores ONCE and evaluate all candidate thresholds at once via NumPy!
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
        return "DATA_UNCERTAIN", None

    test_ignore_idk = params.get('test_ignore_idk', True)
    N, n_0, m, clusters, is_correct_list, pred_label_list = extract_clusters(
        item, ignore_idk=test_ignore_idk
    )
    if not clusters:
        return "DATA_UNCERTAIN", None

    largest_cluster = max(clusters, key=len)
    best_i = largest_cluster[0]
    return pred_label_list[best_i], is_correct_list[best_i]


# ==========================================
# 4. EVALUATION & METRICS PIPELINE
# ==========================================
def calculate_metrics(res):
    matrix = res["matrix_gt_pred_counts"]
    
    tp_a = matrix["ANSWERABLE"]["Pred_ANSWERABLE_correct"]
    fp_a = matrix["ANSWERABLE"]["Pred_ANSWERABLE_wrong"] + matrix["ABSTAIN"]["Pred_ANSWERABLE"]
    ans_count = res['ans_count']
    
    ans_correct_in_samples = res.get('ans_correct_in_samples', 0)
    ans_sample_recall = (ans_correct_in_samples / ans_count) if ans_count > 0 else 0.0
    ans_accuracy = (tp_a / ans_count) if ans_count > 0 else 0.0
    
    pred_pos_a = tp_a + fp_a
    ans_precision = (tp_a / pred_pos_a) if pred_pos_a > 0 else 0.0
    ans_f1 = 2 * (ans_precision * ans_accuracy) / (ans_precision + ans_accuracy) if (ans_precision + ans_accuracy) > 0 else 0.0

    tp_u = matrix["ABSTAIN"]["Pred_Unanswerable"]
    fp_u = matrix["ANSWERABLE"]["Pred_Unanswerable"]
    unans_count = res['unans_count']
    
    unans_correct_in_samples = res.get('unans_correct_in_samples', 0)
    unans_sample_recall = (unans_correct_in_samples / unans_count) if unans_count > 0 else 0.0
    unans_accuracy = (tp_u / unans_count) if unans_count > 0 else 0.0
    
    pred_pos_u = tp_u + fp_u
    unans_precision = (tp_u / pred_pos_u) if pred_pos_u > 0 else 0.0
    unans_f1 = 2 * (unans_precision * unans_accuracy) / (unans_precision + unans_accuracy) if (unans_precision + unans_accuracy) > 0 else 0.0
    
    total_correct = tp_a + tp_u
    total_count = ans_count + unans_count
    overall_accuracy = (total_correct / total_count) if total_count > 0 else 0.0
    balanced_accuracy = (ans_accuracy + unans_accuracy) / 2.0 
    
    res["metrics"] = {
        "answerable": {
            "precision": round(ans_precision, 2),
            "accuracy": round(ans_accuracy, 2), 
            "ans_correct_in_samples": round(ans_sample_recall, 2),
            "f1": round(ans_f1, 2)
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
    return res


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
        }
    }
    
    iterator = tqdm(data, disable=disable_tqdm)
    for item in iterator:
        should_abstain = item.get("should_abstain")
        gt = "ABSTAIN" if should_abstain is True else "ANSWERABLE"
        
        max_freq, total_clusters, correct_present, correct_freq, is_correct_most_freq = get_item_metadata(
            item, ignore_abstention_tokens=ignore_abstention_tokens
        )
        
        if not ignore_abstention_tokens:
            max_freq_ign, _, _, _, _ = get_item_metadata(item, ignore_abstention_tokens=True)
        else:
            max_freq_ign = max_freq
        
        hist_key = str(min(max_freq_ign, 20))
        
        if gt == "ABSTAIN": 
            res['unans_count'] += 1
            if correct_present: res['unans_correct_in_samples'] += 1
            if hist_key in res["error_analysis"]["unanswerable_max_freq_histogram"]:
                res["error_analysis"]["unanswerable_max_freq_histogram"][hist_key] += 1
        else: 
            res['ans_count'] += 1
            if correct_present: res['ans_correct_in_samples'] += 1
            if hist_key in res["error_analysis"]["answerable_max_freq_histogram"]:
                res["error_analysis"]["answerable_max_freq_histogram"][hist_key] += 1

        if method == 'greedy':
            pred_val = item.get("pred_label", [])
            eff_pred = pred_val[0] if isinstance(pred_val, list) and len(pred_val) > 0 else pred_val
            corr_val = item.get("is_correct", [])
            eff_correct = corr_val[0] if isinstance(corr_val, list) and len(corr_val) > 0 else corr_val
        elif method in ['majority', 'most_freq']:
            eff_pred, eff_correct = get_generalized_sc_consensus(
                item, method=method, threshold=threshold,
                ignore_idk=ignore_abstention_tokens, denom_mode=denom_mode
            )
        elif method == 'adapted_sc':
            eff_pred, eff_correct = get_generalized_sc_consensus(
                item, method='majority', threshold=threshold,
                ignore_idk=True, denom_mode=denom_mode
            )
        elif method in ['crp_k', 'crp_theta']:
            eff_pred, eff_correct = get_generalized_crp_consensus(
                item, method=method, threshold=threshold,
                ignore_idk=ignore_abstention_tokens, denom_mode=denom_mode
            )
        elif method == 'Adaptive_Consistency_beta':
            eff_pred, eff_correct = get_Adaptive_Consistency_beta_consensus(
                item, threshold=threshold,
                ignore_idk=ignore_abstention_tokens, denom_mode=denom_mode
            )
        elif method in GENERATIVE_METHODS:
            eff_pred, eff_correct = get_generative_restaurant_consensus(item, params=threshold)
        elif method == 'Oracle_Intermediate':
            eff_pred, eff_correct = get_oracle_intermediate_consensus(
                item, ignore_idk=ignore_abstention_tokens
            )
        else:
            eff_pred, eff_correct = "DATA_UNCERTAIN", None
            
        is_pred_unanswerable = eff_pred in UNANS_LABELS
        is_pred_answerable = (eff_pred == "ANSWERABLE")

        if gt == "ANSWERABLE":
            if is_pred_answerable:
                if eff_correct: res["matrix_gt_pred_counts"][gt]['Pred_ANSWERABLE_correct'] += 1
                else: res["matrix_gt_pred_counts"][gt]['Pred_ANSWERABLE_wrong'] += 1
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

    return calculate_metrics(res)


# ==========================================
# 5. FAST THRESHOLD & ABLATION GRID OPTIMIZATION
# ==========================================
def extract_target_score(res):
    if TARGET_METRIC == 'overall_accuracy': return res["metrics"]["overall_accuracy"]
    elif TARGET_METRIC == 'balanced_accuracy': return res["metrics"]["balanced_accuracy"]
    elif TARGET_METRIC == 'overall_f1': return res["metrics"]["overall_f1"]
    elif TARGET_METRIC == 'ans_precision': return res["metrics"]["answerable"]["precision"]
    elif TARGET_METRIC == 'ans_accuracy': return res["metrics"]["answerable"]["accuracy"]
    elif TARGET_METRIC == 'ans_f1': return res["metrics"]["answerable"]["f1"]
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

    # Use vectorized tau search for instant evaluation when applicable
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

    # For majority, adapted_sc, and Adaptive_Consistency_beta, cached lookups make this loop < 2ms
    best_threshold = test_thresholds[0]
    best_score = -1.0
    for t in tqdm(test_thresholds, desc = method):
        res = process_run(
            val_data, method=method, threshold=t,
            ignore_abstention_tokens=val_ignore_idk, denom_mode=val_denom
        )
        score = extract_target_score(res)
        if score > best_score:
            best_score = score
            best_threshold = t
            
    return best_threshold


def run_16_combination_grid(val_samples, test_samples, base_exp_name, internal_method, fixed_threshold=None):
    grid_results = {}
    learned_thresholds = {}
    for val_idk, val_den in itertools.product([False, True], ['N', 'm']):
        if fixed_threshold is not None:
            learned_thresholds[(val_idk, val_den)] = fixed_threshold
        else:
            learned_thresholds[(val_idk, val_den)] = learn_best_threshold(
                val_samples, internal_method, val_ignore_idk=val_idk, val_denom=val_den
            )

    for (val_idk, val_den), (test_idk, test_den) in itertools.product(
        itertools.product([False, True], ['N', 'm']),
        itertools.product([False, True], ['N', 'm'])
    ):
        opt_t = learned_thresholds[(val_idk, val_den)]
        v_tag = f"V_{'dropIDK' if val_idk else 'keepIDK'}_{val_den}"
        t_tag = f"T_{'dropIDK' if test_idk else 'keepIDK'}_{test_den}"
        combo_key = f"{base_exp_name}__{v_tag}__{t_tag}"

        test_res = process_run(
            test_samples, method=internal_method, threshold=opt_t,
            ignore_abstention_tokens=test_idk, denom_mode=test_den
        )
        val_res = process_run(
            val_samples, method=internal_method, threshold=opt_t,
            ignore_abstention_tokens=val_idk, denom_mode=val_den
        )
        test_res['val_set'] = val_res
        test_res['learned_threshold'] = round(opt_t, 4) if isinstance(opt_t, float) else opt_t
        test_res['ablation_config'] = {
            'val_ignore_idk': val_idk, 'val_denom': val_den,
            'test_ignore_idk': test_idk, 'test_denom': test_den
        }
        grid_results[combo_key] = test_res

    return grid_results


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
                        if isinstance(data, dict) and 'data' in data: data = data['data']
                        for item in data:
                            q = item.get('question', '').strip()
                            if q: identifiers.add(q)
                except Exception:
                    pass
            elif ext == '.jsonl':
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            q = json.loads(line).get('question', '').strip()
                            if q: identifiers.add(q)
                except Exception:
                    pass
            break 
    return identifiers


def split_output_data(output_data, val_questions, test_questions):
    val_data, test_data = [], []
    for item in output_data:
        q = item.get('question', '').strip()
        if q in val_questions: val_data.append(item)
        elif q in test_questions: test_data.append(item)
    return val_data, test_data


def load_data(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f).get('results', [])
    except Exception:
        return []


def process_combination(model, dataset, aware, val_q, test_q, req_exps):
    model_dir = os.path.join(BASE_DIRECTORY, model)
    greedy_file = os.path.join(model_dir, f"{dataset}{aware}_greedy.json")
    samples_file = os.path.join(model_dir, f"{dataset}{aware}_samples.json")
    
    local_result = {}
    
    if 'Greedy' in req_exps:
        greedy_all_data = load_data(greedy_file)
        if greedy_all_data:
            greedy_val_data, greedy_test_data = split_output_data(greedy_all_data, val_q, test_q)
            if greedy_test_data:
                precompute_dataset_caches(greedy_val_data)
                precompute_dataset_caches(greedy_test_data)
                local_result['Greedy'] = process_run(greedy_test_data, method='greedy')
                local_result['Greedy']['val_set'] = process_run(greedy_val_data, method='greedy')
            
    sampling_reqs = [e for e in req_exps if e != 'Greedy']
    if sampling_reqs:
        samples_all_data = load_data(samples_file)
        if samples_all_data:
            val_samples, test_samples = split_output_data(samples_all_data, val_q, test_q)
            
            if val_samples and test_samples:
                # Pre-compute clusters & sequential beta steps ONCE for val and test splits!
                precompute_dataset_caches(val_samples)
                precompute_dataset_caches(test_samples)

                # 1. Unthresholded Self-Consistency
                if 'self Consistency' in req_exps:
                    for test_idk in [False, True]:
                        tag = "dropIDK" if test_idk else "keepIDK"
                        key = 'self Consistency' if not test_idk else f'self Consistency__T_{tag}'
                        local_result[key] = process_run(
                            test_samples, method='most_freq', ignore_abstention_tokens=test_idk
                        )
                        local_result[key]['val_set'] = process_run(
                            val_samples, method='most_freq', ignore_abstention_tokens=test_idk
                        )

                # 2. Existing Thresholded Baselines + 16-Combination Grid
                method_map = [
                    ('self Consistency + T', 'majority', None),
                    ('Majority', 'majority', 0.5),
                    ('scUn', 'adapted_sc', None),
                    ('CRP_k', 'crp_k', None),
                    ('CRP_theta', 'crp_theta', None),
                    ('Adaptive_Consistency_beta', 'Adaptive_Consistency_beta', None),
                ]
                for exp_name, internal_m, fixed_t in method_map:
                    if exp_name in req_exps:
                        opt_t = fixed_t if fixed_t is not None else learn_best_threshold(
                            val_samples, internal_m, val_ignore_idk=True, val_denom='N'
                        )
                        default_test_idk = (internal_m == 'adapted_sc')
                        local_result[exp_name] = process_run(
                            test_samples, method=internal_m, threshold=opt_t,
                            ignore_abstention_tokens=default_test_idk, denom_mode='N'
                        )
                        local_result[exp_name]['val_set'] = process_run(
                            val_samples, method=internal_m, threshold=opt_t,
                            ignore_abstention_tokens=True, denom_mode='N'
                        )
                        local_result[exp_name]['learned_threshold'] = round(opt_t, 4) if isinstance(opt_t, float) else opt_t

                        if RUN_ALL_16_ABLATION_COMBOS:
                            grid_dict = run_16_combination_grid(
                                val_samples, test_samples, exp_name, internal_m, fixed_threshold=fixed_t
                            )
                            local_result.update(grid_dict)

                # 3. Generative Models WITH Dedicated IDK Table
                for gm in IDK_TABLE_GENERATIVE_METHODS:
                    if gm in req_exps:
                        learned_params = train_generative_restaurant_model(val_samples, method=gm)
                        local_result[gm] = process_run(
                            test_samples, method=gm, threshold=learned_params, ignore_abstention_tokens=True
                        )
                        local_result[gm]['val_set'] = process_run(
                            val_samples, method=gm, threshold=learned_params, ignore_abstention_tokens=True
                        )
                        local_result[gm]['learned_params'] = learned_params

                # 4. PYP WITHOUT IDK Table (Single & Two-Restaurant across Keep vs Drop IDK)
                for no_idk_method in ['SingleRest_PYP_NoIDK', 'TwoRest_PYP_NoIDK']:
                    if no_idk_method in req_exps:
                        for val_idk, test_idk in itertools.product([False, True], [False, True]):
                            v_tag = "V_dropIDK" if val_idk else "V_keepIDK"
                            t_tag = "T_dropIDK" if test_idk else "T_keepIDK"
                            exp_key = f"{no_idk_method}__{v_tag}__{t_tag}"

                            learned_params = train_generative_restaurant_model(
                                val_samples, method=no_idk_method,
                                val_ignore_idk=val_idk, test_ignore_idk=test_idk
                            )
                            local_result[exp_key] = process_run(
                                test_samples, method=no_idk_method, threshold=learned_params,
                                ignore_abstention_tokens=test_idk
                            )
                            val_eval_params = dict(learned_params, test_ignore_idk=val_idk)
                            local_result[exp_key]['val_set'] = process_run(
                                val_samples, method=no_idk_method, threshold=val_eval_params,
                                ignore_abstention_tokens=val_idk
                            )
                            local_result[exp_key]['learned_params'] = learned_params

                # 5. Theoretical Upper Bound Oracle (1 < K < N)
                if 'Oracle_Intermediate' in req_exps:
                    for ignore_idk in [False, True]:
                        tag = "dropIDK" if ignore_idk else "keepIDK"
                        oracle_key = f"Oracle_Intermediate__{tag}"
                        local_result[oracle_key] = process_run(
                            test_samples, method='Oracle_Intermediate', ignore_abstention_tokens=ignore_idk
                        )
                        local_result[oracle_key]['val_set'] = process_run(
                            val_samples, method='Oracle_Intermediate', ignore_abstention_tokens=ignore_idk
                        )
                    
    return model, dataset, aware, local_result


if __name__ == "__main__":
    if os.path.exists(OUTPUT_FILE):
        print(f"Loading existing results from {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'r') as f:
            try:
                master_result = json.load(f)
            except json.JSONDecodeError:
                print("Warning: Output file corrupted. Starting fresh.")
                master_result = {model: {} for model in MODELS}
    else:
        master_result = {model: {} for model in MODELS}

    print(f"Running Models: {MODELS}")
    print(f"Running Experiments: {REQUESTED_EXPERIMENTS}")
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
                master_result[model][dataset][aware] = {}
                for exp in REQUESTED_EXPERIMENTS:
                    tasks.append((model, dataset, aware, val_q, test_q, [exp]))
                # tasks.append((model, dataset, aware, val_q, test_q, REQUESTED_EXPERIMENTS))

    print(f"Dispatching {len(tasks)} tasks to process pool...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_combination, *task): task for task in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Experiments"):
            try:
                mod, ds, aw, res_dict = future.result()
                if res_dict:
                    deep_update(master_result[mod][ds][aw], res_dict)
            except Exception as exc:
                task_info = futures[future]
                print(f"Task generated an exception for {task_info[:3]}: {exc}")
                
    print(f"Writing updated results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(master_result, f, indent=2)
    print("Done!")