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
    "brokenmath_type_proof_AU",
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
    # --- New Two-Restaurant Bayesian Models ---
    'TwoRest_CRP',
    'TwoRest_Reserved_CRP',
    'TwoRest_Hurdle_CRP',
    'TwoRest_ZI_PYP',
    # --- Theoretical Upper Bound Oracle ---
    'Oracle_Intermediate'
]

# Set to True to run all 16 combinations of (val_idk, val_denom, test_idk, test_denom)
RUN_ALL_16_ABLATION_COMBOS = True

awareness = ['_abstaintion', '']
# awareness = ['_abstaintion']

UNANS_PHRASES = ["i don't know", "insufficient", "cannot be answered", "not enough information"]
UNANS_LABELS = {"ABSTAIN", "DATA_UNCERTAIN", "MODEL_UNCERTAIN", "UNANSWERABLE"}


# ==========================================
# 1. CORE CLUSTERING & EXTRACTION HELPERS
# ==========================================
def deep_update(d, u):
    """Recursively updates a nested dictionary without overwriting sibling keys."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def is_sample_abstention(lbl, ans):
    """Checks if a single generated sample is an explicit abstention ('I don't know')."""
    lbl_str = str(lbl).strip() if lbl is not None else ""
    ans_str = str(ans).strip().lower() if ans is not None else ""
    if lbl_str in UNANS_LABELS:
        return True
    if any(p in ans_str for p in UNANS_PHRASES):
        return True
    return False


def extract_clusters(item, ignore_idk=False, up_to_k=None):
    """
    Unified cluster extractor for both full samples and sequential prefixes.
    Returns:
        N_total: Total samples considered (up_to_k or len(is_correct))
        n_0: Number of explicit abstention samples
        m: Number of valid/substantive samples included in clustering
        clusters: List of lists of sample indices (for included samples)
        all_clusters_with_idk: List of lists of sample indices (when IDK is kept)
    """
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


def get_mle_theta(denom_size, K):
    """Computes MLE concentration parameter theta (alpha) given sample count denom_size and K clusters."""
    if denom_size <= 0 or K <= 1:
        return 0.0
    if K >= denom_size:
        return 100.0  # Cap finite upper value so thresholding remains well-behaved
    def expected_K(theta):
        return sum(theta / (theta + i) for i in range(denom_size))
    low, high = 1e-5, 1e4
    for _ in range(50):
        mid = (low + high) / 2.0
        if expected_K(mid) > K:
            high = mid
        else:
            low = mid
    return mid


def get_item_metadata(item, ignore_abstention_tokens=True):
    """Extracts analytical data for error analysis and histograms."""
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
        correct_cluster_sizes = [
            len(c) for c in clusters if any(is_correct_list[idx] for idx in c)
        ]
        if correct_cluster_sizes:
            is_correct_most_freq = (max(correct_cluster_sizes) == max_freq)

    return max_freq, total_clusters, correct_present, correct_freq, is_correct_most_freq


# ==========================================
# 2. UNIFIED CONSENSUS & VARIATION METHODS
# ==========================================
def get_generalized_sc_consensus(
    item, method='majority', threshold=0.5, ignore_idk=False, denom_mode='N'
):
    """
    Handles 'most_freq', 'majority', and 'adapted_sc' under all combinations of:
      - ignore_idk: False (keep IDK in clusters) vs True (filter IDK before clustering)
      - denom_mode: 'N' (fraction out of total N=20) vs 'm' (fraction out of substantive m = N - n_0)
    """
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
    """
    Handles 'crp_k' and 'crp_theta' under all combinations of:
      - ignore_idk: False vs True
      - denom_mode: 'N' (normalize K or solve MLE theta with N) vs 'm' (normalize/solve with m = N - n_0)
    """
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
        # When threshold <= 1.0, treat as normalized cluster ratio K / denom; else raw K scaled by (20 / denom)
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
    """
    Sequential Beta stopping criteria supporting ignore_idk and denom_mode variations.
    - When denom_mode == 'm', the Beta test compares top cluster 'a' against remaining substantive mass 'm_k - a'
      (or second cluster 'b'), requiring at least 1 substantive cluster.
    """
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
            # Compare leading cluster against runner-up within active pool
            b = float(cluster_sizes[1]) if len(cluster_sizes) > 1 else 0.0
        else:
            # If normalizing against N even when ignore_idk=True, count dropped IDK tokens as competing mass
            b_second = float(cluster_sizes[1]) if len(cluster_sizes) > 1 else 0.0
            b = max(b_second, float(n_0_k)) if ignore_idk else b_second

        try:
            prob = 1.0 - scipy.special.betainc(a + 1.0, b + 1.0, 0.5)
        except Exception:
            prob = 0.0

        if prob >= threshold:
            largest_cluster = max(clusters, key=len)
            best_i = largest_cluster[0]
            return pred_lbl[best_i], is_corr[best_i]

    return "DATA_UNCERTAIN", None


# ==========================================
# 3.ORACLE UPPER BOUND (1 < K < N)
# ==========================================
def get_oracle_intermediate_consensus(item, ignore_idk=False):
    """
    Computes the best achievable performance by ANY cluster-distribution algorithm:
    - If K == 1 (all 20 samples are identical): No distribution-only detector can reject a 20/20 consensus,
      so it MUST accept the majority cluster.
    - If K == N (all 20 samples are distinct singletons): No distribution-only detector can accept 20 completely
      disjoint answers, so it MUST abstain ("DATA_UNCERTAIN").
    - If 1 < K < N (intermediate cluster count): The oracle uses ground-truth 'should_abstain' to make the
      optimal accept/abstain decision on the dominant cluster.
    """
    N, n_0, m, clusters, is_correct_list, pred_label_list = extract_clusters(
        item, ignore_idk=ignore_idk
    )
    if N == 0:
        return "DATA_UNCERTAIN", None

    # If all tokens were filtered out as IDK, or 0 clusters exist -> Abstain
    if not clusters:
        return "DATA_UNCERTAIN", None

    K = len(clusters)
    denom = m if ignore_idk else N

    largest_cluster = max(clusters, key=len)
    best_i = largest_cluster[0]

    # Boundary Case 1: Unanimous agreement (K == 1 and fills the pool) -> Must Accept
    if K == 1 and denom == N:
        return pred_label_list[best_i], is_correct_list[best_i]

    # Boundary Case 2: Total dispersion (K == N distinct singletons) -> Must Reject/Abstain
    if K >= N:
        return "DATA_UNCERTAIN", None

    # Intermediate Regime (1 < K < N, or partial IDK pool): Oracle makes the optimal decision
    should_abstain = item.get("should_abstain") is True
    if should_abstain:
        return "DATA_UNCERTAIN", None
    else:
        return pred_label_list[best_i], is_correct_list[best_i]


# ==========================================
# 4. TWO-RESTAURANT BAYESIAN MIXTURE MODELS
# ==========================================
def extract_seating_stats(data, ignore_idk_in_substantive=True):
    """Extracts (N, n_0, m, K, cluster_sizes) for every item in a split."""
    stats = []
    for item in data:
        N, n_0, m, clusters, _, _ = extract_clusters(item, ignore_idk=ignore_idk_in_substantive)
        sizes = [len(c) for c in clusters]
        K = len(sizes)
        gt_unans = (item.get("should_abstain") is True)
        stats.append((N, n_0, m, K, sizes, gt_unans))
    return stats


def fit_beta_binomial_mle(n0_list, N_list):
    """Fits Beta-Binomial (a, b) parameters via MLE for the IDK table count n_0 out of N."""
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
    """Fits CRP concentration parameter alpha on (m, K) pairs where m >= 1."""
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
    """
    Fits (alpha, beta) for the Reserved-Atom Dirichlet Process CRP:
    P(pi | alpha, beta) = [Gamma(alpha+beta)/Gamma(alpha+beta+N)] * [Gamma(beta+n0)/Gamma(beta)] * alpha^K * prod(n_k - 1)!
    """
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
    Fits Two-Parameter Pitman-Yor Process (alpha, d) via exact rising-factorial log-likelihood
    so cluster sizes n_k do not cancel out.
    """
    valid = [
        (m, K, sizes)
        for m, K, sizes in zip(m_list, K_list, sizes_list)
        if m >= 2 and K >= 1
    ]
    if not valid:
        return 1.0, 0.1

    def neg_ll(params):
        alpha, d = params
        if d < 1e-5 or d >= 1.0 - 1e-5 or alpha <= -d + 1e-5:
            return 1e12
        total_ll = 0.0
        for m, K, sizes in valid:
            # 1. Table creation weight: sum_{j=1}^{K-1} log(alpha + j*d)
            if K > 1:
                j_vals = np.arange(1, K, dtype=float)
                total_ll += np.sum(np.log(alpha + j_vals * d))
            # 2. Denominator rising factorial: - (gammaln(alpha + m) - gammaln(alpha + 1))
            total_ll -= (scipy.special.gammaln(alpha + m) - scipy.special.gammaln(alpha + 1.0))
            # 3. Cluster size weight: sum_{k=1}^K [gammaln(n_k - d) - gammaln(1 - d)]
            sizes_arr = np.array(sizes, dtype=float)
            gt1 = sizes_arr[sizes_arr > 1]
            if len(gt1) > 0:
                total_ll += np.sum(
                    scipy.special.gammaln(gt1 - d) - scipy.special.gammaln(1.0 - d)
                )
        return -total_ll

    res = scipy.optimize.minimize(
        neg_ll,
        x0=[1.0, 0.15],
        bounds=[(-0.999, 500.0), (1e-4, 0.999)],  # alpha > -d is enforced inside neg_ll
        method='L-BFGS-B'
    )
    return float(res.x[0]), float(res.x[1])


def train_two_restaurant_model(val_data, method='TwoRest_ZI_PYP'):
    """
    Fits Restaurant A (Answerable) and Restaurant U (Unanswerable) parameters on val_data,
    then optimizes the Log-Likelihood Ratio (LLR) decision threshold tau on TARGET_METRIC.
    """
    # Plain TwoRest_CRP treats IDK as regular clusters; Reserved/Hurdle/ZI-PYP route IDK to Table 0
    sep_idk = (method != 'TwoRest_CRP')
    stats = extract_seating_stats(val_data, ignore_idk_in_substantive=sep_idk)

    ans_stats = [s for s in stats if not s[5]]
    unans_stats = [s for s in stats if s[5]]

    # Fallback if validation split only has 1 class
    if not ans_stats:
        ans_stats = stats
    if not unans_stats:
        unans_stats = stats

    params = {'method': method}

    if method == 'TwoRest_CRP':
        params['alpha_A'] = fit_crp_alpha_mle([s[0] for s in ans_stats], [s[3] for s in ans_stats])
        params['alpha_U'] = fit_crp_alpha_mle([s[0] for s in unans_stats], [s[3] for s in unans_stats])
        if abs(params['alpha_U'] - params['alpha_A']) < 1e-4:
            params['alpha_U'] = max(params['alpha_A'] * 2.0, 1.5)

    elif method == 'TwoRest_Reserved_CRP':
        params['alpha_A'], params['beta_A'] = fit_reserved_crp_mle(
            [s[0] for s in ans_stats], [s[1] for s in ans_stats], [s[3] for s in ans_stats]
        )
        params['alpha_U'], params['beta_U'] = fit_reserved_crp_mle(
            [s[0] for s in unans_stats], [s[1] for s in unans_stats], [s[3] for s in unans_stats]
        )

    elif method == 'TwoRest_Hurdle_CRP':
        params['a_A'], params['b_A'] = fit_beta_binomial_mle([s[1] for s in ans_stats], [s[0] for s in ans_stats])
        params['a_U'], params['b_U'] = fit_beta_binomial_mle([s[1] for s in unans_stats], [s[0] for s in unans_stats])
        params['alpha_A'] = fit_crp_alpha_mle([s[2] for s in ans_stats], [s[3] for s in ans_stats])
        params['alpha_U'] = fit_crp_alpha_mle([s[2] for s in unans_stats], [s[3] for s in unans_stats])

    elif method == 'TwoRest_ZI_PYP':
        params['a_A'], params['b_A'] = fit_beta_binomial_mle([s[1] for s in ans_stats], [s[0] for s in ans_stats])
        params['a_U'], params['b_U'] = fit_beta_binomial_mle([s[1] for s in unans_stats], [s[0] for s in unans_stats])
        params['alpha_A'], params['d_A'] = fit_pyp_mle(
            [s[2] for s in ans_stats], [s[3] for s in ans_stats], [s[4] for s in ans_stats]
        )
        params['alpha_U'], params['d_U'] = fit_pyp_mle(
            [s[2] for s in unans_stats], [s[3] for s in unans_stats], [s[4] for s in unans_stats]
        )

    # Optimize LLR decision threshold tau on val_data
    llr_scores = [compute_two_restaurant_llr(item, params) for item in val_data]
    candidate_taus = sorted(list(set(np.percentile(llr_scores, np.linspace(0, 100, 41)))))
    if 0.0 not in candidate_taus:
        candidate_taus.append(0.0)

    best_tau = 0.0
    best_score = -1.0
    for tau in candidate_taus:
        trial_params = dict(params, tau=float(tau))
        res = process_run(val_data, method=method, threshold=trial_params, ignore_abstention_tokens=True)
        score = extract_target_score(res)
        if score > best_score:
            best_score = score
            best_tau = float(tau)

    params['tau'] = round(best_tau, 4)
    return params


def compute_two_restaurant_llr(item, params):
    """Computes Log-Likelihood Ratio log P(pattern | R_U) - log P(pattern | R_A)."""
    method = params['method']
    sep_idk = (method != 'TwoRest_CRP')
    N, n_0, m, clusters, _, _ = extract_clusters(item, ignore_idk=sep_idk)
    if N == 0:
        return 1e6

    sizes = [len(c) for c in clusters]
    K = len(sizes)

    if method == 'TwoRest_CRP':
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
        # 1. Beta-Binomial IDK log-likelihood
        ll_idk_U = scipy.special.betaln(n_0 + aU, m + bU) - scipy.special.betaln(aU, bU)
        ll_idk_A = scipy.special.betaln(n_0 + aA, m + bA) - scipy.special.betaln(aA, bA)
        # 2. Substantive CRP log-likelihood
        ll_crp_U = (K * np.log(alU) + scipy.special.gammaln(alU) - scipy.special.gammaln(m + alU)) if m >= 1 else 0.0
        ll_crp_A = (K * np.log(alA) + scipy.special.gammaln(alA) - scipy.special.gammaln(m + alA)) if m >= 1 else 0.0
        return float((ll_idk_U + ll_crp_U) - (ll_idk_A + ll_crp_A))

    elif method == 'TwoRest_ZI_PYP':
        aU, bU, alU, dU = params['a_U'], params['b_U'], params['alpha_U'], params['d_U']
        aA, bA, alA, dA = params['a_A'], params['b_A'], params['alpha_A'], params['d_A']
        # Term 1: Explicit Abstention (IDK) Signal
        llr = (
            (scipy.special.betaln(n_0 + aU, m + bU) - scipy.special.betaln(aU, bU))
            - (scipy.special.betaln(n_0 + aA, m + bA) - scipy.special.betaln(aA, bA))
        )
        if m >= 1 and K >= 1:
            # Term 2: Semantic Dispersion (Table Creation & Denominator)
            if K > 1:
                j_vals = np.arange(1, K, dtype=float)
                llr += np.sum(np.log((alU + j_vals * dU) / (alA + j_vals * dA)))
            llr -= (
                (scipy.special.gammaln(alU + m) - scipy.special.gammaln(alU + 1.0))
                - (scipy.special.gammaln(alA + m) - scipy.special.gammaln(alA + 1.0))
            )
            # Term 3: Non-Canceling Cluster Concentration (Table Sizes n_k)
            sizes_arr = np.array(sizes, dtype=float)
            gt1 = sizes_arr[sizes_arr > 1]
            if len(gt1) > 0:
                llr += np.sum(
                    (scipy.special.gammaln(gt1 - dU) - scipy.special.gammaln(1.0 - dU))
                    - (scipy.special.gammaln(gt1 - dA) - scipy.special.gammaln(1.0 - dA))
                )
        return float(llr)

    return 0.0


def get_two_restaurant_consensus(item, params):
    """Predicts answerable vs. unanswerable using the learned Two-Restaurant LLR threshold."""
    llr = compute_two_restaurant_llr(item, params)
    tau = params.get('tau', 0.0)
    if llr > tau:
        return "DATA_UNCERTAIN", None

    sep_idk = (params['method'] != 'TwoRest_CRP')
    N, n_0, m, clusters, is_correct_list, pred_label_list = extract_clusters(item, ignore_idk=sep_idk)
    if not clusters:
        return "DATA_UNCERTAIN", None

    largest_cluster = max(clusters, key=len)
    best_i = largest_cluster[0]
    return pred_label_list[best_i], is_correct_list[best_i]


# ==========================================
# 5. EVALUATION & METRICS PIPELINE
# ==========================================
def calculate_metrics(res):
    """Calculates Precision, Accuracy, Sample Recall (Coverage), F1, and Balanced Accuracy."""
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

        # Route Prediction Method
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
        elif method in ['TwoRest_CRP', 'TwoRest_Reserved_CRP', 'TwoRest_Hurdle_CRP', 'TwoRest_ZI_PYP']:
            eff_pred, eff_correct = get_two_restaurant_consensus(item, params=threshold)
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
                if correct_present:
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
# 6. THRESHOLD & ABLATION GRID OPTIMIZATION
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
    best_threshold = None
    best_score = -1.0
    
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


def run_16_combination_grid(val_samples, test_samples, base_exp_name, internal_method, fixed_threshold=None):
    """
    Runs all combinations of:
      1. val_ignore_idk in [False, True]   (keep vs. drop 'I don't know' during threshold optimization)
      2. val_denom in ['N', 'm']           (out of 20 vs. out of substantive m = N - n_0 during val)
      3. test_ignore_idk in [False, True]  (keep vs. drop 'I don't know' during test evaluation)
      4. test_denom in ['N', 'm']          (out of 20 vs. out of substantive m = N - n_0 during test)
    """
    grid_results = {}

    # Pre-learn thresholds for the 4 validation settings (or use fixed_threshold if provided, e.g., Majority=0.5)
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
# 7. DATA SPLIT & EXECUTION UTILS
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
    """Worker function to process a single (model, dataset, awareness) combination."""
    req_exps = req_exps if isinstance(req_exps, list) else [req_exps]

    model_dir = os.path.join(BASE_DIRECTORY, model)
    greedy_file = os.path.join(model_dir, f"{dataset}{aware}_greedy.json")
    samples_file = os.path.join(model_dir, f"{dataset}{aware}_samples.json")
    
    local_result = {}
    
    # 1. Baseline Greedy
    if 'Greedy' in req_exps:
        greedy_all_data = load_data(greedy_file)
        if greedy_all_data:
            greedy_val_data, greedy_test_data = split_output_data(greedy_all_data, val_q, test_q)
            if greedy_test_data:
                local_result['Greedy'] = process_run(greedy_test_data, method='greedy')
                local_result['Greedy']['val_set'] = process_run(greedy_val_data, method='greedy')
            
    # 2. Sampling Experiments
    sampling_reqs = [e for e in req_exps if e != 'Greedy']
    if sampling_reqs:
        samples_all_data = load_data(samples_file)
        if samples_all_data:
            val_samples, test_samples = split_output_data(samples_all_data, val_q, test_q)
            
            if val_samples and test_samples:
                # --- Unthresholded Self-Consistency (Test with and without IDK) ---
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

                # --- Thresholded Baselines + Full 16-Combination Ablation Grid ---
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
                        # Standard default run (preserves backward-compatible key)
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

                        # Full 16-way Combinatorial Grid (Val IDK x Val Denom x Test IDK x Test Denom)
                        if RUN_ALL_16_ABLATION_COMBOS:
                            grid_dict = run_16_combination_grid(
                                val_samples, test_samples, exp_name, internal_m, fixed_threshold=fixed_t
                            )
                            local_result[exp_name]['ablation_grid'] = grid_dict

                # --- New Two-Restaurant Bayesian Mixture Models ---
                for tr_method in ['TwoRest_CRP', 'TwoRest_Reserved_CRP', 'TwoRest_Hurdle_CRP', 'TwoRest_ZI_PYP']:
                    if tr_method in req_exps:
                        learned_params = train_two_restaurant_model(val_samples, method=tr_method)
                        local_result[tr_method] = process_run(
                            test_samples, method=tr_method, threshold=learned_params, ignore_abstention_tokens=True
                        )
                        local_result[tr_method]['val_set'] = process_run(
                            val_samples, method=tr_method, threshold=learned_params, ignore_abstention_tokens=True
                        )
                        local_result[tr_method]['learned_params'] = learned_params

                # --- Theoretical Upper Bound Oracle (1 < K < N) ---
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
                    
    return model, dataset, aware, req_exps[0], local_result


# ==========================================
# MAIN EXECUTION
# ==========================================
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
    print("Pre-fetching dataset splits...")
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
                for exp in REQUESTED_EXPERIMENTS:
                    tasks.append((model, dataset, aware, val_q, test_q, exp))

    print(f"Dispatching {len(tasks)} tasks to process pool...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_combination, *task): task for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Experiments"):
            try:
                mod, ds, aw, exp, res_dict = future.result()
                if res_dict:
                    deep_update(master_result[mod][ds][aw], res_dict)
            except Exception as exc:
                task_info = futures[future]
                print(f"Task generated an exception for {task_info[:4]}: {exc}")
                
    print(f"Writing updated results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(master_result, f, indent=2)
    print("Done!")