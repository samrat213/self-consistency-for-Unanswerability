import os
import json
from tqdm import tqdm
import numpy as np

# import scipy.special
import concurrent.futures


# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIRECTORY = "/home/scai/phd/aiz248314/KBQA/scUn/UA-Bench/"
DATA_DIRECTORY = os.path.join(BASE_DIRECTORY, "data") 
# MODELS = ["result_qwen3", "result_qwen36", "result_qwen38", 'result_gemini36', 'result_gemini38', "result_qwen35", "result_qwen35_base",]
MODELS = ["result_qwen38",]
OUTPUT_FILE = '/home/scai/phd/aiz248314/KBQA/scUn/UA-Bench/results.json'

# TARGET_METRIC = 'overall_accuracy' 
# Change this from 'overall_accuracy' to 'balanced_accuracy'
TARGET_METRIC = 'overall_f1'
datasets = [
    # 'gaia', 
    # 'gsm8k-mip', 
    # 'math-mip', 
    # 'musique_1000', 
    # 'olympiadbench-math', 
    # 'selfaware', 


    # 'ReliableMath', 
    # 'brokenmath', 
    # 'brokenmath_combined', 
    # 'brokenmath_proof_answerable', 
    
    # 'ReliableMath', 
    "brokenmath_type_answer_AU",
    "brokenmath_type_proof_AU"
    # "UA_commonsense",
    # "UA_math"

]

REQUESTED_EXPERIMENTS = [
    'Greedy', 
    'Majority', 'self Consistency', 'self Consistency + T', 
    'scUn',
    'CRP_k','CRP_theta',
    'Adaptive_Consistency_beta'
    ]

awareness = ['_abstaintion', '']
# ==========================================
# --- 1. CORE CONSENSUS & CLUSTERING LOGIC ---
def deep_update(d, u):
    """Recursively updates a nested dictionary without overwriting sibling keys."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def get_mle_theta(N, K):
    if K <= 1: return 0.0
    if K >= N: return float('inf')
    def expected_K(theta): return sum(theta / (theta + i) for i in range(N))
    low, high = 1e-5, 1e4
    for _ in range(50):
        mid = (low + high) / 2
        if expected_K(mid) > K: high = mid
        else: low = mid
    return mid

def get_item_metadata(item):
    """Extracts analytical data about the sample distribution for error analysis."""
    # 1. Type-safe extraction for greedy vs sampling structures
    is_correct_list = item.get("is_correct", [])
    if not isinstance(is_correct_list, list):
        is_correct_list = [is_correct_list]
        
    compare_label = item.get("compare_label", {})
    if not isinstance(compare_label, dict):
        compare_label = {}
        
    pred_answer_list = item.get("pred_answer", [])
    if not isinstance(pred_answer_list, list):
        pred_answer_list = [pred_answer_list]
        
    N = len(is_correct_list)

    if not is_correct_list or N == 0:
        return 0, 0, False, 0, False

    # 2. Map Clusters
    clusters = []
    visited = set()
    for i in range(N):
        if i not in visited:
            matches = compare_label.get(str(i), [])
            cluster_indices = [j for j, val in enumerate(matches) if val == 1]
            
            # Failsafe lexical fallback for greedy runs or malformed matrices
            if not cluster_indices:
                lexical_cluster = []
                if pred_answer_list and i < len(pred_answer_list):
                    ans_i = str(pred_answer_list[i]).strip().lower()
                    for j in range(N):
                        if j < len(pred_answer_list):
                            if ans_i == str(pred_answer_list[j]).strip().lower():
                                lexical_cluster.append(j)
                cluster_indices = lexical_cluster if lexical_cluster else [i]

            clusters.append(cluster_indices)
            visited.update(cluster_indices)

    total_clusters = len(clusters)
    cluster_sizes = [len(c) for c in clusters]
    max_freq = max(cluster_sizes) if cluster_sizes else 1

    # 3. Analyze Correct Answers
    correct_freq = sum(1 for x in is_correct_list if x is True)
    correct_present = correct_freq > 0
    is_correct_most_freq = False

    if correct_present:
        # Find sizes of clusters containing correct answers
        correct_cluster_sizes = [len(c) for c in clusters if any(is_correct_list[idx] for idx in c)]
        if correct_cluster_sizes:
            is_correct_most_freq = max(correct_cluster_sizes) == max_freq

    return max_freq, total_clusters, correct_present, correct_freq, is_correct_most_freq

# --- (Consensus functions remain identical to previous turn) ---
def get_standard_consensus(item, method='majority', threshold=0.5):
    is_correct_list = item.get("is_correct", [])
    pred_label_list = item.get("pred_label", [])
    compare_label = item.get("compare_label", {})
    if not compare_label or not is_correct_list: return "DATA_UNCERTAIN", None
    N = len(is_correct_list)
    votes = {int(i_str): sum(matches) for i_str, matches in compare_label.items()}
    if not votes: return "DATA_UNCERTAIN", None
    best_i = max(votes, key=votes.get)
    max_votes = votes[best_i]
    if method == 'majority':
        if max_votes >= (N * threshold): return pred_label_list[best_i], is_correct_list[best_i]
        else: return "DATA_UNCERTAIN", None 
    elif method == 'most_freq': return pred_label_list[best_i], is_correct_list[best_i]

def get_adapted_sc_consensus(item, threshold=0.5):
    is_correct_list = item.get("is_correct", [])
    pred_label_list = item.get("pred_label", [])
    pred_answer_list = item.get("pred_answer", [])
    compare_label = item.get("compare_label", {})
    if not compare_label or not is_correct_list: return "DATA_UNCERTAIN", None
    N = len(is_correct_list)
    unans_phrases = ["i don't know", "insufficient", "cannot be answered", "not enough information"]
    answerable_indices = []
    for i in range(N):
        lbl = pred_label_list[i] if pred_label_list else ""
        ans = str(pred_answer_list[i]).lower() if pred_answer_list else ""
        is_unans = lbl in ["ABSTAIN", "DATA_UNCERTAIN", "MODEL_UNCERTAIN", "UNANSWERABLE"]
        if any(p in ans for p in unans_phrases): is_unans = True
        if not is_unans: answerable_indices.append(i)
    if not answerable_indices: return "DATA_UNCERTAIN", None
    clusters = []
    visited = set()
    for i in answerable_indices:
        if i not in visited:
            matches = compare_label.get(str(i), [])
            cluster_indices = [j for j, val in enumerate(matches) if val == 1 and j in answerable_indices]
            if not cluster_indices:
                lexical_cluster = []
                if pred_answer_list and i < len(pred_answer_list):
                    ans_i = str(pred_answer_list[i]).strip().lower()
                    for j in answerable_indices:
                        if j < len(pred_answer_list):
                            if ans_i == str(pred_answer_list[j]).strip().lower(): lexical_cluster.append(j)
                cluster_indices = lexical_cluster if lexical_cluster else [i]
            clusters.append(cluster_indices)
            visited.update(cluster_indices)
    if not clusters: return "DATA_UNCERTAIN", None
    largest_cluster = max(clusters, key=len)
    best_i = largest_cluster[0]
    if len(largest_cluster) >= (N * threshold): return pred_label_list[best_i], is_correct_list[best_i]
    else: return "DATA_UNCERTAIN", None

def get_crp_consensus(item, method='crp_k', threshold=None):
    is_correct_list = item.get("is_correct", [])
    pred_label_list = item.get("pred_label", [])
    pred_answer_list = item.get("pred_answer", [])
    compare_label = item.get("compare_label", {})
    if not compare_label or not is_correct_list: return "DATA_UNCERTAIN", None
    N = len(is_correct_list)
    clusters = []
    visited = set()
    for i in range(N):
        if i not in visited:
            matches = compare_label.get(str(i), [])
            cluster_indices = [j for j, val in enumerate(matches) if val == 1]
            if not cluster_indices:
                lexical_cluster = []
                if pred_answer_list and i < len(pred_answer_list):
                    ans_i = str(pred_answer_list[i]).strip().lower()
                    for j in range(N):
                        if j < len(pred_answer_list):
                            if ans_i == str(pred_answer_list[j]).strip().lower(): lexical_cluster.append(j)
                cluster_indices = lexical_cluster if lexical_cluster else [i]
            clusters.append(cluster_indices)
            visited.update(cluster_indices)
    K = len(clusters)
    is_uncertain = False
    if method == 'crp_k' and threshold is not None:
        if K > threshold: is_uncertain = True
    elif method == 'crp_theta' and threshold is not None:
        theta = get_mle_theta(N, K)
        if theta > threshold: is_uncertain = True
    if is_uncertain: return "DATA_UNCERTAIN", None
    largest_cluster = max(clusters, key=len)
    best_i = largest_cluster[0]
    return pred_label_list[best_i], is_correct_list[best_i]

def get_Adaptive_Consistency_beta_consensus(item, threshold=0.95):
    """
    Simulates adaptive consistency sequentially. 
    Stops generating when the Beta probability of the leading cluster exceeds the threshold.
    """
    is_correct_list = item.get("is_correct", [])
    pred_label_list = item.get("pred_label", [])
    pred_answer_list = item.get("pred_answer", [])
    compare_label = item.get("compare_label", {})

    if not compare_label or not is_correct_list:
        return "DATA_UNCERTAIN", None

    N = len(is_correct_list)

    # Simulate sequential generation step-by-step
    for k in range(1, N + 1):
        # 1. Cluster only the first k samples available at this step
        clusters = []
        visited = set()
        for i in range(k):
            if i not in visited:
                # Get matches, but restrict them strictly to the currently generated subset (< k)
                matches = compare_label.get(str(i), [])
                cluster_indices = [j for j, val in enumerate(matches) if val == 1 and j < k]
                
                # Failsafe lexical fallback
                if not cluster_indices:
                    lexical_cluster = []
                    if pred_answer_list and i < len(pred_answer_list):
                        ans_i = str(pred_answer_list[i]).strip().lower()
                        for j in range(k):
                            if j < len(pred_answer_list):
                                if ans_i == str(pred_answer_list[j]).strip().lower():
                                    lexical_cluster.append(j)
                    cluster_indices = lexical_cluster if lexical_cluster else [i]

                clusters.append(cluster_indices)
                visited.update(cluster_indices)

        # 2. Get the sizes of the top 2 clusters (a and b)
        cluster_sizes = sorted([len(c) for c in clusters], reverse=True)
        a = float(cluster_sizes[0])
        b = float(cluster_sizes[1]) if len(cluster_sizes) > 1 else 0.0

        # 3. Calculate Beta Probability
        # P(X > 0.5) for X ~ Beta(a+1, b+1). 
        # betainc(a, b, x) calculates the integral from 0 to x. We want 0.5 to 1, so we subtract from 1.
        try:
            prob = 1.0 - scipy.special.betainc(a + 1.0, b + 1.0, 0.5)
        except Exception:
            prob = 0.0

        # 4. Stopping Condition
        if prob >= threshold:
            largest_cluster = max(clusters, key=len)
            best_i = largest_cluster[0]
            return pred_label_list[best_i], is_correct_list[best_i]

    # If the model exhausts all 20 samples and still cannot reach the confidence threshold,
    # it implies high epistemic uncertainty. We mark it as unanswerable/abstained.
    return "DATA_UNCERTAIN", None

# --- 2. EVALUATION & METRICS PIPELINE ---
def calculate_metrics(res):
    """Calculates Precision, Accuracy, Sample Recall (Coverage), and Balanced Accuracy."""
    matrix = res["matrix_gt_pred_counts"]
    
    # --- ANSWERABLE METRICS ---
    tp_a = matrix["ANSWERABLE"]["Pred_ANSWERABLE_correct"]
    fp_a = matrix["ANSWERABLE"]["Pred_ANSWERABLE_wrong"] + matrix["ABSTAIN"]["Pred_ANSWERABLE"]
    ans_count = res['ans_count']
    
    ans_correct_in_samples = res.get('ans_correct_in_samples', 0)
    ans_sample_recall = (ans_correct_in_samples / ans_count) if ans_count > 0 else 0.0
    ans_accuracy = (tp_a / ans_count) if ans_count > 0 else 0.0
    
    pred_pos_a = tp_a + fp_a
    ans_precision = (tp_a / pred_pos_a) if pred_pos_a > 0 else 0.0
    
    # Safely define F1
    ans_f1 = 2 * (ans_precision * ans_accuracy) / (ans_precision + ans_accuracy) if (ans_precision + ans_accuracy) > 0 else 0.0

    # --- UNANSWERABLE METRICS ---
    tp_u = matrix["ABSTAIN"]["Pred_Unanswerable"]
    fp_u = matrix["ANSWERABLE"]["Pred_Unanswerable"]
    unans_count = res['unans_count']
    
    unans_correct_in_samples = res.get('unans_correct_in_samples', 0)
    unans_sample_recall = (unans_correct_in_samples / unans_count) if unans_count > 0 else 0.0
    unans_accuracy = (tp_u / unans_count) if unans_count > 0 else 0.0
    
    pred_pos_u = tp_u + fp_u
    unans_precision = (tp_u / pred_pos_u) if pred_pos_u > 0 else 0.0
    
    # Safely define F1
    unans_f1 = 2 * (unans_precision * unans_accuracy) / (unans_precision + unans_accuracy) if (unans_precision + unans_accuracy) > 0 else 0.0
    
    # --- OVERALL & BALANCED ---
    total_correct = tp_a + tp_u
    total_count = ans_count + unans_count
    overall_accuracy = (total_correct / total_count) if total_count > 0 else 0.0
    
    # Prevents the "Abstain-All" degenerate strategy during threshold learning
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
        "overall_f1": round((ans_f1+unans_f1)/2, 2),
        "balanced_accuracy": round(balanced_accuracy, 2)
    }
    return res

def process_run(data, method, threshold=None):
    res = {
        'ans_count': 0, 'unans_count': 0,
        'ans_correct_in_samples': 0,
        'unans_correct_in_samples': 0,  # Tracks global presence for unanswerable items
        # "correct_was_most_freq_count": 0,
        "matrix_gt_pred_counts": {
            "ANSWERABLE": {"Pred_ANSWERABLE_correct": 0, "Pred_ANSWERABLE_wrong": 0, "Pred_Unanswerable": 0},
            "ABSTAIN": {"Pred_ANSWERABLE": 0, "Pred_Unanswerable": 0}
        },
        "error_analysis": {
            "FN_missed_answerable": {
                "count": 0, "correct_present_count": 0, "correct_was_most_freq_count": 0, "sum_correct_freq": 0
            },
            "FP_hallucinated_unanswerable": {
                "count": 0, "sum_winning_freq": 0, "sum_total_clusters": 0, 'correct_present_count':0
            },
            "TN_successful_abstention": {
                "count": 0, "sum_total_clusters": 0
            },
            "global_max_freq_histogram": {str(i): 0 for i in range(21)} 
        }
    }
    
    for item in tqdm(data):
        should_abstain = item.get("should_abstain")
        gt = "ABSTAIN" if should_abstain is True else "ANSWERABLE"
        
        # Extract Deep Error Analysis Metadata
        max_freq, total_clusters, correct_present, correct_freq, is_correct_most_freq = get_item_metadata(item)
        
        if gt == "ABSTAIN": 
            res['unans_count'] += 1
            if correct_present:
                res['unans_correct_in_samples'] += 1
        else: 
            res['ans_count'] += 1
            if correct_present:
                res['ans_correct_in_samples'] += 1
        
        # Update Histogram
        hist_key = str(min(max_freq, 20))
        if hist_key in res["error_analysis"]["global_max_freq_histogram"]:
            res["error_analysis"]["global_max_freq_histogram"][hist_key] += 1

        # Determine Consensus
        if method == 'greedy':
            pred_val = item.get("pred_label", [])
            eff_pred = pred_val[0] if isinstance(pred_val, list) and len(pred_val) > 0 else pred_val
            corr_val = item.get("is_correct", [])
            eff_correct = corr_val[0] if isinstance(corr_val, list) and len(corr_val) > 0 else corr_val
        elif method in ['majority', 'most_freq']:
            eff_pred, eff_correct = get_standard_consensus(item, method=method, threshold=threshold)
        elif method == 'adapted_sc':
            eff_pred, eff_correct = get_adapted_sc_consensus(item, threshold=threshold)
        elif method in ['crp_k', 'crp_theta']:
            eff_pred, eff_correct = get_crp_consensus(item, method=method, threshold=threshold)
        elif method == 'Adaptive_Consistency_beta':
            eff_pred, eff_correct = get_Adaptive_Consistency_beta_consensus(item, threshold=threshold)
        is_pred_unanswerable = eff_pred in ["DATA_UNCERTAIN", "MODEL_UNCERTAIN", "ABSTAIN"]
        is_pred_answerable = eff_pred == "ANSWERABLE"

        # Update Matrix & Analytical Stats
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

    # Finalize Averages
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
    
    # Strip zero-value bins from histogram
    # res["error_analysis"]["global_max_freq_histogram"] = {k: v for k, v in analysis["global_max_freq_histogram"].items() if v > 0}

    return calculate_metrics(res)
# --- 3. OPTIMIZATION LOOP ---

def extract_target_score(res):
    """Safely routes the target metric requested by the user at the top of the file."""
    if TARGET_METRIC == 'overall_accuracy': return res["metrics"]["overall_accuracy"]
    elif TARGET_METRIC == 'balanced_accuracy': return res["metrics"]["balanced_accuracy"]
    elif TARGET_METRIC == 'overall_f1': return res["metrics"]["overall_f1"]
    elif TARGET_METRIC == 'ans_precision': return res["metrics"]["answerable"]["precision"]
    elif TARGET_METRIC == 'ans_accuracy': return res["metrics"]["answerable"]["accuracy"]
    elif TARGET_METRIC == 'ans_f1': return res["metrics"]["answerable"]["f1"]
    return res["metrics"]["overall_accuracy"]

# def extract_target_score(res):
#     """Safely routes the target metric requested by the user at the top of the file."""
#     if TARGET_METRIC == 'overall_accuracy': return res["metrics"]["overall_accuracy"]
#     elif TARGET_METRIC == 'ans_precision': return res["metrics"]["answerable"]["precision"]
#     elif TARGET_METRIC == 'ans_accuracy': return res["metrics"]["answerable"]["accuracy"]
#     elif TARGET_METRIC == 'unans_accuracy': return res["metrics"]["unanswerable"]["accuracy"]
#     return res["metrics"]["overall_accuracy"]

def learn_best_threshold(val_data, method):
    best_threshold = None
    best_score = -1.0
    
    if method in ['majority', 'adapted_sc']: test_thresholds = [i / 20.0 for i in range(1, 21)]
    elif method == 'crp_k': test_thresholds = list(range(1, 21))
    elif method == 'crp_theta': test_thresholds =  list(np.arange(0, 20, 0.1))
    elif method == 'Adaptive_Consistency_beta':
        # Test thresholds: 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99
        # test_thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
        test_thresholds =  list(np.arange(0, 1, 0.05))
    else: return None
    
    for t in test_thresholds:
        res = process_run(val_data, method=method, threshold=t)
        score = extract_target_score(res)
        if score > best_score:
            best_score = score
            best_threshold = t
            
    return best_threshold


# --- 4. DATA SPLIT UTILS ---

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
                except Exception as e: pass
            elif ext == '.jsonl':
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            q = json.loads(line).get('question', '').strip()
                            if q: identifiers.add(q)
                except Exception as e: pass
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
        with open(filepath, "r") as f: return json.load(f).get('results', [])
    except Exception: return []

def process_combination(model, dataset, aware, val_q, test_q, req_exps):
    """Worker function to process a single combination conditionally."""
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
                local_result['Greedy']['val_set']  = process_run(greedy_val_data, method='greedy')
            
    # 2. Sampling Experiments
    sampling_reqs = [e for e in req_exps if e != 'greedy']
    if sampling_reqs:
        samples_all_data = load_data(samples_file)
        if samples_all_data:
            val_samples, test_samples = split_output_data(samples_all_data, val_q, test_q)
            
            if val_samples and test_samples:
                if 'self Consistency' in req_exps:
                    local_result['self Consistency'] = process_run(test_samples, method='most_freq')
                    local_result['self Consistency']['val_set'] = process_run(val_samples, method='most_freq')
                
                if 'self Consistency + T' in req_exps:
                    opt_majority = learn_best_threshold(val_samples, 'majority')
                    local_result['self Consistency + T'] = process_run(test_samples, method='majority', threshold=opt_majority)
                    local_result['self Consistency + T']['val_set'] = process_run(val_samples, method='majority', threshold=opt_majority)
                    local_result['self Consistency + T']['learned_threshold'] = opt_majority

                if 'Majority' in req_exps:
                    local_result['Majority'] = process_run(test_samples, method='majority', threshold=0.5)
                    local_result['Majority']['learned_threshold'] = 0.5
                    local_result['Majority']['val_set'] = process_run(val_samples, method='majority', threshold=0.5)
                
                
                if 'scUn' in req_exps:
                    opt_adapted = learn_best_threshold(val_samples, 'adapted_sc')
                    local_result['scUn'] = process_run(test_samples, method='adapted_sc', threshold=opt_adapted)
                    local_result['scUn']['val_set']  = process_run(val_samples, method='adapted_sc', threshold=opt_adapted)
                    local_result['scUn']['learned_threshold'] = opt_adapted
                
                if 'CRP_k' in req_exps:
                    opt_crp_k = learn_best_threshold(val_samples, 'crp_k')
                    local_result['CRP_k'] = process_run(test_samples, method='crp_k', threshold=opt_crp_k)
                    local_result['CRP_k']['val_set'] = process_run(val_samples, method='crp_k', threshold=opt_crp_k)
                    local_result['CRP_k']['learned_threshold'] = opt_crp_k
                
                if 'CRP_theta' in req_exps:
                    opt_crp_theta = learn_best_threshold(val_samples, 'crp_theta')
                    local_result['CRP_theta'] = process_run(test_samples, method='crp_theta', threshold=opt_crp_theta)
                    local_result['CRP_theta']['val_set']  = process_run(val_samples, method='crp_theta', threshold=opt_crp_theta)
                    local_result['CRP_theta']['learned_threshold'] = round(opt_crp_theta, 2)
                    
                if 'Adaptive_Consistency_beta' in req_exps:
                    opt_beta = learn_best_threshold(val_samples, 'Adaptive_Consistency_beta')
                    local_result['Adaptive_Consistency_beta'] = process_run(test_samples, method='Adaptive_Consistency_beta', threshold=opt_beta)
                    local_result['Adaptive_Consistency_beta']['val_set'] = process_run(val_samples, method='Adaptive_Consistency_beta', threshold=opt_beta)
                    local_result['Adaptive_Consistency_beta']['learned_threshold'] = opt_beta
                    
    return model, dataset, aware, local_result
# ==========================================
# MAIN EXECUTION
# ==========================================
# result = {}
# dataset_splits = {ds: {'val': get_split_identifiers('val', ds), 'test': get_split_identifiers('test', ds)} for ds in datasets}

# for model in  tqdm(MODELS, desc="MODELS"):
#     result[model] = {}
#     model_dir = os.path.join(BASE_DIRECTORY, model)
    
#     # --- INDIVIDUAL DATASETS ---
#     for dataset in tqdm(datasets, desc="datasets"):
#         result[model][dataset] = {}
#         val_q, test_q = dataset_splits[dataset]['val'], dataset_splits[dataset]['test']
            
#         for aware in awareness:
#             result[model][dataset][aware] = {}
#             greedy_file = os.path.join(model_dir, f"{dataset}{aware}_greedy.json")
#             samples_file = os.path.join(model_dir, f"{dataset}{aware}_samples.json")
            
#             # Baseline Greedy
#             greedy_all_data = load_data(greedy_file)
#             if greedy_all_data:
#                 _, greedy_test_data = split_output_data(greedy_all_data, val_q, test_q)
#                 if greedy_test_data:
#                     result[model][dataset][aware]['Greedy'] = process_run(greedy_test_data, method='greedy')
                
#             # Sampling Experiments
#             samples_all_data = load_data(samples_file)
#             if samples_all_data:
#                 val_samples, test_samples = split_output_data(samples_all_data, val_q, test_q)
                
#                 if val_samples and test_samples:
#                     opt_majority = learn_best_threshold(val_samples, 'majority')
#                     result[model][dataset][aware]['self Consistency + T'] = process_run(test_samples, method='majority', threshold=opt_majority)
#                     result[model][dataset][aware]['self Consistency + T']['learned_threshold'] = opt_majority
                    
#                     result[model][dataset][aware]['self Consistency'] = process_run(test_samples, method='most_freq')
                    
#                     opt_adapted = learn_best_threshold(val_samples, 'adapted_sc')
#                     result[model][dataset][aware]['scUn'] = process_run(test_samples, method='adapted_sc', threshold=opt_adapted)
#                     result[model][dataset][aware]['scUn']['learned_threshold'] = opt_adapted
                    
#                     opt_crp_k = learn_best_threshold(val_samples, 'crp_k')
#                     result[model][dataset][aware]['CRP_k'] = process_run(test_samples, method='crp_k', threshold=opt_crp_k)
#                     result[model][dataset][aware]['CRP_k']['learned_threshold'] = opt_crp_k
                    
#                     opt_crp_theta = learn_best_threshold(val_samples, 'crp_theta')
#                     result[model][dataset][aware]['CRP_theta'] = process_run(test_samples, method='crp_theta', threshold=opt_crp_theta)
#                     result[model][dataset][aware]['CRP_theta']['learned_threshold'] = round(opt_crp_theta, 2)

    # --- AGGREGATED DATASETS ---
    

# print(f"Successfully processed Models: {list(result.keys())}")


# with open("/home/scai/phd/aiz248314/KBQA/scUn/UA-Bench/results.json","w", encoding="utf-8") as f:
#     json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    result = {model: {ds: {} for ds in datasets} for model in MODELS}
    
    # print("Pre-fetching dataset splits...")
    # dataset_splits = {ds: {'val': get_split_identifiers('val', ds), 'test': get_split_identifiers('test', ds)} for ds in datasets}

    # 1. Build a flat list of all tasks to be executed
    # tasks = []
    # for model in MODELS:
    #     for dataset in datasets:
    #         val_q = dataset_splits[dataset]['val']
    #         test_q = dataset_splits[dataset]['test']
    #         for aware in awareness:
    #             tasks.append((model, dataset, aware, val_q, test_q))
    OUTPUT_FILE = '/home/scai/phd/aiz248314/KBQA/scUn/UA-Bench/results.json'
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
    print(f"Running experiments: {REQUESTED_EXPERIMENTS}")
    print("Pre-fetching dataset splits...")
    dataset_splits = {ds: {'val': get_split_identifiers('val', ds), 'test': get_split_identifiers('test', ds)} for ds in datasets}
    
    tasks = []
    for model in MODELS:
        # Ensure model key exists in case a new model was added to the script
        if model not in master_result: master_result[model] = {}
        for dataset in datasets:
            if dataset not in master_result[model]: master_result[model][dataset] = {}
            val_q, test_q = dataset_splits[dataset]['val'], dataset_splits[dataset]['test']
            for aware in awareness:
                if aware not in master_result[model][dataset]: master_result[model][dataset][aware] = {}
                tasks.append((model, dataset, aware, val_q, test_q, REQUESTED_EXPERIMENTS))

    # 2. Execute tasks concurrently
    # Uses min(32, cpu_count + 4) workers by default. Adjust max_workers if you have high core counts.
    print(f"Dispatching {len(tasks)} tasks to thread pool...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all tasks to the executor
        futures = {executor.submit(process_combination, *task): task for task in tasks}
        
        # Wrap as_completed with tqdm for a single, unified progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Experiments"):
            try:
                mod, ds, aw, res_dict = future.result()
                # Safely update the main dictionary in the main thread
                result[mod][ds][aw] = res_dict
                if res_dict:
                    deep_update(master_result[mod][ds][aw], res_dict)
            except Exception as exc:
                task_info = futures[future]
                print(f"Task generated an exception for {task_info[:3]}: {exc}")
                
    print(f"Writing updated results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(master_result, f, indent=2)

    