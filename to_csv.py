import json
import pandas as pd

# Load the JSON data
with open("results.json", "r") as f:
    data = json.load(f)

# 1. Flatten the data into a list of records
models_map = {'result_qwen38':'Qwen3.8','result_qwen36':'Qwen3.6', 'result_qwen3':'Qwen3', 'result_gemini36':'Gemini3.6'}
aware_map = {'_abstaintion':"Aware", '':"Unaware"}
exp_map = {'_greedy_test':'_greedy_test','_samples_learned_majority':'_samples_learned_majority', '_samples_most_freq':'_samples_most_freq', 
            '_samples_adapted_sc':'_samples_adapted_sc','_samples_crp_k':'_samples_crp_k', '_samples_crp_theta':'_samples_crp_theta'}

datasets = ['ReliableMath', 'brokenmath', 'brokenmath_proof_answerable', 'UA_commonsense', 'UA_math']
rows = []


for model, result in data.items():
    for dataset, aware_res  in result.items():
        for mode, methods in aware_res.items():
            mode_str = "Aware" if mode == "_abstaintion" else "Unaware"
            if not methods:  # Skip empty dictionaries
                continue
            for method,details in methods.items():
                if not details:  # Skip empty dictionaries
                    continue
                
                metrics = details.get("metrics", {})
                ans = metrics.get("answerable", {})
                unans = metrics.get("unanswerable", {})
                # print(details)
                row = {
                    "Model": models_map[model],
                    "Dataset": dataset,
                    "n_ans": details.get("ans_count", 0),
                    "n_unans": details.get("unans_count", 0),
                    "Awareness": aware_map[mode],
                    "Method": method,
                    "Overall_Acc": metrics.get("overall_accuracy"),
                    "Ans_accuracy": ans.get("accuracy"),
                    "Unans_accuracy": unans.get("accuracy"),
                    "ans_f1": ans.get("f1"),
                    "unans_f1": unans.get("f1"),
                    "ans_correct_in_samples":ans.get("ans_correct_in_samples"),
                    "unans_correct_in_samples":unans.get("unans_correct_in_samples"),
                    "FN_correct_present": details["error_analysis"]["FN_missed_answerable"]["correct_present_count"],
                    "FP_correct_present": details["error_analysis"]["FP_hallucinated_unanswerable"]["correct_present_count"]
                    # "Ans_Precision": ans.get("precision"),
                    # "Unans_Precision": unans.get("precision")
                }
                if 'learned_threshold' in details.keys():
                    row['Threshold'] = details['learned_threshold']
                rows.append(row)

# Convert to a flat DataFrame
df = pd.DataFrame(rows)
awareness_order = ["Aware", "Unaware"]

method_order = [
    "Greedy",
    "Majority",
    "self Consistency",
    "self Consistency + T",
    "scUn",
    "CRP_k",
    "CRP_theta",
    "Adaptive_Consistency_beta"
]

# 2. Convert the columns to Categorical types with the ordered=True flag
df['Awareness'] = pd.Categorical(df['Awareness'], categories=awareness_order, ordered=True)
df['Method'] = pd.Categorical(df['Method'], categories=method_order, ordered=True)

# 3. Sort the DataFrame using these newly established rules
df = df.sort_values(by=['Dataset', 'Awareness', 'Method'])



# Optional: Reset the index so it looks clean from 0 to N
df = df.reset_index(drop=True)

df.to_csv("results.csv", index=False)
# 2. Pivot the DataFrame so models become distinct columns
df_pivot = df.pivot_table(
    index=["Dataset", "Awareness", "Method"],
    columns="Model",
    values=[
        "Overall_Acc", "Ans_accuracy", "Unans_accuracy",
        "ans_f1","unans_f1",
        "ans_correct_in_samples","unans_correct_in_samples",
        "FN_correct_present","FP_correct_present" 
        ]
)

# 3. Flatten the MultiIndex columns to single strings (e.g., 'qwen38_Overall_Acc')
df_pivot.columns = [f"{model}_{metric}" for metric, model in df_pivot.columns]
df_pivot = df_pivot.reset_index()


base_cols = ["Dataset", "Awareness", "Method"]

models = [models_map[model] for model in  data.keys()]
metric_order = [
    "Overall_Acc",
    "Ans_accuracy",
    "Unans_accuracy",
    "ans_f1",
    "unans_f1",
    "ans_correct_in_samples",
    "unans_correct_in_samples",
    "FN_correct_present",
    "FP_correct_present",
    'Threshold'
]

new_col_order = [col for col in base_cols if col in df_pivot.columns]

for model in models:
    for metric in metric_order:
        # Construct the expected column name
        expected_col = f"{model}_{metric}"
        
        # If it matches exactly, add it
        if expected_col in df_pivot.columns:
          new_col_order.append(expected_col)
        else:
            # Fallback: Find any column that contains BOTH the model and the metric
            # This handles unexpected spaces like "Gemini 3.6_ Ans_accuracy"
            matching_cols = [c for c in df_pivot.columns if model in c and metric in c]
            if matching_cols:
                new_col_order.append(matching_cols[0])

# 3. Add any leftover columns to the end (so we don't accidentally drop data)
leftover_cols = [c for c in df_pivot.columns if c not in new_col_order]
final_order = new_col_order + leftover_cols
print(df_pivot.columns)
# 4. Apply the new order
df_pivot = df_pivot[final_order]

# 4. Save to CSV
df_pivot.to_csv("model_comparison_results.csv", index=False)