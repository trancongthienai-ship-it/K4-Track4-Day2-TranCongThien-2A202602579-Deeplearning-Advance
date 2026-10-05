import os
import glob
import json
import subprocess
import pandas as pd
import numpy as np

predictions_dir = "submissions/2A202602579_trancongthien/predictions"
eval_out_dir = "temp_eval"
os.makedirs(eval_out_dir, exist_ok=True)

metrics = {}

# Run eval.py for all CSVs
csv_files = glob.glob(f"{predictions_dir}/*.csv")
for f in csv_files:
    basename = os.path.basename(f)
    # e.g., B01_seed0_val.csv or F01_seed0_test.csv
    parts = basename.replace(".csv", "").split("_")
    exp_id = parts[0]
    seed = parts[1]
    split = parts[2]
    
    subset_csv = "data/labels/val_subset0.csv" if split == "val" else "data/labels/test_subset0.csv"
    tag = f"{exp_id}_{seed}_{split}"
    
    cmd = [
        "python3", "eval.py", "score",
        "--pred", f,
        "--test-csv", subset_csv,
        "--labels", "data/labels/labels.csv",
        "--tag", tag,
        "--out", eval_out_dir
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open(f"{eval_out_dir}/{tag}_metrics.json", "r") as jf:
            m = json.load(jf)
            metrics[tag] = m
    except Exception as e:
        print(f"Failed on {f}: {e}")

# Parse into DataFrames
backbones_data = []
training_data = []
final_data = []

for tag, m in metrics.items():
    parts = tag.split("_")
    exp_id = parts[0]
    seed = parts[1]
    split = parts[2]
    
    row = {
        "exp_id": exp_id,
        "seed": seed,
        "split": split,
        "macro-F1": m.get("macro_f1", 0),
        "top-1": m.get("top1_acc", 0),
        "ece": m.get("ece", 0)
    }
    
    if exp_id.startswith("B"):
        backbones_data.append(row)
    elif exp_id.startswith("T"):
        training_data.append(row)
    elif exp_id.startswith("F"):
        final_data.append(row)

# Create Excel writer
excel_path = "submissions/2A202602579_trancongthien/results.xlsx"
with pd.ExcelWriter(excel_path) as writer:
    if backbones_data:
        pd.DataFrame(backbones_data).to_excel(writer, sheet_name="Backbones", index=False)
    else:
        pd.DataFrame(columns=["exp_id", "backbone", "macro-F1", "top-1"]).to_excel(writer, sheet_name="Backbones", index=False)
        
    if training_data:
        pd.DataFrame(training_data).to_excel(writer, sheet_name="Training", index=False)
    else:
        pd.DataFrame(columns=["exp_id", "trục", "macro-F1", "top-1"]).to_excel(writer, sheet_name="Training", index=False)
        
    pd.DataFrame(columns=["exp_id", "phương pháp", "macro-F1", "top-1"]).to_excel(writer, sheet_name="Inference", index=False)
    
    if final_data:
        pd.DataFrame(final_data).to_excel(writer, sheet_name="Final", index=False)
    else:
        pd.DataFrame(columns=["exp_id", "macro-F1 test", "top-1 test"]).to_excel(writer, sheet_name="Final", index=False)
        
    pd.DataFrame(columns=["lớp", "precision", "recall", "F1"]).to_excel(writer, sheet_name="PerClass", index=False)
    pd.DataFrame(columns=["cấu hình", "p50", "p95"]).to_excel(writer, sheet_name="Latency", index=False)
    pd.DataFrame(columns=["exp_id", "macro-F1", "độ trễ"]).to_excel(writer, sheet_name="Summary", index=False)

print("results.xlsx created successfully!")
