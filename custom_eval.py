import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

predictions_dir = "submissions/2A202602579_trancongthien/predictions"

backbones_data = []
training_data = []
final_data = []

csv_files = glob.glob(f"{predictions_dir}/*.csv")
for f in csv_files:
    basename = os.path.basename(f)
    parts = basename.replace(".csv", "").split("_")
    exp_id = parts[0]
    seed = parts[1]
    split = parts[2]
    
    try:
        df = pd.read_csv(f)
        y_true = df["y_true"]
        y_pred = df["y_pred"]
        
        # Calculate top-1 and macro-F1
        top1 = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        
        row = {
            "exp_id": exp_id,
            "seed": seed,
            "split": split,
            "macro-F1": round(macro_f1, 4),
            "top-1": round(top1, 4)
        }
        
        if exp_id.startswith("B"):
            backbones_data.append(row)
        elif exp_id.startswith("T"):
            training_data.append(row)
        elif exp_id.startswith("F"):
            final_data.append(row)
    except Exception as e:
        print(f"Failed {f}: {e}")

excel_path = "submissions/2A202602579_trancongthien/results.xlsx"

# Build final output logic
def make_df(data, cols):
    if not data:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(data).sort_values(by=["exp_id", "seed"])

with pd.ExcelWriter(excel_path) as writer:
    make_df(backbones_data, ["exp_id", "backbone", "macro-F1", "top-1"]).to_excel(writer, sheet_name="Backbones", index=False)
    make_df(training_data, ["exp_id", "trục", "macro-F1", "top-1"]).to_excel(writer, sheet_name="Training", index=False)
    pd.DataFrame(columns=["exp_id", "phương pháp", "macro-F1", "top-1"]).to_excel(writer, sheet_name="Inference", index=False)
    make_df(final_data, ["exp_id", "macro-F1 test", "top-1 test"]).to_excel(writer, sheet_name="Final", index=False)
    pd.DataFrame(columns=["lớp", "precision", "recall", "F1"]).to_excel(writer, sheet_name="PerClass", index=False)
    pd.DataFrame(columns=["cấu hình", "p50", "p95"]).to_excel(writer, sheet_name="Latency", index=False)
    pd.DataFrame(columns=["exp_id", "macro-F1", "độ trễ"]).to_excel(writer, sheet_name="Summary", index=False)

print("results.xlsx created successfully with sklearn metrics!")
