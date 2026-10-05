import os
import glob
import json
import pandas as pd

runs_dir = "code/runs"

backbones_data = []
training_data = []
final_data = []

for exp_dir in glob.glob(f"{runs_dir}/*"):
    if not os.path.isdir(exp_dir): continue
    exp_id = os.path.basename(exp_dir)
    
    for seed_dir in glob.glob(f"{exp_dir}/*"):
        if not os.path.isdir(seed_dir): continue
        seed = os.path.basename(seed_dir)
        
        history_path = os.path.join(seed_dir, "history.csv")
        config_path = os.path.join(seed_dir, "config.json")
        
        if not os.path.exists(history_path) or not os.path.exists(config_path):
            continue
            
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
                
            df = pd.read_csv(history_path)
            if len(df) == 0: continue
            
            best_idx = df["val_f1"].idxmax()
            best_f1 = df.loc[best_idx, "val_f1"]
            avg_time = df["time"].mean()
            
            row = {
                "exp_id": exp_id,
                "seed": seed,
                "backbone": cfg.get("backbone", ""),
                "epoch": len(df),
                "macro-F1": round(best_f1, 4),
                "top-1": "",
                "time/epoch (s)": round(avg_time, 2),
                "config_details": f"loss:{cfg.get('loss')}, aug:{cfg.get('aug')}"
            }
            
            if exp_id.startswith("B"):
                backbones_data.append(row)
            elif exp_id.startswith("T"):
                training_data.append(row)
            elif exp_id.startswith("F"):
                final_row = {
                    "exp_id": exp_id,
                    "seed": seed,
                    "macro-F1 val": round(best_f1, 4),
                    "macro-F1 test": "Chưa có data (file predictions lỗi)",
                    "top-1 test": "Chưa có data"
                }
                final_data.append(final_row)
                
        except Exception as e:
            print(f"Error reading {history_path}: {e}")

excel_path = "submissions/2A202602579_trancongthien/results.xlsx"

def make_df(data, cols):
    if not data:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(data).sort_values(by=["exp_id", "seed"])
    for c in cols:
        if c not in df.columns: df[c] = ""
    return df[cols]

with pd.ExcelWriter(excel_path) as writer:
    make_df(backbones_data, ["exp_id", "backbone", "macro-F1", "top-1", "time/epoch (s)", "epoch"]).to_excel(writer, sheet_name="Backbones", index=False)
    make_df(training_data, ["exp_id", "backbone", "config_details", "macro-F1", "top-1", "time/epoch (s)"]).to_excel(writer, sheet_name="Training", index=False)
    
    # Inference is explicitly marked as not run
    pd.DataFrame([{"exp_id": "I01", "phương pháp": "Chưa chạy thí nghiệm Inference", "macro-F1": "", "top-1": ""}]).to_excel(writer, sheet_name="Inference", index=False)
    
    make_df(final_data, ["exp_id", "seed", "macro-F1 val", "macro-F1 test", "top-1 test"]).to_excel(writer, sheet_name="Final", index=False)
    
    pd.DataFrame([{"lớp": "Chưa có data do test lỗi", "precision": "", "recall": "", "F1": ""}]).to_excel(writer, sheet_name="PerClass", index=False)
    pd.DataFrame([{"cấu hình": "Chưa chạy benchmark", "p50": "", "p95": ""}]).to_excel(writer, sheet_name="Latency", index=False)
    pd.DataFrame([{"exp_id": "F01", "macro-F1": final_data[0]['macro-F1 val'] if final_data else "", "độ trễ": "Chưa đo"}]).to_excel(writer, sheet_name="Summary", index=False)

print("results.xlsx created successfully using history logs!")
