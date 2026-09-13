import subprocess
import sys
import pandas as pd
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

K_VALUES = [1, 2, 3]
GNN_TYPES = ["gcn", "gat"]  # set to just ["gcn"] if you only want the k-hop study, not the GAT comparison

results_file = "outputs/results_log.csv"
if os.path.exists(results_file):
    os.remove(results_file)  # start clean so this sweep's results aren't mixed with earlier runs

for k in K_VALUES:
    print(f"\n{'='*70}\nBuilding ego networks for K_HOPS={k}\n{'='*70}")
    env = os.environ.copy()
    env["K_HOPS"] = str(k)
    subprocess.run([sys.executable, "sampling_and_features.py"], env=env, check=True, cwd=SCRIPT_DIR)

    for gnn_type in GNN_TYPES:
        print(f"\n{'='*70}\nTraining: K_HOPS={k}, gnn_type={gnn_type}\n{'='*70}")
        env = os.environ.copy()
        env["INSTANCES_PATH"] = f"data/instances_k{k}.pt"
        env["GNN_TYPE"] = gnn_type
        env["RESULTS_TAG"] = f"k{k}_{gnn_type}"
        subprocess.run([sys.executable, "train.py"], env=env, check=True, cwd=SCRIPT_DIR)

# ---- Summarize ----
df = pd.read_csv(results_file)
full_model_rows = df[df["model"] == "Full model (GNN + handcrafted)"].copy()
full_model_rows[["k_hops", "gnn_type_from_tag"]] = full_model_rows["tag"].str.extract(r"k(\d+)_(\w+)")

summary = full_model_rows.pivot(index="k_hops", columns="gnn_type_from_tag", values="auc")
print("\n" + "=" * 70)
print("SENSITIVITY SWEEP SUMMARY — Full model AUC-ROC by ego-network size (k)")
print("=" * 70)
print(summary.to_string())
print(f"\nFull results (all models, all runs) saved in: {results_file}")
print("Copy the table above straight into your report's ablation section.")
