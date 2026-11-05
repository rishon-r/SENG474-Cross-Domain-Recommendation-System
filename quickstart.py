"""Sample successful run with no download: synthetic data -> full pipeline -> results.

    python quickstart.py
"""
import subprocess
import sys

steps = [
    [sys.executable, "-m", "src.cmn.synthetic", "--out", "data/raw_synthetic"],
    [sys.executable, "-m", "src.main", "--config", "configs/quickstart.yaml", "--force"],
]
for cmd in steps:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
print("\nDone. See evaluation/results/quickstart/results_summary.csv and charts/ndcg_vs_overlap.png")
