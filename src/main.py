"""Pipeline entry point (Algorithm 2: RunExperiment).

    python -m src.main --config configs/default.yaml

Stages: parse data -> split (O_test removed from books) -> embeddings (movies, books) ->
for each overlap fraction and each strategy: train mapping, evaluate NDCG@n -> results + chart.
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from src.cmn.config import load_config
from src.cmn.seed import set_seed
from src.data.preprocess import config_hash, make_split, parse_and_cache
from src.embeddings.mf import train_mf
from src.evaluation.metrics import build_truth, ndcg_at_n
from src.evaluation.recommend import make_score_fn, popularity_score_fn, random_score_fn
from src.mappings.train import STRATEGIES


def log(msg):
    print(msg, flush=True)


def get_embeddings(cfg, data, split, seed, out_dir, force):
    # The cache name depends on everything the embeddings depend on (parsed data, split and
    # embedding settings, seed), so changing any of them retrains instead of reusing stale files.
    key = config_hash(dict(data=data["cache_key"], split=cfg["split"], embedding=cfg["embedding"], seed=seed))
    path = os.path.join(out_dir, f"embeddings_seed{seed}_{key}.npz")
    if os.path.exists(path) and not force:
        log(f"Loading embeddings from cache {path}")
        z = np.load(path)
        return z["U_movie"], z["U_book"], z["V_book"]
    e = cfg["embedding"]
    log("Training movie embeddings (all users, all movie ratings)")
    U_m, _ = train_mf(data["m_user"], data["m_item"], data["m_rating"], int(data["n_users"]), int(data["n_movies"]),
                      e["k"], e["lam"], e["lr"], e["batch_size"], e["epochs"], seed, log)
    log("Training book embeddings (book ratings with O_test removed)")
    bu, bi, br = split["b_train"]
    U_b, V_b = train_mf(bu, bi, br, int(data["n_users"]), int(data["n_books"]),
                        e["k"], e["lam"], e["lr"], e["batch_size"], e["epochs"], seed + 1, log)
    np.savez(path, U_movie=U_m, U_book=U_b, V_book=V_b)
    return U_m, U_b, V_b


def run(cfg, force=False):
    d, s, ev = cfg["data"], cfg["split"], cfg["eval"]
    out_dir = cfg["output"]["dir"]
    res_dir = cfg["output"]["results_dir"]
    for p in (out_dir, res_dir, os.path.join(res_dir, "charts"), cfg["output"]["ground_truth_dir"]):
        os.makedirs(p, exist_ok=True)
    if "max_rows" in d:
        raise ValueError("data.max_rows was replaced by data.user_frac (e.g. 0.05 for a debug run); "
                         "see configs/default.yaml")
    data = parse_and_cache(d["movies_path"], d["books_path"], d["preprocessed_dir"], d.get("user_frac"), force, log)
    n_books = int(data["n_books"])
    rows = []
    for seed in cfg["seeds"]:
        set_seed(seed)
        split = make_split(data, seed, s["test_frac"], tuple(s["fractions"]))
        np.savez(os.path.join(cfg["output"]["ground_truth_dir"], f"test_truth_seed{seed}.npz"),
                 user=split["truth"][0], item=split["truth"][1], rating=split["truth"][2])
        log(f"[seed {seed}] users={int(data['n_users'])} overlap={len(split['overlap'])} "
            f"rho_true={split['rho_true']:.3f} test_users={len(split['test_users'])}")
        U_m, U_b, V_b = get_embeddings(cfg, data, split, seed, out_dir, force)
        hist = np.bincount(split["b_train"][0], minlength=int(data["n_users"]))   # |H_b(u)| after the split
        test_users = split["test_users"]
        assert (hist[test_users] < ev["T"]).all(), "test users must be cold-start"
        truth = build_truth(split["truth"], test_users, n_books)

        def record(name, frac, size, score_fn, t0):
            v = ndcg_at_n(score_fn, test_users, truth, ev["n"], ev["user_batch"])
            rows.append(dict(seed=seed, strategy=name, fraction=frac, n_train_users=size, ndcg=v,
                             seconds=round(time.time() - t0, 2)))
            log(f"  {name:10s} frac={frac:<4} users={size:<7} NDCG@{ev['n']}={v:.4f}")

        record("popularity", "-", 0, popularity_score_fn(split["b_train"][1], n_books), time.time())
        record("random", "-", 0, random_score_fn(n_books, seed), time.time())
        for frac in s["fractions"]:
            O = split["subsets"][frac]
            X, Y = U_m[O], U_b[O]
            for name, cls in STRATEGIES.items():
                t0 = time.time()
                mapper = cls(cfg["embedding"]["k"], cfg["mapping"]).fit(X, Y, seed)
                record(name, frac, len(O), make_score_fn(mapper, U_m, U_b, V_b, hist, ev["T"]), t0)

    write_results(rows, res_dir, ev["n"])
    return rows


def write_results(rows, res_dir, n):
    path = os.path.join(res_dir, "results.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # mean/std over seeds + chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = {}
    for r in rows:
        summary.setdefault((r["strategy"], r["fraction"]), []).append(r["ndcg"])
    with open(os.path.join(res_dir, "results_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "fraction", "ndcg_mean", "ndcg_std", "n_seeds"])
        for (s, fr), v in summary.items():
            w.writerow([s, fr, round(float(np.mean(v)), 5), round(float(np.std(v)), 5), len(v)])
    fig, ax = plt.subplots(figsize=(5, 3.4))
    fracs = sorted({fr for (_, fr) in summary if fr != "-"})
    for s in ("shared", "cluster", "specific"):
        ys = [np.mean(summary[(s, fr)]) for fr in fracs]
        es = [np.std(summary[(s, fr)]) for fr in fracs]
        ax.errorbar([int(f * 100) for f in fracs], ys, yerr=es, marker="o", label=s, capsize=2)
    for b in ("popularity", "random"):
        if (b, "-") in summary:
            ax.axhline(np.mean(summary[(b, "-")]), ls="--", lw=1, label=b, color="gray" if b == "random" else "black")
    ax.set_xlabel("overlap users used for mapping (% of O_train)")
    ax.set_ylabel(f"NDCG@{n}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, "charts", "ndcg_vs_overlap.png"), dpi=150)
    log(f"Results written to {res_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="recompute cached parsed data and embeddings")
    a = ap.parse_args()
    run(load_config(a.config), a.force)
