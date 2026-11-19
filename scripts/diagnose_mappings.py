"""Diagnostic for the Discussion: why do the mappings rank like random?

    python -m scripts.diagnose_mappings --config configs/default.yaml [--seed 0] [--users 2000]

Reuses the cached embeddings of src.main (same split, seed and tuned settings), trains each
strategy on 100% of the training overlap users and reports, on a sample of test users:
  - mapping error vs. always predicting the mean book embedding (how much the mapping learns),
  - spread of the mapped vectors vs. the spread of real book embeddings,
  - how many distinct top-10 lists the users get and how popular the recommended books are.
Reads data only; writes evaluation/results/diagnostics.json.
"""
import argparse
import json
import os

import numpy as np
import torch

from src.cmn.config import load_config
from src.cmn.seed import set_seed
from src.data.preprocess import make_split, parse_and_cache
from src.evaluation.recommend import history_matrix, make_score_fn
from src.main import get_embeddings, load_tuned
from src.mappings.train import STRATEGIES


def main(cfg, seed, n_users):
    d, s, ev = cfg["data"], cfg["split"], cfg["eval"]
    tuned = load_tuned(cfg)
    data = parse_and_cache(d["movies_path"], d["books_path"], d["preprocessed_dir"], d.get("user_frac"), False, print)
    set_seed(seed)
    split = make_split(data, seed, s["test_frac"], tuple(s["fractions"]))
    U_m, U_b, V_b = get_embeddings(cfg, data, split, seed, cfg["output"]["dir"], False)
    n_u, n_b = int(data["n_users"]), int(data["n_books"])
    bu, bi, _ = split["b_train"]
    hist = np.bincount(bu, minlength=n_u)
    history = history_matrix(bu, bi, n_u, n_b)
    pop = np.bincount(bi, minlength=n_b)
    pop_top = set(np.argsort(-pop)[:ev["n"]].tolist())
    rng = np.random.default_rng(seed)
    test = np.sort(rng.choice(split["test_users"], size=min(n_users, len(split["test_users"])), replace=False))
    O = split["subsets"][1.0]
    X, Y = U_m[O], U_b[O]
    # "true" target embeddings of test users are not available (their books were removed),
    # so the error is measured on the training users the mapping was fit on (an upper bound on fit)
    mean_err = float(((Y - Y.mean(0)) ** 2).sum(1).mean())
    out = {"seed": seed, "n_test_users_sampled": len(test), "k": int(cfg["embedding"]["k"]),
           "book_emb_spread_train_users": round(float(np.sqrt(((Y - Y.mean(0)) ** 2).sum(1)).mean()), 4),
           "mse_predict_mean_book_embedding": round(mean_err, 4),
           "median_train_ratings_of_popularity_top10": int(np.median([pop[b] for b in pop_top]))}
    for name, cls in STRATEGIES.items():
        print(f"fitting {name} on {len(O)} users", flush=True)
        m = cls(cfg["embedding"]["k"], tuned.get(name, cfg["mapping"])).fit(X, Y, seed)
        fit_err = float(((m.map(X) - Y) ** 2).sum(1).mean())
        Z = m.map(U_m[test])
        spread = float(np.sqrt(((Z - Z.mean(0)) ** 2).sum(1)).mean())
        scores = make_score_fn(m, U_m, U_b, V_b, hist, ev["T"], history)
        tops = []
        for i in range(0, len(test), ev["user_batch"]):
            tops.append(torch.topk(scores(test[i:i + ev["user_batch"]]), ev["n"]).indices.numpy())
        tops = np.concatenate(tops)
        lists = {}
        for row in tops:
            lists[tuple(sorted(row.tolist()))] = lists.get(tuple(sorted(row.tolist())), 0) + 1
        books = tops.ravel()
        out[name] = {
            "train_mse": round(fit_err, 4),
            "mse_reduction_vs_mean_pct": round(100 * (1 - fit_err / mean_err), 1),
            "mapped_vector_spread": round(spread, 4),
            "distinct_top10_lists": len(lists),
            "share_users_with_most_common_list_pct": round(100 * max(lists.values()) / len(test), 1),
            "distinct_books_recommended": int(len(np.unique(books))),
            "median_train_ratings_of_recommended_books": int(np.median(pop[books])),
            "share_recommendations_in_popularity_top10_pct": round(100 * np.isin(books, list(pop_top)).mean(), 2),
        }
        print(name, out[name], flush=True)
    path = os.path.join(cfg["output"]["results_dir"], "diagnostics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"written to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--users", type=int, default=2000)
    a = ap.parse_args()
    main(load_config(a.config), a.seed, a.users)
