"""Generate a small synthetic dataset in the Amazon 2018 JSON-lines format.

Users have latent tastes drawn from a few clusters; movie and book ratings are
different (linearly related) views of the same taste, so a mapping can learn it.
Used by the quickstart and the tests (no download needed).
"""
import argparse
import gzip
import json
import os

import numpy as np


def make(out_dir: str, n_users=1500, n_items=300, per_user=25, seed=0, overlap=0.6, clusters=3):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    d = 4
    centers = rng.normal(0, 1.5, (clusters, d))
    z = centers[rng.integers(0, clusters, n_users)] + rng.normal(0, 0.5, (n_users, d))
    A = rng.normal(0, 1, (n_items, d))                      # movie item factors
    B = (A @ rng.normal(0, 0.8, (d, d)).T) + rng.normal(0, 0.3, (n_items, d))  # book item factors
    in_movie = rng.random(n_users) < 0.9
    in_book = rng.random(n_users) < 0.9
    both = rng.random(n_users) < overlap
    in_movie |= both
    in_book |= both

    def write(path, prefix, F, member):
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for u in np.where(member)[0]:
                items = rng.choice(n_items, size=min(per_user, n_items), replace=False)
                for it in items:
                    r = 3.0 + 0.6 * (F[it] @ z[u]) + rng.normal(0, 0.4)
                    r = float(np.clip(np.round(r), 1, 5))
                    f.write(json.dumps({"reviewerID": f"U{u}", "asin": f"{prefix}{it}",
                                        "overall": r, "unixReviewTime": int(rng.integers(1e9, 1.6e9))}) + "\n")

    write(os.path.join(out_dir, "Movies_and_TV_5.json.gz"), "M", A, in_movie)
    write(os.path.join(out_dir, "Books_5.json.gz"), "B", B, in_book)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    make(a.out, seed=a.seed)
    print("synthetic data written to", a.out)
